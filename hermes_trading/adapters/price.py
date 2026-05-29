"""
price.py — real-time price feed with full indicator suite.

Every tick (10s):
  • Kraken Ticker   → current price, bid/ask spread
  • Kraken Depth    → order-book imbalance (bid pressure vs ask pressure)
  • Kraken OHLC     → 1-min candle history, cached 30s, used for all indicators

Indicators returned:
  rsi           — 14-period Wilder RSI
  macd          — MACD line (EMA12 - EMA26)
  macd_signal   — 9-period EMA of MACD
  macd_hist     — MACD histogram (macd - signal); positive = bullish momentum
  bb_upper/middle/lower — Bollinger Bands (20-period, 2 std dev)
  bb_pct        — %B: 0 = at lower band, 1 = at upper band, <0 = below band
  atr           — Average True Range (14-period); measures volatility in $ terms
  momentum      — Rate of change over 10 periods (%)
  spread_pct    — bid-ask spread as % of mid price
  ob_imbalance  — order book imbalance: +1 = all bids, -1 = all asks

schema_version: "2"
"""

import logging
import time
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.price")

SCHEMA_VERSION = "2"
KRAKEN_BASE    = "https://api.kraken.com"
BINANCE_US_BASE = "https://api.binance.us"

KRAKEN_PAIR_MAP = {
    "BTC/USDT": "XBTUSD",
    "ETH/USDT": "ETHUSD",
    "SOL/USDT": "SOLUSD",
    "BNB/USDT": "BNBUSD",
}

# ── OHLC cache (refreshed every 30s, shared across calls) ────────────────────
_ohlc_cache: dict = {"candles": [], "ts": 0.0, "pair": ""}
_OHLC_TTL = 30   # seconds


# ── Public entry point ────────────────────────────────────────────────────────

async def fetch(asset: str) -> dict[str, Any]:
    try:
        return await _fetch_kraken_full(asset)
    except Exception as exc:
        log.warning("Kraken full fetch failed (%s) — trying Binance.US", exc)
        try:
            return await _fetch_binance_us(asset)
        except Exception as exc2:
            log.error("All price adapters failed: %s", exc2)
            raise


# ── Kraken full fetch (ticker + depth + OHLC) ────────────────────────────────

async def _fetch_kraken_full(asset: str) -> dict[str, Any]:
    pair = KRAKEN_PAIR_MAP.get(asset, asset.replace("/", ""))

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Ticker — real-time price, bid, ask, volume
        ticker_resp = await client.get(
            f"{KRAKEN_BASE}/0/public/Ticker",
            params={"pair": pair},
        )
        ticker_resp.raise_for_status()
        ticker_data = ticker_resp.json()
        if ticker_data.get("error"):
            raise ValueError(f"Kraken ticker error: {ticker_data['error']}")

        tk_key = [k for k in ticker_data["result"] if k != "last"][0]
        tk = ticker_data["result"][tk_key]
        current_price = float(tk["c"][0])   # last trade price
        best_bid      = float(tk["b"][0])
        best_ask      = float(tk["a"][0])
        volume_24h    = float(tk["v"][1])   # 24h rolling volume

        # 2. Order book — top 10 levels for imbalance
        ob_imbalance = 0.0
        try:
            depth_resp = await client.get(
                f"{KRAKEN_BASE}/0/public/Depth",
                params={"pair": pair, "count": 10},
            )
            depth_resp.raise_for_status()
            depth_data = depth_resp.json()
            if not depth_data.get("error"):
                dk = [k for k in depth_data["result"] if k != "last"][0]
                bids = depth_data["result"][dk]["bids"]
                asks = depth_data["result"][dk]["asks"]
                bid_vol = sum(float(b[1]) for b in bids)
                ask_vol = sum(float(a[1]) for a in asks)
                total   = bid_vol + ask_vol
                ob_imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
        except Exception as e:
            log.debug("Depth fetch failed (non-fatal): %s", e)

        # 3. OHLC — cached, refreshed every 30s
        candles = await _get_ohlc(client, pair)

    # ── Indicators ────────────────────────────────────────────────────────────
    closes  = [float(c[4]) for c in candles]
    highs   = [float(c[2]) for c in candles]
    lows    = [float(c[3]) for c in candles]
    volumes = [float(c[6]) for c in candles]

    # Append current ticker price as synthetic latest close for fresher signals
    closes.append(current_price)
    highs.append(max(current_price, closes[-2] if len(closes) >= 2 else current_price))
    lows.append(min(current_price, closes[-2] if len(closes) >= 2 else current_price))

    rsi          = _rsi(closes)
    macd, sig, hist = _macd(closes)
    bb_u, bb_m, bb_l, bb_pct = _bollinger(closes)
    atr          = _atr(highs, lows, closes)
    momentum     = _momentum(closes)
    spread_pct   = ((best_ask - best_bid) / ((best_ask + best_bid) / 2)) * 100 if best_ask > 0 else 0.0
    current_volume = volumes[-1] if volumes else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol":         asset,
        "close":          current_price,
        "open":           float(candles[-1][1]) if candles else current_price,
        "high":           float(candles[-1][2]) if candles else current_price,
        "low":            float(candles[-1][3]) if candles else current_price,
        "volume":         current_volume,
        "volume_24h":     volume_24h,
        # ── indicators ──
        "rsi":            rsi,
        "macd":           round(macd, 4),
        "macd_signal":    round(sig, 4),
        "macd_hist":      round(hist, 4),
        "bb_upper":       round(bb_u, 2),
        "bb_middle":      round(bb_m, 2),
        "bb_lower":       round(bb_l, 2),
        "bb_pct":         round(bb_pct, 4),
        "atr":            round(atr, 2),
        "momentum":       round(momentum, 4),
        "spread_pct":     round(spread_pct, 6),
        "ob_imbalance":   round(ob_imbalance, 4),
        "source":         "kraken",
        "timestamp":      int(time.time() * 1000),
    }


async def _get_ohlc(client: httpx.AsyncClient, pair: str) -> list:
    """Return cached 1-min OHLC candles, refreshing if stale."""
    global _ohlc_cache
    now = time.time()
    if (
        _ohlc_cache["pair"] == pair
        and _ohlc_cache["candles"]
        and now - _ohlc_cache["ts"] < _OHLC_TTL
    ):
        return _ohlc_cache["candles"]

    resp = await client.get(
        f"{KRAKEN_BASE}/0/public/OHLC",
        params={"pair": pair, "interval": 1},
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise ValueError(f"Kraken OHLC error: {data['error']}")

    result_key = [k for k in data["result"] if k != "last"][0]
    candles = data["result"][result_key][-100:]   # last 100 1-min candles

    _ohlc_cache = {"candles": candles, "ts": now, "pair": pair}
    return candles


# ── Binance.US fallback ───────────────────────────────────────────────────────

async def _fetch_binance_us(asset: str) -> dict[str, Any]:
    symbol = asset.replace("/", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{BINANCE_US_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 100},
        )
        resp.raise_for_status()
        klines = resp.json()

    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    macd, sig, hist = _macd(closes)
    bb_u, bb_m, bb_l, bb_pct = _bollinger(closes)

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol":      asset,
        "close":       closes[-1],
        "open":        float(klines[-1][1]),
        "high":        highs[-1],
        "low":         lows[-1],
        "volume":      volumes[-1],
        "volume_24h":  0.0,
        "rsi":         _rsi(closes),
        "macd":        round(macd, 4),
        "macd_signal": round(sig, 4),
        "macd_hist":   round(hist, 4),
        "bb_upper":    round(bb_u, 2),
        "bb_middle":   round(bb_m, 2),
        "bb_lower":    round(bb_l, 2),
        "bb_pct":      round(bb_pct, 4),
        "atr":         round(_atr(highs, lows, closes), 2),
        "momentum":    round(_momentum(closes), 4),
        "spread_pct":  0.0,
        "ob_imbalance":0.0,
        "source":      "binance_us",
        "timestamp":   int(klines[-1][0]),
    }


# ── Indicator functions ───────────────────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> float:
    """Wilder RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_g  = sum(gains[:period]) / period
    avg_l  = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_g / avg_l)), 2)


def _ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average — returns full series."""
    if not values:
        return []
    k   = 2.0 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def _macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """MACD line, signal line, histogram."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    m  = macd_line[-1]
    s  = signal_line[-1]
    return m, s, m - s


def _bollinger(
    closes: list[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[float, float, float, float]:
    """Upper, middle, lower bands + %B."""
    if len(closes) < period:
        c = closes[-1] if closes else 0.0
        return c, c, c, 0.5
    window = closes[-period:]
    mean   = sum(window) / period
    std    = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
    upper  = mean + std_dev * std
    lower  = mean - std_dev * std
    pct_b  = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return upper, mean, lower, pct_b


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> float:
    """Average True Range."""
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    # Wilder smoothing
    atr = sum(trs[:period]) / min(period, len(trs))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _momentum(closes: list[float], period: int = 10) -> float:
    """Rate of change (%) over `period` bars."""
    if len(closes) <= period:
        return 0.0
    prev = closes[-(period + 1)]
    if prev == 0:
        return 0.0
    return (closes[-1] - prev) / prev * 100
