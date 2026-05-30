"""
features.py — raw feature extraction + rolling z-scores (Layer A).

Maintains an in-memory deque of recent feature observations. Each tick,
new raw features are appended. Z-scores are computed on the rolling
window and clipped to [-3, +3].

Z-scoring puts every input on the same scale and absorbs the
non-stationarity of crypto, which is the standard quant multi-factor
preprocessing step.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any


EPS = 1e-9
CLIP = 3.0


class FeatureBuffer:
    """Rolling window of feature dicts for z-scoring."""

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self._buf: deque[dict[str, float]] = deque(maxlen=window)

    def append(self, raw: dict[str, float]) -> None:
        self._buf.append(raw)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        """At least 30 samples needed for usable z-scores."""
        return len(self._buf) >= 30

    def zscores(self, raw: dict[str, float]) -> dict[str, float]:
        """Return z-scored values for every key in `raw`, clipped to [-3, +3]."""
        out: dict[str, float] = {}
        if len(self._buf) < 5:
            return {k: 0.0 for k in raw}
        for key, val in raw.items():
            series = [b.get(key) for b in self._buf if b.get(key) is not None]
            if len(series) < 5:
                out[key] = 0.0
                continue
            mean = sum(series) / len(series)
            var = sum((x - mean) ** 2 for x in series) / max(1, len(series) - 1)
            std = math.sqrt(var) + EPS
            z = (val - mean) / std
            out[key] = max(-CLIP, min(CLIP, z))
        return out


# ── Indicator helpers (use cached OHLC candles) ──────────────────────────────

def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema_slope(values: list[float], period: int = 20, lookback: int = 5) -> float:
    """Slope of EMA(period) over the last `lookback` bars."""
    if len(values) < period + lookback:
        return 0.0
    e = ema(values, period)
    if len(e) < lookback + 1:
        return 0.0
    return e[-1] - e[-1 - lookback]


def adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Wilder's ADX. Returns 0 if not enough data."""
    n = len(closes)
    if n < period * 2 + 1:
        return 0.0
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm.append(up if up > dn and up > 0 else 0.0)
        minus_dm.append(dn if dn > up and dn > 0 else 0.0)
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    # Wilder smoothing
    def smooth(xs: list[float]) -> list[float]:
        if len(xs) < period:
            return []
        s = [sum(xs[:period])]
        for x in xs[period:]:
            s.append(s[-1] - s[-1] / period + x)
        return s
    atr_s   = smooth(tr)
    pdm_s   = smooth(plus_dm)
    mdm_s   = smooth(minus_dm)
    if not atr_s or atr_s[-1] == 0:
        return 0.0
    plus_di  = [100 * p / a for p, a in zip(pdm_s, atr_s) if a > 0]
    minus_di = [100 * m / a for m, a in zip(mdm_s, atr_s) if a > 0]
    if not plus_di or not minus_di:
        return 0.0
    dx = [
        100 * abs(p - m) / max(p + m, EPS)
        for p, m in zip(plus_di, minus_di)
    ]
    if len(dx) < period:
        return sum(dx) / len(dx)
    return sum(dx[-period:]) / period


def hurst_exponent(series: list[float], min_lag: int = 2, max_lag: int = 50) -> float:
    """Estimate the Hurst exponent via rescaled-range. H > 0.5 = trending."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    lags = range(min_lag, min(max_lag, n // 2))
    tau: list[float] = []
    valid_lags: list[float] = []
    for lag in lags:
        diffs = [series[i + lag] - series[i] for i in range(n - lag)]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)
        std = math.sqrt(max(var, EPS))
        if std <= 0:
            continue
        tau.append(math.log(std))
        valid_lags.append(math.log(lag))
    if len(tau) < 3:
        return 0.5
    # OLS slope
    xm = sum(valid_lags) / len(valid_lags)
    ym = sum(tau) / len(tau)
    num = sum((x - xm) * (y - ym) for x, y in zip(valid_lags, tau))
    den = sum((x - xm) ** 2 for x in valid_lags) + EPS
    slope = num / den
    return max(0.0, min(1.0, slope))


def realized_vol(closes: list[float], period: int = 20) -> float:
    """Realized vol of log returns over `period` bars (annualized roughly)."""
    if len(closes) < period + 1:
        return 0.0
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < period:
        return 0.0
    recent = rets[-period:]
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / max(1, len(recent) - 1)
    return math.sqrt(max(var, 0.0))


def vwap_distance(closes: list[float], volumes: list[float], period: int = 30) -> float:
    """(price − VWAP) / std_W(price). Positive means above VWAP."""
    if len(closes) < period or len(volumes) < period:
        return 0.0
    p = closes[-period:]
    v = volumes[-period:]
    vsum = sum(v)
    if vsum <= 0:
        return 0.0
    vwap = sum(pi * vi for pi, vi in zip(p, v)) / vsum
    mean = sum(p) / len(p)
    var = sum((x - mean) ** 2 for x in p) / max(1, len(p) - 1)
    std = math.sqrt(max(var, EPS))
    return (closes[-1] - vwap) / std


def rate_of_change(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 0.0
    prev = closes[-(period + 1)]
    if prev == 0:
        return 0.0
    return (closes[-1] - prev) / prev


# ── Top-level: build the full feature dict from price_data + extras ──────────

def extract_raw_features(
    price_data: dict[str, Any],
    candles: list[list[Any]] | None,
    funding_rate: float = 0.0,
    fng_value: float = 50.0,
    ls_ratio: float = 1.0,
    top_ls_ratio: float = 1.0,
    taker_buy_ratio: float = 0.5,
    oi_pct_change: float = 0.0,
) -> dict[str, float]:
    """
    Compute the raw (un-z-scored) feature dict.

    `candles` is the list of [time, open, high, low, close, vwap, volume, count]
    from Kraken (the same format the price adapter caches). If unavailable
    or too short, falls back to only the single-bar values from price_data.
    """
    closes  = [float(c[4]) for c in (candles or [])]
    highs   = [float(c[2]) for c in (candles or [])]
    lows    = [float(c[3]) for c in (candles or [])]
    volumes = [float(c[6]) for c in (candles or [])]

    # Append the live price so freshness is captured
    live_price = float(price_data.get("close", 0.0))
    if live_price > 0:
        closes.append(live_price)
        highs.append(max(live_price, closes[-2] if len(closes) >= 2 else live_price))
        lows.append(min(live_price, closes[-2] if len(closes) >= 2 else live_price))
        volumes.append(float(price_data.get("volume", volumes[-1] if volumes else 0.0)))

    # Momentum features
    macd_hist = float(price_data.get("macd_hist", 0.0))
    roc_14    = rate_of_change(closes, 14)
    ema20_sl  = ema_slope(closes, 20, 5)
    adx_val   = adx(highs, lows, closes, 14) if len(closes) >= 30 else 0.0
    if len(closes) >= 50:
        e20 = ema(closes, 20)[-1]
        e50 = ema(closes, 50)[-1]
        signed_adx = adx_val * (1.0 if e20 >= e50 else -1.0)
    else:
        signed_adx = 0.0

    # Mean-reversion features
    rsi    = float(price_data.get("rsi", 50.0))
    rsi_mr = (50.0 - rsi) / 50.0
    bb_pct = float(price_data.get("bb_pct", 0.5))
    bb_mr  = (0.5 - bb_pct) * 2.0
    vwap_d = vwap_distance(closes, volumes, 30) if len(closes) >= 30 else 0.0

    # Microstructure
    ob_imb = float(price_data.get("ob_imbalance", 0.0))
    # Taker buy/sell delta not exposed by Kraken OHLC — leave 0
    vol_delta = 0.0

    # Sentiment (contrarian + positioning)
    funding_neg    = -funding_rate
    fng_contrarian = -(fng_value - 50.0) / 50.0

    # Retail long/short: > 1 means crowd is net long → contrarian bearish
    # Normalise: ratio of 1.5 (50% more longs than shorts) → signal of -0.5
    ls_contra      = -(ls_ratio - 1.0)

    # Top-trader (smart money): follow, not contrarian
    # ratio > 1 = whales net long → bullish
    top_ls_smart   = top_ls_ratio - 1.0

    # Taker aggression: > 0.5 = more aggressive buyers → bullish microstructure
    taker_buy      = taker_buy_ratio - 0.5

    # Open interest change: positive = new money entering (conviction)
    # keep as-is; z-score will scale it
    oi_change      = oi_pct_change

    # Context (never directs C; feeds K and sizing)
    rv     = realized_vol(closes, 20)
    atr_p  = float(price_data.get("atr", 0.0)) / max(live_price, EPS) if live_price > 0 else 0.0
    vol_surge = 0.0  # populated upstream from volume history

    return {
        # momentum
        "macd_hist":   macd_hist,
        "roc_14":      roc_14,
        "ema20_slope": ema20_sl,
        "signed_adx":  signed_adx,
        # mean reversion
        "rsi_mr":      rsi_mr,
        "bb_mr":       bb_mr,
        "vwap_dist":   -vwap_d,   # negate so far-above-vwap reads bearish (mean reversion)
        # microstructure
        "ob_imb":      ob_imb,
        "vol_delta":   vol_delta,
        "taker_buy":   taker_buy,   # taker aggression: +ve = more aggressive buyers
        "oi_change":   oi_change,   # OI % change: +ve = new money entering
        # sentiment (already contrarian-signed)
        "funding_neg": funding_neg,
        "fng_contra":  fng_contrarian,
        "ls_contra":   ls_contra,   # retail crowd contrarian: +ve = crowd short = bullish
        "top_ls_smart": top_ls_smart, # smart money: +ve = whales long = bullish
        # context (not in C, only in K / sizing)
        "realized_vol": rv,
        "atr_pct":      atr_p,
        "vol_surge":    vol_surge,
    }
