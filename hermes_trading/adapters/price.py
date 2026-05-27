"""
price.py — pulls OHLCV + RSI for the configured asset.

Free public endpoint: Binance public REST API (no key needed for spot klines).
Premium override: set EXCHANGE_API_KEY + EXCHANGE_API_SECRET in .env.

schema_version: "1"
"""
import os
import logging
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.price")

SCHEMA_VERSION = "1"

# Binance.com is geo-blocked on US IPs (HTTP 451).
# Binance.US is the compliant endpoint for US-hosted servers.
# Kraken is the fallback if Binance.US also fails.
BINANCE_US_BASE = "https://api.binance.us"
KRAKEN_BASE = "https://api.kraken.com"

# Map common tickers to Kraken pairs
KRAKEN_PAIR_MAP = {
    "BTC/USDT": "XBTUSD",
    "ETH/USDT": "ETHUSD",
    "SOL/USDT": "SOLUSD",
    "BNB/USDT": "BNBUSD",
}


async def fetch(asset: str) -> dict[str, Any]:
    """
    Returns:
      schema_version, symbol, close, high, low, open, volume,
      rsi (14-period, calculated from last 15 closes), timestamp

    Tries Binance.US first; falls back to Kraken on failure.
    """
    try:
        return await _fetch_binance_us(asset)
    except Exception as exc:
        log.warning("Binance.US failed (%s) — trying Kraken", exc)
        return await _fetch_kraken(asset)


async def _fetch_binance_us(asset: str) -> dict[str, Any]:
    symbol = asset.replace("/", "")  # BTC/USDT → BTCUSDT

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{BINANCE_US_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 15},
        )
        resp.raise_for_status()
        klines = resp.json()

    closes = [float(k[4]) for k in klines]
    highs  = [float(k[2]) for k in klines]
    lows   = [float(k[3]) for k in klines]
    opens  = [float(k[1]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    rsi = _rsi14(closes)

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": asset,
        "close":  closes[-1],
        "open":   opens[-1],
        "high":   highs[-1],
        "low":    lows[-1],
        "volume": volumes[-1],
        "rsi":    rsi,
        "timestamp": klines[-1][0],
        "source": "binance_us",
    }


async def _fetch_kraken(asset: str) -> dict[str, Any]:
    pair = KRAKEN_PAIR_MAP.get(asset, asset.replace("/", ""))

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{KRAKEN_BASE}/0/public/OHLC",
            params={"pair": pair, "interval": 1},
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")

    # Kraken returns {result: {PAIR: [[time,open,high,low,close,vwap,vol,count], ...]}}
    result_key = [k for k in data["result"] if k != "last"][0]
    candles = data["result"][result_key][-15:]  # last 15 candles

    closes  = [float(c[4]) for c in candles]
    opens   = [float(c[1]) for c in candles]
    highs   = [float(c[2]) for c in candles]
    lows    = [float(c[3]) for c in candles]
    volumes = [float(c[6]) for c in candles]

    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": asset,
        "close":  closes[-1],
        "open":   opens[-1],
        "high":   highs[-1],
        "low":    lows[-1],
        "volume": volumes[-1],
        "rsi":    _rsi14(closes),
        "timestamp": int(candles[-1][0]) * 1000,
        "source": "kraken",
    }


def _rsi14(closes: list[float], period: int = 14) -> float:
    """Standard Wilder RSI from a list of closing prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)
