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
BINANCE_BASE = "https://api.binance.com"


async def fetch(asset: str) -> dict[str, Any]:
    """
    Returns:
      schema_version, symbol, close, high, low, open, volume,
      rsi (14-period, calculated from last 15 closes), timestamp
    """
    symbol = asset.replace("/", "")  # BTC/USDT → BTCUSDT

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{BINANCE_BASE}/api/v3/klines",
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
