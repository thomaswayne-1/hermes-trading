"""
funding.py — Binance USDT-perp funding rate (free, no key).

Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1

Funding rate is per 8h period. Returns the most recent settled rate.
Cached for 5 minutes — funding doesn't change tick to tick.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.funding")

BINANCE_FAPI = "https://fapi.binance.com"
SCHEMA_VERSION = "1"

_cache: dict[str, Any] = {"rate": 0.0, "ts": 0.0, "symbol": ""}
_TTL = 300  # 5 minutes

PAIR_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "BNB/USDT": "BNBUSDT",
}


async def fetch(asset: str) -> dict[str, Any]:
    symbol = PAIR_MAP.get(asset, asset.replace("/", ""))
    now = time.time()
    if (
        _cache["symbol"] == symbol
        and now - _cache["ts"] < _TTL
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": asset,
            "funding_rate": _cache["rate"],
            "source": "binance_fapi_cached",
            "timestamp": int(_cache["ts"] * 1000),
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1},
            )
            resp.raise_for_status()
            data = resp.json()
        if not data:
            raise ValueError("Empty funding response")
        rate = float(data[0]["fundingRate"])
        _cache.update({"rate": rate, "ts": now, "symbol": symbol})
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": asset,
            "funding_rate": rate,
            "source": "binance_fapi",
            "timestamp": int(now * 1000),
        }
    except Exception as exc:
        log.warning("Funding fetch failed: %s — falling back to last cached/zero", exc)
        return {
            "schema_version": SCHEMA_VERSION,
            "symbol": asset,
            "funding_rate": _cache.get("rate", 0.0),
            "source": "fallback",
            "timestamp": int(now * 1000),
        }
