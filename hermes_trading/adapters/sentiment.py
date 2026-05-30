"""
sentiment.py — multi-source sentiment + positioning data.

Sources (all free, no API key required):
  1. Crypto Fear & Greed Index  — alternative.me, updates once/day,   TTL 1h
  2. Global Long/Short Ratio    — Binance futures,  5-min periods,     TTL 5m
  3. Top-Trader Long/Short      — Binance futures,  5-min periods,     TTL 5m
  4. Taker Buy/Sell Ratio       — Binance futures,  5-min periods,     TTL 5m
  5. Open Interest              — Binance futures,  real-time,         TTL 5m

Returned keys
─────────────
  fng_value        float  0-100  (50=neutral)
  fng_classification str
  ls_ratio         float  global long/short account ratio  (>1 = more longs)
  top_ls_ratio     float  top-trader long/short ratio
  taker_buy_ratio  float  taker buy / (buy+sell) volume    (>0.5 = buy-heavy)
  oi_usd           float  open interest in USD
  oi_pct_change    float  % change vs prior sample         (+ve = positions opening)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.sentiment")

SCHEMA_VERSION = "2"
BINANCE_FDATA  = "https://fapi.binance.com/futures/data"
BINANCE_FAPI   = "https://fapi.binance.com/fapi/v1"
FNG_URL        = "https://api.alternative.me/fng/"

_TTL_FNG  = 3600   # 1 hour  — FNG updates once/day
_TTL_LIVE = 300    # 5 min   — positioning data

PAIR_MAP = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
}

# ── per-source caches ─────────────────────────────────────────────────────────

_fng_cache:  dict[str, Any] = {"value": 50.0, "classification": "neutral", "ts": 0.0}
_ls_cache:   dict[str, Any] = {"ratio": 1.0, "ts": 0.0, "symbol": ""}
_top_cache:  dict[str, Any] = {"ratio": 1.0, "ts": 0.0, "symbol": ""}
_taker_cache: dict[str, Any] = {"buy_ratio": 0.5, "ts": 0.0, "symbol": ""}
_oi_cache:   dict[str, Any] = {"oi": 0.0, "prev_oi": 0.0, "ts": 0.0, "symbol": ""}


# ── individual fetchers ───────────────────────────────────────────────────────

async def _fetch_fng(client: httpx.AsyncClient) -> None:
    now = time.time()
    if _fng_cache["ts"] > 0 and now - _fng_cache["ts"] < _TTL_FNG:
        return
    try:
        r = await client.get(FNG_URL, params={"limit": 1}, timeout=8.0)
        r.raise_for_status()
        node = r.json().get("data", [])
        if node:
            _fng_cache["value"]          = float(node[0]["value"])
            _fng_cache["classification"] = node[0].get("value_classification", "neutral")
            _fng_cache["ts"]             = now
    except Exception as exc:
        log.debug("FNG fetch failed (using cached %.0f): %s", _fng_cache["value"], exc)


async def _fetch_ls(client: httpx.AsyncClient, symbol: str) -> None:
    now = time.time()
    if _ls_cache["symbol"] == symbol and now - _ls_cache["ts"] < _TTL_LIVE:
        return
    try:
        r = await client.get(
            f"{BINANCE_FDATA}/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": "5m", "limit": 1},
            timeout=8.0,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            _ls_cache["ratio"]  = float(data[0]["longShortRatio"])
            _ls_cache["ts"]     = now
            _ls_cache["symbol"] = symbol
    except Exception as exc:
        log.debug("Global L/S fetch failed: %s", exc)


async def _fetch_top_ls(client: httpx.AsyncClient, symbol: str) -> None:
    now = time.time()
    if _top_cache["symbol"] == symbol and now - _top_cache["ts"] < _TTL_LIVE:
        return
    try:
        r = await client.get(
            f"{BINANCE_FDATA}/topLongShortAccountRatio",
            params={"symbol": symbol, "period": "5m", "limit": 1},
            timeout=8.0,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            _top_cache["ratio"]  = float(data[0]["longShortRatio"])
            _top_cache["ts"]     = now
            _top_cache["symbol"] = symbol
    except Exception as exc:
        log.debug("Top trader L/S fetch failed: %s", exc)


async def _fetch_taker(client: httpx.AsyncClient, symbol: str) -> None:
    now = time.time()
    if _taker_cache["symbol"] == symbol and now - _taker_cache["ts"] < _TTL_LIVE:
        return
    try:
        r = await client.get(
            f"{BINANCE_FDATA}/takerlongshortRatio",
            params={"symbol": symbol, "period": "5m", "limit": 1},
            timeout=8.0,
        )
        r.raise_for_status()
        data = r.json()
        if data:
            buy_vol  = float(data[0].get("buyVol",  0.0))
            sell_vol = float(data[0].get("sellVol", 0.0))
            total = buy_vol + sell_vol
            _taker_cache["buy_ratio"] = buy_vol / total if total > 0 else 0.5
            _taker_cache["ts"]        = now
            _taker_cache["symbol"]    = symbol
    except Exception as exc:
        log.debug("Taker ratio fetch failed: %s", exc)


async def _fetch_oi(client: httpx.AsyncClient, symbol: str) -> None:
    now = time.time()
    if _oi_cache["symbol"] == symbol and now - _oi_cache["ts"] < _TTL_LIVE:
        return
    try:
        r = await client.get(
            f"{BINANCE_FAPI}/openInterest",
            params={"symbol": symbol},
            timeout=8.0,
        )
        r.raise_for_status()
        oi = float(r.json()["openInterest"])
        prev = _oi_cache["oi"] if _oi_cache["oi"] > 0 else oi
        _oi_cache["prev_oi"] = prev
        _oi_cache["oi"]      = oi
        _oi_cache["ts"]      = now
        _oi_cache["symbol"]  = symbol
    except Exception as exc:
        log.debug("OI fetch failed: %s", exc)


# ── public entry point ────────────────────────────────────────────────────────

async def fetch(asset: str) -> dict[str, Any]:
    symbol = PAIR_MAP.get(asset, asset.replace("/", ""))
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            _fetch_fng(client),
            _fetch_ls(client, symbol),
            _fetch_top_ls(client, symbol),
            _fetch_taker(client, symbol),
            _fetch_oi(client, symbol),
            return_exceptions=True,   # never raise — sentiment is non-blocking
        )

    oi_now  = _oi_cache["oi"]
    oi_prev = _oi_cache["prev_oi"] if _oi_cache["prev_oi"] > 0 else oi_now
    oi_pct  = (oi_now - oi_prev) / max(oi_prev, 1e-9)

    return {
        "schema_version":    SCHEMA_VERSION,
        # Fear & Greed
        "fng_value":         _fng_cache["value"],
        "fng_classification": _fng_cache["classification"],
        # Positioning
        "ls_ratio":          _ls_cache["ratio"],
        "top_ls_ratio":      _top_cache["ratio"],
        # Taker aggression
        "taker_buy_ratio":   _taker_cache["buy_ratio"],
        # Open interest
        "oi_usd":            oi_now,
        "oi_pct_change":     oi_pct,
    }
