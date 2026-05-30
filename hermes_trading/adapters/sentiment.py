"""
sentiment.py — Crypto Fear & Greed index (free, no key).

Endpoint: GET https://api.alternative.me/fng/?limit=1

Returns the most recent index value (0-100). Cached for 1 hour since
the index updates once daily.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.sentiment")

FNG_URL = "https://api.alternative.me/fng/"
SCHEMA_VERSION = "1"

_cache: dict[str, Any] = {"value": 50.0, "classification": "neutral", "ts": 0.0}
_TTL = 300    # 5 minutes


async def fetch(asset: str) -> dict[str, Any]:   # asset unused — FNG is BTC-wide
    now = time.time()
    if _cache["ts"] > 0 and (now - _cache["ts"] < _TTL):
        return {
            "schema_version": SCHEMA_VERSION,
            "fng_value": _cache["value"],
            "fng_classification": _cache["classification"],
            "source": "alternative_me_cached",
            "timestamp": int(_cache["ts"] * 1000),
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(FNG_URL, params={"limit": 1})
            resp.raise_for_status()
            data = resp.json()
        node = data.get("data", [])
        if not node:
            raise ValueError("Empty FNG response")
        value = float(node[0]["value"])
        classification = node[0].get("value_classification", "neutral")
        _cache.update({"value": value, "classification": classification, "ts": now})
        return {
            "schema_version": SCHEMA_VERSION,
            "fng_value": value,
            "fng_classification": classification,
            "source": "alternative_me",
            "timestamp": int(now * 1000),
        }
    except Exception as exc:
        log.warning("F&G fetch failed: %s — forward-fill last value (%.0f)", exc, _cache["value"])
        return {
            "schema_version": SCHEMA_VERSION,
            "fng_value": _cache["value"],
            "fng_classification": _cache["classification"],
            "source": "fallback",
            "timestamp": int(now * 1000),
        }
