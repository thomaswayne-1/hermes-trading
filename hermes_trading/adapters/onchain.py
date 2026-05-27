"""
onchain.py — on-chain metrics (BTC only by default).

Free public endpoint: blockchain.info public stats.
Premium override: set GLASSNODE_API_KEY in .env for richer data.

schema_version: "1"
"""
import os
import logging
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.onchain")

SCHEMA_VERSION = "1"
GLASSNODE_KEY = os.getenv("GLASSNODE_API_KEY", "")


async def fetch(asset: str) -> dict[str, Any]:
    """
    Returns basic on-chain metrics.
    Falls back to blockchain.info if no Glassnode key present.
    """
    if GLASSNODE_KEY and "BTC" in asset.upper():
        return await _fetch_glassnode(asset)
    return await _fetch_blockchain_info(asset)


async def _fetch_blockchain_info(asset: str) -> dict[str, Any]:
    """Free fallback — blockchain.info public stats (BTC only)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get("https://blockchain.info/stats?format=json")
            resp.raise_for_status()
            data = resp.json()
            return {
                "schema_version": SCHEMA_VERSION,
                "source": "blockchain.info",
                "asset": asset,
                "hash_rate_ghs": data.get("hash_rate", None),
                "difficulty": data.get("difficulty", None),
                "mempool_size": data.get("mempool_size", None),
                "n_tx_today": data.get("n_tx", None),
                "total_fees_btc": data.get("total_fees_btc", None),
            }
        except Exception as exc:
            log.warning("blockchain.info fetch failed: %s — returning stub", exc)
            return _stub(asset)


async def _fetch_glassnode(asset: str) -> dict[str, Any]:
    """Glassnode premium endpoint."""
    base = "https://api.glassnode.com/v1/metrics"
    ticker = asset.split("/")[0].upper()
    headers = {"X-Api-Key": GLASSNODE_KEY}
    params = {"a": ticker, "i": "24h"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            sopr_resp = await client.get(f"{base}/indicators/sopr", headers=headers, params=params)
            sopr_resp.raise_for_status()
            sopr_data = sopr_resp.json()
            sopr = sopr_data[-1]["v"] if sopr_data else None

            nvt_resp = await client.get(f"{base}/indicators/nvt", headers=headers, params=params)
            nvt_resp.raise_for_status()
            nvt_data = nvt_resp.json()
            nvt = nvt_data[-1]["v"] if nvt_data else None

            return {
                "schema_version": SCHEMA_VERSION,
                "source": "glassnode",
                "asset": asset,
                "sopr": sopr,
                "nvt": nvt,
            }
        except Exception as exc:
            log.warning("Glassnode fetch failed: %s — falling back to blockchain.info", exc)
            return await _fetch_blockchain_info(asset)


def _stub(asset: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "stub",
        "asset": asset,
        "hash_rate_ghs": None,
        "difficulty": None,
        "mempool_size": None,
    }
