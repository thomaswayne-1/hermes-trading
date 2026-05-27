"""
macro.py — macro / market context signals.

Free public endpoint: Federal Reserve FRED API (no key for basic) +
                      Yahoo Finance via yfinance for DXY, VIX, SPY.

schema_version: "1"
"""
import logging
from typing import Any

log = logging.getLogger("hermes.adapters.macro")

SCHEMA_VERSION = "1"


async def fetch(asset: str) -> dict[str, Any]:
    """
    Returns: dxy_close, vix_close, spx_close, btc_dominance, timestamp.
    Uses yfinance (sync) wrapped in executor to stay async-compatible.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_sync, asset)


def _fetch_sync(asset: str) -> dict[str, Any]:
    try:
        import yfinance as yf
        tickers = yf.download(
            tickers="DX-Y.NYB VIX ^GSPC",
            period="2d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        def last_close(t: str) -> float | None:
            try:
                col = ("Close", t)
                if col in tickers.columns:
                    series = tickers[col].dropna()
                    return float(series.iloc[-1]) if not series.empty else None
                return None
            except Exception:
                return None

        return {
            "schema_version": SCHEMA_VERSION,
            "source": "yfinance",
            "asset": asset,
            "dxy_close":  last_close("DX-Y.NYB"),
            "vix_close":  last_close("VIX"),
            "spx_close":  last_close("^GSPC"),
        }
    except Exception as exc:
        log.warning("yfinance macro fetch failed: %s — returning stub", exc)
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "stub",
            "asset": asset,
            "dxy_close":  None,
            "vix_close":  None,
            "spx_close":  None,
        }
