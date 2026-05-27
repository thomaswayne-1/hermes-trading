"""
news.py — crypto news sentiment.

Free public endpoint: CryptoPanic public API (no key needed for basic feed).
Premium override: set NEWS_API_KEY in .env for full NewsAPI.org access.

schema_version: "1"
"""
import os
import logging
from typing import Any

import httpx

log = logging.getLogger("hermes.adapters.news")

SCHEMA_VERSION = "1"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1"


async def fetch(asset: str) -> dict[str, Any]:
    """
    Returns a sentiment snapshot: bullish_count, bearish_count, neutral_count,
    sentiment_ratio (bullish / total), top_headline.
    """
    currency = asset.split("/")[0].upper()  # BTC/USDT → BTC

    if NEWS_API_KEY:
        return await _fetch_newsapi(currency)
    return await _fetch_cryptopanic(currency)


async def _fetch_cryptopanic(currency: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                f"{CRYPTOPANIC_BASE}/posts/",
                params={
                    "auth_token": "public",  # public read-only token
                    "currencies": currency,
                    "public": "true",
                    "filter": "trending",
                    "limit": 20,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            bullish = sum(1 for r in results if r.get("votes", {}).get("positive", 0) > r.get("votes", {}).get("negative", 0))
            bearish = sum(1 for r in results if r.get("votes", {}).get("negative", 0) > r.get("votes", {}).get("positive", 0))
            neutral = len(results) - bullish - bearish
            total = len(results) or 1

            top_headline = results[0].get("title", "") if results else ""

            return {
                "schema_version": SCHEMA_VERSION,
                "source": "cryptopanic",
                "currency": currency,
                "bullish_count": bullish,
                "bearish_count": bearish,
                "neutral_count": neutral,
                "sentiment_ratio": round(bullish / total, 3),
                "top_headline": top_headline,
                "article_count": len(results),
            }
        except Exception as exc:
            log.warning("CryptoPanic fetch failed: %s — returning stub", exc)
            return _stub(currency)


async def _fetch_newsapi(currency: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": currency + " cryptocurrency",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "apiKey": NEWS_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            top_headline = articles[0].get("title", "") if articles else ""

            return {
                "schema_version": SCHEMA_VERSION,
                "source": "newsapi",
                "currency": currency,
                "article_count": len(articles),
                "top_headline": top_headline,
                "bullish_count": None,   # NewsAPI requires NLP pass for sentiment
                "bearish_count": None,
                "neutral_count": None,
                "sentiment_ratio": None,
            }
        except Exception as exc:
            log.warning("NewsAPI fetch failed: %s — returning stub", exc)
            return _stub(currency)


def _stub(currency: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "stub",
        "currency": currency,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "sentiment_ratio": 0.5,
        "top_headline": "",
        "article_count": 0,
    }
