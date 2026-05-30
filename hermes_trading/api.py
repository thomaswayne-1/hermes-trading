"""
api.py — lightweight read-only HTTP API over the state directory.

Railway exposes this on a public URL so excel_tracker.py (or any client)
can read live state without SSH or file access.

Endpoints:
  GET /health          → {"status": "ok"}               (no auth)
  GET /state           → heartbeat + strategy + goal + model as one JSON blob
  GET /trades          → full trades.jsonl as JSON array

Auth: set API_SECRET env var in Railway. All protected endpoints require:
  X-API-Key: <API_SECRET>
If API_SECRET is empty, auth is disabled (local dev only).
"""

import json
import logging
import os
from pathlib import Path

import yaml
from aiohttp import web

log = logging.getLogger("hermes.api")

_state_env = os.getenv("STATE_DIR", "")
STATE_DIR = Path(_state_env) if _state_env else Path(__file__).parent.parent / "state"
API_SECRET = os.getenv("API_SECRET", "")
PORT = int(os.getenv("PORT", "8080"))   # Railway injects PORT automatically


# ── Auth ──────────────────────────────────────────────────────────────────────

def _authed(request: web.Request) -> bool:
    if not API_SECRET:
        return True
    return request.headers.get("X-API-Key") == API_SECRET


# ── State readers ─────────────────────────────────────────────────────────────

def _rj(name: str) -> dict:
    try:
        return json.loads((STATE_DIR / name).read_text())
    except Exception:
        return {}


def _rjsonl(name: str) -> list:
    p = STATE_DIR / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _ryaml(name: str) -> dict:
    try:
        return yaml.safe_load((STATE_DIR / name).read_text()) or {}
    except Exception:
        return {}


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_state(request: web.Request) -> web.Response:
    if not _authed(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return web.json_response({
        "heartbeat": _rj("heartbeat.json"),
        "strategy":  _ryaml("strategy.yaml"),
        "goal":      _ryaml("goal.yaml"),
        "model":     _rj("model.json"),
    })


async def handle_trades(request: web.Request) -> web.Response:
    if not _authed(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return web.json_response(_rjsonl("trades.jsonl"))


# ── Server lifecycle ──────────────────────────────────────────────────────────

async def start_server() -> None:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/state",  handle_state)
    app.router.add_get("/trades", handle_trades)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("API server listening on 0.0.0.0:%d", PORT)

    # Keep alive — this coroutine must never return while the server runs
    import asyncio
    while True:
        await asyncio.sleep(3600)
