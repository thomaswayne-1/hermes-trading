"""
backup.py — GitHub Gist backup for trade history.

Stores the entire trades.jsonl as a private Gist so trade history
survives Railway volume resets, redeploys, and any other data-loss event.

Environment:
    GITHUB_TOKEN — personal access token with `gist` scope.
                   If unset, backup is silently skipped everywhere.

Gist management:
    - On first push  : creates a new private Gist; saves its ID to gist_id.txt.
    - Subsequent push: PATCHes the existing Gist (overwrites file content).
    - Startup restore: if trades.jsonl is empty, fetches from the Gist.
    - Self-healing   : if gist_id.txt is missing (volume wiped), scans the
                       authenticated user's Gists for one containing
                       hermes_trades.jsonl and rebuilds gist_id.txt.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx

# Module-level lock: prevents two concurrent push_to_gist calls from both
# seeing gist_id = "" and independently POSTing to create duplicate Gists.
_gist_create_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Return (creating if needed) the module-level asyncio.Lock."""
    global _gist_create_lock
    if _gist_create_lock is None:
        _gist_create_lock = asyncio.Lock()
    return _gist_create_lock

log = logging.getLogger("hermes.backup")

GIST_FILENAME = "hermes_trades.jsonl"
GIST_DESC     = "Hermes Trading — trade history backup"
GITHUB_API    = "https://api.github.com"
TIMEOUT       = 20.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gist_id_path(trades_file: Path) -> Path:
    return trades_file.parent / "gist_id.txt"


def _find_existing_gist(token: str) -> str | None:
    """
    Scan the authenticated user's Gists (up to 300) for one containing
    hermes_trades.jsonl.  Used when gist_id.txt is missing after a volume wipe.
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            for page in range(1, 4):
                resp = client.get(
                    f"{GITHUB_API}/gists",
                    headers=_headers(token),
                    params={"per_page": 100, "page": page},
                )
                resp.raise_for_status()
                gists = resp.json()
                if not gists:
                    break
                for g in gists:
                    if GIST_FILENAME in g.get("files", {}):
                        log.info("BACKUP | Found existing Gist %s via scan", g["id"])
                        return g["id"]
    except Exception as exc:
        log.warning("BACKUP | Gist scan failed: %s", exc)
    return None


def _resolve_gist_id(trades_file: Path, token: str) -> str:
    """
    Return the known Gist ID from gist_id.txt, or search GitHub if the file
    is missing (e.g. after a volume wipe).  Returns "" if none found.
    """
    id_path = _gist_id_path(trades_file)
    if id_path.exists():
        gist_id = id_path.read_text().strip()
        if gist_id:
            return gist_id

    # gist_id.txt is gone — try to recover via API scan
    gist_id = _find_existing_gist(token) or ""
    if gist_id:
        id_path.write_text(gist_id)   # rebuild the file
    return gist_id


# ── Synchronous restore (called at startup, before the async event loop) ──────

def restore_from_gist(trades_file: Path, *, init_byte_size: int = -1) -> int:
    """
    If trades_file is empty AND GITHUB_TOKEN is set, fetch trade history from
    the backup Gist and write it to trades_file.

    init_byte_size: byte count of trades_file measured at process start
        (before the trading loop began writing).  Pass this from the caller
        to avoid a race where tick 1 writes a trade before this function runs,
        causing the guard to incorrectly skip restore.
        If -1 (default / caller didn't pass it), fall back to current size.

    Returns the number of trade lines restored (0 if nothing done).
    """
    token = _token()
    if not token:
        return 0

    # Guard: only restore into a file that was empty when this process started.
    # Using init_byte_size (snapshotted before the trading loop started) rather
    # than current size prevents a race where tick 1 beats _startup_restore and
    # permanently locks out the Gist restore.
    check_size = init_byte_size if init_byte_size >= 0 else (
        trades_file.stat().st_size if trades_file.exists() else 0
    )
    if check_size > 0:
        log.debug("BACKUP | trades.jsonl had %d bytes at startup — skipping restore", check_size)
        return 0

    gist_id = _resolve_gist_id(trades_file, token)
    if not gist_id:
        log.info("BACKUP | No existing Gist found — starting fresh (first run or no history)")
        return 0

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                f"{GITHUB_API}/gists/{gist_id}",
                headers=_headers(token),
            )
            resp.raise_for_status()
            data = resp.json()

        file_obj = data.get("files", {}).get(GIST_FILENAME)
        if not file_obj:
            log.warning("BACKUP | Gist %s has no %s file", gist_id, GIST_FILENAME)
            return 0

        content = file_obj.get("content", "")
        if not content.strip():
            return 0

        trades_file.parent.mkdir(parents=True, exist_ok=True)
        trades_file.write_text(content)
        n = len([l for l in content.splitlines() if l.strip()])
        log.info("BACKUP | ✓ Restored %d trade(s) from Gist %s", n, gist_id)
        return n

    except Exception as exc:
        log.warning("BACKUP | Restore failed (non-fatal): %s", exc)
        return 0


# ── Asynchronous push (called fire-and-forget after each trade close) ─────────

async def push_to_gist(trades_file: Path) -> None:
    """
    Push the current trades.jsonl to GitHub Gist.
    Creates a new private Gist on first call; PATCHes the existing one thereafter.
    All errors are logged as warnings — this function never raises.
    """
    token = _token()
    if not token:
        return

    if not trades_file.exists():
        return

    content = trades_file.read_text()
    if not content.strip():
        return

    # _resolve_gist_id uses synchronous httpx — run in a thread so we don't
    # block the event loop while scanning GitHub Gists.
    try:
        loop = asyncio.get_running_loop()
        gist_id = await loop.run_in_executor(None, _resolve_gist_id, trades_file, token)
    except Exception:
        gist_id = ""
    n_lines = len([l for l in content.splitlines() if l.strip()])

    payload: dict = {
        "files": {GIST_FILENAME: {"content": content}}
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if gist_id:
                resp = await client.patch(
                    f"{GITHUB_API}/gists/{gist_id}",
                    headers=_headers(token),
                    json=payload,
                )
                resp.raise_for_status()
                log.info("BACKUP | ✓ Pushed %d trade(s) to Gist %s", n_lines, gist_id)
            else:
                # Serialise creation: hold the module-level lock so two
                # concurrent fire-and-forget push_to_gist calls cannot both
                # see gist_id="" and independently POST two new Gists.
                async with _get_lock():
                    # Re-check inside the lock — another coroutine may have
                    # just created the Gist and written gist_id.txt.
                    try:
                        gist_id = await loop.run_in_executor(
                            None, _resolve_gist_id, trades_file, token
                        )
                    except Exception:
                        gist_id = ""

                    if gist_id:
                        # Another coroutine beat us; just PATCH instead.
                        resp = await client.patch(
                            f"{GITHUB_API}/gists/{gist_id}",
                            headers=_headers(token),
                            json=payload,
                        )
                        resp.raise_for_status()
                        log.info("BACKUP | ✓ Pushed %d trade(s) to Gist %s (lock re-check)", n_lines, gist_id)
                    else:
                        payload["description"] = GIST_DESC
                        payload["public"]      = False
                        resp = await client.post(
                            f"{GITHUB_API}/gists",
                            headers=_headers(token),
                            json=payload,
                        )
                        resp.raise_for_status()
                        new_id = resp.json().get("id", "")
                        if new_id:
                            _gist_id_path(trades_file).write_text(new_id)
                            log.info("BACKUP | ✓ Created Gist %s with %d trade(s)", new_id, n_lines)

    except Exception as exc:
        log.warning("BACKUP | Push failed (non-fatal): %s", exc)
