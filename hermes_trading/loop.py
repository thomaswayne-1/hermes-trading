"""
loop.py — 24/7 async reliability loop.

Every 60 s:
  1. Pull data from all adapters (with per-adapter retries + circuit-breaker).
  2. Load current strategy from state/strategy.yaml.
  3. Evaluate entry condition; paper-trade if it fires.
  4. Score the trade via score.py.
  5. Append to state/trades.jsonl.
  6. Write heartbeat to state/heartbeat.json.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.adapters.macro import fetch as fetch_macro
from hermes_trading.adapters.news import fetch as fetch_news
from hermes_trading.adapters.onchain import fetch as fetch_onchain
from hermes_trading.adapters.price import fetch as fetch_price
from hermes_trading.score import score

log = logging.getLogger("hermes.loop")

TICK_SECONDS = 60
MAX_ADAPTER_RETRIES = 3
CIRCUIT_BREAK_THRESHOLD = 5


class SchemaError(Exception):
    pass


class TradingLoop:
    def __init__(self, asset: str, mode: str, state_dir: Path, goal: dict) -> None:
        self.asset = asset
        self.mode = mode
        self.state_dir = state_dir
        self.goal = goal
        self.strategy_file = state_dir / "strategy.yaml"
        self.trades_file = state_dir / "trades.jsonl"
        self.heartbeat_file = state_dir / "heartbeat.json"
        self._consecutive_failures = 0
        self._open_trade: dict | None = None

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        log.info("Loop started — tick every %ds", TICK_SECONDS)
        while True:
            try:
                await self._tick()
                self._consecutive_failures = 0
            except SchemaError as exc:
                log.error("SCHEMA ERROR — halting loop: %s", exc)
                raise
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                log.warning(
                    "Tick error (%d/%d): %s",
                    self._consecutive_failures,
                    CIRCUIT_BREAK_THRESHOLD,
                    exc,
                )
                if self._consecutive_failures >= CIRCUIT_BREAK_THRESHOLD:
                    log.error("Circuit breaker tripped — halting. Fix the underlying issue.")
                    raise RuntimeError("Circuit breaker tripped") from exc
            await asyncio.sleep(TICK_SECONDS)

    # ------------------------------------------------------------------ #
    #  Single tick                                                         #
    # ------------------------------------------------------------------ #

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc).isoformat()

        # 1. Fetch all adapter data with retries
        data = await self._fetch_all()

        # 2. Load current strategy
        strategy = self._load_strategy()

        # 3. Evaluate entry / exit
        price_data = data.get("price", {})
        current_price = price_data.get("close", 0.0)
        rsi = price_data.get("rsi", 50.0)

        trade_event = None

        if self._open_trade is None:
            # Look for entry
            if self._entry_fires(strategy, rsi):
                self._open_trade = {
                    "id": f"T{int(time.time())}",
                    "asset": self.asset,
                    "entry_price": current_price,
                    "entry_time": now,
                    "direction": strategy["entry"]["direction"],
                    "stop_loss_pct": strategy["stop_loss_pct"],
                    "position_size_r": strategy["position_size_r"],
                    "strategy_version": strategy["version"],
                    "mode": self.mode,
                }
                log.info("ENTRY | price=%.2f rsi=%.1f trade_id=%s", current_price, rsi, self._open_trade["id"])
        else:
            # Check for stop-loss exit
            entry_price = self._open_trade["entry_price"]
            stop_pct = self._open_trade["stop_loss_pct"] / 100.0
            direction = self._open_trade["direction"]

            if direction == "long":
                stop_price = entry_price * (1 - stop_pct)
                hit_stop = current_price <= stop_price
            else:
                stop_price = entry_price * (1 + stop_pct)
                hit_stop = current_price >= stop_price

            # Simple time-based exit after 24h if no stop hit (paper simplification)
            open_seconds = time.time() - int(self._open_trade["id"][1:])
            time_exit = open_seconds >= 86400

            if hit_stop or time_exit:
                exit_reason = "stop_loss" if hit_stop else "time_exit"
                pnl_pct = (
                    (current_price - entry_price) / entry_price
                    if direction == "long"
                    else (entry_price - current_price) / entry_price
                )
                trade_event = {
                    **self._open_trade,
                    "exit_price": current_price,
                    "exit_time": now,
                    "exit_reason": exit_reason,
                    "pnl_pct": round(pnl_pct, 6),
                    "closed": True,
                }
                log.info(
                    "EXIT | reason=%s pnl=%.2f%% trade_id=%s",
                    exit_reason,
                    pnl_pct * 100,
                    self._open_trade["id"],
                )
                self._open_trade = None

        # 4. Log closed trade
        if trade_event:
            self._append_trade(trade_event)

        # 5. Write heartbeat
        self._write_heartbeat(now, current_price, rsi, strategy["version"])

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _entry_fires(self, strategy: dict, rsi: float) -> bool:
        entry = strategy["entry"]
        indicator = entry.get("indicator", "rsi")
        threshold = entry.get("threshold", 30)
        direction = entry.get("direction", "long")

        if indicator == "rsi":
            if direction == "long":
                return rsi < threshold
            else:
                return rsi > threshold
        return False

    def _load_strategy(self) -> dict:
        with open(self.strategy_file) as f:
            return yaml.safe_load(f)

    def _append_trade(self, trade: dict) -> None:
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _write_heartbeat(self, ts: str, price: float, rsi: float, strategy_version: str) -> None:
        hb = {
            "ts": ts,
            "asset": self.asset,
            "price": price,
            "rsi": rsi,
            "strategy_version": strategy_version,
            "open_trade": self._open_trade is not None,
            "consecutive_failures": self._consecutive_failures,
        }
        with open(self.heartbeat_file, "w") as f:
            json.dump(hb, f, indent=2)

    async def _fetch_with_retry(self, name: str, fetch_fn) -> dict:
        last_exc = None
        for attempt in range(1, MAX_ADAPTER_RETRIES + 1):
            try:
                result = await fetch_fn(self.asset)
                if result.get("schema_version") is None:
                    raise SchemaError(f"Adapter '{name}' returned no schema_version field")
                return result
            except SchemaError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt
                log.warning("Adapter '%s' attempt %d/%d failed: %s — retry in %ds", name, attempt, MAX_ADAPTER_RETRIES, exc, wait)
                await asyncio.sleep(wait)
        log.error("Adapter '%s' exhausted retries — using empty fallback", name)
        return {"schema_version": "0", "error": str(last_exc)}

    async def _fetch_all(self) -> dict:
        results = await asyncio.gather(
            self._fetch_with_retry("price", fetch_price),
            self._fetch_with_retry("onchain", fetch_onchain),
            self._fetch_with_retry("news", fetch_news),
            self._fetch_with_retry("macro", fetch_macro),
            return_exceptions=False,
        )
        return {
            "price": results[0],
            "onchain": results[1],
            "news": results[2],
            "macro": results[3],
        }
