"""
loop.py — 24/7 async reliability loop.

Every 60 s:
  1. Pull data from all adapters (with per-adapter retries + circuit-breaker).
  2. Load current strategy from state/strategy.yaml.
  3. Evaluate entry / exit conditions.
  4. Paper-trade if entry fires; manage open trade with full exit logic.
  5. Append closed trades to state/trades.jsonl.
  6. Write heartbeat to state/heartbeat.json.

Strategy variables honoured:
  entry.threshold              RSI level to trigger entry
  entry.direction              long / short
  entry.volume_surge_multiplier  min ratio of current vol to 20-bar avg (0 = disabled)
  entry.rsi_exit_threshold     exit long when RSI rises above this (overbought)
  stop_loss_pct                hard stop, % from entry
  take_profit_pct              hard take-profit, % from entry
  trailing_stop_pct            trailing stop from peak price (0 = disabled)
  leverage                     position multiplier (e.g. 2.0 = 2x)
  reentry_cooldown_minutes     minutes to wait after a closed trade before re-entering
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
VOLUME_LOOKBACK = 20   # bars for average volume calculation


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
        self._peak_price: float = 0.0          # for trailing stop
        self._last_close_time: float = 0.0     # for reentry cooldown
        self._volume_history: list[float] = [] # for volume surge filter

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

        data = await self._fetch_all()
        strategy = self._load_strategy()

        price_data = data.get("price", {})
        current_price = price_data.get("close", 0.0)
        rsi = price_data.get("rsi", 50.0)
        current_volume = price_data.get("volume", 0.0)

        if current_price == 0.0:
            log.warning("Price is 0 — skipping tick (adapter failure)")
            self._write_heartbeat(now, current_price, rsi, strategy.get("version", "?"))
            return

        # Update volume history for surge filter
        if current_volume > 0:
            self._volume_history.append(current_volume)
            if len(self._volume_history) > VOLUME_LOOKBACK:
                self._volume_history.pop(0)

        # Read strategy variables with safe defaults
        stop_loss_pct       = float(strategy.get("stop_loss_pct", 2.0))
        take_profit_pct     = float(strategy.get("take_profit_pct", 0.0))     # 0 = disabled
        trailing_stop_pct   = float(strategy.get("trailing_stop_pct", 0.0))   # 0 = disabled
        leverage            = float(strategy.get("leverage", 1.0))
        cooldown_minutes    = float(strategy.get("reentry_cooldown_minutes", 0.0))
        entry_cfg           = strategy.get("entry", {})
        rsi_exit_threshold  = float(entry_cfg.get("rsi_exit_threshold", 0.0)) # 0 = disabled
        vol_surge_mult      = float(entry_cfg.get("volume_surge_multiplier", 0.0)) # 0 = disabled

        trade_event = None

        if self._open_trade is None:
            # ---- ENTRY logic ----
            in_cooldown = (
                cooldown_minutes > 0
                and self._last_close_time > 0
                and (time.time() - self._last_close_time) < cooldown_minutes * 60
            )

            volume_ok = True
            if vol_surge_mult > 0 and len(self._volume_history) >= 5:
                avg_vol = sum(self._volume_history[:-1]) / len(self._volume_history[:-1])
                volume_ok = current_volume >= avg_vol * vol_surge_mult

            if not in_cooldown and volume_ok and self._entry_fires(strategy, rsi):
                self._open_trade = {
                    "id":               f"T{int(time.time())}",
                    "asset":            self.asset,
                    "entry_price":      current_price,
                    "entry_time":       now,
                    "direction":        entry_cfg.get("direction", "long"),
                    "stop_loss_pct":    stop_loss_pct,
                    "take_profit_pct":  take_profit_pct,
                    "trailing_stop_pct": trailing_stop_pct,
                    "leverage":         leverage,
                    "position_size_r":  float(strategy.get("position_size_r", 0.5)),
                    "strategy_version": strategy.get("version", "?"),
                    "mode":             self.mode,
                }
                self._peak_price = current_price
                log.info(
                    "ENTRY | price=%.2f rsi=%.1f leverage=%.1fx vol_ok=%s trade_id=%s",
                    current_price, rsi, leverage, volume_ok, self._open_trade["id"],
                )

        else:
            # ---- EXIT logic ----
            entry_price = self._open_trade["entry_price"]
            direction   = self._open_trade["direction"]
            lev         = self._open_trade.get("leverage", 1.0)
            tp_pct      = self._open_trade.get("take_profit_pct", 0.0)
            ts_pct      = self._open_trade.get("trailing_stop_pct", 0.0)
            sl_pct      = self._open_trade["stop_loss_pct"]

            # Update trailing peak
            if direction == "long" and current_price > self._peak_price:
                self._peak_price = current_price
            elif direction == "short" and (self._peak_price == 0 or current_price < self._peak_price):
                self._peak_price = current_price

            # Raw PnL (before leverage)
            if direction == "long":
                raw_pnl = (current_price - entry_price) / entry_price
            else:
                raw_pnl = (entry_price - current_price) / entry_price

            # Exit conditions
            exit_reason = None

            # 1. Hard stop-loss
            if raw_pnl <= -(sl_pct / 100.0):
                exit_reason = "stop_loss"

            # 2. Take-profit
            elif tp_pct > 0 and raw_pnl >= tp_pct / 100.0:
                exit_reason = "take_profit"

            # 3. Trailing stop (from peak)
            elif ts_pct > 0 and self._peak_price > 0:
                if direction == "long":
                    drop_from_peak = (self._peak_price - current_price) / self._peak_price
                    if drop_from_peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"
                else:
                    rise_from_peak = (current_price - self._peak_price) / self._peak_price
                    if rise_from_peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"

            # 4. RSI overbought exit (for longs)
            elif rsi_exit_threshold > 0 and direction == "long" and rsi >= rsi_exit_threshold:
                exit_reason = "rsi_overbought"

            # 5. 24-hour time exit
            elif (time.time() - int(self._open_trade["id"][1:])) >= 86400:
                exit_reason = "time_exit"

            if exit_reason:
                # Apply leverage to PnL
                leveraged_pnl = raw_pnl * lev

                trade_event = {
                    **self._open_trade,
                    "exit_price":    current_price,
                    "exit_time":     now,
                    "exit_reason":   exit_reason,
                    "pnl_pct":       round(raw_pnl, 6),
                    "pnl_pct_levered": round(leveraged_pnl, 6),
                    "peak_price":    self._peak_price,
                    "closed":        True,
                }
                log.info(
                    "EXIT | reason=%s price=%.2f pnl=%.2f%% (%.2f%% levered) trade_id=%s",
                    exit_reason,
                    current_price,
                    raw_pnl * 100,
                    leveraged_pnl * 100,
                    self._open_trade["id"],
                )
                self._last_close_time = time.time()
                self._open_trade = None
                self._peak_price = 0.0

        if trade_event:
            self._append_trade(trade_event)

        self._write_heartbeat(now, current_price, rsi, strategy.get("version", "?"))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _entry_fires(self, strategy: dict, rsi: float) -> bool:
        entry = strategy.get("entry", {})
        indicator = entry.get("indicator", "rsi")
        threshold = entry.get("threshold", 30)
        direction = entry.get("direction", "long")

        if indicator == "rsi":
            return rsi < threshold if direction == "long" else rsi > (100 - threshold)
        return False

    def _load_strategy(self) -> dict:
        with open(self.strategy_file) as f:
            return yaml.safe_load(f)

    def _append_trade(self, trade: dict) -> None:
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _write_heartbeat(self, ts: str, price: float, rsi: float, strategy_version: str) -> None:
        hb = {
            "ts":                   ts,
            "asset":                self.asset,
            "price":                price,
            "rsi":                  rsi,
            "strategy_version":     strategy_version,
            "open_trade":           self._open_trade is not None,
            "peak_price":           self._peak_price,
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
                log.warning(
                    "Adapter '%s' attempt %d/%d failed: %s — retry in %ds",
                    name, attempt, MAX_ADAPTER_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)
        log.error("Adapter '%s' exhausted retries — using empty fallback", name)
        return {"schema_version": "0", "error": str(last_exc)}

    async def _fetch_all(self) -> dict:
        results = await asyncio.gather(
            self._fetch_with_retry("price",   fetch_price),
            self._fetch_with_retry("onchain", fetch_onchain),
            self._fetch_with_retry("news",    fetch_news),
            self._fetch_with_retry("macro",   fetch_macro),
            return_exceptions=False,
        )
        return {
            "price":   results[0],
            "onchain": results[1],
            "news":    results[2],
            "macro":   results[3],
        }
