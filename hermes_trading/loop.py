"""
loop.py — 24/7 async reliability loop with multiple concurrent positions.

Every 10 s:
  1. Pull data from all adapters (with per-adapter retries + circuit-breaker).
  2. Load current strategy from state/strategy.yaml.
  3. Evaluate exit conditions on every open trade.
  4. Evaluate entry conditions — open new trade if signal fires and capacity allows.
  5. Append closed trades to state/trades.jsonl.
  6. Write heartbeat to state/heartbeat.json.

Strategy variables honoured:
  entry.direction                  long / short / both
  entry.long_threshold             enter long when RSI < this
  entry.short_threshold            enter short when RSI > this
  entry.long_rsi_exit              exit long when RSI > this (only if profitable)
  entry.short_rsi_exit             exit short when RSI < this (only if profitable)
  entry.macd_confirm               MACD hist direction adds +1 to score
  entry.bb_confirm                 BB %B position adds +1 to score
  entry.ob_confirm                 order-book imbalance adds +1 to score
  entry.volume_surge_multiplier    min ratio of current vol to 20-bar avg (0 = disabled)
  stop_loss_pct                    hard stop, % from entry
  take_profit_pct                  hard take-profit, % from entry
  take_profit_mode                 'fixed' or 'atr'
  take_profit_atr_multiplier       ATR × this = TP distance (atr mode)
  trailing_stop_pct                always-on trailing stop from peak (0 = disabled)
  profit_lock_activate_pct         activate profit-lock trail once profit >= this %
  profit_lock_trail_pct            profit-lock trails peak by this % (0 = disabled)
  leverage_base / leverage_max     scale with confirmation score 1→4
  position_size_base / _max        scale with confirmation score 1→4
  position_size_atr_dampen         reduce size when ATR is elevated
  max_open_positions               max concurrent open trades (default 3)
  reentry_cooldown_minutes         min gap between any two entries

Confirmation scoring:
  Score 1 — RSI alone fired (gatekeeper, always required)
  Score 2 — + MACD histogram aligned
  Score 3 — + Bollinger %B aligned
  Score 4 — + Order-book imbalance aligned

Exit hierarchy (checked in order, first match wins):
  1. Hard stop-loss
  2. Take-profit
  3. Always-on trailing stop
  4. Profit-lock trailing stop  ← NEW
  5. RSI exit (only if profitable)
  6. MACD reversal exit         ← NEW (only if profitable)
  7. 24-hour time exit
"""
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.adapters.macro import fetch as fetch_macro
from hermes_trading.adapters.news import fetch as fetch_news
from hermes_trading.adapters.onchain import fetch as fetch_onchain
from hermes_trading.adapters.price import fetch as fetch_price
from hermes_trading.monte_carlo import run_model as run_mc_model

log = logging.getLogger("hermes.loop")

TICK_SECONDS = 10
MAX_ADAPTER_RETRIES = 3
CIRCUIT_BREAK_THRESHOLD = 5
VOLUME_LOOKBACK = 20


class SchemaError(Exception):
    pass


class TradingLoop:
    def __init__(self, asset: str, mode: str, state_dir: Path, goal: dict) -> None:
        self.asset = asset
        self.mode = mode
        self.state_dir = state_dir
        self.goal = goal
        self.strategy_file = state_dir / "strategy.yaml"
        self.trades_file   = state_dir / "trades.jsonl"
        self.heartbeat_file = state_dir / "heartbeat.json"
        self._consecutive_failures = 0

        # Multiple concurrent positions
        self._open_trades: list[dict] = []
        self._peak_prices: dict[str, float] = {}   # trade_id → peak price seen

        self._last_entry_time: float = 0.0   # for entry cooldown
        self._volume_history: list[float] = []
        self._last_reflected_at: int = 0
        self._tick_count: int = 0

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
                log.warning("Tick error (%d/%d): %s",
                            self._consecutive_failures, CIRCUIT_BREAK_THRESHOLD, exc)
                if self._consecutive_failures >= CIRCUIT_BREAK_THRESHOLD:
                    log.error("Circuit breaker tripped — halting.")
                    raise RuntimeError("Circuit breaker tripped") from exc
            await asyncio.sleep(TICK_SECONDS)

    # ------------------------------------------------------------------ #
    #  Single tick                                                         #
    # ------------------------------------------------------------------ #

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc).isoformat()

        data     = await self._fetch_all()
        strategy = self._load_strategy()

        price_data     = data.get("price", {})
        current_price  = price_data.get("close", 0.0)
        rsi            = price_data.get("rsi", 50.0)
        current_volume = price_data.get("volume", 0.0)
        macd_hist      = price_data.get("macd_hist", 0.0)
        bb_pct         = price_data.get("bb_pct", 0.5)
        atr            = price_data.get("atr", 0.0)
        ob_imbalance   = price_data.get("ob_imbalance", 0.0)

        if current_price == 0.0:
            log.warning("Price is 0 — skipping tick (adapter failure)")
            self._write_heartbeat(now, price_data, strategy.get("version", "?"))
            return

        if current_volume > 0:
            self._volume_history.append(current_volume)
            if len(self._volume_history) > VOLUME_LOOKBACK:
                self._volume_history.pop(0)

        # ── Strategy config ────────────────────────────────────────────────────
        stop_loss_pct     = float(strategy.get("stop_loss_pct",          0.5))
        trailing_stop_pct = float(strategy.get("trailing_stop_pct",      0.0))
        cooldown_minutes  = float(strategy.get("reentry_cooldown_minutes", 1.0))
        max_positions     = int(strategy.get("max_open_positions",        3))
        entry_cfg         = strategy.get("entry", {})

        direction_mode  = entry_cfg.get("direction", "both")
        long_threshold  = float(entry_cfg.get("long_threshold",  55))
        short_threshold = float(entry_cfg.get("short_threshold", 65))
        long_rsi_exit   = float(entry_cfg.get("long_rsi_exit",   78))
        short_rsi_exit  = float(entry_cfg.get("short_rsi_exit",  25))
        vol_surge_mult  = float(entry_cfg.get("volume_surge_multiplier", 0.0))

        lev_base  = float(strategy.get("leverage_base",         1.5))
        lev_max   = float(strategy.get("leverage_max",          3.0))
        ps_base   = float(strategy.get("position_size_base",    0.15))
        ps_max    = float(strategy.get("position_size_max",     0.40))
        ps_dampen = bool(strategy.get("position_size_atr_dampen", True))

        # Profit-lock trailing stop params
        pl_activate = float(strategy.get("profit_lock_activate_pct", 0.25))
        pl_trail    = float(strategy.get("profit_lock_trail_pct",    0.15))

        # Dynamic take profit
        tp_mode     = strategy.get("take_profit_mode", "fixed")
        tp_fixed    = float(strategy.get("take_profit_pct", 0.5))
        tp_atr_mult = float(strategy.get("take_profit_atr_multiplier", 2.5))
        tp_atr_min  = float(strategy.get("take_profit_atr_min_pct",   0.5))
        tp_atr_max  = float(strategy.get("take_profit_atr_max_pct",   3.0))
        if tp_mode == "atr" and atr > 0 and current_price > 0:
            take_profit_pct = (atr * tp_atr_mult / current_price) * 100
            take_profit_pct = round(max(tp_atr_min, min(tp_atr_max, take_profit_pct)), 3)
        else:
            take_profit_pct = tp_fixed

        closed_events: list[dict] = []

        # ── EXIT logic — iterate every open trade ──────────────────────────────
        for trade in list(self._open_trades):
            trade_id    = trade["id"]
            entry_price = trade["entry_price"]
            direction   = trade["direction"]
            lev         = trade.get("leverage", 1.0)
            sl_pct      = float(strategy.get("stop_loss_pct",     0.5))
            tp_pct      = float(strategy.get("take_profit_pct",   0.5))
            ts_pct      = float(strategy.get("trailing_stop_pct", 0.0))

            # Update per-trade peak price
            if trade_id not in self._peak_prices:
                self._peak_prices[trade_id] = entry_price
            peak = self._peak_prices[trade_id]
            if direction == "long":
                if current_price > peak:
                    self._peak_prices[trade_id] = current_price
                    peak = current_price
                raw_pnl = (current_price - entry_price) / entry_price
            else:
                if current_price < peak:
                    self._peak_prices[trade_id] = current_price
                    peak = current_price
                raw_pnl = (entry_price - current_price) / entry_price

            exit_reason = None

            # 1. Hard stop-loss
            if raw_pnl <= -(sl_pct / 100.0):
                exit_reason = "stop_loss"

            # 2. Take-profit
            if not exit_reason and tp_pct > 0 and raw_pnl >= tp_pct / 100.0:
                exit_reason = "take_profit"

            # 3. Always-on trailing stop (from peak)
            if not exit_reason and ts_pct > 0 and peak > 0:
                if direction == "long":
                    if (peak - current_price) / peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"
                else:
                    if (current_price - peak) / peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"

            # 4. Profit-lock trailing stop
            # Activates once peak profit >= pl_activate%. Then if price
            # pulls back pl_trail% from that peak, exit and lock in profit.
            if not exit_reason and pl_activate > 0 and pl_trail > 0:
                peak_pnl = abs(peak - entry_price) / entry_price
                if peak_pnl >= pl_activate / 100.0:
                    if direction == "long":
                        if (peak - current_price) / peak >= pl_trail / 100.0:
                            exit_reason = "profit_lock"
                    else:
                        if (current_price - peak) / peak >= pl_trail / 100.0:
                            exit_reason = "profit_lock"

            # 5. RSI exit — only fires when trade is profitable
            if not exit_reason:
                if direction == "long" and rsi >= long_rsi_exit and raw_pnl > 0:
                    exit_reason = "rsi_overbought"
                elif direction == "short" and rsi <= short_rsi_exit and raw_pnl > 0:
                    exit_reason = "rsi_oversold"

            # 6. MACD reversal exit — momentum flipped against us while profitable
            # Long: entered with bullish MACD, now bearish → momentum reversed
            # Short: entered with bearish MACD, now bullish → momentum reversed
            if not exit_reason and raw_pnl > 0:
                entry_macd = trade.get("entry_macd_hist", 0.0)
                if direction == "long" and entry_macd > 0 and macd_hist < 0:
                    exit_reason = "macd_reversal"
                elif direction == "short" and entry_macd < 0 and macd_hist > 0:
                    exit_reason = "macd_reversal"

            # 7. 24-hour time exit
            if not exit_reason:
                if (time.time() - int(trade_id[1:])) >= 86400:
                    exit_reason = "time_exit"

            if exit_reason:
                leveraged_pnl = raw_pnl * lev
                closed_trade = {
                    **trade,
                    "exit_price":      current_price,
                    "exit_time":       now,
                    "exit_reason":     exit_reason,
                    "pnl_pct":         round(raw_pnl, 6),
                    "pnl_pct_levered": round(leveraged_pnl, 6),
                    "peak_price":      peak,
                    "closed":          True,
                }
                log.info(
                    "EXIT | reason=%s dir=%s price=%.2f pnl=%.3f%% (%.3f%% lev) id=%s",
                    exit_reason, direction, current_price,
                    raw_pnl * 100, leveraged_pnl * 100, trade_id,
                )
                self._open_trades.remove(trade)
                del self._peak_prices[trade_id]
                closed_events.append(closed_trade)

        # ── ENTRY logic ────────────────────────────────────────────────────────
        in_cooldown = (
            cooldown_minutes > 0
            and self._last_entry_time > 0
            and (time.time() - self._last_entry_time) < cooldown_minutes * 60
        )

        volume_ok = True
        if vol_surge_mult > 0 and len(self._volume_history) >= 5:
            avg_vol   = sum(self._volume_history[:-1]) / max(1, len(self._volume_history) - 1)
            volume_ok = current_volume >= avg_vol * vol_surge_mult

        can_enter = (
            not in_cooldown
            and volume_ok
            and len(self._open_trades) < max_positions
        )

        entry_direction = None
        entry_score     = 0

        if can_enter:
            if direction_mode in ("long", "both") and rsi < long_threshold:
                entry_direction = "long"
                entry_score = 1
                if entry_cfg.get("macd_confirm", False) and macd_hist > 0:
                    entry_score += 1
                if entry_cfg.get("bb_confirm", False) and bb_pct < 0.35:
                    entry_score += 1
                if entry_cfg.get("ob_confirm", False) and ob_imbalance > 0:
                    entry_score += 1

            elif direction_mode in ("short", "both") and rsi > short_threshold:
                entry_direction = "short"
                entry_score = 1
                if entry_cfg.get("macd_confirm", False) and macd_hist < 0:
                    entry_score += 1
                if entry_cfg.get("bb_confirm", False) and bb_pct > 0.65:
                    entry_score += 1
                if entry_cfg.get("ob_confirm", False) and ob_imbalance < 0:
                    entry_score += 1

        if entry_direction is not None:
            score_frac = (entry_score - 1) / 3.0
            leverage   = round(lev_base + (lev_max - lev_base) * score_frac, 2)
            pos_size_r = ps_base + (ps_max - ps_base) * score_frac

            if ps_dampen and atr > 0 and current_price > 0:
                atr_pct = (atr / current_price) * 100
                if atr_pct > 0.5:
                    dampen = max(0.5, 1.0 - (atr_pct - 0.5) * 0.3)
                    pos_size_r *= dampen
            pos_size_r = round(max(ps_base, min(ps_max, pos_size_r)), 4)

            new_trade = {
                "id":                  f"T{int(time.time())}",
                "asset":               self.asset,
                "entry_price":         current_price,
                "entry_time":          now,
                "direction":           entry_direction,
                "stop_loss_pct":       stop_loss_pct,
                "take_profit_pct":     take_profit_pct,
                "trailing_stop_pct":   trailing_stop_pct,
                "leverage":            leverage,
                "position_size_r":     pos_size_r,
                "strategy_version":    strategy.get("version", "?"),
                "mode":                self.mode,
                "entry_score":         entry_score,
                "entry_rsi":           rsi,
                "entry_macd_hist":     macd_hist,
                "entry_bb_pct":        bb_pct,
                "entry_atr":           atr,
                "entry_ob_imb":        ob_imbalance,
                "entry_tp_mode":       tp_mode,
            }
            self._open_trades.append(new_trade)
            self._peak_prices[new_trade["id"]] = current_price
            self._last_entry_time = time.time()
            log.info(
                "ENTRY | dir=%s score=%d/4 price=%.2f rsi=%.1f lev=%.2fx "
                "pos=%.1f%% tp=%.3f%% open_positions=%d id=%s",
                entry_direction, entry_score, current_price, rsi,
                leverage, pos_size_r * 100, take_profit_pct,
                len(self._open_trades), new_trade["id"],
            )

        # ── Persist closed trades ──────────────────────────────────────────────
        for event in closed_events:
            self._append_trade(event)
            await self._maybe_reflect()
            asyncio.create_task(self._run_model())

        self._tick_count += 1
        if self._tick_count % 30 == 0:
            asyncio.create_task(self._run_model())

        self._write_heartbeat(now, price_data, strategy.get("version", "?"))

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_strategy(self) -> dict:
        with open(self.strategy_file) as f:
            return yaml.safe_load(f)

    def _append_trade(self, trade: dict) -> None:
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    async def _maybe_reflect(self) -> None:
        cadence = int(self.goal.get("reflection_every", 10))
        if cadence <= 0:
            return
        closed_count = 0
        if self.trades_file.exists():
            for line in self.trades_file.read_text().splitlines():
                if line.strip():
                    try:
                        if json.loads(line).get("closed"):
                            closed_count += 1
                    except Exception:
                        pass
        if closed_count == 0 or closed_count % cadence != 0:
            return
        if closed_count == self._last_reflected_at:
            return
        self._last_reflected_at = closed_count
        log.info("AUTO-REFLECT | trade #%d hit cadence of %d", closed_count, cadence)
        asyncio.create_task(self._run_reflect())

    async def _run_model(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            starting = float(self.goal.get("starting_balance", 100_000))
            await loop.run_in_executor(None, run_mc_model, starting)
        except Exception as exc:
            log.debug("Model run failed (non-fatal): %s", exc)

    async def _run_reflect(self) -> None:
        python = sys.executable
        base   = Path(__file__).parent.parent
        for mode in ("--hermes", "--fallback"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    python, "-m", "hermes_trading.reflect", mode,
                    cwd=str(base),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=180)
                output = stdout.decode() if stdout else ""
                if proc.returncode == 0:
                    log.info("REFLECT DONE (%s):\n%s", mode, output.strip())
                    return
                log.warning("reflect %s exited %d", mode, proc.returncode)
            except asyncio.TimeoutError:
                log.error("reflect %s timed out", mode)
            except Exception as exc:
                log.error("reflect %s failed: %s", mode, exc)
        log.error("All reflection modes failed — strategy unchanged")

    def _write_heartbeat(self, ts: str, price_data: dict, strategy_version: str) -> None:
        hb = {
            "ts":                   ts,
            "asset":                self.asset,
            "price":                price_data.get("close", 0.0),
            "rsi":                  price_data.get("rsi", 50.0),
            "macd_hist":            price_data.get("macd_hist", 0.0),
            "bb_pct":               price_data.get("bb_pct", 0.5),
            "atr":                  price_data.get("atr", 0.0),
            "momentum":             price_data.get("momentum", 0.0),
            "ob_imbalance":         price_data.get("ob_imbalance", 0.0),
            "strategy_version":     strategy_version,
            "open_trade":           len(self._open_trades) > 0,
            "open_trade_count":     len(self._open_trades),
            "open_trades":          [
                {
                    "id":          t["id"],
                    "direction":   t["direction"],
                    "entry_price": t["entry_price"],
                    "entry_score": t.get("entry_score", 1),
                    "leverage":    t.get("leverage", 1.5),
                }
                for t in self._open_trades
            ],
            "peak_price":           max(self._peak_prices.values(), default=0.0),
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
                log.warning("Adapter '%s' attempt %d/%d failed: %s — retry in %ds",
                            name, attempt, MAX_ADAPTER_RETRIES, exc, wait)
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
