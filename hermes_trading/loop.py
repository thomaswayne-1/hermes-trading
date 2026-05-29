"""
loop.py — 24/7 async reliability loop.

Every 10 s:
  1. Pull data from all adapters (with per-adapter retries + circuit-breaker).
  2. Load current strategy from state/strategy.yaml.
  3. Evaluate entry / exit conditions.
  4. Paper-trade if entry fires; manage open trade with full exit logic.
  5. Append closed trades to state/trades.jsonl.
  6. Write heartbeat to state/heartbeat.json.

Strategy variables honoured:
  entry.direction                  long / short / both
  entry.long_threshold             enter long when RSI < this
  entry.short_threshold            enter short when RSI > this
  entry.long_rsi_exit              exit long when RSI > this
  entry.short_rsi_exit             exit short when RSI < this
  entry.macd_confirm               if true, MACD hist direction adds +1 to score
  entry.bb_confirm                 if true, BB %B position adds +1 to score
  entry.ob_confirm                 if true, order-book imbalance adds +1 to score
  entry.volume_surge_multiplier    min ratio of current vol to 20-bar avg (0 = disabled)
  stop_loss_pct                    hard stop, % from entry
  take_profit_pct                  hard take-profit, % from entry  (fixed mode)
  take_profit_mode                 'fixed' or 'atr'
  take_profit_atr_multiplier       ATR × this = TP distance (atr mode)
  trailing_stop_pct                trailing stop from peak price (0 = disabled)
  leverage_base / leverage_max     scale with confirmation score 1→4
  position_size_base / position_size_max  scale with confirmation score 1→4
  position_size_atr_dampen         reduce size when ATR is elevated
  reentry_cooldown_minutes         wait after a closed trade before re-entering

Confirmation scoring:
  Score 1 — RSI alone fired (gatekeeper, always required)
  Score 2 — + MACD histogram aligned
  Score 3 — + Bollinger %B aligned
  Score 4 — + Order-book imbalance aligned
  Position size and leverage scale linearly from *_base (score 1) → *_max (score 4).
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
VOLUME_LOOKBACK = 20   # bars for average volume


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
        self._peak_price: float = 0.0
        self._last_close_time: float = 0.0
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

        # Update volume history for surge filter
        if current_volume > 0:
            self._volume_history.append(current_volume)
            if len(self._volume_history) > VOLUME_LOOKBACK:
                self._volume_history.pop(0)

        # ── Strategy config ────────────────────────────────────────────────────
        stop_loss_pct     = float(strategy.get("stop_loss_pct", 0.5))
        trailing_stop_pct = float(strategy.get("trailing_stop_pct", 0.0))
        cooldown_minutes  = float(strategy.get("reentry_cooldown_minutes", 1.0))
        entry_cfg         = strategy.get("entry", {})

        # Entry thresholds
        direction_mode   = entry_cfg.get("direction", "both")   # long / short / both
        long_threshold   = float(entry_cfg.get("long_threshold",  55))
        short_threshold  = float(entry_cfg.get("short_threshold", 70))
        long_rsi_exit    = float(entry_cfg.get("long_rsi_exit",   78))
        short_rsi_exit   = float(entry_cfg.get("short_rsi_exit",  25))
        vol_surge_mult   = float(entry_cfg.get("volume_surge_multiplier", 0.0))

        # Dynamic leverage range
        lev_base  = float(strategy.get("leverage_base",  1.5))
        lev_max   = float(strategy.get("leverage_max",   3.0))

        # Dynamic position size range
        ps_base   = float(strategy.get("position_size_base",  0.15))
        ps_max    = float(strategy.get("position_size_max",   0.40))
        ps_dampen = bool(strategy.get("position_size_atr_dampen", True))

        # Dynamic take profit
        tp_mode     = strategy.get("take_profit_mode", "fixed")
        tp_fixed    = float(strategy.get("take_profit_pct", 1.0))
        tp_atr_mult = float(strategy.get("take_profit_atr_multiplier", 2.5))
        tp_atr_min  = float(strategy.get("take_profit_atr_min_pct", 0.5))
        tp_atr_max  = float(strategy.get("take_profit_atr_max_pct", 3.0))
        if tp_mode == "atr" and atr > 0 and current_price > 0:
            take_profit_pct = (atr * tp_atr_mult / current_price) * 100
            take_profit_pct = round(max(tp_atr_min, min(tp_atr_max, take_profit_pct)), 3)
        else:
            take_profit_pct = tp_fixed

        trade_event = None

        if self._open_trade is None:
            # ── ENTRY logic ───────────────────────────────────────────────────

            in_cooldown = (
                cooldown_minutes > 0
                and self._last_close_time > 0
                and (time.time() - self._last_close_time) < cooldown_minutes * 60
            )

            # Volume filter — shared for both directions
            volume_ok = True
            if vol_surge_mult > 0 and len(self._volume_history) >= 5:
                avg_vol   = sum(self._volume_history[:-1]) / max(1, len(self._volume_history) - 1)
                volume_ok = current_volume >= avg_vol * vol_surge_mult

            entry_direction = None
            entry_score = 0

            if not in_cooldown and volume_ok:
                # ── Long signal check ─────────────────────────────────────────
                if direction_mode in ("long", "both") and rsi < long_threshold:
                    entry_direction = "long"
                    entry_score = 1   # RSI fired (gatekeeper)
                    if entry_cfg.get("macd_confirm", False) and macd_hist > 0:
                        entry_score += 1
                    if entry_cfg.get("bb_confirm", False) and bb_pct < 0.35:
                        entry_score += 1
                    if entry_cfg.get("ob_confirm", False) and ob_imbalance > 0:
                        entry_score += 1

                # ── Short signal check ─────────────────────────────────────────
                elif direction_mode in ("short", "both") and rsi > short_threshold:
                    entry_direction = "short"
                    entry_score = 1   # RSI fired (gatekeeper)
                    if entry_cfg.get("macd_confirm", False) and macd_hist < 0:
                        entry_score += 1
                    if entry_cfg.get("bb_confirm", False) and bb_pct > 0.65:
                        entry_score += 1
                    if entry_cfg.get("ob_confirm", False) and ob_imbalance < 0:
                        entry_score += 1

            if entry_direction is not None:
                # ── Scale leverage + position with score (1→base, 4→max) ──────
                score_frac = (entry_score - 1) / 3.0   # 0.0 at score=1, 1.0 at score=4
                leverage   = round(lev_base + (lev_max - lev_base) * score_frac, 2)
                pos_size_r = ps_base + (ps_max - ps_base) * score_frac

                # ATR dampening: if ATR > 0.5% of price, scale down
                if ps_dampen and atr > 0 and current_price > 0:
                    atr_pct = (atr / current_price) * 100
                    if atr_pct > 0.5:
                        dampen = max(0.5, 1.0 - (atr_pct - 0.5) * 0.3)
                        pos_size_r *= dampen
                pos_size_r = round(max(ps_base, min(ps_max, pos_size_r)), 4)

                self._open_trade = {
                    "id":                f"T{int(time.time())}",
                    "asset":             self.asset,
                    "entry_price":       current_price,
                    "entry_time":        now,
                    "direction":         entry_direction,
                    "stop_loss_pct":     stop_loss_pct,
                    "take_profit_pct":   take_profit_pct,
                    "trailing_stop_pct": trailing_stop_pct,
                    "leverage":          leverage,
                    "position_size_r":   pos_size_r,
                    "strategy_version":  strategy.get("version", "?"),
                    "mode":              self.mode,
                    # Snapshot at entry for later analysis
                    "entry_score":       entry_score,
                    "entry_rsi":         rsi,
                    "entry_macd_hist":   macd_hist,
                    "entry_bb_pct":      bb_pct,
                    "entry_atr":         atr,
                    "entry_ob_imb":      ob_imbalance,
                    "entry_tp_mode":     tp_mode,
                }
                self._peak_price = current_price
                log.info(
                    "ENTRY | dir=%s score=%d/4 price=%.2f rsi=%.1f lev=%.2fx pos=%.1f%% "
                    "tp=%.3f%% macd_hist=%.4f bb_pct=%.3f ob_imb=%.3f trade_id=%s",
                    entry_direction, entry_score, current_price, rsi,
                    leverage, pos_size_r * 100, take_profit_pct,
                    macd_hist, bb_pct, ob_imbalance, self._open_trade["id"],
                )

        else:
            # ── EXIT logic ────────────────────────────────────────────────────
            # Always read exit params from CURRENT strategy so changes take
            # effect immediately without needing to close the trade first.
            entry_price = self._open_trade["entry_price"]
            direction   = self._open_trade["direction"]
            lev         = self._open_trade.get("leverage", 1.0)
            tp_pct      = float(strategy.get("take_profit_pct",   0.0))
            ts_pct      = float(strategy.get("trailing_stop_pct", 0.0))
            sl_pct      = float(strategy.get("stop_loss_pct",     0.5))

            # Update trailing peak
            if direction == "long":
                if current_price > self._peak_price:
                    self._peak_price = current_price
            else:  # short — peak is the lowest price seen
                if self._peak_price == 0.0 or current_price < self._peak_price:
                    self._peak_price = current_price

            # Raw PnL (before leverage)
            if direction == "long":
                raw_pnl = (current_price - entry_price) / entry_price
            else:
                raw_pnl = (entry_price - current_price) / entry_price

            exit_reason = None

            # 1. Hard stop-loss
            if raw_pnl <= -(sl_pct / 100.0):
                exit_reason = "stop_loss"

            # 2. Take-profit
            if not exit_reason and tp_pct > 0 and raw_pnl >= tp_pct / 100.0:
                exit_reason = "take_profit"

            # 3. Trailing stop (from peak)
            if not exit_reason and ts_pct > 0 and self._peak_price > 0:
                if direction == "long":
                    drop_from_peak = (self._peak_price - current_price) / self._peak_price
                    if drop_from_peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"
                else:
                    rise_from_peak = (current_price - self._peak_price) / self._peak_price
                    if rise_from_peak >= ts_pct / 100.0:
                        exit_reason = "trailing_stop"

            # 4. RSI exit (direction-aware)
            if not exit_reason:
                if direction == "long" and rsi >= long_rsi_exit:
                    exit_reason = "rsi_overbought"
                elif direction == "short" and rsi <= short_rsi_exit:
                    exit_reason = "rsi_oversold"

            # 5. 24-hour time exit
            if not exit_reason:
                trade_open_ts = int(self._open_trade["id"][1:])
                if (time.time() - trade_open_ts) >= 86400:
                    exit_reason = "time_exit"

            if exit_reason:
                leveraged_pnl = raw_pnl * lev

                trade_event = {
                    **self._open_trade,
                    "exit_price":      current_price,
                    "exit_time":       now,
                    "exit_reason":     exit_reason,
                    "pnl_pct":         round(raw_pnl, 6),
                    "pnl_pct_levered": round(leveraged_pnl, 6),
                    "peak_price":      self._peak_price,
                    "closed":          True,
                }
                log.info(
                    "EXIT | reason=%s dir=%s price=%.2f pnl=%.2f%% (%.2f%% levered) trade_id=%s",
                    exit_reason, direction, current_price,
                    raw_pnl * 100, leveraged_pnl * 100,
                    self._open_trade["id"],
                )
                self._last_close_time = time.time()
                self._open_trade = None
                self._peak_price = 0.0

        if trade_event:
            self._append_trade(trade_event)
            await self._maybe_reflect()
            asyncio.create_task(self._run_model())

        # Update model every 30 ticks (~5 min at 10s ticks)
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
        """Trigger a reflection cycle if we've hit the cadence."""
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
        log.info(
            "AUTO-REFLECT | trade #%d hit cadence of %d — spawning reflection",
            closed_count, cadence,
        )
        asyncio.create_task(self._run_reflect())

    async def _run_model(self) -> None:
        """Run Monte Carlo model in background without blocking the loop."""
        try:
            loop = asyncio.get_event_loop()
            starting = float(self.goal.get("starting_balance", 100_000))
            await loop.run_in_executor(None, run_mc_model, starting)
        except Exception as exc:
            log.debug("Model run failed (non-fatal): %s", exc)

    async def _run_reflect(self) -> None:
        """Spawn reflect.py as a subprocess (tries --hermes, falls back to --fallback)."""
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
                log.warning("reflect %s exited %d — trying next mode", mode, proc.returncode)
            except asyncio.TimeoutError:
                log.error("reflect %s timed out after 180s", mode)
            except Exception as exc:
                log.error("reflect %s failed: %s", mode, exc)

        log.error("All reflection modes failed — strategy unchanged this cycle")

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
