"""
loop.py — 24/7 async reliability loop with the Directional Coefficient Engine.

Every 10 s:
  1. Pull data from all adapters (price + funding + sentiment + extras).
  2. Load current strategy from state/strategy.yaml.
  3. Tick the coefficient engine — compute C, K, regime, sub-signals.
  4. Evaluate exits on every open trade (coefficient-aware first, then legacy).
  5. Evaluate entry:
       - If coefficient_engine_enabled: |C| > tau_enter → Kelly sizing
       - Else: legacy RSI-gatekeeper + scoring
  6. Append closed trades. Trigger improvement cycles if cadence hit.
  7. Write heartbeat with engine telemetry.

The engine ALWAYS computes and writes to heartbeat regardless of the master
switch so you can watch its behavior on live data before flipping it on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.adapters.macro import fetch as fetch_macro
from hermes_trading.adapters.news import fetch as fetch_news
from hermes_trading.adapters.onchain import fetch as fetch_onchain
from hermes_trading.adapters.price import fetch as fetch_price, get_cached_candles
from hermes_trading.adapters.funding import fetch as fetch_funding
from hermes_trading.adapters.sentiment import fetch as fetch_sentiment
from hermes_trading.adapters import backup
from hermes_trading.monte_carlo import run_model as run_mc_model

from hermes_trading.engine.coefficient import CoefficientEngine
from hermes_trading.engine.sizing import (
    kelly_buckets, fractional_kelly_size, leverage_from_K, respects_budget,
)
from hermes_trading.engine.exits import vol_scaled_stop_pct, evaluate_coefficient_exits
from hermes_trading.engine.costs import net_pnl

from hermes_trading.improvement.guardrails import (
    CircuitBreaker, current_drawdown, total_leveraged_exposure,
)
from hermes_trading.improvement.cycle import run_fast_cycle

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
        self.strategy_file     = state_dir / "strategy.yaml"
        self.weights_file      = state_dir / "weights.json"
        self.trades_file       = state_dir / "trades.jsonl"
        self.heartbeat_file    = state_dir / "heartbeat.json"
        self.open_trades_file  = state_dir / "open_trades.json"
        self.engine_state_file = state_dir / "engine_state.json"

        self._consecutive_failures = 0

        # Multiple concurrent positions — restored from disk on startup
        self._open_trades: list[dict] = []
        self._peak_prices: dict[str, float] = {}
        self._trough_prices: dict[str, float] = {}   # for MAE tracking

        # Gist restore is deferred to _startup_restore() which runs async
        # inside run_forever() — keeps __init__ fast so Railway health check passes.
        self._restore_open_trades()

        self._last_entry_time: float = (
            max((int(t["id"][1:]) for t in self._open_trades), default=0)
        )
        self._volume_history: list[float] = []
        self._last_reflected_at: int = 0
        self._last_fast_cycle_at: int = 0
        self._tick_count: int = 0

        # Engine + guardrails
        initial_strategy = self._load_strategy()
        coef_cfg = initial_strategy.get("coefficient", {})
        regime_weights = self._load_weights(initial_strategy)
        self.engine = CoefficientEngine(coef_cfg, regime_weights=regime_weights)

        # Restore engine state from disk — skips the 33-minute warmup on redeploy
        if self.engine.load_state(self.engine_state_file):
            log.info(
                "ENGINE RESTORED | %d samples loaded from disk — ready immediately",
                len(self.engine.features),
            )
        else:
            log.info("ENGINE COLD START | warming up (%d samples needed)",
                     self.engine.zscore_window)
        guard_cfg = initial_strategy.get("guardrails", {})
        self.circuit = CircuitBreaker(
            dd_soft=float(guard_cfg.get("dd_soft", 0.06)),
            dd_hard=float(guard_cfg.get("dd_hard", 0.08)),
        )
        # Hold the most recent engine snapshot for exit evaluation
        self._last_snapshot: dict = {"C": 0.0, "K": 0.0, "regime": "warmup", "ready": False}

        # Auto-engage: True once we've flipped the engine on (or if it was already on)
        self._engine_auto_engaged: bool = bool(
            initial_strategy.get("coefficient_engine_enabled", False)
        )

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    async def run_forever(self) -> None:
        log.info("Loop started — tick every %ds", TICK_SECONDS)
        # Fire-and-forget: Gist restore runs in the background so the main
        # trading loop starts immediately and heartbeats are written from tick 1.
        asyncio.create_task(self._startup_restore())
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
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts  = time.time()

        # Write a minimal heartbeat immediately so the monitor always shows a
        # fresh timestamp — even if _fetch_all hangs for a full tick cycle.
        self._write_heartbeat_alive(now_iso)

        # Core data — hard 25s timeout so a hung adapter can't stall the loop.
        try:
            data = await asyncio.wait_for(self._fetch_all(), timeout=25.0)
        except asyncio.TimeoutError:
            log.warning("_fetch_all timed out after 25s — skipping tick")
            self._write_heartbeat(now_iso, {}, "?", self._last_snapshot)
            return

        strategy = self._load_strategy()

        price_data     = data.get("price", {})
        funding_data   = data.get("funding", {})
        sentiment_data = data.get("sentiment", {})

        current_price  = float(price_data.get("close", 0.0))
        rsi            = float(price_data.get("rsi", 50.0))
        current_volume = float(price_data.get("volume", 0.0))
        macd_hist      = float(price_data.get("macd_hist", 0.0))
        bb_pct         = float(price_data.get("bb_pct", 0.5))
        atr            = float(price_data.get("atr", 0.0))
        ob_imbalance   = float(price_data.get("ob_imbalance", 0.0))
        funding_rate    = float(funding_data.get("funding_rate", 0.0))
        fng_value       = float(sentiment_data.get("fng_value",       50.0))
        ls_ratio        = float(sentiment_data.get("ls_ratio",         1.0))
        top_ls_ratio    = float(sentiment_data.get("top_ls_ratio",     1.0))
        taker_buy_ratio = float(sentiment_data.get("taker_buy_ratio",  0.5))
        oi_pct_change   = float(sentiment_data.get("oi_pct_change",    0.0))

        if current_price == 0.0:
            log.warning("Price is 0 — skipping tick (adapter failure)")
            self._write_heartbeat(now_iso, price_data, strategy.get("version", "?"), {})
            return

        if current_volume > 0:
            self._volume_history.append(current_volume)
            if len(self._volume_history) > VOLUME_LOOKBACK:
                self._volume_history.pop(0)

        # ── Tick the engine (always — for telemetry) ──────────────────────────
        candles = get_cached_candles()
        snapshot = self.engine.tick(
            price_data, candles=candles,
            funding_rate=funding_rate,
            fng_value=fng_value,
            ls_ratio=ls_ratio,
            top_ls_ratio=top_ls_ratio,
            taker_buy_ratio=taker_buy_ratio,
            oi_pct_change=oi_pct_change,
        )
        self._last_snapshot = snapshot
        C = float(snapshot["C"])
        K = float(snapshot["K"])
        regime = snapshot["regime"]

        # ── Master switch ─────────────────────────────────────────────────────
        engine_on = bool(strategy.get("coefficient_engine_enabled", False))

        # ── Drawdown / circuit breaker ────────────────────────────────────────
        closed_trades_so_far = self._load_closed_trades()
        starting = float(self.goal.get("starting_balance", 100_000))
        dd = current_drawdown(closed_trades_so_far, starting)
        cb_state = self.circuit.evaluate(dd)
        if cb_state == "hard_trip" and self._open_trades:
            log.error("CIRCUIT BREAKER HARD — flattening %d open positions (DD=%.2f%%)",
                      len(self._open_trades), dd * 100)
            for trade in list(self._open_trades):
                self._close_trade(trade, current_price, now_iso, "circuit_breaker",
                                  funding_rate, strategy)

        # ── Strategy config (shared) ──────────────────────────────────────────
        entry_cfg = strategy.get("entry", {})

        # ── EXIT logic ────────────────────────────────────────────────────────
        for trade in list(self._open_trades):
            reason = self._evaluate_exits(
                trade=trade, current_price=current_price,
                strategy=strategy, snapshot=snapshot, rsi=rsi,
                macd_hist=macd_hist, atr=atr, engine_on=engine_on,
            )
            if reason:
                self._close_trade(trade, current_price, now_iso, reason,
                                  funding_rate, strategy)

        # ── ENTRY logic ───────────────────────────────────────────────────────
        if cb_state != "hard_trip":
            if engine_on and snapshot.get("ready", False):
                self._maybe_open_engine(
                    snapshot=snapshot, current_price=current_price,
                    strategy=strategy, now_iso=now_iso, now_ts=now_ts,
                    funding_rate=funding_rate, fng_value=fng_value,
                    closed_trades=closed_trades_so_far,
                )
            elif not engine_on:
                self._maybe_open_legacy(
                    strategy=strategy, entry_cfg=entry_cfg,
                    rsi=rsi, macd_hist=macd_hist, bb_pct=bb_pct,
                    ob_imbalance=ob_imbalance, atr=atr,
                    current_price=current_price, current_volume=current_volume,
                    now_iso=now_iso, now_ts=now_ts, snapshot=snapshot,
                )

        # ── Background tasks ──────────────────────────────────────────────────
        self._tick_count += 1
        if self._tick_count % 30 == 0:
            asyncio.create_task(self._run_model())

        self._write_heartbeat(now_iso, price_data, strategy.get("version", "?"), snapshot)
        self.engine.save_state(self.engine_state_file)

    # ------------------------------------------------------------------ #
    #  Exit evaluation                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate_exits(
        self, *, trade: dict, current_price: float, strategy: dict,
        snapshot: dict, rsi: float, macd_hist: float, atr: float, engine_on: bool,
    ) -> str | None:
        trade_id    = trade["id"]
        entry_price = float(trade["entry_price"])
        direction   = trade["direction"]

        # Update per-trade peak (for trailing/profit-lock) and trough (for MAE)
        if trade_id not in self._peak_prices:
            self._peak_prices[trade_id] = entry_price
        if trade_id not in self._trough_prices:
            self._trough_prices[trade_id] = entry_price

        peak = self._peak_prices[trade_id]
        trough = self._trough_prices[trade_id]
        if direction == "long":
            if current_price > peak:
                self._peak_prices[trade_id] = current_price
                peak = current_price
            if current_price < trough:
                self._trough_prices[trade_id] = current_price
                trough = current_price
            raw_pnl = (current_price - entry_price) / entry_price
        else:
            if current_price < peak:
                self._peak_prices[trade_id] = current_price
                peak = current_price
            if current_price > trough:
                self._trough_prices[trade_id] = current_price
                trough = current_price
            raw_pnl = (entry_price - current_price) / entry_price

        # ── NEW: vol-scaled stop / R-multiple TP (only if engine_on) ──────────
        if engine_on and snapshot.get("ready", False):
            exit_cfg = strategy.get("exits", {})
            k_sl       = float(exit_cfg.get("k_sl", 1.0))
            r_multiple = float(exit_cfg.get("r_multiple", 1.5))
            bounds     = tuple(exit_cfg.get("stop_pct_bounds", [0.003, 0.015]))
            horizon    = int(exit_cfg.get("trade_horizon_bars", 30))
            vol_horizon = self.engine.vol.forecast_horizon_std(horizon)
            vs_stop = vol_scaled_stop_pct(vol_horizon, k_sl=k_sl, bounds=bounds)
            if raw_pnl <= -vs_stop:
                return "vol_scaled_stop"
            if raw_pnl >= r_multiple * vs_stop:
                return "r_multiple_tp"

            # Coefficient exits (flip + collapse)
            coef_reason = evaluate_coefficient_exits(
                trade, current_price, float(snapshot["C"]),
                tau_exit=float(strategy.get("coefficient", {}).get("tau_exit", 0.05)),
            )
            if coef_reason:
                return coef_reason

        # ── LEGACY cascade ────────────────────────────────────────────────────
        sl_pct = float(strategy.get("stop_loss_pct", 0.5))
        tp_pct = float(strategy.get("take_profit_pct", 0.5))
        ts_pct = float(strategy.get("trailing_stop_pct", 0.0))

        # If engine_on we may have already hit vs_stop/r_multiple_tp above; only fall
        # back to legacy fixed SL/TP when the engine is off, to avoid double gating.
        if not engine_on:
            if raw_pnl <= -(sl_pct / 100.0):
                return "stop_loss"
            if tp_pct > 0 and raw_pnl >= tp_pct / 100.0:
                return "take_profit"

        # Always-on trailing stop
        if ts_pct > 0 and peak > 0:
            if direction == "long":
                if (peak - current_price) / peak >= ts_pct / 100.0:
                    return "trailing_stop"
            else:
                if (current_price - peak) / peak >= ts_pct / 100.0:
                    return "trailing_stop"

        # Profit lock
        pl_activate = float(strategy.get("profit_lock_activate_pct", 0.25))
        pl_trail    = float(strategy.get("profit_lock_trail_pct",    0.15))
        if pl_activate > 0 and pl_trail > 0:
            peak_pnl = abs(peak - entry_price) / entry_price
            if peak_pnl >= pl_activate / 100.0:
                if direction == "long":
                    if (peak - current_price) / peak >= pl_trail / 100.0:
                        return "profit_lock"
                else:
                    if (current_price - peak) / peak >= pl_trail / 100.0:
                        return "profit_lock"

        # RSI exit — only when profitable
        ecfg = strategy.get("entry", {})
        long_rsi_exit  = float(ecfg.get("long_rsi_exit", 78))
        short_rsi_exit = float(ecfg.get("short_rsi_exit", 25))
        if direction == "long" and rsi >= long_rsi_exit and raw_pnl > 0:
            return "rsi_overbought"
        if direction == "short" and rsi <= short_rsi_exit and raw_pnl > 0:
            return "rsi_oversold"

        # MACD reversal — only when profitable
        if raw_pnl > 0:
            entry_macd = float(trade.get("entry_macd_hist", 0.0))
            if direction == "long" and entry_macd > 0 and macd_hist < 0:
                return "macd_reversal"
            if direction == "short" and entry_macd < 0 and macd_hist > 0:
                return "macd_reversal"

        # Time exit
        if (time.time() - int(trade_id[1:])) >= 86400:
            return "time_exit"

        return None

    # ------------------------------------------------------------------ #
    #  Trade lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def _close_trade(
        self, trade: dict, current_price: float, now_iso: str, reason: str,
        funding_rate: float, strategy: dict,
    ) -> None:
        trade_id = trade["id"]
        entry_price = float(trade["entry_price"])
        direction = trade["direction"]
        lev = float(trade.get("leverage", 1.0))

        if direction == "long":
            raw_pnl = (current_price - entry_price) / entry_price
        else:
            raw_pnl = (entry_price - current_price) / entry_price

        gross_lev = raw_pnl * lev

        # Holding time
        opened_ts = int(trade_id[1:])
        holding_min = max(0.0, (time.time() - opened_ts) / 60.0)

        # Net PnL after fees + funding
        costs_cfg = strategy.get("costs", {})
        fee_bps   = float(costs_cfg.get("taker_fee_bps", 5.0))
        fp_hours  = float(costs_cfg.get("funding_period_hours", 8.0))
        net_lev   = net_pnl(
            gross_lev,
            taker_fee_bps=fee_bps,
            funding_rate=funding_rate * lev,   # funding scales with leverage
            holding_minutes=holding_min,
            direction=direction,
        )

        peak = self._peak_prices.get(trade_id, entry_price)
        trough = self._trough_prices.get(trade_id, entry_price)
        if direction == "long":
            mfe = max(0.0, (peak - entry_price) / entry_price)
            mae = max(0.0, (entry_price - trough) / entry_price)
        else:
            mfe = max(0.0, (entry_price - peak) / entry_price)
            mae = max(0.0, (trough - entry_price) / entry_price)

        closed = {
            **trade,
            "exit_price":       current_price,
            "exit_time":        now_iso,
            "exit_reason":      reason,
            "pnl_pct":          round(raw_pnl, 6),
            "pnl_pct_levered":  round(gross_lev, 6),
            "pnl_pct_gross":    round(gross_lev, 6),
            "pnl_pct_net":      round(net_lev, 6),
            "peak_price":       peak,
            "trough_price":     trough,
            "mfe":              round(mfe, 6),
            "mae":              round(mae, 6),
            "holding_minutes":  round(holding_min, 2),
            "funding_at_exit":  round(funding_rate, 6),
            "closed":           True,
        }

        log.info(
            "EXIT | reason=%s dir=%s price=%.2f gross=%.3f%% net=%.3f%% lev=%.2fx id=%s",
            reason, direction, current_price,
            gross_lev * 100, net_lev * 100, lev, trade_id,
        )

        try:
            self._open_trades.remove(trade)
        except ValueError:
            pass
        self._peak_prices.pop(trade_id, None)
        self._trough_prices.pop(trade_id, None)
        self._save_open_trades()

        self._append_trade(closed)
        # Fire-and-forget: push updated trade history to GitHub Gist.
        # Non-blocking — a push failure never affects trading.
        asyncio.create_task(backup.push_to_gist(self.trades_file))
        asyncio.create_task(self._post_close(strategy))

    async def _post_close(self, strategy: dict) -> None:
        """Run improvement cycles + model refresh after a trade closes."""
        engine_on = bool(strategy.get("coefficient_engine_enabled", False))
        cadence = int(strategy.get("improvement", {}).get("fast_cadence_trades", 10))

        closed = self._load_closed_trades()
        n = len(closed)

        # ── Auto-engage: flip engine on when threshold reached ────────────────
        if not engine_on:
            self._maybe_auto_engage_engine(strategy, closed)
            # Re-check in case we just flipped it
            engine_on = bool(self._load_strategy().get("coefficient_engine_enabled", False))

        if engine_on and n > 0 and n % cadence == 0 and n != self._last_fast_cycle_at:
            self._last_fast_cycle_at = n
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    run_fast_cycle,
                    self.state_dir, closed, self.goal,
                )
                # Reload weights into the engine
                self.engine.regime_weights = self._load_weights(self._load_strategy())
            except Exception as exc:
                log.warning("fast_cycle failed (non-fatal): %s", exc)
        elif not engine_on:
            # Legacy reflect path
            await self._maybe_reflect()

        asyncio.create_task(self._run_model())

    # ------------------------------------------------------------------ #
    #  Entry — engine path                                                 #
    # ------------------------------------------------------------------ #

    def _maybe_open_engine(
        self, *, snapshot: dict, current_price: float, strategy: dict,
        now_iso: str, now_ts: float, funding_rate: float, fng_value: float,
        closed_trades: list[dict],
    ) -> None:
        C = float(snapshot["C"])
        K = float(snapshot["K"])
        coef_cfg = strategy.get("coefficient", {})
        siz_cfg  = strategy.get("sizing", {})
        lev_cfg  = strategy.get("leverage", {})
        tau_enter = float(coef_cfg.get("tau_enter", 0.12))

        if abs(C) <= tau_enter:
            return

        # Cooldown
        cooldown_min = float(siz_cfg.get("reentry_cooldown_min", 2.0))
        if (now_ts - self._last_entry_time) < cooldown_min * 60:
            return

        # Position cap
        max_pos = int(strategy.get("max_open_positions", 3))
        if len(self._open_trades) >= max_pos:
            return

        # Circuit-breaker soft trip: tighten further
        if self.circuit.state == "soft_trip":
            tau_enter = tau_enter * 1.5
            if abs(C) <= tau_enter:
                return

        # Kelly size
        buckets = kelly_buckets(closed_trades)
        guard_cfg = strategy.get("guardrails", {})
        size, source = fractional_kelly_size(
            C, K, buckets,
            lambda_kelly=float(siz_cfg.get("lambda_kelly", 0.35))
                * (0.5 if self.circuit.state == "soft_trip" else 1.0),
            size_floor=float(siz_cfg.get("size_floor", 0.05)),
            size_cap=float(siz_cfg.get("size_cap", 0.40)),
            bucket_min_samples=int(guard_cfg.get("kelly_bucket_min_samples", 50)),
        )
        if size <= 0:
            return

        # Kelly budget
        kelly_max = float(siz_cfg.get("kelly_budget_max", 0.60))
        if not respects_budget(size, self._open_trades, kelly_budget_max=kelly_max):
            return

        # Leverage
        lev = leverage_from_K(
            K,
            vol_penalty=float(snapshot.get("vol_penalty", 1.0)),
            lev_min=float(lev_cfg.get("min", 1.0)),
            lev_max=float(lev_cfg.get("max", 3.0)),
        )

        direction = "long" if C > 0 else "short"
        trade = {
            "id":                  f"T{int(now_ts)}",
            "asset":               self.asset,
            "entry_price":         current_price,
            "entry_time":          now_iso,
            "direction":           direction,
            "leverage":            round(lev, 3),
            "position_size_r":     round(size, 4),
            "size_source":         source,
            "strategy_version":    strategy.get("version", "?"),
            "mode":                self.mode,
            # Engine snapshot
            "entry_C":             round(C, 4),
            "entry_K":             round(K, 4),
            "entry_regime":        snapshot["regime"],
            "entry_sub_signals":   snapshot["sub_signals"],
            "entry_vol_forecast":  snapshot["vol_forecast"],
            "entry_funding":       funding_rate,
            "entry_fng":           fng_value,
            # Indicators for legacy MACD-reversal exit + analysis
            "entry_macd_hist":     None,   # filled below from snapshot if available
            "entry_rsi":           None,
            "entry_bb_pct":        None,
            "entry_ob_imb":        None,
        }

        self._open_trades.append(trade)
        self._peak_prices[trade["id"]]   = current_price
        self._trough_prices[trade["id"]] = current_price
        self._last_entry_time = now_ts
        self._save_open_trades()

        log.info(
            "ENGINE ENTRY | dir=%s C=%+0.3f K=%.3f regime=%s size=%.2f%% lev=%.2fx "
            "source=%s open=%d id=%s",
            direction, C, K, snapshot["regime"],
            size * 100, lev, source, len(self._open_trades), trade["id"],
        )

    # ------------------------------------------------------------------ #
    #  Entry — legacy path (preserved for fallback)                        #
    # ------------------------------------------------------------------ #

    def _maybe_open_legacy(
        self, *, strategy: dict, entry_cfg: dict,
        rsi: float, macd_hist: float, bb_pct: float, ob_imbalance: float,
        atr: float, current_price: float, current_volume: float,
        now_iso: str, now_ts: float, snapshot: dict,
    ) -> None:
        cooldown_minutes  = float(strategy.get("reentry_cooldown_minutes", 1.0))
        max_positions     = int(strategy.get("max_open_positions",        3))

        direction_mode  = entry_cfg.get("direction", "both")
        long_threshold  = float(entry_cfg.get("long_threshold",  55))
        short_threshold = float(entry_cfg.get("short_threshold", 65))
        vol_surge_mult  = float(entry_cfg.get("volume_surge_multiplier", 0.0))

        in_cooldown = (
            cooldown_minutes > 0
            and self._last_entry_time > 0
            and (now_ts - self._last_entry_time) < cooldown_minutes * 60
        )
        volume_ok = True
        if vol_surge_mult > 0 and len(self._volume_history) >= 5:
            avg_vol   = sum(self._volume_history[:-1]) / max(1, len(self._volume_history) - 1)
            volume_ok = current_volume >= avg_vol * vol_surge_mult

        can_enter = (
            not in_cooldown and volume_ok and len(self._open_trades) < max_positions
        )
        if not can_enter:
            return

        entry_direction = None
        entry_score = 0
        if direction_mode in ("long", "both") and rsi < long_threshold:
            entry_direction = "long"
            entry_score = 1
            if entry_cfg.get("macd_confirm", False) and macd_hist > 0: entry_score += 1
            if entry_cfg.get("bb_confirm",   False) and bb_pct < 0.35: entry_score += 1
            if entry_cfg.get("ob_confirm",   False) and ob_imbalance > 0: entry_score += 1
        elif direction_mode in ("short", "both") and rsi > short_threshold:
            entry_direction = "short"
            entry_score = 1
            if entry_cfg.get("macd_confirm", False) and macd_hist < 0: entry_score += 1
            if entry_cfg.get("bb_confirm",   False) and bb_pct > 0.65: entry_score += 1
            if entry_cfg.get("ob_confirm",   False) and ob_imbalance < 0: entry_score += 1

        if entry_direction is None:
            return

        lev_base  = float(strategy.get("leverage_base",         1.5))
        lev_max   = float(strategy.get("leverage_max",          3.0))
        ps_base   = float(strategy.get("position_size_base",    0.15))
        ps_max    = float(strategy.get("position_size_max",     0.40))
        ps_dampen = bool(strategy.get("position_size_atr_dampen", True))

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
            "id":                  f"T{int(now_ts)}",
            "asset":               self.asset,
            "entry_price":         current_price,
            "entry_time":          now_iso,
            "direction":           entry_direction,
            "stop_loss_pct":       float(strategy.get("stop_loss_pct", 0.5)),
            "take_profit_pct":     float(strategy.get("take_profit_pct", 0.5)),
            "trailing_stop_pct":   float(strategy.get("trailing_stop_pct", 0.0)),
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
            # Also snapshot the engine's read at entry so attribution still has data
            "entry_C":             snapshot.get("C", 0.0),
            "entry_K":             snapshot.get("K", 0.0),
            "entry_regime":        snapshot.get("regime", "unknown"),
            "entry_sub_signals":   snapshot.get("sub_signals", {}),
        }
        self._open_trades.append(new_trade)
        self._peak_prices[new_trade["id"]]   = current_price
        self._trough_prices[new_trade["id"]] = current_price
        self._last_entry_time = now_ts
        self._save_open_trades()
        log.info(
            "LEGACY ENTRY | dir=%s score=%d/4 price=%.2f rsi=%.1f lev=%.2fx "
            "pos=%.1f%% open=%d id=%s",
            entry_direction, entry_score, current_price, rsi,
            leverage, pos_size_r * 100, len(self._open_trades), new_trade["id"],
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Auto-engage                                                         #
    # ------------------------------------------------------------------ #

    def _maybe_auto_engage_engine(self, strategy: dict, closed_trades: list[dict]) -> None:
        """
        Flip coefficient_engine_enabled → true once auto_engage_trades closed trades
        have been recorded.  Fires exactly once per process lifetime (guarded by
        _engine_auto_engaged).  Set improvement.auto_engage_trades = 0 to disable.
        """
        if self._engine_auto_engaged:
            return
        if bool(strategy.get("coefficient_engine_enabled", False)):
            # Already on — someone flipped it manually or a prior run engaged it.
            self._engine_auto_engaged = True
            return

        threshold = int(strategy.get("improvement", {}).get("auto_engage_trades", 50))
        if threshold <= 0:
            return   # disabled

        n = len(closed_trades)
        if n < threshold:
            return

        # ── Threshold crossed ─────────────────────────────────────────────────
        self._engine_auto_engaged = True
        log.info(
            "AUTO-ENGAGE | %d closed trades reached threshold=%d — "
            "flipping coefficient_engine_enabled = true",
            n, threshold,
        )

        # Fresh read so we don't race with cycle.py writes
        current = self._load_strategy()
        current["coefficient_engine_enabled"] = True
        with open(self.strategy_file, "w") as f:
            yaml.dump(current, f, default_flow_style=False, sort_keys=False)

        # Reload engine weights now that the engine is live
        self.engine.regime_weights = self._load_weights(current)

        # Record the event in hypotheses.jsonl
        hyp = {
            "ts":        datetime.now(timezone.utc).isoformat(),
            "version":   current.get("version", "?"),
            "layer":     "auto_engage",
            "event":     "coefficient_engine_enabled → true",
            "reason":    (
                f"Auto-engagement: {n} closed trades reached threshold of {threshold}. "
                "Coefficient engine now drives entries, sizing, and exits."
            ),
            "trades_at_engage": n,
            "threshold":        threshold,
        }
        hyp_file = self.state_dir / "hypotheses.jsonl"
        with open(hyp_file, "a") as f:
            f.write(json.dumps(hyp) + "\n")

        log.info("ENGINE ENGAGED — strategy.yaml updated, weights reloaded")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_strategy(self) -> dict:
        with open(self.strategy_file) as f:
            return yaml.safe_load(f)

    def _load_weights(self, strategy: dict) -> dict[str, dict[str, float]]:
        if self.weights_file.exists():
            try:
                return json.loads(self.weights_file.read_text())
            except Exception:
                pass
        return strategy.get("regime", {}).get("weights",
            CoefficientEngine._default_regime_weights())

    def _load_closed_trades(self) -> list[dict]:
        if not self.trades_file.exists():
            return []
        out: list[dict] = []
        for line in self.trades_file.read_text().splitlines():
            if line.strip():
                try:
                    t = json.loads(line)
                    if t.get("closed"):
                        out.append(t)
                except Exception:
                    pass
        return out

    def _append_trade(self, trade: dict) -> None:
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def _save_open_trades(self) -> None:
        """Persist open positions + peak/trough prices to disk so they survive restarts."""
        snapshot = {
            "open_trades":   self._open_trades,
            "peak_prices":   self._peak_prices,
            "trough_prices": self._trough_prices,
        }
        with open(self.open_trades_file, "w") as f:
            json.dump(snapshot, f)

    def _restore_open_trades(self) -> None:
        """Reload open positions from disk on startup (survives Railway redeploys)."""
        if not self.open_trades_file.exists():
            return
        try:
            data = json.loads(self.open_trades_file.read_text())
            self._open_trades   = data.get("open_trades", [])
            self._peak_prices   = {k: float(v) for k, v in data.get("peak_prices", {}).items()}
            self._trough_prices = {k: float(v) for k, v in data.get("trough_prices", {}).items()}
            if self._open_trades:
                log.info(
                    "RESTORED %d open trade(s) from disk: %s",
                    len(self._open_trades),
                    [t["id"] for t in self._open_trades],
                )
        except Exception as exc:
            log.warning("Could not restore open trades (starting fresh): %s", exc)
            self._open_trades   = []
            self._peak_prices   = {}
            self._trough_prices = {}

    async def _startup_restore(self) -> None:
        """
        Restore trade history from GitHub Gist if trades.jsonl is empty.
        Called once at the start of run_forever() — after the event loop is
        running — so __init__ stays fast and Railway health checks pass.
        """
        try:
            loop = asyncio.get_event_loop()
            restored = await loop.run_in_executor(
                None, backup.restore_from_gist, self.trades_file
            )
            if restored:
                log.info("STARTUP | Restored %d trade(s) from GitHub Gist", restored)
                # Reload open trades in case they reference restored history
                self._restore_open_trades()
        except Exception as exc:
            log.warning("STARTUP | Gist restore failed (non-fatal): %s", exc)

    async def _maybe_reflect(self) -> None:
        """Legacy reflect cycle — only used when coefficient_engine_enabled = false."""
        cadence = int(self.goal.get("reflection_every", 10))
        if cadence <= 0:
            return
        closed_count = len(self._load_closed_trades())
        if closed_count == 0 or closed_count % cadence != 0:
            return
        if closed_count == self._last_reflected_at:
            return
        self._last_reflected_at = closed_count
        log.info("LEGACY REFLECT | trade #%d hit cadence of %d", closed_count, cadence)
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

    def _write_heartbeat(self, ts: str, price_data: dict, strategy_version: str, snapshot: dict) -> None:
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
                    "id":           t["id"],
                    "direction":    t["direction"],
                    "entry_price":  t["entry_price"],
                    "entry_C":      t.get("entry_C"),
                    "entry_K":      t.get("entry_K"),
                    "entry_score":  t.get("entry_score"),
                    "leverage":     t.get("leverage", 1.5),
                    "size":         t.get("position_size_r"),
                } for t in self._open_trades
            ],
            "peak_price":           max(self._peak_prices.values(), default=0.0),
            "consecutive_failures": self._consecutive_failures,
            # Engine telemetry — visible regardless of master switch
            "engine": {
                "C":            snapshot.get("C", 0.0),
                "K":            snapshot.get("K", 0.0),
                "regime":       snapshot.get("regime", "warmup"),
                "regime_cert":  snapshot.get("regime_cert", 0.0),
                "sub_signals":  snapshot.get("sub_signals", {}),
                "weights_used": snapshot.get("weights_used", {}),
                "agreement":    snapshot.get("agreement", 0.0),
                "vol_penalty":  snapshot.get("vol_penalty", 1.0),
                "vol_forecast": snapshot.get("vol_forecast", 0.0),
                "ready":        snapshot.get("ready", False),
                "samples":      snapshot.get("samples", len(self.engine.features)),
            },
            "circuit_breaker": {
                "state":         self.circuit.state,
                "tripped_at":    self.circuit.tripped_at,
            },
        }
        with open(self.heartbeat_file, "w") as f:
            json.dump(hb, f, indent=2)

    def _write_heartbeat_alive(self, ts: str) -> None:
        """
        Write a minimal heartbeat at the START of each tick — before waiting
        for adapters — so the monitor always shows a fresh timestamp even if
        _fetch_all() hangs.  Preserves the previous price/engine data so the
        monitor never goes blank.
        """
        try:
            existing = json.loads(self.heartbeat_file.read_text())
        except Exception:
            existing = {}
        existing["ts"] = ts   # stamp as of NOW
        try:
            with open(self.heartbeat_file, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception:
            pass   # non-fatal

    # ------------------------------------------------------------------ #
    #  Adapter fetches                                                     #
    # ------------------------------------------------------------------ #

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

    async def _fetch_tolerant(self, name: str, fetch_fn) -> dict:
        """Non-blocking fetch — never raises, returns empty dict on failure.
        Used for funding/sentiment where a hiccup shouldn't gate trading."""
        try:
            return await asyncio.wait_for(fetch_fn(self.asset), timeout=8.0)
        except Exception as exc:
            log.debug("Tolerant adapter '%s' failed (non-fatal): %s", name, exc)
            return {}

    async def _fetch_all(self) -> dict:
        results = await asyncio.gather(
            self._fetch_with_retry("price",   fetch_price),
            self._fetch_with_retry("onchain", fetch_onchain),
            self._fetch_with_retry("news",    fetch_news),
            self._fetch_with_retry("macro",   fetch_macro),
            self._fetch_tolerant("funding",   fetch_funding),
            self._fetch_tolerant("sentiment", fetch_sentiment),
            return_exceptions=False,
        )
        return {
            "price":     results[0],
            "onchain":   results[1],
            "news":      results[2],
            "macro":     results[3],
            "funding":   results[4],
            "sentiment": results[5],
        }
