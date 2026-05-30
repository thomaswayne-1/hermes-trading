"""
cycle.py — Layer 1 fast attribution cycle orchestration.

Runs after every N closed trades. Computes IC per block over the rolling
window, updates the live regime weights with shrinkage + clamps, runs
the shadow validator, and persists the change to:
  - state/weights.json
  - state/strategy.yaml (bumped version)
  - state/attribution.jsonl (per-cycle IC snapshot)
  - state/hypotheses.jsonl (the change record)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .attribution import block_ic
from .adapt import (
    update_block_weights,
    update_tau_enter,
    update_lambda_kelly,
    update_k_sl,
    update_r_multiple,
)
from .guardrails import (
    current_drawdown,
    rolling_sharpe,
    shadow_validate,
)

log = logging.getLogger("hermes.improvement.cycle")


def _bump_version(v: str) -> str:
    try:
        return f"{int(v) + 1:02d}"
    except (ValueError, TypeError):
        return "01"


def run_fast_cycle(
    state_dir: Path,
    closed_trades: list[dict],
    goal: dict,
) -> dict[str, Any] | None:
    """
    Run one Layer 1 attribution cycle. Returns the change record dict if
    a change was applied, None otherwise.
    """
    strategy_file    = state_dir / "strategy.yaml"
    weights_file     = state_dir / "weights.json"
    attribution_file = state_dir / "attribution.jsonl"
    hypotheses_file  = state_dir / "hypotheses.jsonl"
    history_dir      = state_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    with open(strategy_file) as f:
        strategy = yaml.safe_load(f)

    improvement_cfg = strategy.get("improvement", {})
    window      = int(improvement_cfg.get("ic_window", 40))
    min_trades  = int(improvement_cfg.get("ic_min_trades", 20))
    rho         = float(improvement_cfg.get("rho", 0.25))
    weight_step = float(improvement_cfg.get("weight_step", 0.10))

    ics = block_ic(closed_trades, window=window, min_trades=min_trades)
    if ics is None:
        log.info("fast_cycle: insufficient data (need %d trades with entry_sub_signals)", min_trades)
        return None

    # Determine which regime to update (use the most common regime in window,
    # or just 'ranging' as a default since per-regime is Layer 2's job)
    recent = [t for t in closed_trades if t.get("entry_sub_signals")][-window:]
    regimes = [t.get("entry_regime", "ranging") for t in recent]
    if regimes:
        regime_to_update = max(set(regimes), key=regimes.count)
    else:
        regime_to_update = "ranging"

    # Load live weights
    if weights_file.exists():
        weights = json.loads(weights_file.read_text())
    else:
        weights = strategy.get("regime", {}).get("weights", {})

    prior = dict(weights.get(regime_to_update, {"mom": 0.25, "rev": 0.25, "mic": 0.25, "sen": 0.25}))
    new = update_block_weights(prior, ics, rho=rho, step=weight_step)

    # Scalar adaptations
    dd = current_drawdown(closed_trades, float(goal.get("starting_balance", 100_000)))
    sharpe = rolling_sharpe(closed_trades, window=window)
    hit_rate = sum(1 for t in recent if t.get("pnl_pct_net", t.get("pnl_pct", 0)) > 0) / len(recent) if recent else 0.0

    coef_cfg = strategy.get("coefficient", {})
    siz_cfg  = strategy.get("sizing", {})
    exit_cfg = strategy.get("exits", {})

    old_tau_enter = float(coef_cfg.get("tau_enter", 0.12))
    old_lambda    = float(siz_cfg.get("lambda_kelly", 0.35))
    old_k_sl      = float(exit_cfg.get("k_sl", 1.0))
    old_r_mult    = float(exit_cfg.get("r_multiple", 1.5))

    new_tau_enter = update_tau_enter(
        old_tau_enter,
        trades_in_window=len(recent),
        trades_target=window,
        hit_rate=hit_rate,
    )
    new_lambda = update_lambda_kelly(
        old_lambda,
        rolling_sharpe=sharpe,
        rolling_drawdown=dd,
        dd_soft=float(strategy.get("guardrails", {}).get("dd_soft", 0.06)),
    )

    # k_sl and r_multiple need MAE/MFE — skip if not tracked yet
    new_k_sl   = old_k_sl
    new_r_mult = old_r_mult
    mae_vals = [t.get("mae") for t in recent if t.get("mae") is not None]
    mfe_vals = [t.get("mfe") for t in recent if t.get("mfe") is not None]
    if mae_vals:
        mae_p50 = sorted(mae_vals)[len(mae_vals) // 2]
        stop_pct_p50 = float(strategy.get("stop_loss_pct", 0.5)) / 100.0
        new_k_sl = update_k_sl(old_k_sl, mae_p50=mae_p50, stop_pct_p50=stop_pct_p50)
    if mfe_vals:
        mfe_p50 = sorted(mfe_vals)[len(mfe_vals) // 2]
        stop_pct_p50 = float(strategy.get("stop_loss_pct", 0.5)) / 100.0
        new_r_mult = update_r_multiple(old_r_mult, mfe_p50=mfe_p50, stop_pct_p50=stop_pct_p50)

    # Compose the change record
    old_version = strategy.get("version", "00")
    new_version = _bump_version(old_version)
    record = {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "version":      new_version,
        "layer":        "fast_attribution",
        "regime":       regime_to_update,
        "block_ic":     {k: round(v, 4) for k, v in ics.items()},
        "weights_before": prior,
        "weights_after":  {k: round(v, 4) for k, v in new.items()},
        "scalars_changed": {
            k: [round(o, 4), round(n, 4)]
            for k, (o, n) in [
                ("tau_enter",    (old_tau_enter, new_tau_enter)),
                ("lambda_kelly", (old_lambda, new_lambda)),
                ("k_sl",         (old_k_sl, new_k_sl)),
                ("r_multiple",   (old_r_mult, new_r_mult)),
            ]
            if abs(o - n) > 1e-6
        },
        "reasoning":    f"Block IC: {ics}. Drawdown {dd:.4f}. Sharpe {sharpe:.3f}. "
                        f"Hit rate {hit_rate:.2%} over {len(recent)} trades.",
    }

    # Shadow validation
    passed, msg = shadow_validate(record, closed_trades, window=int(strategy.get("guardrails", {}).get("shadow_window", 100)))
    record["shadow_validation"] = ("passed: " if passed else "rejected: ") + msg

    if not passed:
        log.warning("fast_cycle: shadow validation rejected change (%s) — strategy unchanged", msg)
        # still log the attempt
        with open(hypotheses_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        return None

    # Apply: archive old, write new
    archive_path = history_dir / f"v{old_version}.yaml"
    with open(archive_path, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)

    weights[regime_to_update] = new
    weights_file.write_text(json.dumps(weights, indent=2))

    strategy["version"] = new_version
    strategy.setdefault("coefficient", {})["tau_enter"]    = new_tau_enter
    strategy.setdefault("sizing",      {})["lambda_kelly"] = new_lambda
    strategy.setdefault("exits",       {})["k_sl"]         = new_k_sl
    strategy.setdefault("exits",       {})["r_multiple"]   = new_r_mult
    strategy.setdefault("regime", {}).setdefault("weights", {})[regime_to_update] = new

    with open(strategy_file, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)

    # Log
    with open(attribution_file, "a") as f:
        f.write(json.dumps({
            "ts":     record["ts"],
            "regime": regime_to_update,
            "ic":     record["block_ic"],
            "n":      len(recent),
        }) + "\n")
    with open(hypotheses_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    log.info(
        "fast_cycle applied: v%s → v%s. Regime=%s. Weights %s. Scalars %s",
        old_version, new_version, regime_to_update,
        record["weights_after"], record["scalars_changed"],
    )
    return record
