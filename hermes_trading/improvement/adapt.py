"""
adapt.py — weight and scalar updates with shrinkage and clamps (Layer 1, 2).

For each directional block, move its weight toward the IC profile,
blended with the prior, rate-limited, and renormalized so the regime
row still sums to 1.0.

  W_IC_b   = max(IC_b, 0) / Σ_b max(IC_b, 0)
  W_new_b  = (1 − rho) · W_prior_b + rho · W_IC_b
  W_new_b  = clip(W_new_b, W_prior_b − step, W_prior_b + step)
  renormalize

Defaults: rho = 0.25, step = 0.10. A block with negative IC is driven
toward zero rather than sign-flipped, since flips on a noisy estimate
are how adaptive weighters self-destruct.

Also adapts a small set of scalars from realized distributions, each
rate-limited and clamped.
"""
from __future__ import annotations

from typing import Any


# ── Block weight update ──────────────────────────────────────────────────────

def update_block_weights(
    prior_weights: dict[str, float],   # {mom, rev, mic, sen} summing to 1
    block_ics: dict[str, float],       # {mom, rev, mic, sen}
    rho: float = 0.25,
    step: float = 0.10,
) -> dict[str, float]:
    """
    Shrinkage-blended weight update. Always returns weights summing to 1.0.
    """
    # Zero-floor and normalize IC
    pos_ic = {b: max(ic, 0.0) for b, ic in block_ics.items()}
    total_ic = sum(pos_ic.values())
    if total_ic <= 0:
        # No positive-IC blocks: contract toward equal weights gently
        equal = 1.0 / len(prior_weights)
        target = {b: equal for b in prior_weights}
    else:
        target = {b: v / total_ic for b, v in pos_ic.items()}

    # Blend toward target
    blended = {
        b: (1 - rho) * prior_weights.get(b, 0.0) + rho * target.get(b, 0.0)
        for b in prior_weights
    }

    # Rate-limit per-step movement
    clamped = {
        b: max(
            prior_weights[b] - step,
            min(prior_weights[b] + step, blended[b]),
        )
        for b in prior_weights
    }

    # Renormalize so row sums to 1
    total = sum(clamped.values())
    if total <= 0:
        return prior_weights
    return {b: v / total for b, v in clamped.items()}


# ── Scalar updates ───────────────────────────────────────────────────────────

def update_tau_enter(
    current: float,
    *,
    trades_in_window: int,
    trades_target: int,
    hit_rate: float,
    step: float = 0.01,
    bounds: tuple[float, float] = (0.06, 0.25),
) -> float:
    """If under target volume, loosen. If over target and hit rate weak, tighten."""
    if trades_in_window < trades_target:
        new = current - step
    elif trades_in_window > trades_target * 1.5 and hit_rate < 0.45:
        new = current + step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_k_sl(
    current: float,
    *,
    mae_p50: float,        # median Max Adverse Excursion (fraction)
    stop_pct_p50: float,   # median stop distance used (fraction)
    step: float = 0.1,
    bounds: tuple[float, float] = (0.5, 2.0),
) -> float:
    """
    If trades are routinely getting stopped well before MAE recovers, widen.
    If MAE rarely approaches the stop, tighten.
    """
    if stop_pct_p50 <= 0:
        return current
    ratio = mae_p50 / stop_pct_p50
    if ratio > 0.85:
        new = current + step
    elif ratio < 0.4:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_r_multiple(
    current: float,
    *,
    mfe_p50: float,        # median Max Favorable Excursion
    stop_pct_p50: float,
    step: float = 0.1,
    bounds: tuple[float, float] = (1.0, 3.0),
) -> float:
    """Push TP toward where winners typically peak."""
    if stop_pct_p50 <= 0:
        return current
    target_R = mfe_p50 / stop_pct_p50
    if target_R > current + 0.2:
        new = current + step
    elif target_R < current - 0.2:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


# ── Whipsaw-gating parameter updates (6 new scalars) ─────────────────────────
#
# Priority when churn_rate is high: raise c_exit_band first, then flip_persist,
# then min_hold_bars.  This matches the fallback logic in reflect.py.
#
# INVARIANT: c_exit_band < c_enter_band at all times.
# Enforced in update_c_exit_band — the clamping happens in code, not just in
# the prompt, so the deterministic fallback can never violate it either.

def update_c_enter_band(
    current: float,
    *,
    churn_rate: float,
    avg_hold_bars: float,
    step: float = 0.02,
    bounds: tuple[float, float] = (0.08, 0.30),
) -> float:
    """
    If churn_rate is high AND avg_hold is short, tighten c_enter_band so
    fewer near-zero signals open trades.  If trade frequency has become
    very low (avg_hold high + churn low), loosen slightly.
    """
    if churn_rate > 0.25 and avg_hold_bars < 2.0:
        new = current + step
    elif churn_rate < 0.05 and avg_hold_bars > 5.0:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_c_exit_band(
    current: float,
    *,
    churn_rate: float,
    c_enter_band: float,
    step: float = 0.02,
    bounds: tuple[float, float] = (0.05, 0.25),
) -> float:
    """
    Primary anti-churn lever.  If churn is high, widen the exit band so
    small reversals don't trigger flips.  Always enforces
    c_exit_band < c_enter_band (INVARIANT).
    """
    if churn_rate > 0.20:
        new = current + step
    elif churn_rate < 0.03:
        new = current - step
    else:
        new = current
    new = max(bounds[0], min(bounds[1], new))
    # Enforce invariant: exit band must be strictly below enter band
    new = min(new, c_enter_band - 0.02)
    return max(bounds[0], new)


def update_flip_persist(
    current: int,
    *,
    churn_rate: float,
    step: int = 1,
    bounds: tuple[int, int] = (1, 6),
) -> int:
    """
    If churn is still high after c_exit_band adjustment, require more
    consecutive confirming bars.
    """
    if churn_rate > 0.25:
        new = current + step
    elif churn_rate < 0.05:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_min_hold_bars(
    current: int,
    *,
    churn_rate: float,
    avg_hold_bars: float,
    step: int = 1,
    bounds: tuple[int, int] = (1, 10),
) -> int:
    """
    If avg_hold is shorter than min_hold (exits happening at the floor),
    raise the floor.  Only tighten if churn is low and holds are long.
    """
    if churn_rate > 0.20 or avg_hold_bars < float(current) + 0.5:
        new = current + step
    elif churn_rate < 0.03 and avg_hold_bars > float(current) * 3:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_reentry_lock_bars(
    current: int,
    *,
    churn_rate: float,
    step: int = 1,
    bounds: tuple[int, int] = (0, 15),
) -> int:
    """Widen lockout when churn is still present; narrow when churn is gone."""
    if churn_rate > 0.20:
        new = current + step
    elif churn_rate < 0.03:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_entry_persist(
    current: int,
    *,
    churn_rate: float,
    step: int = 1,
    bounds: tuple[int, int] = (1, 5),
) -> int:
    """
    Require more sustained conviction at entry when churn is high.
    Lighter than flip_persist — quicker entries than exits.
    """
    if churn_rate > 0.25:
        new = current + step
    elif churn_rate < 0.03:
        new = current - step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))


def update_lambda_kelly(
    current: float,
    *,
    rolling_sharpe: float,
    rolling_drawdown: float,
    dd_soft: float = 0.06,
    step: float = 0.025,
    bounds: tuple[float, float] = (0.20, 0.50),
) -> float:
    """Raise toward 0.5 when Sharpe is healthy; cut toward 0.25 when DD grows."""
    if rolling_drawdown > dd_soft:
        new = current - step
    elif rolling_sharpe > 1.0:
        new = current + step
    else:
        new = current
    return max(bounds[0], min(bounds[1], new))
