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
