"""
score.py — scores a list of closed trades against goal.yaml.

Returns a float in [-1.0, +1.0]:
  +1.0  = all three dimensions hit target
   0.0  = breakeven / flat
  -1.0  = worst-case across all dimensions
  < failure_below → score is clamped to -1.0 (steeply negative)

Composite of three equally-weighted sub-scores:
  1. Realised return vs target_return_30d
  2. Drawdown vs max_drawdown
  3. Sharpe vs min_sharpe
"""
import math
from typing import Any


def score(trades: list[dict], goal: dict) -> float:
    """
    trades: list of dicts from trades.jsonl (closed trades only).
    goal:   dict from goal.yaml.
    Returns float in [-1.0, +1.0].
    """
    closed = [t for t in trades if t.get("closed")]
    if not closed:
        return 0.0

    pnls = [t["pnl_pct"] for t in closed]

    # --- 1. Return score ---
    total_return = sum(pnls)
    target = goal["target_return_30d"]
    failure_below = goal.get("failure_below", -0.04)

    if total_return <= failure_below:
        return -1.0

    return_score = _clamp(total_return / target, -1.0, 1.0)

    # --- 2. Drawdown score ---
    max_dd = _max_drawdown(pnls)
    max_allowed = goal["max_drawdown"]
    if max_dd >= max_allowed:
        drawdown_score = -1.0
    else:
        # 0 drawdown → +1, approaching max_allowed → 0
        drawdown_score = 1.0 - (max_dd / max_allowed)

    # --- 3. Sharpe score ---
    sharpe = _sharpe(pnls)
    min_sharpe = goal["min_sharpe"]
    sharpe_score = _clamp(sharpe / min_sharpe - 0.5, -1.0, 1.0)

    # Equal-weight composite
    composite = (return_score + drawdown_score + sharpe_score) / 3.0
    return round(_clamp(composite, -1.0, 1.0), 4)


# ------------------------------------------------------------------ #
#  Internals                                                           #
# ------------------------------------------------------------------ #

def _max_drawdown(pnls: list[float]) -> float:
    """Peak-to-trough drawdown over cumulative PnL series."""
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from per-trade PnL list."""
    if len(pnls) < 2:
        return 0.0
    n = len(pnls)
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    # Annualise assuming ~252 trading days, one trade per day (conservative)
    return ((mean - risk_free) / std) * math.sqrt(252)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
