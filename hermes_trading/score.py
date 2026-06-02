"""
score.py — scores a list of closed trades against goal.yaml.

Returns a float in [-1.0, +1.0]:
  +1.0  = all three dimensions hit target
   0.0  = breakeven / flat
  -1.0  = worst-case across all dimensions
  < failure_below → score is clamped to -1.0 (steeply negative)

Composite of four sub-scores (weights 1 / 1 / 1 / 0.5 → normalised by 3.5):
  1. Realised return vs target_return_30d
  2. Drawdown vs max_drawdown
  3. Sharpe vs min_sharpe
  4. Churn penalty — sub-fee-move coefficient exits drag the score toward -1
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

    # Use levered PnL if available (reflects actual capital at risk)
    pnls = [t.get("pnl_pct_levered", t["pnl_pct"]) for t in closed]

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

    # --- 4. Churn score (whipsaw-gating patch) ---
    # churn_rate is the fraction of coefficient exits where |price move| < fee
    # round-trip.  0 churn → +1; ≥ 50% churn → -1.  Weight 0.5 so it informs
    # the tuner without dominating the other three dimensions.
    fee_rt     = 0.001   # 0.10% default; matches churn_monitor.fee_roundtrip
    churn_rate = _churn_rate(closed, fee_roundtrip=fee_rt)
    churn_score = _clamp(1.0 - churn_rate * 2.0, -1.0, 1.0)

    # Weighted composite (3.5 = 1+1+1+0.5)
    composite = (return_score + drawdown_score + sharpe_score + churn_score * 0.5) / 3.5
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


def _churn_rate(closed: list[dict], fee_roundtrip: float = 0.001) -> float:
    """
    Fraction of trades that are coefficient exits with |price move| < fee_roundtrip.
    Used as the churn sub-score input.
    """
    _COEFF = {"coefficient_flip", "coefficient_collapse"}
    n = len(closed)
    if n == 0:
        return 0.0
    sub_fee = 0
    for t in closed:
        if t.get("exit_reason") in _COEFF:
            ep = float(t.get("entry_price", 0) or 0)
            xp = float(t.get("exit_price",  0) or 0)
            if ep > 0 and xp > 0 and abs(xp - ep) / ep < fee_roundtrip:
                sub_fee += 1
    return sub_fee / n
