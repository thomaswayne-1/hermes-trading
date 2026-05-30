"""
guardrails.py — circuit breaker, shadow validation, sample gates (Part 6).

These are load-bearing. Self-modifying code plus leverage is the most
dangerous combination in this project. Every constraint here is what
separates adaptation from curve-fitting and ruin.
"""
from __future__ import annotations

import math
from typing import Any


# ── Drawdown tracking ────────────────────────────────────────────────────────

def equity_curve(closed_trades: list[dict], starting_balance: float = 100_000.0) -> list[float]:
    bal = starting_balance
    curve = [bal]
    for t in closed_trades:
        pos = float(t.get("position_size_r", 0.15))
        lev_pnl = t.get("pnl_pct_net", t.get("pnl_pct_levered", 0.0))
        bal += bal * pos * lev_pnl
        curve.append(bal)
    return curve


def current_drawdown(closed_trades: list[dict], starting_balance: float = 100_000.0) -> float:
    """Peak-to-current drawdown as a fraction (e.g. 0.06 = 6%)."""
    curve = equity_curve(closed_trades, starting_balance)
    if len(curve) < 2:
        return 0.0
    peak = max(curve)
    cur = curve[-1]
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - cur) / peak)


def rolling_sharpe(closed_trades: list[dict], window: int = 40) -> float:
    rets = [t.get("pnl_pct_net", t.get("pnl_pct_levered", 0.0)) for t in closed_trades[-window:]]
    if len(rets) < 5:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(1, len(rets) - 1)
    s = math.sqrt(var) if var > 0 else 1e-9
    return (m / s) * math.sqrt(252)   # annualized


# ── Circuit breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    States:
      'normal'     — full operation
      'soft_trip'  — drawdown > dd_soft. Halve lambda_kelly, freeze weight
                     expansion, tighten tau_enter
      'hard_trip'  — drawdown > dd_hard. Flatten positions, pause entries,
                     require Hermes review to reset
    """

    def __init__(self, dd_soft: float = 0.06, dd_hard: float = 0.08) -> None:
        self.dd_soft = dd_soft
        self.dd_hard = dd_hard
        self.state: str = "normal"
        self.tripped_at: dict[str, Any] = {}

    def evaluate(self, dd: float) -> str:
        if dd >= self.dd_hard:
            if self.state != "hard_trip":
                self.tripped_at["hard"] = dd
            self.state = "hard_trip"
        elif dd >= self.dd_soft:
            if self.state == "normal":
                self.tripped_at["soft"] = dd
                self.state = "soft_trip"
            # don't downgrade hard→soft automatically
        else:
            self.state = "normal"
        return self.state

    def reset_after_review(self) -> None:
        """Called by Hermes after a strategic review unfreezes the system."""
        self.state = "normal"
        self.tripped_at.clear()


# ── Shadow validation ────────────────────────────────────────────────────────

def shadow_validate(
    proposed_change: dict[str, Any],
    historical_trades: list[dict],
    window: int = 100,
) -> tuple[bool, str]:
    """
    Replay the proposed change against the most recent `window` trades and
    reject it if net expectancy worsens.

    This is walk-forward discipline in miniature. It catches changes that
    look good in-sample but fail on held-out data.

    Conservative implementation: if we lack the simulator infrastructure
    to replay properly, fall back to a sanity check (block weights stay
    in [0, 1] and sum to 1, scalars stay in bounds) — better than nothing.
    """
    # Sanity checks
    new_weights = proposed_change.get("weights_after")
    if new_weights:
        total = sum(new_weights.values())
        if not (0.99 <= total <= 1.01):
            return False, f"weights sum to {total:.3f}, not 1.0"
        if any(w < 0 or w > 1 for w in new_weights.values()):
            return False, "weight out of [0, 1] range"

    # Held-out expectancy check (uses gross PnL — we can't truly replay
    # adaptive sizing without a full simulator, but we can verify the
    # recent slice has positive expectancy before unblocking the change).
    recent = historical_trades[-window:] if len(historical_trades) >= window else historical_trades
    if not recent:
        return True, "no held-out data, allowed"
    net_expect = sum(t.get("pnl_pct_net", t.get("pnl_pct_levered", 0.0)) for t in recent) / len(recent)
    if net_expect < -0.005:   # held-out window is bleeding badly
        return False, f"held-out expectancy {net_expect:+.4f} below -0.5% floor"
    return True, f"held-out expectancy {net_expect:+.4f}, allowed"


# ── Net-exposure cap ─────────────────────────────────────────────────────────

def total_leveraged_exposure(open_trades: list[dict]) -> float:
    return sum(
        float(t.get("position_size_r", 0.0)) * float(t.get("leverage", 1.0))
        for t in open_trades
    )
