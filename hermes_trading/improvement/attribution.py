"""
attribution.py — per-block Information Coefficient (Layer 1).

IC is the Spearman rank correlation between a block's sub-signal value at
entry and the realized net return of that trade. The IC is the standard
measure of how well a signal predicts forward returns; near 0 is noise,
0.05-0.10 is a real edge in this domain.

For each directional block, compute IC over the last M closed trades
(default 40, min 20).
"""
from __future__ import annotations

import math
from typing import Iterable


def _ranks(xs: list[float]) -> list[float]:
    """Fractional ranks (average rank for ties), 1-indexed."""
    n = len(xs)
    indexed = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. Returns 0 if insufficient data."""
    n = len(x)
    if n < 5 or n != len(y):
        return 0.0
    rx = _ranks(x)
    ry = _ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    den = dx * dy
    if den <= 0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def block_ic(closed_trades: list[dict], window: int = 40, min_trades: int = 20) -> dict[str, float] | None:
    """
    Return {block: IC} over the most recent `window` trades.

    Trades are expected to carry 'entry_sub_signals' (a dict {mom, rev,
    mic, sen} captured at entry) and 'pnl_pct_net' (or 'pnl_pct' as fallback).
    """
    recent = [
        t for t in closed_trades
        if t.get("entry_sub_signals") and t.get("closed")
    ][-window:]
    if len(recent) < min_trades:
        return None

    rets = [t.get("pnl_pct_net", t.get("pnl_pct", 0.0)) for t in recent]
    out: dict[str, float] = {}
    for block in ("mom", "rev", "mic", "sen"):
        signals = [t["entry_sub_signals"].get(block, 0.0) for t in recent]
        # Sign-align with trade direction (long: signal × +1; short: × -1)
        # because a bullish signal that produces a profitable short should
        # count negatively to IC.
        aligned = [
            s * (1.0 if t.get("direction") == "long" else -1.0)
            for s, t in zip(signals, recent)
        ]
        out[block] = spearman(aligned, rets)
    return out


def regime_ic(closed_trades: list[dict], regime: str, window: int = 40, min_trades: int = 20) -> dict[str, float] | None:
    """IC restricted to trades that occurred in a specific regime."""
    filtered = [t for t in closed_trades if t.get("entry_regime") == regime]
    return block_ic(filtered, window, min_trades)


# ── Churn metrics (whipsaw-gating patch) ─────────────────────────────────────

_COEFF_EXIT_REASONS = frozenset({"coefficient_flip", "coefficient_collapse"})


def churn_metrics(
    closed_trades: list[dict],
    window: int = 40,
    fee_roundtrip: float = 0.001,
) -> dict:
    """
    Compute churn-rate and average hold bars over the most recent *window*
    closed trades.

    churn_rate  — fraction of trades that are coefficient exits where the
                  |entry→exit price move| < fee_roundtrip (sub-fee-move exits).
    avg_hold_bars — mean bars_held over the window (0 if field absent).
    coeff_exit_rate — fraction of trades exited by any coefficient reason.

    Returns a dict with keys: churn_rate, avg_hold_bars, coeff_exit_rate, n.
    """
    recent = [t for t in closed_trades if t.get("closed")][-window:]
    n = len(recent)
    if n == 0:
        return {"churn_rate": 0.0, "avg_hold_bars": 0.0, "coeff_exit_rate": 0.0, "n": 0}

    sub_fee_count    = 0
    coeff_exit_count = 0
    hold_bars_list: list[float] = []

    for t in recent:
        reason = t.get("exit_reason", "")
        if reason in _COEFF_EXIT_REASONS:
            coeff_exit_count += 1
            ep = float(t.get("entry_price", 0) or 0)
            xp = float(t.get("exit_price",  0) or 0)
            if ep > 0 and xp > 0 and abs(xp - ep) / ep < fee_roundtrip:
                sub_fee_count += 1

        bh = t.get("bars_held")
        if bh is not None:
            hold_bars_list.append(float(bh))

    avg_hold = sum(hold_bars_list) / len(hold_bars_list) if hold_bars_list else 0.0

    return {
        "churn_rate":       sub_fee_count    / n,
        "avg_hold_bars":    avg_hold,
        "coeff_exit_rate":  coeff_exit_count / n,
        "n":                n,
    }
