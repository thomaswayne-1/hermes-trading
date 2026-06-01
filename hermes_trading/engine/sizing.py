"""
sizing.py — fractional Kelly position sizing (Part 2).

f* = μ / σ²   (continuous form)
f  = lambda_kelly × f_raw
f  = clip(f, size_floor, size_cap)

Bucket past closed trades by |C| × K at entry. For each bucket compute
EWMA mean return μ̂ and variance σ̂². Use the bucket of the live signal
to estimate f_raw. Fall back to binary Kelly with the nearest populated
bucket if samples < min.

Total Kelly budget across open positions capped at kelly_budget_max.
"""
from __future__ import annotations

import math
from typing import Any


EPS = 1e-9
N_BUCKETS = 10   # deciles of |C|·K


def bucket_index(signal_strength: float) -> int:
    """Map |C|·K ∈ [0, 1] to a decile index 0..9."""
    s = max(0.0, min(0.999999, signal_strength))
    return int(s * N_BUCKETS)


def kelly_buckets(closed_trades: list[dict], ewma_lam: float = 0.9) -> dict[int, dict[str, float]]:
    """
    Build per-bucket (μ̂, σ̂², n) from history.

    Each trade is expected to carry 'entry_C', 'entry_K', and 'pnl_pct_net'
    (or fall back to 'pnl_pct' if net not present).
    """
    buckets: dict[int, list[float]] = {i: [] for i in range(N_BUCKETS)}
    for t in closed_trades:
        C = t.get("entry_C")
        K = t.get("entry_K")
        if C is None or K is None:
            continue
        idx = bucket_index(abs(C) * K)
        ret = t.get("pnl_pct_net", t.get("pnl_pct", 0.0))
        buckets[idx].append(ret)

    out: dict[int, dict[str, float]] = {}
    for idx, returns in buckets.items():
        n = len(returns)
        if n == 0:
            out[idx] = {"mu": 0.0, "var": 0.0, "n": 0, "win_prob": 0.0, "payoff": 0.0}
            continue
        # EWMA mean (recent emphasis)
        mu = returns[0]
        for r in returns[1:]:
            mu = ewma_lam * mu + (1 - ewma_lam) * r
        # Sample variance
        m = sum(returns) / n
        var = sum((r - m) ** 2 for r in returns) / max(1, n - 1)
        # Binary Kelly inputs as a fallback
        wins  = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        win_prob = len(wins) / n
        avg_w = sum(wins) / len(wins) if wins else 0.0
        avg_l = abs(sum(losses) / len(losses)) if losses else EPS
        payoff = avg_w / avg_l if avg_l > 0 else 0.0
        out[idx] = {
            "mu": mu, "var": var, "n": n,
            "win_prob": win_prob, "payoff": payoff,
        }
    return out


def fractional_kelly_size(
    C: float,
    K: float,
    buckets: dict[int, dict[str, float]],
    *,
    lambda_kelly: float = 0.35,
    size_floor: float = 0.05,
    size_cap: float = 0.40,
    bucket_min_samples: int = 50,
) -> tuple[float, str]:
    """
    Return (position_size_fraction, source_label).

    source_label is one of: 'kelly_continuous', 'kelly_binary_neighbor',
    'floor_strong_signal', 'skip_no_edge'.
    """
    if C == 0.0:
        return (0.0, "skip_no_edge")

    signal = abs(C) * K
    idx = bucket_index(signal)
    bucket = buckets.get(idx, {})

    n = int(bucket.get("n", 0))
    if n >= bucket_min_samples and bucket.get("var", 0) > EPS:
        # Continuous Kelly
        f_raw = bucket["mu"] / (bucket["var"] + EPS)
        if f_raw <= 0:
            # No historical edge in this bucket. If signal is strong, take floor; else skip.
            if signal > 0.5:
                return (size_floor, "floor_strong_signal")
            return (0.0, "skip_no_edge")
        f = lambda_kelly * f_raw
        f = max(size_floor, min(size_cap, f))
        return (f, "kelly_continuous")

    # Fallback: nearest populated bucket for binary Kelly
    nearest = _find_nearest_bucket(idx, buckets, bucket_min_samples)
    if nearest is not None:
        b = buckets[nearest]
        p = b["win_prob"]
        bpayoff = b["payoff"]
        if p > 0 and bpayoff > 0:
            f_raw = (p * (1 + bpayoff) - 1) / bpayoff   # binary Kelly
            if f_raw > 0:
                f = lambda_kelly * f_raw
                return (max(size_floor, min(size_cap, f)), "kelly_binary_neighbor")

    # Insufficient data or negative Kelly edge.
    # If the directional signal is strong (|C| > 0.30), still take the floor
    # size so the engine can build up trade history. signal = |C|·K is usually
    # 0.05–0.15 so a signal-based threshold of 0.4 is unreachable in practice.
    if abs(C) > 0.30:
        return (size_floor, "floor_strong_signal")
    return (0.0, "skip_no_edge")


def _find_nearest_bucket(idx: int, buckets: dict[int, dict[str, float]], min_n: int) -> int | None:
    """Scan outward from idx for a bucket with at least min_n trades."""
    for offset in range(1, N_BUCKETS):
        for cand in (idx - offset, idx + offset):
            if 0 <= cand < N_BUCKETS and buckets.get(cand, {}).get("n", 0) >= min_n:
                return cand
    # Fall back to most-populated bucket if nothing meets min_n
    if not buckets:
        return None
    populated = [(i, b.get("n", 0)) for i, b in buckets.items() if b.get("n", 0) > 0]
    if not populated:
        return None
    return max(populated, key=lambda x: x[1])[0]


def leverage_from_K(K: float, vol_penalty: float, *, lev_min: float = 1.0, lev_max: float = 3.0) -> float:
    """leverage = clip(1 + K × (lev_max − 1), lev_min, lev_max) × vol_penalty"""
    raw = 1.0 + K * (lev_max - 1.0)
    return max(lev_min, min(lev_max, raw)) * vol_penalty


def respects_budget(proposed_size: float, open_trades: list[dict], kelly_budget_max: float = 0.60) -> bool:
    """Check the proposed new position keeps total budget within cap."""
    total = proposed_size + sum(t.get("position_size_r", 0.0) for t in open_trades)
    return total <= kelly_budget_max
