"""
exits.py — coefficient-aware exits + volatility-scaled stop/target.

These complement (do NOT replace) the existing exit cascade in loop.py.
New exits are checked *first* so they fire before legacy exits when active.

New exit priority (above legacy cascade):
  1. Vol-scaled stop loss
  2. R-multiple take profit (= r_multiple × vol-scaled stop)
  3. Coefficient flip  (sign of C reversed against position)
  4. Coefficient collapse (|C| < tau_exit while profitable)

Then the legacy cascade continues:
  5. Always-on trailing stop
  6. Profit lock
  7. RSI exit (only if profitable)
  8. MACD reversal
  9. 24h time exit

── Whipsaw-gating (Change 1–3) ──────────────────────────────────────────────

The raw coefficient-flip exit fires on any sign cross of C — the noisiest
regime.  This patch adds three guards:

  1. Hysteresis band: a long position flips only when C < -c_exit_band
     (not at any C < 0).  c_exit_band < c_enter_band creates a dead zone
     where C can wobble without triggering either a new entry or a flip exit.

  2. Persistence requirement: the flip condition must hold for flip_persist
     consecutive bars.  bars_against is incremented each bar the condition
     holds; reset to 0 any bar it does not.  Caller must pass the current
     counter and store the returned updated counter.

  3. Minimum hold: coefficient exits (flip AND collapse) may not fire within
     min_hold_bars ticks of entry.  Price-based exits (stop-loss, TP) are
     NEVER gated by this — they remain instant.
"""
from __future__ import annotations

from typing import Any


def vol_scaled_stop_pct(
    vol_forecast_horizon: float,
    *,
    k_sl: float = 1.0,
    bounds: tuple[float, float] = (0.003, 0.015),
) -> float:
    """
    stop_pct = clip(k_sl × σ_horizon, lo, hi).

    vol_forecast_horizon is the EWMA std forecast scaled to the trade horizon
    (per-bar × sqrt(horizon_bars)). Returns the stop distance as a *fraction*
    (e.g. 0.005 = 0.5%).
    """
    raw = k_sl * vol_forecast_horizon
    lo, hi = bounds
    return max(lo, min(hi, raw))


def evaluate_coefficient_exits(
    trade: dict,
    current_price: float,
    C: float,
    tau_exit: float = 0.05,
    *,
    c_exit_band: float = 0.10,
    min_hold_bars: int = 3,
    flip_persist: int = 3,
    bars_held: int = 0,
    bars_against: int = 0,
) -> tuple[str | None, int]:
    """
    Evaluate coefficient-based exits for an open position.

    Returns (exit_reason | None, updated_bars_against).

    The caller MUST store the returned bars_against back to its per-trade
    state (e.g. self._bars_against[trade_id]) on every call, even when no
    exit is triggered — the counter accumulates across consecutive bars.

    Parameters
    ----------
    trade          : Open trade dict (needs 'direction', 'entry_price').
    current_price  : Current market price.
    C              : Current coefficient value ∈ [-1, +1].
    tau_exit       : Collapse threshold — |C| must fall below this (default 0.05).
    c_exit_band    : Hysteresis exit band.  C must cross past the *opposite*
                     c_exit_band before a flip exit fires (default 0.10).
                     Must be ≤ c_enter_band (enforced by the tuner invariant).
    min_hold_bars  : Minimum ticks held before any coefficient exit may fire.
                     Stop-loss and take-profit are NEVER subject to this gate.
    flip_persist   : Consecutive bars the flip condition must hold before the
                     flip exit fires (Schmitt-trigger debounce).
    bars_held      : Ticks elapsed since entry (caller computes this).
    bars_against   : Current accumulated consecutive-bars-against counter for
                     this trade (caller loads from its state dict).
    """
    direction = trade.get("direction")
    entry_price = float(trade.get("entry_price", 0))
    if entry_price == 0 or direction not in ("long", "short"):
        return None, 0

    if direction == "long":
        raw_pnl = (current_price - entry_price) / entry_price
        # Flip condition: C must cross past the *exit* band on the short side
        flip_condition = C < -c_exit_band
    else:
        raw_pnl = (entry_price - current_price) / entry_price
        # Flip condition: C must cross past the exit band on the long side
        flip_condition = C > c_exit_band

    # ── Advance or reset the bars_against counter ─────────────────────────────
    if flip_condition:
        new_bars_against = bars_against + 1
    else:
        new_bars_against = 0

    # ── Gate 1: minimum hold — no coefficient exit before min_hold_bars ───────
    if bars_held < min_hold_bars:
        return None, new_bars_against

    # ── Flip exit: condition held for flip_persist consecutive bars ───────────
    if flip_condition and new_bars_against >= flip_persist:
        return "coefficient_flip", new_bars_against

    # ── Collapse exit: |C| too weak while in profit ───────────────────────────
    # No additional persistence check — collapse is a gentler, deliberate exit
    # that already requires being in profit.  The min_hold gate is sufficient.
    if abs(C) < tau_exit and raw_pnl > 0:
        return "coefficient_collapse", new_bars_against

    return None, new_bars_against
