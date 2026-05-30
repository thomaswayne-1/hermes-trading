"""
exits.py — coefficient-aware exits + volatility-scaled stop/target.

These complement (do NOT replace) the existing exit cascade in loop.py.
New exits are checked *first* so they fire before legacy exits when active.

New exit priority (above legacy cascade):
  1. Vol-scaled stop loss
  2. R-multiple take profit (= r_multiple × vol-scaled stop)
  3. Coefficient flip (sign(C) flipped against position)
  4. Coefficient collapse (|C| < tau_exit while profitable)

Then the legacy cascade continues:
  5. Always-on trailing stop
  6. Profit lock
  7. RSI exit (only if profitable)
  8. MACD reversal
  9. 24h time exit
"""
from __future__ import annotations

from typing import Any


def vol_scaled_stop_pct(vol_forecast_horizon: float, *, k_sl: float = 1.0, bounds: tuple[float, float] = (0.003, 0.015)) -> float:
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
) -> str | None:
    """
    Returns an exit_reason string if a coefficient-based exit should fire,
    or None.
    """
    direction = trade.get("direction")
    entry_price = float(trade.get("entry_price", 0))
    if entry_price == 0 or direction not in ("long", "short"):
        return None

    if direction == "long":
        raw_pnl = (current_price - entry_price) / entry_price
        # Flip: C turned negative against a long
        if C < 0:
            return "coefficient_flip"
    else:
        raw_pnl = (entry_price - current_price) / entry_price
        # Flip: C turned positive against a short
        if C > 0:
            return "coefficient_flip"

    # Collapse: signal too weak while in profit — recycle capital
    if abs(C) < tau_exit and raw_pnl > 0:
        return "coefficient_collapse"

    return None
