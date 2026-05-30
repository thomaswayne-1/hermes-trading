"""
costs.py — trading-cost model (Part 2).

Subtract taker fees (both legs) and expected funding over the holding
period from every expectancy and Kelly calculation. Frequent perp
trading is fee/funding-sensitive — a gross-profitable strategy can
easily turn net-negative.

Logged into trades.jsonl as pnl_pct_gross and pnl_pct_net.
"""
from __future__ import annotations


def round_trip_fee(taker_fee_bps: float = 5.0) -> float:
    """Both legs of a round-trip taker fill. Default 5 bps × 2 = 10 bps."""
    return 2 * taker_fee_bps / 10_000.0


def expected_funding_cost(funding_rate: float, holding_minutes: float, direction: str, funding_period_hours: float = 8.0) -> float:
    """
    Pro-rata funding cost over the trade's holding window.

    funding_rate is per funding period (typically 8h on Binance).
    A long *pays* when funding > 0 (and receives when negative); short is mirror.
    Returns a fraction subtracted from PnL for longs / added for shorts.
    """
    if funding_period_hours <= 0:
        return 0.0
    periods = holding_minutes / 60.0 / funding_period_hours
    cost = funding_rate * periods
    return cost if direction == "long" else -cost


def net_pnl(
    pnl_gross: float,
    *,
    taker_fee_bps: float = 5.0,
    funding_rate: float = 0.0,
    holding_minutes: float = 0.0,
    direction: str = "long",
) -> float:
    """
    Return PnL after fees and funding (as a fraction).
    """
    fees = round_trip_fee(taker_fee_bps)
    funding = expected_funding_cost(funding_rate, holding_minutes, direction)
    return pnl_gross - fees - funding
