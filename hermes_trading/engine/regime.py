"""
regime.py — regime detection (Part 3).

Primary: Markov regime-switching (TODO — requires statsmodels and refit
machinery, scheduled for Layer 2 follow-up).

Active: Hurst-exponent + ADX rules classifier. Hurst > 0.55 plus ADX > 25
is Trending; Hurst < 0.45 is Ranging; realized vol above its 80th
percentile overrides to High-vol.

Returns (label, certainty) where certainty ∈ [0, 1] feeds K.
"""
from __future__ import annotations

from collections import deque
from typing import Literal

from .features import hurst_exponent, adx, realized_vol


Regime = Literal["trending", "ranging", "high_vol"]


class RegimeClassifier:
    def __init__(self, vol_history_window: int = 200) -> None:
        self._vol_history: deque[float] = deque(maxlen=vol_history_window)

    def classify(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> tuple[Regime, float]:
        if len(closes) < 60:
            return ("ranging", 0.33)   # cold start — low certainty

        rv = realized_vol(closes, 20)
        if rv > 0:
            self._vol_history.append(rv)

        # High-vol override: realized vol > 80th percentile of recent history
        if len(self._vol_history) >= 50:
            sorted_vol = sorted(self._vol_history)
            p80 = sorted_vol[int(0.8 * len(sorted_vol))]
            if rv > p80:
                # certainty rises with how far above p80 we are
                excess = (rv - p80) / max(p80, 1e-9)
                certainty = min(1.0, 0.55 + excess)
                return ("high_vol", certainty)

        h = hurst_exponent(closes[-200:] if len(closes) >= 200 else closes)
        a = adx(highs, lows, closes, 14)

        if h > 0.55 and a > 25:
            # certainty: how far above the threshold on both axes
            cert = min(1.0, 0.5 + (h - 0.55) * 2.0 + (a - 25) / 100.0)
            return ("trending", cert)
        if h < 0.45:
            cert = min(1.0, 0.5 + (0.45 - h) * 2.0)
            return ("ranging", cert)
        # In between — call it ranging with lower certainty
        return ("ranging", 0.4)
