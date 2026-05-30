"""
volatility.py — EWMA variance forecast (Part 4).

RiskMetrics-style:  σ²_t = λ·σ²_{t−1} + (1−λ)·r²_{t−1}, λ = 0.94 default.

For value-at-risk style forecasting, simple EWMA is about as accurate as
full GARCH as long as it captures heavy tails. EGARCH is supported behind
a config flag (arch library), default off.

The forecast feeds three places:
  • Kelly denominator
  • Vol-scaled stop/target
  • vol_penalty in K and leverage
"""
from __future__ import annotations

import math
from collections import deque


class EWMAForecaster:
    def __init__(self, lam: float = 0.94, warmup: int = 30) -> None:
        self.lam = lam
        self.warmup = warmup
        self._var: float = 0.0
        self._initialized = False
        self._return_buf: deque[float] = deque(maxlen=500)
        self._var_history: deque[float] = deque(maxlen=500)

    def update(self, log_return: float) -> None:
        self._return_buf.append(log_return)
        if not self._initialized:
            if len(self._return_buf) >= self.warmup:
                # Seed with sample variance of warmup window
                mean = sum(self._return_buf) / len(self._return_buf)
                self._var = sum((r - mean) ** 2 for r in self._return_buf) / max(1, len(self._return_buf) - 1)
                self._initialized = True
        else:
            self._var = self.lam * self._var + (1 - self.lam) * (log_return ** 2)
        if self._initialized:
            self._var_history.append(self._var)

    def forecast_per_bar(self) -> float:
        """One-bar-ahead variance forecast. Returns 0 until warmed up."""
        return self._var if self._initialized else 0.0

    def forecast_std_per_bar(self) -> float:
        return math.sqrt(self.forecast_per_bar())

    def forecast_horizon_std(self, horizon_bars: int) -> float:
        """sqrt-of-time scaling for a horizon."""
        return self.forecast_std_per_bar() * math.sqrt(max(1, horizon_bars))

    def median_forecast(self) -> float:
        """Median of the recent vol forecast history. Feeds vol_target in K."""
        if len(self._var_history) < 20:
            return self.forecast_std_per_bar()
        stds = sorted(math.sqrt(v) for v in self._var_history)
        return stds[len(stds) // 2]
