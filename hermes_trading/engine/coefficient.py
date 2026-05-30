"""
coefficient.py — Layer C blend + Layer D conviction → C, K.

Layer C: regime-conditional blend of the four sub-signals.
    raw = Σ_block W_block(regime) · s_block
    C   = tanh(g × raw)

Layer D: conviction scalar.
    agreement       = |raw| / (|s_mom| + |s_rev| + |s_mic| + |s_sen| + eps)
    vol_penalty     = clip(vol_target / max(vol_forecast, eps), 0, 1)
    regime_certainty = max smoothed regime probability
    K = agreement × vol_penalty × regime_certainty
"""
from __future__ import annotations

import math
from typing import Any

from .features import FeatureBuffer, extract_raw_features
from .blocks import compute_blocks
from .regime import RegimeClassifier
from .volatility import EWMAForecaster


EPS = 1e-9


class CoefficientEngine:
    """Top-level engine. Stateful — holds the rolling buffers."""

    def __init__(self, config: dict[str, Any], regime_weights: dict[str, dict[str, float]] | None = None) -> None:
        self.config = config
        self.regime_weights = regime_weights or self._default_regime_weights()
        self.zscore_window = int(config.get("zscore_window", 200))
        self.gain          = float(config.get("gain", 1.5))
        self.tau_enter     = float(config.get("tau_enter", 0.12))
        self.tau_exit      = float(config.get("tau_exit", 0.05))

        self.features  = FeatureBuffer(window=self.zscore_window)
        self.regime    = RegimeClassifier()
        self.vol       = EWMAForecaster(lam=0.94)
        self._last_price: float = 0.0

    @staticmethod
    def _default_regime_weights() -> dict[str, dict[str, float]]:
        return {
            "trending":  {"mom": 0.55, "rev": 0.10, "mic": 0.20, "sen": 0.15},
            "ranging":   {"mom": 0.10, "rev": 0.55, "mic": 0.20, "sen": 0.15},
            "high_vol":  {"mom": 0.20, "rev": 0.25, "mic": 0.20, "sen": 0.35},
        }

    def tick(
        self,
        price_data: dict[str, Any],
        candles: list[list[Any]] | None = None,
        funding_rate: float = 0.0,
        fng_value: float = 50.0,
        ls_ratio: float = 1.0,
        top_ls_ratio: float = 1.0,
        taker_buy_ratio: float = 0.5,
        oi_pct_change: float = 0.0,
    ) -> dict[str, Any]:
        """
        Process one tick. Returns a snapshot dict with C, K, regime,
        sub-signals, and diagnostics — to be written to heartbeat and
        consumed by the trading layer.
        """
        # Update volatility forecast on log returns
        live_price = float(price_data.get("close", 0.0))
        if live_price > 0 and self._last_price > 0:
            try:
                lr = math.log(live_price / self._last_price)
                self.vol.update(lr)
            except (ValueError, ZeroDivisionError):
                pass
        if live_price > 0:
            self._last_price = live_price

        # Extract raw features
        raw = extract_raw_features(
            price_data, candles,
            funding_rate=funding_rate,
            fng_value=fng_value,
            ls_ratio=ls_ratio,
            top_ls_ratio=top_ls_ratio,
            taker_buy_ratio=taker_buy_ratio,
            oi_pct_change=oi_pct_change,
        )

        # Append to buffer and z-score
        self.features.append(raw)
        if not self.features.ready:
            return self._cold_start_snapshot(raw)

        z = self.features.zscores(raw)
        sub = compute_blocks(z)   # {mom, rev, mic, sen} each in [-1, +1]

        # Classify regime from cached candles
        closes = [float(c[4]) for c in (candles or [])]
        highs  = [float(c[2]) for c in (candles or [])]
        lows   = [float(c[3]) for c in (candles or [])]
        if live_price > 0:
            closes.append(live_price)
            highs.append(live_price)
            lows.append(live_price)
        regime, regime_cert = self.regime.classify(closes, highs, lows)

        # Apply regime-specific block weights
        W = self.regime_weights.get(regime, self.regime_weights["ranging"])
        raw_score = (
            W["mom"] * sub["mom"]
            + W["rev"] * sub["rev"]
            + W["mic"] * sub["mic"]
            + W["sen"] * sub["sen"]
        )
        # Cut gain in high-vol regime
        g = self.gain * (0.667 if regime == "high_vol" else 1.0)
        C = math.tanh(g * raw_score)

        # Conviction K
        agreement = abs(raw_score) / (
            abs(sub["mom"]) + abs(sub["rev"]) + abs(sub["mic"]) + abs(sub["sen"]) + EPS
        )
        agreement = max(0.0, min(1.0, agreement))

        vol_forecast = self.vol.forecast_std_per_bar()
        vol_target   = self.vol.median_forecast()
        if vol_forecast <= EPS:
            vol_penalty = 1.0
        else:
            vol_penalty = max(0.0, min(1.0, vol_target / vol_forecast))

        K = agreement * vol_penalty * regime_cert

        return {
            "C":             round(C, 4),
            "K":             round(K, 4),
            "regime":        regime,
            "regime_cert":   round(regime_cert, 4),
            "sub_signals":   {k: round(v, 4) for k, v in sub.items()},
            "weights_used":  W,
            "agreement":     round(agreement, 4),
            "vol_penalty":   round(vol_penalty, 4),
            "vol_forecast":  round(vol_forecast, 6),
            "vol_target":    round(vol_target, 6),
            "raw_score":     round(raw_score, 4),
            "ready":         True,
        }

    def _cold_start_snapshot(self, raw: dict[str, float]) -> dict[str, Any]:
        return {
            "C": 0.0,
            "K": 0.0,
            "regime": "warmup",
            "regime_cert": 0.0,
            "sub_signals": {"mom": 0.0, "rev": 0.0, "mic": 0.0, "sen": 0.0},
            "weights_used": self.regime_weights["ranging"],
            "agreement": 0.0,
            "vol_penalty": 1.0,
            "vol_forecast": 0.0,
            "vol_target": 0.0,
            "raw_score": 0.0,
            "ready": False,
            "samples": len(self.features),
        }
