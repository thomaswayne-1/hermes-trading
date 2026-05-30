"""
blocks.py — Layer B: the four directional sub-signals.

Each block collapses to a bounded sub-signal:
    s_block = tanh( Σ feature_weight × z )

Default inner weights are equal within a block. These are mostly left
alone by the fast loop; the fast loop adapts the *block-level* weights
in Layer C, where the attribution signal is more statistically reliable.
"""
from __future__ import annotations

import math

# Inner feature weights per block. Equal weights by default.
BLOCK_FEATURES = {
    "mom": ["macd_hist", "roc_14", "ema20_slope", "signed_adx"],
    "rev": ["rsi_mr", "bb_mr", "vwap_dist"],
    "mic": ["ob_imb", "vol_delta"],
    "sen": ["funding_neg", "fng_contra"],
}


def compute_blocks(z: dict[str, float], inner_weights: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """
    Compute the four sub-signals from the z-scored feature dict.
    Each sub-signal is in [-1, +1] courtesy of the tanh.
    """
    out: dict[str, float] = {}
    for block, features in BLOCK_FEATURES.items():
        if inner_weights and block in inner_weights:
            w = inner_weights[block]
        else:
            w = {f: 1.0 / len(features) for f in features}
        total = 0.0
        for f in features:
            total += w.get(f, 0.0) * z.get(f, 0.0)
        out[block] = math.tanh(total)
    return out
