"""
engine/ — Directional Coefficient Engine.

Pipeline:
    raw features → rolling z-scores → 4 signal blocks → regime-weighted blend → tanh → C
                                          │
                                          └→ agreement × vol_penalty × regime_certainty → K

C ∈ [-1, +1] is the directional coefficient. +1 maximally bullish, -1 maximally bearish.
K ∈ [0, 1] is the conviction scalar driving Kelly sizing.
"""
