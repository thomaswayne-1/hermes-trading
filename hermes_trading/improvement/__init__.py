"""
improvement/ — multilayer self-improvement (Part 5).

Three layers:
  Layer 1 (fast, every 10 trades)  — attribution-driven weight updates.
  Layer 2 (regime, every 50 trades) — refit regime weights from per-regime IC.
  Layer 3 (Hermes, every 50 trades) — strategic review with multi-variable changes.

Each layer logs every change to hypotheses.jsonl and bumps the strategy
version, preserving the existing version/archive mechanism.
"""
