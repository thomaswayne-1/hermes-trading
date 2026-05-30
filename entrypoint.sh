#!/bin/sh
# entrypoint.sh — initialises persistent data dir on first run,
# then updates config files on every redeploy without touching trade history.
#
# /app/state  — bundled in Docker image (config templates, updated each deploy)
# /app/data   — persistent volume (runtime state, survives redeploys)
#
# CONFIG (overwritten every deploy so the user can push changes):
#   strategy.yaml, goal.yaml, weights.json
#
# RUNTIME (created if missing, never overwritten):
#   trades.jsonl, model.json, heartbeat.json, history/, hypotheses.jsonl,
#   attribution.jsonl

DATA=/app/data
STATE=/app/state

mkdir -p "$DATA" "$DATA/history"

# ── Config files: always overwrite from image ─────────────────────────────────
cp "$STATE/strategy.yaml" "$DATA/strategy.yaml"
cp "$STATE/goal.yaml"     "$DATA/goal.yaml"
[ -f "$STATE/weights.json" ] && cp "$STATE/weights.json" "$DATA/weights.json"

# ── Runtime files: only create if missing ─────────────────────────────────────
[ -f "$DATA/trades.jsonl"      ] || touch "$DATA/trades.jsonl"
[ -f "$DATA/model.json"        ] || echo '{}' > "$DATA/model.json"
[ -f "$DATA/heartbeat.json"    ] || echo '{}' > "$DATA/heartbeat.json"
[ -f "$DATA/hypotheses.jsonl"  ] || touch "$DATA/hypotheses.jsonl"
[ -f "$DATA/attribution.jsonl" ] || touch "$DATA/attribution.jsonl"
[ -f "$DATA/open_trades.json"  ] || echo '{}' > "$DATA/open_trades.json"

echo "[entrypoint] Data dir ready. Trades: $(wc -l < "$DATA/trades.jsonl") lines"

exec uv run python -m hermes_trading.run
