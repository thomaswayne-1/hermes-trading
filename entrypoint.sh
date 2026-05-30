#!/bin/sh
# entrypoint.sh — initialises persistent data dir on first run,
# then updates config files on every redeploy without touching trade history.
#
# Volume mount: Railway mounts a persistent disk at /app/data
# /app/state  — bundled in Docker image (config templates, updated each deploy)
# /app/data   — persistent volume (runtime state, survives redeploys)

DATA=/app/data
STATE=/app/state

mkdir -p "$DATA"

# ── Config files: always overwrite from image so deploys take effect ──────────
cp "$STATE/strategy.yaml" "$DATA/strategy.yaml"
cp "$STATE/goal.yaml"     "$DATA/goal.yaml"

# ── Runtime files: only create if missing, never overwrite ────────────────────
[ -f "$DATA/trades.jsonl"   ] || touch "$DATA/trades.jsonl"
[ -f "$DATA/model.json"     ] || echo '{}' > "$DATA/model.json"
[ -f "$DATA/heartbeat.json" ] || echo '{}' > "$DATA/heartbeat.json"
[ -d "$DATA/history"        ] || mkdir -p "$DATA/history"
[ -f "$DATA/hypotheses.jsonl" ] || touch "$DATA/hypotheses.jsonl"

echo "[entrypoint] Data dir ready. Trades: $(wc -l < "$DATA/trades.jsonl") lines"

exec uv run python -m hermes_trading.run
