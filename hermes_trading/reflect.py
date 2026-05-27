"""
reflect.py — reflection cycle runner.

Two modes:
  --fallback   Deterministic rule-based reflection (used before Hermes is installed).
  --hermes     Production mode: calls `hermes` subprocess with a structured prompt.

Both modes:
  - Change exactly ONE variable per cycle.
  - Bump strategy version.
  - Save prior version to state/history/v{NNNN}.yaml.
  - Append hypothesis to state/hypotheses.jsonl.
"""
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading.score import score

log = logging.getLogger("hermes.reflect")

STATE_DIR = Path(__file__).parent.parent / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
TRADES_FILE = STATE_DIR / "trades.jsonl"
GOAL_FILE = STATE_DIR / "goal.yaml"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

HERMES_TRADE_WINDOW = 25  # trades sent to Hermes for context


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    trades = []
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return [t for t in trades if t.get("closed")]


def bump_version(version_str: str) -> str:
    try:
        n = int(version_str)
        return f"{n + 1:02d}"
    except ValueError:
        return version_str + "_next"


def archive_strategy(strategy: dict) -> None:
    version = strategy.get("version", "00")
    archive_path = HISTORY_DIR / f"v{version}.yaml"
    save_yaml(archive_path, strategy)
    log.info("Archived strategy → %s", archive_path)


def append_hypothesis(hypothesis: dict) -> None:
    with open(HYPOTHESES_FILE, "a") as f:
        f.write(json.dumps(hypothesis) + "\n")


# ------------------------------------------------------------------ #
#  Fallback: deterministic reflection                                 #
# ------------------------------------------------------------------ #

def reflect_fallback(strategy: dict, goal: dict, trades: list[dict]) -> tuple[dict, dict]:
    """
    Simple deterministic rule:
      - If realised return < target → loosen entry.threshold by 2 (more trades).
      - If max_drawdown exceeded → tighten stop_loss_pct by 0.2 (less risk).
    Always changes exactly ONE variable.
    """
    pnls = [t["pnl_pct"] for t in trades]
    total_return = sum(pnls) if pnls else 0.0

    # Calculate max drawdown
    from hermes_trading.score import _max_drawdown
    max_dd = _max_drawdown(pnls) if pnls else 0.0

    changed_var = None
    old_val = None
    new_val = None
    reason = ""

    if max_dd > goal["max_drawdown"]:
        # Priority: tighten stop loss to protect capital
        old_val = strategy["stop_loss_pct"]
        new_val = round(max(0.5, old_val - 0.2), 2)
        strategy["stop_loss_pct"] = new_val
        changed_var = "stop_loss_pct"
        reason = (
            f"Max drawdown {max_dd:.2%} exceeded limit {goal['max_drawdown']:.2%}. "
            f"Tightening stop_loss_pct: {old_val} → {new_val}."
        )
    elif total_return < goal["target_return_30d"]:
        # Secondary: loosen entry threshold to capture more trades
        old_val = strategy["entry"]["threshold"]
        new_val = min(50, old_val + 2)
        strategy["entry"]["threshold"] = new_val
        changed_var = "entry.threshold"
        reason = (
            f"Realised return {total_return:.2%} below target {goal['target_return_30d']:.2%}. "
            f"Loosening entry.threshold: {old_val} → {new_val} (more entries)."
        )
    else:
        # On track — slightly tighten position sizing for risk management
        old_val = strategy["position_size_r"]
        new_val = round(min(1.0, old_val + 0.05), 2)
        strategy["position_size_r"] = new_val
        changed_var = "position_size_r"
        reason = (
            f"Strategy on track (return={total_return:.2%}, dd={max_dd:.2%}). "
            f"Incrementally increasing position_size_r: {old_val} → {new_val}."
        )

    hypothesis = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "fallback",
        "strategy_version_before": strategy.get("version"),
        "changed_variable": changed_var,
        "old_value": old_val,
        "new_value": new_val,
        "reasoning": reason,
        "score_before": score(trades, goal),
    }

    return strategy, hypothesis


# ------------------------------------------------------------------ #
#  Hermes: LLM-driven reflection                                      #
# ------------------------------------------------------------------ #

def reflect_hermes(strategy: dict, goal: dict, trades: list[dict]) -> tuple[dict, dict]:
    """
    Sends the last HERMES_TRADE_WINDOW trades + current strategy to `hermes`
    via subprocess stdin, parses a JSON hypothesis from stdout.
    """
    recent_trades = trades[-HERMES_TRADE_WINDOW:]
    score_before = score(recent_trades, goal)

    prompt = f"""You are the brain of a self-improving trading agent. Your job is to reflect
on recent trade outcomes and propose exactly ONE change to the strategy.

GOAL:
{json.dumps(goal, indent=2)}

CURRENT STRATEGY:
{json.dumps(strategy, indent=2)}

RECENT TRADES (last {len(recent_trades)}):
{json.dumps(recent_trades, indent=2)}

CURRENT SCORE: {score_before:.4f} (range -1.0 to +1.0)

Instructions:
1. Analyse the trades. Identify what's working and what isn't.
2. Generate 1–3 hypotheses. Each must name exactly ONE variable in the strategy and predict the score direction.
3. Pick the hypothesis with the highest confidence.
4. Respond ONLY with a JSON object in this exact format:
{{
  "changed_variable": "<dot.path to variable e.g. entry.threshold or stop_loss_pct>",
  "old_value": <current value>,
  "new_value": <proposed value>,
  "reasoning": "<one paragraph explaining why>",
  "confidence": <0.0 to 1.0>,
  "pending_hypotheses": ["<other hypothesis 1>", "<other hypothesis 2>"]
}}

HARD CONSTRAINT: change exactly ONE variable. Put any others in pending_hypotheses.
"""

    try:
        result = subprocess.run(
            ["hermes"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout.strip()
        # Extract JSON from output (Hermes may include prose before/after)
        start = output.find("{")
        end = output.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in Hermes output")
        parsed = json.loads(output[start:end])
    except Exception as exc:
        log.error("Hermes call failed: %s — falling back to deterministic reflect", exc)
        return reflect_fallback(strategy, goal, trades)

    # Apply the change to strategy
    var_path = parsed["changed_variable"].split(".")
    target = strategy
    for key in var_path[:-1]:
        target = target[key]
    target[var_path[-1]] = parsed["new_value"]

    hypothesis = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "hermes",
        "strategy_version_before": strategy.get("version"),
        "changed_variable": parsed["changed_variable"],
        "old_value": parsed["old_value"],
        "new_value": parsed["new_value"],
        "reasoning": parsed.get("reasoning", ""),
        "confidence": parsed.get("confidence", 0.0),
        "pending_hypotheses": parsed.get("pending_hypotheses", []),
        "score_before": score_before,
    }

    return strategy, hypothesis


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="Hermes reflection cycle")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--fallback", action="store_true", help="Deterministic reflection")
    mode_group.add_argument("--hermes", action="store_true", help="LLM-driven reflection via Hermes")
    args = parser.parse_args()

    strategy = load_yaml(STRATEGY_FILE)
    goal = load_yaml(GOAL_FILE)
    trades = load_trades()

    log.info(
        "Reflection starting | mode=%s trades=%d strategy_version=%s",
        "fallback" if args.fallback else "hermes",
        len(trades),
        strategy.get("version"),
    )

    # Archive current strategy before mutating
    archive_strategy(strategy)

    # Run the appropriate reflection
    if args.fallback:
        updated_strategy, hypothesis = reflect_fallback(strategy, goal, trades)
    else:
        updated_strategy, hypothesis = reflect_hermes(strategy, goal, trades)

    # Bump version
    old_version = updated_strategy.get("version", "00")
    updated_strategy["version"] = bump_version(old_version)
    hypothesis["strategy_version_after"] = updated_strategy["version"]
    hypothesis["score_after"] = None  # will be filled after next trades

    # Persist
    save_yaml(STRATEGY_FILE, updated_strategy)
    append_hypothesis(hypothesis)

    log.info(
        "Reflection complete | version %s → %s | changed=%s %s → %s",
        old_version,
        updated_strategy["version"],
        hypothesis["changed_variable"],
        hypothesis["old_value"],
        hypothesis["new_value"],
    )
    log.info("Reasoning: %s", hypothesis["reasoning"])


if __name__ == "__main__":
    main()
