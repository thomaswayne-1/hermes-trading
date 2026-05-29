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
    Deterministic reflection covering all strategy variables.
    Checks conditions in priority order and changes exactly ONE variable.

    Priority order:
      1.  Drawdown too high             → tighten stop_loss_pct
      2.  Leverage too aggressive       → reduce leverage_base
      3.  RSI exits before TP (longs)   → raise long_rsi_exit
      4.  RSI exits before TP (shorts)  → lower short_rsi_exit
      5.  TP never reached              → lower take_profit_pct
      6.  Too many stop-loss hits       → widen stop_loss_pct
      7.  Consecutive losses            → increase reentry_cooldown_minutes
      8.  Low win rate (long)           → tighten long_threshold (require deeper oversold)
      9.  Low win rate (short)          → tighten short_threshold (require more overbought)
     10.  Too few trades                → loosen long_threshold
     11.  Volume filter blocking        → lower volume_surge_multiplier
     12.  On track, good returns        → increase position_size_max
     13.  Default                       → nudge leverage_base up slightly
    """
    from hermes_trading.score import _max_drawdown

    pnls     = [t.get("pnl_pct", 0)         for t in trades]
    pnls_lev = [t.get("pnl_pct_levered", 0) for t in trades]
    reasons  = [t.get("exit_reason", "")     for t in trades]

    total_return = sum(pnls_lev) if pnls_lev else 0.0
    max_dd       = _max_drawdown(pnls_lev)   if pnls_lev else 0.0
    win_rate     = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0
    n            = len(trades)

    stop_loss_exits = reasons.count("stop_loss")
    rsi_ob_exits    = reasons.count("rsi_overbought")   # long direction
    rsi_os_exits    = reasons.count("rsi_oversold")     # short direction
    rsi_exits       = rsi_ob_exits + rsi_os_exits
    tp_exits        = reasons.count("take_profit")
    time_exits      = reasons.count("time_exit")

    long_trades  = [t for t in trades if t.get("direction") == "long"]
    short_trades = [t for t in trades if t.get("direction") == "short"]

    # running streak of losses
    loss_streak = 0
    for t in reversed(trades):
        if t.get("pnl_pct", 0) < 0:
            loss_streak += 1
        else:
            break

    ecfg = strategy.get("entry", {})

    changed_var = None
    old_val     = None
    new_val     = None
    reason      = ""

    # ── 1. Drawdown too high ──────────────────────────────────────────────────
    if max_dd > goal.get("max_drawdown", 0.08):
        old_val = float(strategy.get("stop_loss_pct", 0.5))
        new_val = round(max(0.2, old_val - 0.1), 2)
        strategy["stop_loss_pct"] = new_val
        changed_var = "stop_loss_pct"
        reason = (
            f"Max drawdown {max_dd:.2%} exceeded limit {goal.get('max_drawdown', 0.08):.2%}. "
            f"Tightening stop_loss_pct {old_val} → {new_val} to protect capital."
        )

    # ── 2. Leverage too aggressive (drawdown > 60% of limit) ─────────────────
    elif max_dd > goal.get("max_drawdown", 0.08) * 0.6:
        old_val = float(strategy.get("leverage_base", 1.5))
        new_val = round(max(1.0, old_val - 0.25), 2)
        strategy["leverage_base"] = new_val
        changed_var = "leverage_base"
        reason = (
            f"Drawdown {max_dd:.2%} approaching limit. "
            f"Reducing leverage_base {old_val} → {new_val} to dampen loss magnitude."
        )

    # ── 3. Long RSI exit fires before take-profit ────────────────────────────
    elif (n >= 3 and len(long_trades) >= 2
          and rsi_ob_exits / max(1, len(long_trades)) > 0.6
          and tp_exits / n < 0.2):
        old_val = float(ecfg.get("long_rsi_exit", 78))
        new_val = round(min(92, old_val + 3), 1)
        ecfg["long_rsi_exit"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.long_rsi_exit"
        reason = (
            f"{rsi_ob_exits}/{len(long_trades)} long trades exited via RSI before take_profit. "
            f"Raising long_rsi_exit {old_val} → {new_val} to give longs more room."
        )

    # ── 4. Short RSI exit fires before take-profit ───────────────────────────
    elif (n >= 3 and len(short_trades) >= 2
          and rsi_os_exits / max(1, len(short_trades)) > 0.6
          and tp_exits / n < 0.2):
        old_val = float(ecfg.get("short_rsi_exit", 25))
        new_val = round(max(8, old_val - 3), 1)
        ecfg["short_rsi_exit"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.short_rsi_exit"
        reason = (
            f"{rsi_os_exits}/{len(short_trades)} short trades exited via RSI before take_profit. "
            f"Lowering short_rsi_exit {old_val} → {new_val} to give shorts more room."
        )

    # ── 5. Take-profit never reached — target too ambitious ──────────────────
    elif n >= 5 and tp_exits == 0 and rsi_exits + time_exits > n * 0.7:
        old_val = float(strategy.get("take_profit_pct", 1.0))
        new_val = round(max(0.3, old_val - 0.15), 2)
        strategy["take_profit_pct"] = new_val
        changed_var = "take_profit_pct"
        reason = (
            f"Take-profit hit 0 times in {n} trades (target may be too high). "
            f"Lowering take_profit_pct {old_val} → {new_val} for more achievable exits."
        )

    # ── 6. Stop-loss hit too often — entry into noisy moves ──────────────────
    elif n >= 5 and stop_loss_exits / n > 0.5:
        old_val = float(strategy.get("stop_loss_pct", 0.5))
        new_val = round(min(2.0, old_val + 0.15), 2)
        strategy["stop_loss_pct"] = new_val
        changed_var = "stop_loss_pct"
        reason = (
            f"Stop-loss triggered {stop_loss_exits}/{n} times — too much noise at current level. "
            f"Widening stop_loss_pct {old_val} → {new_val} to reduce premature cuts."
        )

    # ── 7. Loss streak — cool down re-entry ──────────────────────────────────
    elif loss_streak >= 3:
        old_val = float(strategy.get("reentry_cooldown_minutes", 1))
        new_val = round(min(30, old_val + 2), 1)
        strategy["reentry_cooldown_minutes"] = new_val
        changed_var = "reentry_cooldown_minutes"
        reason = (
            f"{loss_streak} consecutive losses. "
            f"Increasing reentry_cooldown_minutes {old_val} → {new_val} to avoid revenge trading."
        )

    # ── 8. Low win rate on longs — tighten long entry ────────────────────────
    elif (n >= 5 and win_rate < 0.4 and len(long_trades) > len(short_trades)):
        old_val = float(ecfg.get("long_threshold", 55))
        new_val = round(max(30, old_val - 3), 1)
        ecfg["long_threshold"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.long_threshold"
        reason = (
            f"Win rate {win_rate:.0%} below 40%, dominated by longs. "
            f"Tightening long_threshold {old_val} → {new_val} (require deeper oversold)."
        )

    # ── 9. Low win rate on shorts — tighten short entry ──────────────────────
    elif (n >= 5 and win_rate < 0.4 and len(short_trades) >= len(long_trades)):
        old_val = float(ecfg.get("short_threshold", 70))
        new_val = round(min(85, old_val + 3), 1)
        ecfg["short_threshold"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.short_threshold"
        reason = (
            f"Win rate {win_rate:.0%} below 40%, dominated by shorts. "
            f"Tightening short_threshold {old_val} → {new_val} (require more overbought)."
        )

    # ── 10. Too few trades — loosen long entry threshold ─────────────────────
    elif n < 3 and total_return < goal.get("target_return_30d", 0.05):
        old_val = float(ecfg.get("long_threshold", 55))
        new_val = round(min(65, old_val + 3), 1)
        ecfg["long_threshold"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.long_threshold"
        reason = (
            f"Only {n} trades recorded — entry threshold may be too tight. "
            f"Loosening long_threshold {old_val} → {new_val} to increase trade frequency."
        )

    # ── 11. Volume filter blocking entries ───────────────────────────────────
    elif n < 3 and float(ecfg.get("volume_surge_multiplier", 0)) > 1.0:
        old_val = float(ecfg.get("volume_surge_multiplier", 1.2))
        new_val = round(max(1.0, old_val - 0.1), 2)
        ecfg["volume_surge_multiplier"] = new_val
        strategy["entry"] = ecfg
        changed_var = "entry.volume_surge_multiplier"
        reason = (
            f"Low trade count — volume filter may be too strict. "
            f"Lowering volume_surge_multiplier {old_val} → {new_val}."
        )

    # ── 12. On track — increase position size ceiling ────────────────────────
    elif total_return >= goal.get("target_return_30d", 0.05) * 0.5 and win_rate >= 0.5:
        old_val = float(strategy.get("position_size_max", 0.40))
        new_val = round(min(0.60, old_val + 0.05), 2)
        strategy["position_size_max"] = new_val
        changed_var = "position_size_max"
        reason = (
            f"Strategy performing well (return={total_return:.2%}, win_rate={win_rate:.0%}). "
            f"Incrementally raising position_size_max {old_val} → {new_val}."
        )

    # ── 13. Default — nudge leverage_base up slightly ────────────────────────
    else:
        old_val = float(strategy.get("leverage_base", 1.5))
        new_val = round(min(float(strategy.get("leverage_max", 3.0)), old_val + 0.25), 2)
        strategy["leverage_base"] = new_val
        changed_var = "leverage_base"
        reason = (
            f"No critical issues detected (return={total_return:.2%}, dd={max_dd:.2%}). "
            f"Incrementally raising leverage_base {old_val} → {new_val} to boost returns."
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
