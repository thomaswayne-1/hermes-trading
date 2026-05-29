"""
monte_carlo.py — live decision model with dynamic inputs.

Runs after every trade close and every 5 minutes otherwise.
Combines Monte Carlo simulation with Kelly sizing, market regime
detection, and concrete strategy recommendations.

Output written to state/model.json — read by the tracker, the
reflection engine, and the trading loop.

Components:
  1. Monte Carlo equity simulation      — 10,000 paths, N future trades
  2. Kelly criterion                    — optimal position size
  3. Market regime detection            — trending/choppy/volatile/reverting
  4. Risk assessment                    — drawdown proximity, ruin probability
  5. Decision engine                    — concrete lever adjustments
"""

import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("hermes.monte_carlo")

STATE_DIR     = Path(__file__).parent.parent / "state"
TRADES_FILE   = STATE_DIR / "trades.jsonl"
HEARTBEAT     = STATE_DIR / "heartbeat.json"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE     = STATE_DIR / "goal.yaml"
MODEL_OUT     = STATE_DIR / "model.json"

N_PATHS       = 10_000
N_FUTURE      = 100        # trades to project forward
MIN_TRADES    = 3          # need at least this many before running


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_trades() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    out = []
    for line in TRADES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                t = json.loads(line)
                if t.get("closed"):
                    out.append(t)
            except Exception:
                pass
    return out


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _pct(v: float) -> float:
    return v * 100


# ── 1. Monte Carlo simulation ─────────────────────────────────────────────────

def monte_carlo(
    returns: list[float],
    starting_balance: float,
    n_paths: int = N_PATHS,
    n_trades: int = N_FUTURE,
    ruin_threshold: float = 0.92,   # balance < 92% = ruin (−8%)
) -> dict:
    """
    Bootstrap resample historical trade returns to simulate future equity paths.
    Returns percentile distribution + risk metrics.
    """
    if not returns:
        return {}

    final_balances: list[float] = []
    max_drawdowns:  list[float] = []
    ruined = 0

    for _ in range(n_paths):
        bal  = starting_balance
        peak = starting_balance
        max_dd = 0.0

        for _ in range(n_trades):
            r    = random.choice(returns)
            bal *= (1 + r)
            peak = max(peak, bal)
            dd   = (peak - bal) / peak
            if dd > max_dd:
                max_dd = dd

        final_balances.append(bal)
        max_drawdowns.append(max_dd)
        if bal < starting_balance * ruin_threshold:
            ruined += 1

    final_balances.sort()
    max_drawdowns.sort()

    def pct(lst: list[float], p: float) -> float:
        idx = int(len(lst) * p / 100)
        return round(lst[max(0, min(idx, len(lst) - 1))], 2)

    return {
        "equity_p5":       pct(final_balances, 5),
        "equity_p25":      pct(final_balances, 25),
        "equity_p50":      pct(final_balances, 50),
        "equity_p75":      pct(final_balances, 75),
        "equity_p95":      pct(final_balances, 95),
        "max_dd_median":   round(pct(max_drawdowns, 50), 4),
        "max_dd_p95":      round(pct(max_drawdowns, 95), 4),
        "ruin_probability": round(ruined / n_paths, 4),
        "expected_return":  round((sum(final_balances) / n_paths - starting_balance) / starting_balance, 4),
    }


# ── 2. Kelly criterion ────────────────────────────────────────────────────────

def kelly(returns: list[float], fraction: float = 0.25) -> dict:
    """
    Compute Kelly-optimal position size.
    Uses fractional Kelly (default 25%) for conservatism.

    Kelly formula: f* = (b*p - q) / b
      b = avg_win / avg_loss   (payoff ratio)
      p = win probability
      q = 1 - p
    """
    if not returns:
        return {"kelly_full": 0.0, "kelly_quarter": 0.0, "payoff_ratio": 0.0}

    wins   = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r < 0]

    if not wins or not losses:
        return {"kelly_full": 0.0, "kelly_quarter": 0.0, "payoff_ratio": 0.0}

    p     = len(wins) / len(returns)
    q     = 1 - p
    b     = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    kelly_f = (b * p - q) / b if b > 0 else 0.0
    kelly_f = max(0.0, min(1.0, kelly_f))

    return {
        "kelly_full":     round(kelly_f, 4),
        "kelly_quarter":  round(kelly_f * fraction, 4),
        "payoff_ratio":   round(b, 4),
        "win_probability": round(p, 4),
    }


# ── 3. Trade statistics ───────────────────────────────────────────────────────

def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}

    returns_raw = [t.get("pnl_pct", 0)         for t in trades]
    returns_lev = [t.get("pnl_pct_levered", 0) for t in trades]
    exits       = [t.get("exit_reason", "")     for t in trades]

    wins   = [r for r in returns_lev if r > 0]
    losses = [r for r in returns_lev if r <= 0]

    avg_win  = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    win_rate = len(wins)   / len(returns_lev)

    # Sharpe (annualised assuming 30 trades/day → 10,950/year)
    avg  = sum(returns_lev) / len(returns_lev)
    var  = sum((r - avg) ** 2 for r in returns_lev) / len(returns_lev)
    std  = var ** 0.5
    sharpe = (avg / std * math.sqrt(10950)) if std > 0 else 0.0

    # Max drawdown on actual sequence
    balance = 1.0
    peak    = 1.0
    max_dd  = 0.0
    for r in returns_lev:
        balance *= (1 + r)
        peak     = max(peak, balance)
        max_dd   = max(max_dd, (peak - balance) / peak)

    # Loss streak
    streak = 0
    for t in reversed(trades):
        if t.get("pnl_pct", 0) < 0:
            streak += 1
        else:
            break

    return {
        "n_trades":       len(trades),
        "win_rate":       round(win_rate, 4),
        "avg_win_pct":    round(_pct(avg_win), 4),
        "avg_loss_pct":   round(_pct(avg_loss), 4),
        "sharpe":         round(sharpe, 4),
        "max_drawdown":   round(max_dd, 4),
        "total_return":   round(sum(returns_lev), 4),
        "loss_streak":    streak,
        "exit_reasons":   {r: exits.count(r) for r in set(exits)},
    }


# ── 4. Market regime detection ────────────────────────────────────────────────

def detect_regime(hb: dict) -> dict:
    """
    Classify current market conditions from live indicators.

    Regimes:
      trending_up     — momentum + RSI elevated, MACD positive
      trending_down   — momentum negative, MACD negative, RSI falling
      oversold_bounce — RSI < 35, MACD turning positive (entry zone)
      overbought_fade — RSI > 70, MACD histogram declining
      choppy          — low momentum, RSI near 50, MACD near 0
      high_volatility — ATR elevated relative to price
    """
    rsi        = hb.get("rsi", 50.0)
    macd_hist  = hb.get("macd_hist", 0.0)
    bb_pct     = hb.get("bb_pct", 0.5)
    momentum   = hb.get("momentum", 0.0)
    ob_imb     = hb.get("ob_imbalance", 0.0)
    atr        = hb.get("atr", 0.0)
    price      = hb.get("price", 1.0)
    atr_pct    = (atr / price * 100) if price > 0 else 0.0

    regime  = "choppy"
    quality = 0.5   # 0 = bad to trade, 1 = great to trade

    if atr_pct > 1.0:
        regime  = "high_volatility"
        quality = 0.3

    elif rsi < 35 and macd_hist > -5:
        regime  = "oversold_bounce"
        quality = 0.9    # best entry zone

    elif rsi > 70 and macd_hist < 0:
        regime  = "overbought_fade"
        quality = 0.2    # avoid entering longs

    elif momentum > 0.3 and macd_hist > 0 and rsi > 55:
        regime  = "trending_up"
        quality = 0.7

    elif momentum < -0.3 and macd_hist < 0 and rsi < 45:
        regime  = "trending_down"
        quality = 0.1    # avoid longs

    elif abs(momentum) < 0.1 and abs(macd_hist) < 10:
        regime  = "choppy"
        quality = 0.4

    # Order book boost: strong bid pressure improves quality
    if ob_imb > 0.5:
        quality = min(1.0, quality + 0.1)
    elif ob_imb < -0.3:
        quality = max(0.0, quality - 0.1)

    return {
        "regime":               regime,
        "quality":              round(quality, 3),
        "atr_pct":              round(atr_pct, 4),
        "description":          _regime_description(regime),
    }


def _regime_description(regime: str) -> str:
    return {
        "oversold_bounce":  "RSI oversold, momentum turning — prime entry zone",
        "trending_up":      "Strong upward trend — momentum entries valid",
        "overbought_fade":  "Overbought, MACD fading — avoid new longs",
        "trending_down":    "Downtrend — long entries high risk",
        "high_volatility":  "Elevated volatility — reduce position size",
        "choppy":           "Directionless — wait for clearer signal",
    }.get(regime, "Unknown regime")


# ── 5. Decision engine ────────────────────────────────────────────────────────

def decisions(
    stats: dict,
    mc: dict,
    kelly_data: dict,
    regime: dict,
    strategy: dict,
    goal: dict,
    starting_balance: float,
) -> dict:
    """
    Translate model outputs into concrete lever recommendations.
    Each recommendation has: field, current, suggested, reason, urgency.
    """
    recs   = []
    action = "hold"   # hold / tighten / loosen / pause

    if not stats:
        return {"recommendations": [], "action": "insufficient_data",
                "signal": "neutral", "confidence": 0.0}

    win_rate     = stats.get("win_rate", 0.5)
    max_dd       = stats.get("max_drawdown", 0.0)
    sharpe       = stats.get("sharpe", 0.0)
    loss_streak  = stats.get("loss_streak", 0)
    ruin_prob    = mc.get("ruin_probability", 0.0)
    dd_p95       = mc.get("max_dd_p95", 0.0)
    kelly_q      = kelly_data.get("kelly_quarter", 0.2)
    regime_name  = regime.get("regime", "choppy")
    regime_q     = regime.get("quality", 0.5)
    max_dd_limit = goal.get("max_drawdown", 0.08)

    cur_pos_base = float(strategy.get("position_size_base", 0.15))
    cur_pos_max  = float(strategy.get("position_size_max", 0.40))
    cur_lev_base = float(strategy.get("leverage_base", 1.5))
    cur_lev_max  = float(strategy.get("leverage_max", 4.0))
    cur_sl       = float(strategy.get("stop_loss_pct", 0.5))
    cur_tp       = float(strategy.get("take_profit_pct", 1.0))

    # ── RISK CHECKS (urgent) ──────────────────────────────────────────────────

    if ruin_prob > 0.10:
        recs.append({
            "field":     "position_size_base",
            "current":   cur_pos_base,
            "suggested": round(max(0.05, cur_pos_base * 0.7), 3),
            "reason":    f"Ruin probability {ruin_prob:.1%} is dangerously high. Reduce size immediately.",
            "urgency":   "critical",
        })
        action = "tighten"

    elif ruin_prob > 0.05:
        recs.append({
            "field":     "leverage_max",
            "current":   cur_lev_max,
            "suggested": round(max(1.5, cur_lev_max - 0.5), 1),
            "reason":    f"Ruin probability {ruin_prob:.1%} elevated. Reduce max leverage.",
            "urgency":   "high",
        })
        action = "tighten"

    if max_dd > max_dd_limit * 0.8:
        recs.append({
            "field":     "stop_loss_pct",
            "current":   cur_sl,
            "suggested": round(max(0.2, cur_sl - 0.1), 2),
            "reason":    f"Drawdown {max_dd:.1%} approaching {max_dd_limit:.0%} limit. Tighten stop.",
            "urgency":   "high",
        })
        action = "tighten"

    if loss_streak >= 4:
        recs.append({
            "field":     "reentry_cooldown_minutes",
            "current":   float(strategy.get("reentry_cooldown_minutes", 1)),
            "suggested": 5.0,
            "reason":    f"{loss_streak} consecutive losses. Pause and wait for cleaner setup.",
            "urgency":   "high",
        })
        action = "pause"

    # ── REGIME-BASED ADJUSTMENTS ──────────────────────────────────────────────

    if regime_name == "high_volatility":
        suggested_pos = round(max(0.05, cur_pos_base * 0.6), 3)
        if suggested_pos < cur_pos_base:
            recs.append({
                "field":     "position_size_base",
                "current":   cur_pos_base,
                "suggested": suggested_pos,
                "reason":    "High volatility detected. Reduce base position size to manage risk.",
                "urgency":   "medium",
            })

    elif regime_name == "trending_down":
        recs.append({
            "field":     "entry.macd_confirm",
            "current":   str(strategy.get("entry", {}).get("macd_confirm", False)),
            "suggested": "true",
            "reason":    "Downtrend active. Require MACD confirmation to avoid catching falling knife.",
            "urgency":   "medium",
        })
        action = "tighten" if action == "hold" else action

    elif regime_name == "oversold_bounce" and win_rate > 0.5:
        # Good conditions — consider sizing up
        kelly_suggested = round(min(cur_pos_max, max(cur_pos_base, kelly_q)), 3)
        if kelly_suggested > cur_pos_base + 0.03:
            recs.append({
                "field":     "position_size_base",
                "current":   cur_pos_base,
                "suggested": kelly_suggested,
                "reason":    f"Oversold regime + {win_rate:.0%} win rate. Kelly suggests {kelly_q:.1%} base size.",
                "urgency":   "low",
            })
            action = "loosen" if action == "hold" else action

    # ── PERFORMANCE TUNING ────────────────────────────────────────────────────

    if stats.get("n_trades", 0) >= 5:
        exits = stats.get("exit_reasons", {})
        n     = stats.get("n_trades", 1)

        # RSI exit fires too early
        if exits.get("rsi_overbought", 0) / n > 0.6 and exits.get("take_profit", 0) == 0:
            cur_rsi_exit = float(strategy.get("entry", {}).get("rsi_exit_threshold", 75))
            recs.append({
                "field":     "entry.rsi_exit_threshold",
                "current":   cur_rsi_exit,
                "suggested": round(min(90, cur_rsi_exit + 3), 1),
                "reason":    f"RSI exit firing {exits.get('rsi_overbought',0)}/{n} times before take_profit. Raise threshold.",
                "urgency":   "medium",
            })

        # Stop loss too tight (firing often)
        if exits.get("stop_loss", 0) / n > 0.45:
            recs.append({
                "field":     "stop_loss_pct",
                "current":   cur_sl,
                "suggested": round(min(2.0, cur_sl + 0.15), 2),
                "reason":    f"Stop loss hit {exits.get('stop_loss',0)}/{n} times. May be too tight for current volatility.",
                "urgency":   "medium",
            })

        # TP never hit — target too ambitious
        if exits.get("take_profit", 0) == 0 and n >= 8:
            recs.append({
                "field":     "take_profit_pct",
                "current":   cur_tp,
                "suggested": round(max(0.3, cur_tp - 0.2), 2),
                "reason":    f"Take profit not hit in {n} trades. Consider lowering target.",
                "urgency":   "medium",
            })

        # Good Sharpe + win rate → size up
        if sharpe > 1.5 and win_rate > 0.6 and ruin_prob < 0.03:
            kelly_suggested = round(min(cur_pos_max, kelly_q * 1.5), 3)
            if kelly_suggested > cur_pos_max - 0.02:
                recs.append({
                    "field":     "position_size_max",
                    "current":   cur_pos_max,
                    "suggested": round(min(0.8, cur_pos_max + 0.05), 3),
                    "reason":    f"Sharpe {sharpe:.2f}, win rate {win_rate:.0%}. Strategy robust — size up.",
                    "urgency":   "low",
                })
                action = "loosen" if action == "hold" else action

    # ── Overall signal ────────────────────────────────────────────────────────
    # Combine regime quality and performance into a single trading signal
    perf_score = (win_rate - 0.5) * 2   # -1 to +1
    signal_score = (regime_q * 0.6 + (perf_score * 0.4))
    if signal_score > 0.6:
        signal = "strong_buy"
    elif signal_score > 0.3:
        signal = "buy"
    elif signal_score < -0.2:
        signal = "avoid"
    else:
        signal = "neutral"

    return {
        "recommendations": recs,
        "action":          action,
        "signal":          signal,
        "signal_score":    round(signal_score, 3),
        "confidence":      round(regime_q, 3),
    }


# ── 6. Main model runner ──────────────────────────────────────────────────────

def run_model(starting_balance: float = 100_000) -> dict:
    """Run the full model and write model.json. Returns the model output."""
    trades   = _load_trades()
    hb       = _load_json(HEARTBEAT)
    strategy = _load_yaml(STRATEGY_FILE)
    goal     = _load_yaml(GOAL_FILE)
    balance  = goal.get("starting_balance", starting_balance)

    if len(trades) < MIN_TRADES:
        result = {
            "ts":           datetime.now(timezone.utc).isoformat(),
            "status":       f"insufficient_data ({len(trades)}/{MIN_TRADES} trades needed)",
            "n_trades":     len(trades),
        }
        MODEL_OUT.write_text(json.dumps(result, indent=2))
        return result

    # Compute running balance for each trade (compound)
    running = balance
    levered_returns = []
    for t in trades:
        ps  = float(t.get("position_size_r", 0.2))
        ret = t.get("pnl_pct_levered", 0)
        dollar_pnl = running * ps * ret
        running   += dollar_pnl
        levered_returns.append(ret)

    current_balance = running

    # Run all components
    stats      = trade_stats(trades)
    mc         = monte_carlo(levered_returns, current_balance)
    kelly_data = kelly(levered_returns)
    regime     = detect_regime(hb)
    dec        = decisions(stats, mc, kelly_data, regime, strategy, goal, balance)

    result = {
        "ts":               datetime.now(timezone.utc).isoformat(),
        "status":           "ok",
        "n_trades":         len(trades),
        "current_balance":  round(current_balance, 2),
        "pnl_dollar":       round(current_balance - balance, 2),
        "pnl_pct":          round((current_balance - balance) / balance, 4),

        # Trade stats
        "stats":            stats,

        # Monte Carlo (projected over next 100 trades)
        "monte_carlo": {
            **mc,
            "n_paths":      N_PATHS,
            "n_future":     N_FUTURE,
        },

        # Kelly position sizing
        "kelly":            kelly_data,

        # Market regime
        "regime":           regime,

        # Decisions
        "signal":           dec["signal"],
        "signal_score":     dec["signal_score"],
        "confidence":       dec["confidence"],
        "action":           dec["action"],
        "recommendations":  dec["recommendations"],
    }

    MODEL_OUT.write_text(json.dumps(result, indent=2))
    log.info(
        "MODEL | signal=%s regime=%s ruin=%.1f%% kelly=%.1f%% recs=%d",
        dec["signal"],
        regime["regime"],
        mc.get("ruin_probability", 0) * 100,
        kelly_data.get("kelly_quarter", 0) * 100,
        len(dec["recommendations"]),
    )
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    result = run_model()
    print(json.dumps(result, indent=2))
