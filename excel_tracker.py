#!/usr/bin/env python3
"""
Usage:
    cd ~/hermes-trading
    python3 excel_tracker.py

Prints a status line every 10 seconds and writes three CSV files:
    position.csv   — live position + agent health
    stats.csv      — session P&L summary
    trades.csv     — full trade history
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

BASE    = Path(__file__).parent
STATE   = BASE / "state"
REFRESH = 10


def rj(name):
    try:
        return json.loads((STATE / name).read_text())
    except:
        return {}


def rjsonl(name):
    p = STATE / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except:
                pass
    return out


def rstrat():
    try:
        return yaml.safe_load((STATE / "strategy.yaml").read_text()) or {}
    except:
        return {}


def rgoal():
    try:
        return yaml.safe_load((STATE / "goal.yaml").read_text()) or {}
    except:
        return {}


def rmodel():
    try:
        return json.loads((BASE / "state" / "model.json").read_text())
    except:
        return {}


def pct(v):
    return f"{'+' if v > 0 else ''}{v * 100:.2f}%"


def usd(v):
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.2f}"


def age(ts):
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        s = int((datetime.now(timezone.utc) - t).total_seconds())
        if s < 60:   return f"{s}s ago"
        if s < 3600: return f"{s // 60}m ago"
        return f"{s // 3600}h {(s % 3600) // 60}m ago"
    except:
        return "?"


def dollar_pnl(pnl_levered, position_size_r, balance):
    return balance * position_size_r * pnl_levered


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def run():
    hb     = rj("heartbeat.json")
    trades = rjsonl("trades.jsonl")
    s      = rstrat()
    g      = rgoal()
    closed = [t for t in trades if t.get("closed")]
    ecfg   = s.get("entry", {})

    starting_balance = float(g.get("starting_balance", 100000))
    pos_size_r       = float(s.get("position_size_r", 0.8))

    price   = hb.get("price", 0)
    rsi     = hb.get("rsi", 0)
    peak    = hb.get("peak_price", 0)
    open_tr = hb.get("open_trade", False)
    fails   = hb.get("consecutive_failures", 0)
    ver     = hb.get("strategy_version", "?")
    ts      = hb.get("ts", "")
    sl      = float(s.get("stop_loss_pct", 2.0))
    tp      = float(s.get("take_profit_pct", 0.0))
    tsp     = float(s.get("trailing_stop_pct", 0.0))
    lev     = float(s.get("leverage", 1.0))
    thr     = ecfg.get("threshold", 30)
    rsi_ex  = float(ecfg.get("rsi_exit_threshold", 0.0))
    trail   = round(peak * (1 - tsp / 100), 2) if open_tr and peak and tsp else ""

    # running balance and per-trade dollar PnL
    balance = starting_balance
    trade_dollar_pnls = []
    for t in closed:
        t_pos_size = float(t.get("position_size_r", pos_size_r))
        d = dollar_pnl(t.get("pnl_pct_levered", 0), t_pos_size, balance)
        trade_dollar_pnls.append(d)
        balance += d

    total_dollar_pnl = balance - starting_balance

    # ── terminal line ─────────────────────────────────────────────────────────
    now = datetime.now().strftime("%H:%M:%S")
    status = "IN TRADE" if open_tr else "waiting"
    print(f"[{now}] updated  price=${price:,.2f}  rsi={rsi:.2f}  "
          f"status={status}  trades={len(closed)}  "
          f"balance=${balance:,.2f}  pnl={usd(total_dollar_pnl)}")

    # ── position.csv ──────────────────────────────────────────────────────────
    write_csv(BASE / "position.csv", [
        ["Field",                "Value"],
        ["Updated",              datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Asset",                hb.get("asset", "BTC/USDT")],
        ["Status",               "In trade" if open_tr else "No open position"],
        ["Direction",            ecfg.get("direction", "long").upper() if open_tr else ""],
        ["Price",                price],
        ["RSI",                  round(rsi, 2)],
        ["Peak price",           peak if open_tr else ""],
        ["Trailing stop level",  trail],
        ["Hard stop",            f"-{sl}% from entry"],
        ["Take profit",          f"+{tp}% from entry"],
        ["RSI exits above",      rsi_ex if rsi_ex else "off"],
        ["Leverage",             f"{lev}x"],
        ["Entry fires when",     f"RSI < {thr}"],
        [""],
        ["Starting balance",     f"${starting_balance:,.2f}"],
        ["Current balance",      f"${balance:,.2f}"],
        ["Total PnL $",          usd(total_dollar_pnl)],
        ["Total PnL %",          pct(total_dollar_pnl / starting_balance)],
        [""],
        ["Strategy version",     f"v{ver}"],
        ["Last tick",            age(ts)],
        ["Consecutive failures", fails],
        ["Agent",                "Healthy" if fails == 0 else "Degraded"],
    ])

    # ── stats.csv ─────────────────────────────────────────────────────────────
    if not closed:
        stats_rows = [
            ["Field",              "Value"],
            ["Updated",            datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",   f"${starting_balance:,.2f}"],
            ["Current balance",    f"${balance:,.2f}"],
            ["Total PnL $",        "$0.00"],
            ["Total PnL %",        "0.00%"],
            ["Trades closed",      0],
            ["Win rate",           ""],
            ["Best trade $",       ""],
            ["Worst trade $",      ""],
            ["Max drawdown",       ""],
        ]
    else:
        wins   = [t for t in closed if t.get("pnl_pct", 0) > 0]
        losses = [t for t in closed if t.get("pnl_pct", 0) <= 0]
        wr     = len(wins) / len(closed) * 100
        best_i = max(range(len(closed)), key=lambda i: closed[i].get("pnl_pct_levered", 0))
        worst_i= min(range(len(closed)), key=lambda i: closed[i].get("pnl_pct_levered", 0))

        cum = 1.0; pk = 1.0; dd = 0.0
        for t in closed:
            cum *= (1 + t.get("pnl_pct_levered", 0))
            pk   = max(pk, cum)
            dd   = max(dd, (pk - cum) / pk)

        stats_rows = [
            ["Field",              "Value"],
            ["Updated",            datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",   f"${starting_balance:,.2f}"],
            ["Current balance",    f"${balance:,.2f}"],
            ["Total PnL $",        usd(total_dollar_pnl)],
            ["Total PnL %",        pct(total_dollar_pnl / starting_balance)],
            ["Trades closed",      f"{len(closed)}  ({len(wins)}W / {len(losses)}L)"],
            ["Win rate",           f"{wr:.0f}%"],
            ["Best trade $",       usd(trade_dollar_pnls[best_i])],
            ["Worst trade $",      usd(trade_dollar_pnls[worst_i])],
            ["Max drawdown",       f"-{dd * 100:.2f}%"],
        ]
    write_csv(BASE / "stats.csv", stats_rows)

    # ── model.csv ─────────────────────────────────────────────────────────────
    m = rmodel()
    if m.get("status") == "ok":
        mc  = m.get("monte_carlo", {})
        k   = m.get("kelly", {})
        reg = m.get("regime", {})
        st  = m.get("stats", {})
        rec = m.get("recommendations", [])
        model_rows = [
            ["Field",                   "Value"],
            ["Updated",                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Signal",                  m.get("signal", "").replace("_", " ").upper()],
            ["Signal score",            f"{m.get('signal_score', 0):.3f}  (-1 avoid → +1 strong buy)"],
            ["Action",                  m.get("action", "").upper()],
            ["Confidence",              f"{m.get('confidence', 0):.1%}"],
            [""],
            ["── REGIME ──",            ""],
            ["Market regime",           reg.get("regime", "").replace("_", " ")],
            ["Regime quality",          f"{reg.get('quality', 0):.1%}"],
            ["ATR %",                   f"{reg.get('atr_pct', 0):.3f}%"],
            ["Description",             reg.get("description", "")],
            [""],
            ["── MONTE CARLO (next 100 trades) ──", ""],
            ["Paths simulated",         f"{mc.get('n_paths', 0):,}"],
            ["Equity P5  (bad)",        f"${mc.get('equity_p5', 0):,.0f}"],
            ["Equity P25",              f"${mc.get('equity_p25', 0):,.0f}"],
            ["Equity P50 (median)",     f"${mc.get('equity_p50', 0):,.0f}"],
            ["Equity P75",              f"${mc.get('equity_p75', 0):,.0f}"],
            ["Equity P95 (good)",       f"${mc.get('equity_p95', 0):,.0f}"],
            ["Expected return",         pct(mc.get("expected_return", 0))],
            ["Max drawdown (median)",   f"{mc.get('max_dd_median', 0)*100:.2f}%"],
            ["Max drawdown (P95)",      f"{mc.get('max_dd_p95', 0)*100:.2f}%"],
            ["Risk of ruin (−8%)",      f"{mc.get('ruin_probability', 0):.1%}"],
            [""],
            ["── KELLY SIZING ──",      ""],
            ["Win probability",         f"{k.get('win_probability', 0):.1%}"],
            ["Payoff ratio",            f"{k.get('payoff_ratio', 0):.2f}x"],
            ["Kelly full",              f"{k.get('kelly_full', 0):.1%}"],
            ["Kelly quarter (recommended)", f"{k.get('kelly_quarter', 0):.1%}"],
            [""],
            ["── TRADE STATS ──",       ""],
            ["Win rate",                f"{st.get('win_rate', 0):.1%}"],
            ["Avg win",                 f"{st.get('avg_win_pct', 0):+.3f}%"],
            ["Avg loss",                f"{st.get('avg_loss_pct', 0):+.3f}%"],
            ["Sharpe ratio",            f"{st.get('sharpe', 0):.3f}"],
            ["Max drawdown (actual)",   f"{st.get('max_drawdown', 0)*100:.2f}%"],
            ["Loss streak",             st.get("loss_streak", 0)],
            [""],
            ["── RECOMMENDATIONS ──",   ""],
        ]
        if not rec:
            model_rows.append(["No recommendations", "Strategy within normal bounds"])
        else:
            for i, r in enumerate(rec, 1):
                model_rows.append([f"[{r.get('urgency','').upper()}] {r.get('field','')}",
                                    f"{r.get('current','')} → {r.get('suggested','')}"])
                model_rows.append(["  Reason", r.get("reason", "")])
    else:
        model_rows = [
            ["Field", "Value"],
            ["Status", m.get("status", "model not yet run")],
            ["Note", "Need 3+ closed trades to activate model"],
        ]
    write_csv(BASE / "model.csv", model_rows)

    # ── trades.csv ────────────────────────────────────────────────────────────
    header = ["Trade ID", "Direction", "Entry Price", "Exit Price",
              "PnL $", "PnL %", "Balance After", "Exit Reason",
              "Strategy", "Leverage", "Entry Time", "Exit Time"]
    if not closed:
        trade_rows = [header, ["No closed trades yet"] + [""] * 11]
    else:
        running = starting_balance
        rows = []
        for i, t in enumerate(closed):
            d = trade_dollar_pnls[i]
            running += d
            rows.append([
                t.get("id", ""),
                t.get("direction", "").upper(),
                f"${t.get('entry_price', 0):,.2f}",
                f"${t.get('exit_price', 0):,.2f}",
                usd(d),
                pct(t.get("pnl_pct_levered", 0)),
                f"${running:,.2f}",
                t.get("exit_reason", "").replace("_", " "),
                "v" + str(t.get("strategy_version", "?")),
                f'{t.get("leverage", 1)}x',
                t.get("entry_time", ""),
                t.get("exit_time", ""),
            ])
        trade_rows = [header] + list(reversed(rows))
    write_csv(BASE / "trades.csv", trade_rows)


if __name__ == "__main__":
    print("Hermes tracker running — Ctrl+C to stop\n")
    while True:
        try:
            run()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] error: {e}")
        time.sleep(REFRESH)
