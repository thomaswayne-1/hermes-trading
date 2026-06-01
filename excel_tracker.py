#!/usr/bin/env python3
"""
excel_tracker.py — live CSV tracker for Hermes Trading.

Two modes (auto-detected):
  Remote  — set RAILWAY_URL and API_SECRET env vars; fetches from Railway API.
  Local   — reads state/ files directly (fallback / local dev).

Usage:
    # Remote (Railway is primary):
    RAILWAY_URL=https://your-app.up.railway.app API_SECRET=yourkey python3 excel_tracker.py

    # Local dev:
    cd ~/hermes-trading && python3 excel_tracker.py

Writes three CSV files every 10 seconds:
    position.csv   — live position + agent health
    stats.csv      — session P&L summary
    model.csv      — Monte Carlo + Kelly output
    trades.csv     — full trade history
"""

import csv
import json
import os
import ssl
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

BASE    = Path(__file__).parent
STATE   = BASE / "state"
REFRESH = 10

RAILWAY_URL = os.getenv("RAILWAY_URL", "").rstrip("/")
API_SECRET  = os.getenv("API_SECRET", "")


# ── Data fetching (remote or local) ──────────────────────────────────────────

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _http_get(path: str) -> dict | list:
    url = f"{RAILWAY_URL}{path}"
    req = urllib.request.Request(url)
    if API_SECRET:
        req.add_header("X-API-Key", API_SECRET)
    with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())


def rj(name):
    if RAILWAY_URL:
        return {}  # fetched via /state blob
    try:
        return json.loads((STATE / name).read_text())
    except Exception:
        return {}


def rjsonl(name):
    if RAILWAY_URL:
        return []  # fetched via /trades
    p = STATE / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def rstrat():
    if RAILWAY_URL:
        return {}  # fetched via /state blob
    try:
        return yaml.safe_load((STATE / "strategy.yaml").read_text()) or {}
    except Exception:
        return {}


def rgoal():
    if RAILWAY_URL:
        return {}
    try:
        return yaml.safe_load((STATE / "goal.yaml").read_text()) or {}
    except Exception:
        return {}


def rmodel():
    if RAILWAY_URL:
        return {}
    try:
        return json.loads((BASE / "state" / "model.json").read_text())
    except Exception:
        return {}


def fetch_all():
    """Return (heartbeat, trades, strategy, goal, model) regardless of mode."""
    if RAILWAY_URL:
        try:
            blob   = _http_get("/state")
            trades = _http_get("/trades")
            return (
                blob.get("heartbeat", {}),
                [t for t in trades if t.get("closed")],
                blob.get("strategy", {}),
                blob.get("goal", {}),
                blob.get("model", {}),
            )
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Railway fetch error: {e}")
            return {}, [], {}, {}, {}
    else:
        hb     = rj("heartbeat.json")
        trades = [t for t in rjsonl("trades.jsonl") if t.get("closed")]
        s      = rstrat()
        g      = rgoal()
        m      = rmodel()
        return hb, trades, s, g, m


# ── Formatting ────────────────────────────────────────────────────────────────

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
    except Exception:
        return "?"


def dollar_pnl(pnl_levered, position_size_r, balance):
    return balance * position_size_r * pnl_levered


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ── Main render loop ──────────────────────────────────────────────────────────

def unrealised_pnl(open_trades: list, current_price: float, balance: float) -> tuple[float, list]:
    """
    Calculate unrealised dollar PnL across all open positions.
    Returns (total_unrealised_usd, list of per-trade dicts).
    """
    details = []
    total = 0.0
    for t in open_trades:
        entry = float(t.get("entry_price", current_price) or current_price)
        lev   = float(t.get("leverage", 1.0))
        size  = float(t.get("size") or t.get("position_size_r", 0.15))
        direction = t.get("direction", "long")
        if entry <= 0 or current_price <= 0:
            continue
        raw = (current_price - entry) / entry if direction == "long" \
              else (entry - current_price) / entry
        gross = raw * lev
        dollar = gross * size * balance
        total += dollar
        details.append({
            "id":        t.get("id", "?"),
            "direction": direction.upper(),
            "entry":     entry,
            "raw_pct":   raw,
            "gross_pct": gross,
            "dollar":    dollar,
            "leverage":  lev,
            "size":      size,
        })
    return total, details


def run():
    hb, closed, s, g, m = fetch_all()

    ecfg   = s.get("entry", {})

    starting_balance = float(g.get("starting_balance", 100000))
    pos_size_r       = float(s.get("position_size_base", 0.15))

    price      = hb.get("price", 0)
    rsi        = hb.get("rsi", 0)
    peak       = hb.get("peak_price", 0)
    open_tr    = hb.get("open_trade", False)
    open_trades_hb = hb.get("open_trades", [])
    fails      = hb.get("consecutive_failures", 0)
    ver        = hb.get("strategy_version", "?")
    ts         = hb.get("ts", "")
    sl         = float(s.get("stop_loss_pct", 0.5))
    tp         = float(s.get("take_profit_pct", 0.0))
    tsp        = float(s.get("trailing_stop_pct", 0.0))
    trail      = round(peak * (1 - tsp / 100), 2) if open_tr and peak and tsp else ""

    # Realised balance from closed trade history
    # Use pnl_pct_net (after taker fees + funding) — falls back to pnl_pct_levered
    # for old trades recorded before net PnL was tracked.
    balance = starting_balance
    trade_dollar_pnls = []
    for t in closed:
        t_pos  = float(t.get("position_size_r", pos_size_r))
        pnl    = t.get("pnl_pct_net") if t.get("pnl_pct_net") is not None \
                 else t.get("pnl_pct_levered", 0)
        d      = dollar_pnl(pnl, t_pos, balance)
        trade_dollar_pnls.append(d)
        balance += d

    realised_pnl = balance - starting_balance

    # Unrealised PnL from open positions (uses live price)
    unrealised, open_details = unrealised_pnl(open_trades_hb, price, balance)
    total_dollar_pnl = realised_pnl + unrealised
    effective_balance = balance + unrealised

    # ── Terminal line ──────────────────────────────────────────────────────────
    src  = "railway" if RAILWAY_URL else "local"
    now  = datetime.now().strftime("%H:%M:%S")

    if open_details:
        pos_str = "  ".join(
            f"{d['direction']}@${d['entry']:,.0f}({'+' if d['dollar']>=0 else ''}{d['dollar']:,.0f})"
            for d in open_details
        )
        status = f"IN TRADE [{pos_str}]"
    else:
        status = "waiting"

    print(f"[{now}] [{src}]  price=${price:,.2f}  rsi={rsi:.2f}  "
          f"status={status}  "
          f"closed={len(closed)}  "
          f"balance=${effective_balance:,.2f}  "
          f"realised={usd(realised_pnl)}  unrealised={usd(unrealised)}")

    # ── position.csv ──────────────────────────────────────────────────────────
    pos_rows = [
        ["Field",                "Value"],
        ["Updated",              datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Source",               "Railway (cloud)" if RAILWAY_URL else "Local"],
        ["Asset",                hb.get("asset", "BTC/USDT")],
        ["Current price",        f"${price:,.2f}"],
        ["RSI",                  round(rsi, 2)],
        ["MACD hist",            round(hb.get("macd_hist", 0), 4)],
        ["BB %B",                round(hb.get("bb_pct", 0), 4)],
        ["OB imbalance",         round(hb.get("ob_imbalance", 0), 4)],
        [""],
        ["── OPEN POSITIONS ──", f"{len(open_details)} open"],
    ]
    if open_details:
        for i, d in enumerate(open_details, 1):
            move_pct = d["raw_pct"] * 100
            pos_rows += [
                [f"  Position {i}",        f"{d['direction']}  {d['size']*100:.0f}%  {d['leverage']:.1f}x lev"],
                [f"  Entry price",          f"${d['entry']:,.2f}"],
                [f"  Move vs entry",        f"{'+' if move_pct>=0 else ''}{move_pct:.3f}%"],
                [f"  Gross PnL (levered)",  f"{'+' if d['gross_pct']>=0 else ''}{d['gross_pct']*100:.3f}%"],
                [f"  Dollar PnL",           usd(d["dollar"])],
                [f"  Stop loss at",         f"${d['entry'] * (1 + sl/100) if d['direction']=='SHORT' else d['entry'] * (1 - sl/100):,.2f}  ({'-' if d['direction']=='LONG' else '+'}{sl}%)"],
            ]
    else:
        pos_rows.append(["  No open positions", ""])

    pos_rows += [
        [""],
        ["── BALANCE ──",        ""],
        ["Starting balance",     f"${starting_balance:,.2f}"],
        ["Realised PnL",         usd(realised_pnl)],
        ["Unrealised PnL",       usd(unrealised)],
        ["Effective balance",    f"${effective_balance:,.2f}"],
        ["Total PnL $",          usd(total_dollar_pnl)],
        ["Total PnL %",          pct(total_dollar_pnl / starting_balance)],
        [""],
        ["── STRATEGY ──",       ""],
        ["Long entry RSI <",     ecfg.get("long_threshold", 55)],
        ["Long RSI exit >",      ecfg.get("long_rsi_exit", 78)],
        ["Short entry RSI >",    ecfg.get("short_threshold", 65)],
        ["Short RSI exit <",     ecfg.get("short_rsi_exit", 25)],
        ["Leverage range",       f"{s.get('leverage_base', 1.5)}x – {s.get('leverage_max', 3.0)}x"],
        ["Position size range",  f"{s.get('position_size_base', 0.15)*100:.0f}% – {s.get('position_size_max', 0.40)*100:.0f}%"],
        ["Stop loss",            f"-{sl}%"],
        ["Take profit",          f"+{tp}%"],
        [""],
        ["Strategy version",     f"v{ver}"],
        ["Last tick",            age(ts)],
        ["Consecutive failures", fails],
        ["Agent",                "Healthy" if fails == 0 else "Degraded"],
    ]
    write_csv(BASE / "position.csv", pos_rows)

    # ── stats.csv ─────────────────────────────────────────────────────────────
    if not closed:
        stats_rows = [
            ["Field",              "Value"],
            ["Updated",            datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",   f"${starting_balance:,.2f}"],
            ["Effective balance",  f"${effective_balance:,.2f}"],
            ["Realised PnL",       usd(realised_pnl)],
            ["Unrealised PnL",     usd(unrealised)],
            ["Total PnL $",        usd(total_dollar_pnl)],
            ["Total PnL %",        pct(total_dollar_pnl / starting_balance)],
            ["Trades closed",      0],
            ["Open positions",     len(open_details)],
            ["Win rate",           ""],
            ["Best trade $",       ""],
            ["Worst trade $",      ""],
            ["Max drawdown",       ""],
        ]
    else:
        wins    = [t for t in closed if t.get("pnl_pct", 0) > 0]
        losses  = [t for t in closed if t.get("pnl_pct", 0) <= 0]
        longs   = [t for t in closed if t.get("direction") == "long"]
        shorts  = [t for t in closed if t.get("direction") == "short"]
        wr      = len(wins) / len(closed) * 100
        best_i  = max(range(len(closed)), key=lambda i: closed[i].get("pnl_pct_levered", 0))
        worst_i = min(range(len(closed)), key=lambda i: closed[i].get("pnl_pct_levered", 0))

        cum = 1.0; pk = 1.0; dd = 0.0
        for t in closed:
            cum *= (1 + t.get("pnl_pct_levered", 0))
            pk   = max(pk, cum)
            dd   = max(dd, (pk - cum) / pk)

        # score breakdown by entry score
        score_counts = {}
        for t in closed:
            sc = t.get("entry_score", "?")
            score_counts[sc] = score_counts.get(sc, 0) + 1
        score_str = "  ".join(f"{k}/4×{v}" for k, v in sorted(score_counts.items()) if k != "?")

        stats_rows = [
            ["Field",              "Value"],
            ["Updated",            datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",   f"${starting_balance:,.2f}"],
            ["Effective balance",  f"${effective_balance:,.2f}"],
            ["Realised PnL",       usd(realised_pnl)],
            ["Unrealised PnL",     usd(unrealised)],
            ["Total PnL $",        usd(total_dollar_pnl)],
            ["Total PnL %",        pct(total_dollar_pnl / starting_balance)],
            ["Open positions",     len(open_details)],
            ["Trades closed",      f"{len(closed)}  ({len(wins)}W / {len(losses)}L)"],
            ["Long / Short",       f"{len(longs)}L / {len(shorts)}S"],
            ["Win rate",           f"{wr:.0f}%"],
            ["Score distribution", score_str or "n/a"],
            ["Best trade $",       usd(trade_dollar_pnls[best_i])],
            ["Worst trade $",      usd(trade_dollar_pnls[worst_i])],
            ["Max drawdown",       f"-{dd * 100:.2f}%"],
        ]
    write_csv(BASE / "stats.csv", stats_rows)

    # ── model.csv ─────────────────────────────────────────────────────────────
    if m.get("status") == "ok":
        mc  = m.get("monte_carlo", {})
        k   = m.get("kelly", {})
        reg = m.get("regime", {})
        st  = m.get("stats", {})
        rec = m.get("recommendations", [])
        model_rows = [
            ["Field",                        "Value"],
            ["Updated",                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Signal",                       m.get("signal", "").replace("_", " ").upper()],
            ["Signal score",                 f"{m.get('signal_score', 0):.3f}  (-1 avoid → +1 strong buy)"],
            ["Action",                       m.get("action", "").upper()],
            ["Confidence",                   f"{m.get('confidence', 0):.1%}"],
            [""],
            ["── REGIME ──",                 ""],
            ["Market regime",                reg.get("regime", "").replace("_", " ")],
            ["Regime quality",               f"{reg.get('quality', 0):.1%}"],
            ["ATR %",                        f"{reg.get('atr_pct', 0):.3f}%"],
            ["Description",                  reg.get("description", "")],
            [""],
            ["── MONTE CARLO (next 100 trades) ──", ""],
            ["Paths simulated",              f"{mc.get('n_paths', 0):,}"],
            ["Equity P5  (bad)",             f"${mc.get('equity_p5', 0):,.0f}"],
            ["Equity P25",                   f"${mc.get('equity_p25', 0):,.0f}"],
            ["Equity P50 (median)",          f"${mc.get('equity_p50', 0):,.0f}"],
            ["Equity P75",                   f"${mc.get('equity_p75', 0):,.0f}"],
            ["Equity P95 (good)",            f"${mc.get('equity_p95', 0):,.0f}"],
            ["Expected return",              pct(mc.get("expected_return", 0))],
            ["Max drawdown (median)",        f"{mc.get('max_dd_median', 0)*100:.2f}%"],
            ["Max drawdown (P95)",           f"{mc.get('max_dd_p95', 0)*100:.2f}%"],
            ["Risk of ruin (−8%)",           f"{mc.get('ruin_probability', 0):.1%}"],
            [""],
            ["── KELLY SIZING ──",           ""],
            ["Win probability",              f"{k.get('win_probability', 0):.1%}"],
            ["Payoff ratio",                 f"{k.get('payoff_ratio', 0):.2f}x"],
            ["Kelly full",                   f"{k.get('kelly_full', 0):.1%}"],
            ["Kelly quarter (recommended)",  f"{k.get('kelly_quarter', 0):.1%}"],
            [""],
            ["── TRADE STATS ──",            ""],
            ["Win rate",                     f"{st.get('win_rate', 0):.1%}"],
            ["Avg win",                      f"{st.get('avg_win_pct', 0):+.3f}%"],
            ["Avg loss",                     f"{st.get('avg_loss_pct', 0):+.3f}%"],
            ["Sharpe ratio",                 f"{st.get('sharpe', 0):.3f}"],
            ["Max drawdown (actual)",        f"{st.get('max_drawdown', 0)*100:.2f}%"],
            ["Loss streak",                  st.get("loss_streak", 0)],
            [""],
            ["── RECOMMENDATIONS ──",        ""],
        ]
        if not rec:
            model_rows.append(["No recommendations", "Strategy within normal bounds"])
        else:
            for r in rec:
                model_rows.append([
                    f"[{r.get('urgency','').upper()}] {r.get('field','')}",
                    f"{r.get('current','')} → {r.get('suggested','')}",
                ])
                model_rows.append(["  Reason", r.get("reason", "")])
    else:
        model_rows = [
            ["Field",  "Value"],
            ["Status", m.get("status", "model not yet run")],
            ["Note",   "Need 3+ closed trades to activate model"],
        ]
    write_csv(BASE / "model.csv", model_rows)

    # ── trades.csv ────────────────────────────────────────────────────────────
    header = ["Trade ID", "Direction", "Score", "Entry Price", "Exit Price",
              "PnL $", "PnL %", "Balance After", "Exit Reason",
              "Strategy", "Leverage", "Entry Time", "Exit Time"]
    if not closed:
        trade_rows = [header, ["No closed trades yet"] + [""] * 12]
    else:
        running = starting_balance
        rows = []
        for i, t in enumerate(closed):
            d = trade_dollar_pnls[i]
            running += d
            rows.append([
                t.get("id", ""),
                t.get("direction", "").upper(),
                f"{t.get('entry_score', '?')}/4",
                f"${t.get('entry_price', 0):,.2f}",
                f"${t.get('exit_price', 0):,.2f}",
                usd(d),
                pct(t.get("pnl_pct_net") if t.get("pnl_pct_net") is not None else t.get("pnl_pct_levered", 0)),
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
    mode = "railway" if RAILWAY_URL else "local"
    print(f"Hermes tracker running [{mode}] — Ctrl+C to stop\n")
    if RAILWAY_URL:
        print(f"  Fetching from: {RAILWAY_URL}")
        print(f"  Auth:          {'enabled' if API_SECRET else 'DISABLED (set API_SECRET)'}\n")
    while True:
        try:
            run()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] error: {e}")
        time.sleep(REFRESH)
