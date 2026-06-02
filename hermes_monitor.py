#!/usr/bin/env python3
"""
hermes_monitor.py — combined live monitor + CSV tracker for Hermes Trading.

Replaces both watch_engine.py and excel_tracker.py.

Run:
    python3 hermes_monitor.py

Every 10 seconds:
  - Renders full engine monitor to terminal (regime, C/K, sub-signals,
    open positions, performance)
  - Writes position.csv, stats.csv, model.csv, trades.csv

Environment (optional — auto-detected):
    RAILWAY_URL   default: https://hermes-trading-production-169f.up.railway.app
    API_SECRET    default: hermes2024
"""

import csv
import json
import os
import ssl
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────
RAILWAY_URL = os.getenv("RAILWAY_URL", "https://hermes-trading-production-169f.up.railway.app").rstrip("/")
API_SECRET  = os.getenv("API_SECRET",  "hermes2024")
STARTING    = 100_000.0
INTERVAL    = 10
BASE        = Path(__file__).parent
W           = 60   # terminal display width

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(path: str):
    req = urllib.request.Request(
        f"{RAILWAY_URL}{path}",
        headers={"X-API-Key": API_SECRET},
    )
    with urllib.request.urlopen(req, context=_SSL, timeout=10) as r:
        return json.loads(r.read())


# ── Last-known-good cache ─────────────────────────────────────────────────────

_cache: dict = {
    "state": {}, "trades": [],
    "stale": False, "stale_since": None,
}


def fetch_all():
    global _cache
    try:
        state  = _get("/state")
        trades = _get("/trades")
        if not isinstance(trades, list):
            trades = trades.get("trades", [])
        closed = [t for t in trades if t.get("closed")]

        # Never trust a response with fewer closed trades than cache
        if len(closed) < len(_cache["trades"]):
            raise ValueError(
                f"Response has {len(closed)} closed trades but cache has "
                f"{len(_cache['trades'])} — likely a redeploy blip"
            )

        _cache.update({"state": state, "trades": closed,
                       "stale": False, "stale_since": None})
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        if not _cache["stale"]:
            _cache["stale"] = True
            _cache["stale_since"] = now
        print(f"  [{now}] ⚠ Using cached data — {e}")

    return _cache["state"], _cache["trades"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def bar(value: float, width: int = 22) -> str:
    mid    = width // 2
    filled = min(int(abs(value) * mid + 0.5), mid)
    if value >= 0:
        l, r = " " * mid, "█" * filled + " " * (mid - filled)
    else:
        l, r = " " * (mid - filled) + "█" * filled, " " * mid
    return f"[{l}│{r}]"


def _net(t: dict) -> float:
    v = t.get("pnl_pct_net")
    return v if v is not None else t.get("pnl_pct_levered", 0)


def _calc_balance(closed: list) -> tuple[float, list]:
    """Cumulative balance + per-trade dollar PnL list. Identical logic in both tools."""
    bal   = STARTING
    dolls = []
    for t in closed:
        size = float(t.get("position_size_r", 0.15))
        d    = _net(t) * size * bal
        dolls.append(d)
        bal  += d
    return bal, dolls


def _usd(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _pct(v: float) -> str:
    return f"{'+' if v > 0 else ''}{v * 100:.2f}%"


def _age(ts: str) -> str:
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


def _write_csv(name: str, rows: list) -> None:
    with open(BASE / name, "w", newline="") as f:
        csv.writer(f).writerows(rows)


# ── Terminal render ───────────────────────────────────────────────────────────

def render_terminal(state: dict, closed: list) -> None:
    hb    = state.get("heartbeat", {})
    eng   = hb.get("engine", {})
    cb    = hb.get("circuit_breaker", {})
    strat = state.get("strategy", {})
    sigs  = eng.get("sub_signals") or {}
    wts   = eng.get("weights_used") or {}

    price   = float(hb.get("price",        0))
    rsi     = float(hb.get("rsi",          50))
    macd    = float(hb.get("macd_hist",    0))
    bb      = float(hb.get("bb_pct",       0.5))
    atr     = float(hb.get("atr",          0))
    ob_imb  = float(hb.get("ob_imbalance", 0))
    # Convert UTC heartbeat timestamp → local time; compute age
    _ts_raw = hb.get("ts", "")
    ts = "?"
    data_age_s = 0
    try:
        _dt = datetime.fromisoformat(_ts_raw)
        if _dt.tzinfo is None:
            _dt = _dt.replace(tzinfo=timezone.utc)
        ts = _dt.astimezone().strftime("%H:%M:%S")   # time only — same day as local
        data_age_s = int((datetime.now(timezone.utc) - _dt).total_seconds())
    except Exception:
        ts = _ts_raw[:19].replace("T", " ")

    C       = float(eng.get("C",           0))
    K       = float(eng.get("K",           0))
    ready   = bool(eng.get("ready",        False))
    samps   = int(eng.get("samples",       0))
    reg     = str(eng.get("regime",        "warmup"))
    cert    = float(eng.get("regime_cert", 0))
    agree   = float(eng.get("agreement",   0))
    volp    = float(eng.get("vol_penalty", 1))
    volf    = float(eng.get("vol_forecast",0))

    engine_live = bool(strat.get("coefficient_engine_enabled", False))
    coef_cfg    = strat.get("coefficient", {})
    tau_enter   = float(coef_cfg.get("tau_enter", 0.12))
    max_pos     = int(strat.get("max_open_positions", 7))
    version     = strat.get("version", "?")
    cb_state    = cb.get("state", "normal").upper()

    open_trades = hb.get("open_trades", [])

    # Balance + performance
    balance, dolls = _calc_balance(closed)
    n       = len(closed)
    wins    = sum(1 for t in closed if _net(t) > 0)
    wr      = wins / n if n else 0.0
    net_pnl = balance - STARTING

    # Unrealised
    total_unreal = 0.0
    for t in open_trades:
        ep  = float(t.get("entry_price", price))
        lev = float(t.get("leverage",    1.5))
        sz  = float(t.get("size",        0.15))
        raw = (price - ep) / ep if t["direction"] == "long" else (ep - price) / ep
        total_unreal += raw * lev * sz * balance

    now_local = datetime.now().strftime("%H:%M:%S")
    stale_tag = f"  ⚠ STALE since {_cache['stale_since']}" if _cache["stale"] else ""

    # Heartbeat freshness tag
    if data_age_s <= 15:
        age_tag = f" ({data_age_s}s)"
    elif data_age_s <= 60:
        age_tag = f" ⚠ {data_age_s}s old"
    else:
        _m, _s = data_age_s // 60, data_age_s % 60
        age_tag = f" ⚠⚠ {_m}m{_s}s old"

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\033[2J\033[H", end="")   # clear screen
    print("━" * W)
    mode = "LIVE" if engine_live else "SHADOW"
    print(f"  HERMES  [{mode}]  v{version}   local {now_local}{stale_tag}")
    print("━" * W)

    # One-liner status (matches tracker format)
    if open_trades:
        pos_str = "  ".join(
            f"{t['direction'].upper()}@${float(t.get('entry_price',0)):,.0f}"
            for t in open_trades
        )
        status = f"IN TRADE [{pos_str}]"
    else:
        status = "waiting"
    eq = balance + total_unreal
    sym = "+" if eq >= STARTING else ""
    print(f"  {ts}{age_tag}   BTC ${price:,.2f}   RSI {rsi:.1f}")
    print(f"  Status: {status}   closed={n}   balance=${eq:,.2f}  ({sym}${eq-STARTING:,.2f})")
    print()

    # Engine status
    if not ready:
        e_status = f"⏳ WARMING UP  ({samps}/200 samples — ~{max(0,(200-samps)*INTERVAL)//60}min)"
    elif engine_live:
        e_status = "✅ LIVE & READY"
    else:
        e_status = "👁  SHADOW MODE"
    print(f"  Engine:  {e_status}")
    print(f"  Regime:  {reg.upper():12s}  certainty={cert:.2f}   Circuit: {cb_state}")
    print()

    # Signal
    print(f"  ── Directional Signal {'─'*(W-23)}")
    direction = "LONG  ▲" if C > 0.05 else ("SHORT ▼" if C < -0.05 else "FLAT  —")
    print(f"  C = {C:+.4f}  {bar(C, 24)}  {direction}")
    print(f"  K = {K:.4f}   Agreement={agree:.2f}  VolPenalty={volp:.2f}  VolFcast={volf:.5f}")
    print()

    if ready:
        if engine_live:
            if abs(C) > tau_enter:
                print(f"  ► ENTERING {'LONG' if C>0 else 'SHORT'}  "
                      f"(|C|={abs(C):.4f} > τ={tau_enter})")
            else:
                print(f"  ► NO ENTRY  — need +{tau_enter-abs(C):.4f} more  "
                      f"(|C|={abs(C):.4f} ≤ τ={tau_enter})")
        else:
            side = "LONG" if C > 0 else "SHORT"
            print(f"  ► SHADOW: {'signal — ' + side if abs(C)>tau_enter else 'no signal'}  "
                  f"(|C|={abs(C):.4f})")
    else:
        print(f"  ► Warming up — {200-samps} ticks remaining")
    print()

    # Sub-signals
    print(f"  ── Sub-signals  [{reg.upper()}] {'─'*max(0,W-20-len(reg))}")
    for key, label in [("mom","Momentum  "),("rev","Mean-Rev  "),
                       ("mic","Microstr  "),("sen","Sentiment ")]:
        v = float(sigs.get(key, 0))
        w = float(wts.get(key, 0))
        lean = "▲ bullish" if v > 0.15 else ("▼ bearish" if v < -0.15 else "  neutral")
        print(f"  {label} {v:+.3f}  {bar(v, 20)}  w={w:.2f}  {lean}")
    print()

    # Market indicators
    print(f"  ── Market Indicators {'─'*(W-22)}")
    ob_lbl = "buy pressure" if ob_imb>0.1 else ("sell pressure" if ob_imb<-0.1 else "balanced")
    bb_lbl = "overbought" if bb>0.8 else ("oversold" if bb<0.2 else f"{bb:.2f}")
    print(f"  RSI {rsi:5.1f}   MACD {macd:+.4f}   ATR ${atr:.2f}   BB% {bb_lbl}")
    print(f"  OB imbalance {ob_imb:+.3f}  ({ob_lbl})")
    print()

    # Open positions
    print(f"  ── Open Positions ({len(open_trades)}/{max_pos}) {'─'*max(0,W-22)}")
    if open_trades:
        for t in open_trades:
            ep    = float(t.get("entry_price", price))
            lev   = float(t.get("leverage",    1.5))
            sz    = float(t.get("size",        0.15))
            ec    = t.get("entry_C");  ek = t.get("entry_K")
            raw   = (price-ep)/ep if t["direction"]=="long" else (ep-price)/ep
            unr_pct = raw * lev * 100
            unr_usd = raw * lev * sz * balance
            c_str = f"C={ec:+.3f} " if ec is not None else ""
            k_str = f"K={ek:.3f}"   if ek is not None else ""
            sym   = "+" if unr_pct >= 0 else ""
            print(f"  {t['direction'].upper():5s} @ ${ep:>10,.2f}  "
                  f"lev={lev:.1f}x  sz={sz*100:.0f}%  "
                  f"{c_str}{k_str}  {sym}{unr_pct:.2f}%  (${sym}{unr_usd:,.0f})")
        sym = "+" if total_unreal >= 0 else ""
        print(f"  {'':48s}total {sym}${total_unreal:,.0f}")
    else:
        print("  No open positions")
    print()

    # Performance
    print(f"  ── Performance {'─'*(W-17)}")
    if n > 0:
        sym = "+" if net_pnl >= 0 else ""
        print(f"  Closed={n}  WR={wr*100:.0f}%  "
              f"Realised=${balance:,.2f} ({sym}${net_pnl:,.2f})  "
              f"Unrealised=${total_unreal:+,.0f}")
        print(f"  Total equity: ${eq:,.2f}")
    else:
        print(f"  No closed trades yet   Starting: ${STARTING:,.0f}")
    print()
    print(f"  Refreshing every {INTERVAL}s — Ctrl+C to stop")
    print("━" * W)


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_csvs(state: dict, closed: list) -> None:
    hb    = state.get("heartbeat", {})
    strat = state.get("strategy", {})
    goal  = state.get("goal",     {})
    model = state.get("model",    {})
    eng   = hb.get("engine", {})

    price         = float(hb.get("price", 0))
    rsi           = float(hb.get("rsi",   0))
    open_trades   = hb.get("open_trades", [])
    starting      = float(goal.get("starting_balance", STARTING))
    ecfg          = strat.get("entry", {})
    sl            = float(strat.get("stop_loss_pct", 0.5))
    tp            = float(strat.get("take_profit_pct", 0.0))
    ver           = hb.get("strategy_version", "?")
    ts            = hb.get("ts", "")
    fails         = hb.get("consecutive_failures", 0)

    balance, dolls = _calc_balance(closed)
    net_pnl        = balance - starting

    # Unrealised
    unrealised = 0.0
    open_details = []
    for t in open_trades:
        ep  = float(t.get("entry_price", price) or price)
        lev = float(t.get("leverage", 1.0))
        sz  = float(t.get("size") or t.get("position_size_r", 0.15))
        if ep <= 0 or price <= 0:
            continue
        raw   = (price-ep)/ep if t.get("direction")=="long" else (ep-price)/ep
        gross = raw * lev
        dol   = gross * sz * balance
        unrealised += dol
        open_details.append({
            "id": t.get("id","?"), "direction": t.get("direction","long").upper(),
            "entry": ep, "raw_pct": raw, "gross_pct": gross,
            "dollar": dol, "leverage": lev, "size": sz,
        })

    eq              = balance + unrealised
    total_pnl       = eq - starting

    # ── position.csv ─────────────────────────────────────────────────────────
    rows = [
        ["Field",                "Value"],
        ["Updated",              datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Source",               RAILWAY_URL],
        ["Asset",                hb.get("asset", "BTC/USDT")],
        ["Current price",        f"${price:,.2f}"],
        ["RSI",                  round(rsi, 2)],
        ["MACD hist",            round(hb.get("macd_hist", 0), 4)],
        ["BB %B",                round(hb.get("bb_pct", 0), 4)],
        ["OB imbalance",         round(hb.get("ob_imbalance", 0), 4)],
        ["Engine C",             round(float(eng.get("C", 0)), 4)],
        ["Engine K",             round(float(eng.get("K", 0)), 4)],
        ["Regime",               eng.get("regime", "?")],
        [""],
        ["── OPEN POSITIONS ──", f"{len(open_details)} open"],
    ]
    if open_details:
        for i, d in enumerate(open_details, 1):
            rows += [
                [f"  Position {i}",         f"{d['direction']}  {d['size']*100:.0f}%  {d['leverage']:.1f}x lev"],
                [f"  Entry price",           f"${d['entry']:,.2f}"],
                [f"  Move vs entry",         f"{'+' if d['raw_pct']>=0 else ''}{d['raw_pct']*100:.3f}%"],
                [f"  Gross PnL (levered)",   f"{'+' if d['gross_pct']>=0 else ''}{d['gross_pct']*100:.3f}%"],
                [f"  Dollar PnL",            _usd(d["dollar"])],
                [f"  Stop loss at",
                 f"${d['entry']*(1+sl/100) if d['direction']=='SHORT' else d['entry']*(1-sl/100):,.2f}"],
            ]
    else:
        rows.append(["  No open positions", ""])
    rows += [
        [""],
        ["── BALANCE ──",        ""],
        ["Starting balance",     f"${starting:,.2f}"],
        ["Realised PnL",         _usd(net_pnl)],
        ["Unrealised PnL",       _usd(unrealised)],
        ["Total PnL $",          _usd(total_pnl)],
        ["Total PnL %",          _pct(total_pnl / starting)],
        ["Effective balance",    f"${eq:,.2f}"],
        [""],
        ["── ENGINE ──",         ""],
        ["C (direction)",        round(float(eng.get("C", 0)), 4)],
        ["K (conviction)",       round(float(eng.get("K", 0)), 4)],
        ["Regime",               eng.get("regime", "?")],
        ["Regime certainty",     round(float(eng.get("regime_cert", 0)), 4)],
        ["Engine enabled",       strat.get("coefficient_engine_enabled", False)],
        ["Samples",              eng.get("samples", 0)],
        [""],
        ["── STRATEGY ──",       ""],
        ["Long entry RSI <",     ecfg.get("long_threshold", 55)],
        ["Short entry RSI >",    ecfg.get("short_threshold", 65)],
        ["Max positions",        strat.get("max_open_positions", 7)],
        ["Stop loss",            f"-{sl}%"],
        ["Take profit",          f"+{tp}%"],
        ["Strategy version",     f"v{ver}"],
        ["Last tick",            _age(hb.get("ts",""))],
        ["Consecutive failures", fails],
        ["Agent",                "Healthy" if fails == 0 else "Degraded"],
    ]
    _write_csv("position.csv", rows)

    # ── stats.csv ─────────────────────────────────────────────────────────────
    if not closed:
        stats = [
            ["Field",             "Value"],
            ["Updated",           datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",  f"${starting:,.2f}"],
            ["Effective balance", f"${eq:,.2f}"],
            ["Realised PnL",      _usd(net_pnl)],
            ["Unrealised PnL",    _usd(unrealised)],
            ["Total PnL $",       _usd(total_pnl)],
            ["Total PnL %",       _pct(total_pnl / starting)],
            ["Trades closed",     0],
            ["Open positions",    len(open_details)],
        ]
    else:
        wins    = [t for t in closed if _net(t) > 0]
        losses  = [t for t in closed if _net(t) <= 0]
        longs   = [t for t in closed if t.get("direction") == "long"]
        shorts  = [t for t in closed if t.get("direction") == "short"]
        wr      = len(wins) / len(closed) * 100
        best_i  = max(range(len(closed)), key=lambda i: _net(closed[i]))
        worst_i = min(range(len(closed)), key=lambda i: _net(closed[i]))

        cum = 1.0; pk = 1.0; dd = 0.0
        for t in closed:
            cum *= (1 + _net(t))
            pk   = max(pk, cum)
            dd   = max(dd, (pk - cum) / pk)

        score_counts: dict = {}
        for t in closed:
            sc = t.get("entry_score", "eng")
            score_counts[sc] = score_counts.get(sc, 0) + 1
        score_str = "  ".join(f"{k}×{v}" for k, v in sorted(score_counts.items(), key=lambda x: str(x[0])))

        stats = [
            ["Field",             "Value"],
            ["Updated",           datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Starting balance",  f"${starting:,.2f}"],
            ["Realised balance",  f"${balance:,.2f}"],
            ["Unrealised PnL",    _usd(unrealised)],
            ["Total PnL $",       _usd(total_pnl)],
            ["Total PnL %",       _pct(total_pnl / starting)],
            ["Effective balance", f"${eq:,.2f}"],
            ["Open positions",    len(open_details)],
            ["Trades closed",     f"{len(closed)}  ({len(wins)}W / {len(losses)}L)"],
            ["Long / Short",      f"{len(longs)}L / {len(shorts)}S"],
            ["Win rate",          f"{wr:.0f}%"],
            ["Score dist",        score_str or "n/a"],
            ["Best trade $",      _usd(dolls[best_i])],
            ["Worst trade $",     _usd(dolls[worst_i])],
            ["Max drawdown",      f"-{dd * 100:.2f}%"],
        ]
    _write_csv("stats.csv", stats)

    # ── trades.csv ────────────────────────────────────────────────────────────
    header = ["Trade ID","Direction","Score","Entry Price","Exit Price",
              "PnL $","PnL %","Balance After","Exit Reason",
              "Strategy","Leverage","Entry Time","Exit Time"]
    if not closed:
        trade_rows = [header, ["No closed trades yet"] + [""]*12]
    else:
        running = starting
        rows2 = []
        for i, t in enumerate(closed):
            running += dolls[i]
            rows2.append([
                t.get("id",""),
                t.get("direction","").upper(),
                f"{t.get('entry_score','eng')}",
                f"${t.get('entry_price',0):,.2f}",
                f"${t.get('exit_price',0):,.2f}",
                _usd(dolls[i]),
                _pct(_net(t)),
                f"${running:,.2f}",
                t.get("exit_reason","").replace("_"," "),
                "v" + str(t.get("strategy_version","?")),
                f'{t.get("leverage",1)}x',
                t.get("entry_time",""),
                t.get("exit_time",""),
            ])
        trade_rows = [header] + list(reversed(rows2))
    _write_csv("trades.csv", trade_rows)

    # ── model.csv ─────────────────────────────────────────────────────────────
    if model.get("status") == "ok":
        mc  = model.get("monte_carlo", {})
        k   = model.get("kelly", {})
        reg = model.get("regime", {})
        st  = model.get("stats", {})
        rec = model.get("recommendations", [])
        model_rows = [
            ["Field",                        "Value"],
            ["Updated",                      datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Signal",                       model.get("signal","").replace("_"," ").upper()],
            ["Action",                       model.get("action","").upper()],
            ["Confidence",                   f"{model.get('confidence',0):.1%}"],
            [""],
            ["── REGIME ──",                 ""],
            ["Market regime",                reg.get("regime","")],
            ["Regime quality",               f"{reg.get('quality',0):.1%}"],
            ["ATR %",                        f"{reg.get('atr_pct',0):.3f}%"],
            [""],
            ["── MONTE CARLO ──",            ""],
            ["Paths",                        f"{mc.get('n_paths',0):,}"],
            ["P5  (bad case)",               f"${mc.get('equity_p5',0):,.0f}"],
            ["P50 (median)",                 f"${mc.get('equity_p50',0):,.0f}"],
            ["P95 (good case)",              f"${mc.get('equity_p95',0):,.0f}"],
            ["Expected return",              _pct(mc.get("expected_return",0))],
            ["Max drawdown P95",             f"{mc.get('max_dd_p95',0)*100:.2f}%"],
            ["Risk of ruin",                 f"{mc.get('ruin_probability',0):.1%}"],
            [""],
            ["── KELLY ──",                  ""],
            ["Win probability",              f"{k.get('win_probability',0):.1%}"],
            ["Payoff ratio",                 f"{k.get('payoff_ratio',0):.2f}x"],
            ["Kelly quarter (recommended)",  f"{k.get('kelly_quarter',0):.1%}"],
            [""],
            ["── TRADE STATS ──",            ""],
            ["Win rate",                     f"{st.get('win_rate',0):.1%}"],
            ["Avg win",                      f"{st.get('avg_win_pct',0):+.3f}%"],
            ["Avg loss",                     f"{st.get('avg_loss_pct',0):+.3f}%"],
            ["Sharpe",                       f"{st.get('sharpe',0):.3f}"],
            ["Max drawdown (actual)",        f"{st.get('max_drawdown',0)*100:.2f}%"],
            [""],
            ["── RECOMMENDATIONS ──",        ""],
        ]
        if not rec:
            model_rows.append(["No recommendations", "Within normal bounds"])
        else:
            for r in rec:
                model_rows.append([
                    f"[{r.get('urgency','').upper()}] {r.get('field','')}",
                    f"{r.get('current','')} → {r.get('suggested','')}",
                ])
                model_rows.append(["  Reason", r.get("reason","")])
    else:
        model_rows = [
            ["Field",  "Value"],
            ["Status", model.get("status","model not yet run")],
            ["Note",   "Need 3+ closed trades to activate"],
        ]
    _write_csv("model.csv", model_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

ERROR_LOG = BASE / "hermes_error.log"


def main() -> None:
    print(f"HERMES Monitor — connecting to {RAILWAY_URL}")
    print("Writing position.csv / stats.csv / trades.csv / model.csv\n")
    while True:
        try:
            state, closed = fetch_all()
            render_terminal(state, closed)
            write_csvs(state, closed)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            now = datetime.now().strftime("%H:%M:%S")
            tb_str = traceback.format_exc()
            # Write to log file (persists across screen clears)
            with open(ERROR_LOG, "a") as f:
                f.write(f"\n[{now}] {exc}\n{tb_str}\n")
            # Print WITHOUT triggering a screen clear
            print(f"\n  [{now}] error — {exc}")
            print(tb_str)
            print(f"  (full traceback saved to hermes_error.log)")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
