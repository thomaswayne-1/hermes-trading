"""
watch_engine.py — live monitor for the Hermes Directional Coefficient Engine.

Run:  python3 watch_engine.py

Shows the full signal chain every 10 seconds:
  regime → sub-signals → C/K → entry decision → open positions → performance
"""

import json
import os
import ssl
import time
import urllib.request
from datetime import datetime, timezone

RAILWAY_URL = os.getenv("RAILWAY_URL", "https://hermes-trading-production-169f.up.railway.app")
API_SECRET  = os.getenv("API_SECRET",  "hermes2024")
STARTING    = 100_000.0
INTERVAL    = 10

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode    = ssl.CERT_NONE

W = 58   # display width


def _fetch(path: str):
    req = urllib.request.Request(
        f"{RAILWAY_URL}{path}",
        headers={"X-API-Key": API_SECRET},
    )
    with urllib.request.urlopen(req, context=_SSL, timeout=10) as r:
        return json.loads(r.read())


def bar(value: float, width: int = 22) -> str:
    """ASCII bar chart for a value in [-1, +1]."""
    mid    = width // 2
    filled = int(abs(value) * mid + 0.5)
    filled = min(filled, mid)
    if value >= 0:
        left  = " " * mid
        right = "█" * filled + " " * (mid - filled)
    else:
        left  = " " * (mid - filled) + "█" * filled
        right = " " * mid
    return f"[{left}│{right}]"


def clear():
    print("\033[2J\033[H", end="")


def _pnl_color(v: float) -> str:
    if v > 0:  return f"+{v:.2f}%"
    if v < 0:  return f"{v:.2f}%"
    return "0.00%"


def render(state: dict, trades: list) -> None:
    hb    = state.get("heartbeat", {})
    eng   = hb.get("engine", {})
    cb    = hb.get("circuit_breaker", {})
    strat = state.get("strategy", {})
    sigs  = eng.get("sub_signals") or {}
    wts   = eng.get("weights_used") or {}

    # ── Market data ───────────────────────────────────────────────────────────
    ts      = hb.get("ts", "")[:19].replace("T", " ")
    price   = float(hb.get("price",       0))
    rsi     = float(hb.get("rsi",         50))
    macd    = float(hb.get("macd_hist",   0))
    bb      = float(hb.get("bb_pct",      0.5))
    atr     = float(hb.get("atr",         0))
    ob_imb  = float(hb.get("ob_imbalance",0))

    # ── Engine state ──────────────────────────────────────────────────────────
    C       = float(eng.get("C",           0))
    K       = float(eng.get("K",           0))
    ready   = bool(eng.get("ready",        False))
    samps   = int(eng.get("samples",       0))
    reg     = str(eng.get("regime",        "warmup"))
    cert    = float(eng.get("regime_cert", 0))
    agree   = float(eng.get("agreement",   0))
    volp    = float(eng.get("vol_penalty", 1))
    volf    = float(eng.get("vol_forecast",0))
    raw_sc  = float(eng.get("raw_score",   C))  # pre-tanh score if available

    # ── Strategy config ───────────────────────────────────────────────────────
    engine_live = bool(strat.get("coefficient_engine_enabled", False))
    coef_cfg    = strat.get("coefficient", {})
    tau_enter   = float(coef_cfg.get("tau_enter", 0.12))
    tau_exit    = float(coef_cfg.get("tau_exit",  0.05))
    max_pos     = int(strat.get("max_open_positions", 7))
    cb_state    = cb.get("state", "normal").upper()
    version     = strat.get("version", "?")

    # ── Open positions ────────────────────────────────────────────────────────
    open_trades = hb.get("open_trades", [])

    # ── Performance from closed trades ────────────────────────────────────────
    # Uses identical logic to excel_tracker.py:
    #   - cumulative balance (each trade's dollar compounds on the running balance)
    #   - pnl_pct_net (after fees+funding); fallback to pnl_pct_levered for old trades
    #   - win = pnl_pct_net > 0 (after fees, not raw)
    closed  = [t for t in trades if t.get("closed")]
    n       = len(closed)
    balance = STARTING
    wins    = 0
    for t in closed:
        pnl  = t.get("pnl_pct_net") if t.get("pnl_pct_net") is not None \
               else t.get("pnl_pct_levered", 0)
        size = float(t.get("position_size_r", 0.15))
        balance += pnl * size * balance   # cumulative — same as tracker
        if pnl > 0:
            wins += 1
    wr      = wins / n if n else 0.0
    net_pnl = balance - STARTING

    # ── Unrealised PnL on open positions ─────────────────────────────────────
    total_unreal_usd = 0.0
    for t in open_trades:
        ep  = float(t.get("entry_price", price))
        lev = float(t.get("leverage",    1.5))
        sz  = float(t.get("size",        0.15))
        raw = (price - ep) / ep if t["direction"] == "long" else (ep - price) / ep
        total_unreal_usd += raw * lev * sz * balance

    # ── Render ────────────────────────────────────────────────────────────────
    clear()
    print("━" * W)
    mode_tag = "LIVE" if engine_live else "SHADOW MODE"
    print(f"  HERMES  —  Engine Monitor  [{mode_tag}]  v{version}")
    print("━" * W)
    print(f"  {ts}   BTC ${price:,.2f}   RSI {rsi:.1f}")
    print()

    # Engine / regime / circuit
    if not ready:
        status = f"⏳ WARMING UP  ({samps}/200 samples)"
    elif engine_live:
        status = "✅ LIVE & READY"
    else:
        status = "👁  SHADOW — watching only"
    print(f"  Engine:   {status}")
    print(f"  Regime:   {reg.upper():12s}  certainty={cert:.2f}")
    print(f"  Circuit:  {cb_state}")
    print()

    # Directional signal
    print(f"  ── Directional Signal {'─' * (W - 23)}")
    direction = "LONG  ▲" if C > 0.05 else ("SHORT ▼" if C < -0.05 else "FLAT  —")
    print(f"  C = {C:+.4f}  {bar(C, 24)}  {direction}")
    print(f"  K = {K:.4f}   conviction")
    print(f"  Agreement={agree:.2f}  VolPenalty={volp:.2f}  VolForecast={volf:.5f}")
    print()

    # Entry decision line — no more "if engine were live"
    if ready:
        above = abs(C) > tau_enter
        side  = "LONG" if C > 0 else "SHORT"
        if engine_live:
            if above:
                print(f"  ► ENTERING {side}  (|C|={abs(C):.4f} > τ={tau_enter})")
            else:
                gap = tau_enter - abs(C)
                print(f"  ► NO ENTRY  — C needs +{gap:.4f} more to trigger")
        else:
            if above:
                print(f"  ► SHADOW: signal says {side}  (|C|={abs(C):.4f} > τ={tau_enter})")
            else:
                print(f"  ► SHADOW: no signal  (|C|={abs(C):.4f} ≤ τ={tau_enter})")
    else:
        zscore_window = int(coef_cfg.get("zscore_window", 200))
        print(f"  ► Warming up — need {zscore_window - samps} more ticks (~{((zscore_window - samps) * INTERVAL) // 60} min)")
    print()

    # Sub-signals
    print(f"  ── Sub-signals  (regime: {reg.upper()}) {'─' * max(0, W - 28 - len(reg))}")
    blocks = [
        ("mom", "Momentum  "),
        ("rev", "Mean-Rev  "),
        ("mic", "Microstr  "),
        ("sen", "Sentiment "),
    ]
    for key, label in blocks:
        v = float(sigs.get(key, 0))
        w = float(wts.get(key, 0))
        lean = "▲ bullish" if v > 0.15 else ("▼ bearish" if v < -0.15 else "  neutral")
        print(f"  {label} {v:+.3f}  {bar(v, 20)}  w={w:.2f}  {lean}")
    print()

    # Market indicators
    print(f"  ── Market Indicators {'─' * (W - 22)}")
    ob_label = "buy pressure" if ob_imb > 0.1 else ("sell pressure" if ob_imb < -0.1 else "balanced")
    bb_label = "upper (overbought)" if bb > 0.8 else ("lower (oversold)" if bb < 0.2 else f"{bb:.2f}")
    print(f"  RSI {rsi:5.1f}   MACD hist {macd:+.4f}   ATR ${atr:.2f}")
    print(f"  BB%  {bb_label}   OB imb {ob_imb:+.3f} ({ob_label})")
    print()

    # Open positions
    print(f"  ── Open Positions ({len(open_trades)}/{max_pos}) {'─' * max(0, W - 22)}")
    if open_trades:
        for t in open_trades:
            ep    = float(t.get("entry_price", price))
            lev   = float(t.get("leverage",    1.5))
            sz    = float(t.get("size",        0.15))
            ec    = t.get("entry_C")
            ek    = t.get("entry_K")
            raw   = (price - ep) / ep if t["direction"] == "long" else (ep - price) / ep
            unr_pct = raw * lev * 100
            unr_usd = raw * lev * sz * balance
            c_str = f"C={ec:+.3f} " if ec is not None else ""
            k_str = f"K={ek:.3f}"   if ek is not None else ""
            sym   = "+" if unr_pct >= 0 else ""
            print(f"  {t['direction'].upper():5s} @ ${ep:>10,.2f}  "
                  f"lev={lev:.1f}x  sz={sz*100:.0f}%  "
                  f"{c_str}{k_str}  "
                  f"{sym}{unr_pct:.2f}%  (${sym}{unr_usd:,.0f})")
        sym = "+" if total_unreal_usd >= 0 else ""
        print(f"  {'':46s}total {sym}${total_unreal_usd:,.0f}")
    else:
        print("  No open positions")
    print()

    # Performance
    print(f"  ── Performance {'─' * (W - 17)}")
    if n > 0:
        sym = "+" if net_pnl >= 0 else ""
        print(f"  Closed: {n}   Win rate: {wr*100:.0f}%   "
              f"Balance: ${balance:,.2f}  ({sym}${net_pnl:,.2f})")
        if total_unreal_usd != 0:
            total_eq = balance + total_unreal_usd
            print(f"  Unrealised: ${total_unreal_usd:+,.0f}   "
                  f"Total equity: ${total_eq:,.2f}")
    else:
        print(f"  No closed trades yet   Starting balance: ${STARTING:,.0f}")
    print()
    print(f"  Refreshing every {INTERVAL}s   Ctrl+C to stop")
    print("━" * W)


def main() -> None:
    print("Connecting to Railway...")
    while True:
        try:
            state  = _fetch("/state")
            trades = _fetch("/trades")
            if not isinstance(trades, list):
                trades = trades.get("trades", [])
            render(state, trades)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"  [{now}] error — {exc}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
