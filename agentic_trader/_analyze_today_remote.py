#!/usr/bin/env python3
"""Quick analysis of today's trades - run on EC2"""
import json, os, sys
from datetime import datetime

BASE = "/home/ubuntu/titan/agentic_trader"
TODAY = "2026-03-10"

# Load trades
trades = json.load(open(os.path.join(BASE, "active_trades.json")))
today = [t for t in trades if t.get("timestamp", "").startswith(TODAY)]

print(f"=== MARCH 10 TRADE ANALYSIS ({len(today)} trades) ===\n")

if not today:
    print("NO TRADES TODAY")
    sys.exit(0)

# Separate by source
watcher = [t for t in today if "watcher" in t.get("setup_type", "").lower() or "breakout" in t.get("setup_type", "").lower()]
scan = [t for t in today if t not in watcher]

print(f"Watcher trades: {len(watcher)} | Scan trades: {len(scan)}\n")

def show_trades(trades_list, label):
    if not trades_list:
        return
    print(f"--- {label} ---")
    total_pnl = 0
    for t in sorted(trades_list, key=lambda x: x.get("timestamp", "")):
        ul = t.get("underlying", "").replace("NSE:", "")
        st = t.get("status", "?")
        d = t.get("direction", "?")
        su = t.get("setup_type", "?")
        ep = t.get("avg_price", 0)
        xp = t.get("exit_price", 0)
        q = t.get("quantity", 0)
        pnl = t.get("pnl", 0)
        ts = t.get("timestamp", "")[-8:]
        xts = (t.get("exit_timestamp", "") or "")[-8:]
        xr = t.get("exit_reason", "")
        opt = t.get("option_symbol", "")
        score = t.get("watcher_score", t.get("score", "?"))
        trigger = t.get("trigger_type", "?")

        print(f"  {ts} {ul:14s} {d:5s} {su:22s} score={str(score):>4s} {st:6s}")
        print(f"         opt={opt}  trigger={trigger}")
        print(f"         entry={ep:.2f}  exit={xp:.2f}  qty={q}  pnl={pnl:+,.0f}  {xr} {xts}")

        if st == "CLOSED":
            total_pnl += pnl

    closed = [t for t in trades_list if t.get("status") == "CLOSED"]
    opn = [t for t in trades_list if t.get("status") == "OPEN"]
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    losses = len(closed) - wins

    print(f"\n  Summary: {wins}W/{losses}L closed | {len(opn)} open | Realized: {total_pnl:+,.0f}")
    if opn:
        print(f"  Open positions:")
        for t in opn:
            ul = t.get("underlying", "").replace("NSE:", "")
            ep = t.get("avg_price", 0)
            q = t.get("quantity", 0)
            d = t.get("direction", "?")
            print(f"    {ul} {d} entry={ep:.2f} qty={q}")
    print()

show_trades(watcher, "WATCHER TRADES")
show_trades(scan, "SCAN TRADES")

# Overall
all_closed = [t for t in today if t.get("status") == "CLOSED"]
total_realized = sum(t.get("pnl", 0) for t in all_closed)
print(f"=== TOTAL REALIZED P&L: {total_realized:+,.0f} ===")
