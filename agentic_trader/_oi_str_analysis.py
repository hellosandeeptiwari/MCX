#!/usr/bin/env python3
"""Analyze OI_Watcher trades for today - strength bifurcation."""
import json, re, sys

LEDGER = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-24.jsonl"

entries = []
exits = []
with open(LEDGER) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("source") != "OI_WATCHER":
            continue
        if d["event"] == "ENTRY":
            entries.append(d)
        elif d["event"] == "EXIT":
            exits.append(d)

entry_map = {e["order_id"]: e for e in entries}
trades = []

for ex in exits:
    oid = ex["order_id"]
    en = entry_map.get(oid, {})
    rat = en.get("rationale", "")
    m = re.search(r"str=([0-9.]+)", rat)
    strength = float(m.group(1)) if m else 0
    trades.append({
        "symbol": ex["underlying"],
        "option": ex["symbol"],
        "strength": strength,
        "entry": ex["entry_price"],
        "exit": ex["exit_price"],
        "pnl": ex["pnl"],
        "pnl_pct": ex["pnl_pct"],
        "exit_type": ex["exit_type"],
        "hold_min": ex["hold_minutes"],
        "entry_time": en.get("ts", "")[:19],
        "oi_signal": en.get("oi_signal", ""),
    })

# Still open entries
for oid, en in entry_map.items():
    if not any(e["order_id"] == oid for e in exits):
        rat = en.get("rationale", "")
        m = re.search(r"str=([0-9.]+)", rat)
        strength = float(m.group(1)) if m else 0
        trades.append({
            "symbol": en["underlying"],
            "option": en["symbol"],
            "strength": strength,
            "entry": en["entry_price"],
            "exit": "OPEN",
            "pnl": "OPEN",
            "pnl_pct": "OPEN",
            "exit_type": "OPEN",
            "hold_min": "OPEN",
            "entry_time": en.get("ts", "")[:19],
            "oi_signal": en.get("oi_signal", ""),
        })

trades.sort(key=lambda x: x["entry_time"])

winners = [t for t in trades if isinstance(t["pnl"], (int, float)) and t["pnl"] > 0]
losers = [t for t in trades if isinstance(t["pnl"], (int, float)) and t["pnl"] <= 0]
still_open = [t for t in trades if t["pnl"] == "OPEN"]


def fmt(t):
    if isinstance(t["pnl"], (int, float)):
        pnl_s = f"{t['pnl']:+.0f}"
        pnl_p = f"{t['pnl_pct']:+.1f}%"
    else:
        pnl_s = t["pnl"]
        pnl_p = t["pnl_pct"]
    return (
        f"  {t['entry_time'][11:]}  {t['symbol']:<22}  str={t['strength']:.2f}  "
        f"{t['oi_signal']:<18}  PnL={pnl_s:>7} ({pnl_p:>7})  "
        f"exit={t['exit_type']}  hold={t['hold_min']}min"
    )


print(f"=== OI_WATCHER TRADES TODAY: {len(trades)} total ===")
print(f"Winners: {len(winners)} | Losers: {len(losers)} | Still Open: {len(still_open)}")
print()

print("--- WINNERS ---")
for t in winners:
    print(fmt(t))
w_strs = [t["strength"] for t in winners]
if w_strs:
    print(f"  Strength range: {min(w_strs):.2f} - {max(w_strs):.2f} (avg {sum(w_strs)/len(w_strs):.2f})")
w_pnl = sum(t["pnl"] for t in winners)
print(f"  Total PnL: +{w_pnl:.0f}")

print()
print("--- LOSERS ---")
for t in losers:
    print(fmt(t))
l_strs = [t["strength"] for t in losers]
if l_strs:
    print(f"  Strength range: {min(l_strs):.2f} - {max(l_strs):.2f} (avg {sum(l_strs)/len(l_strs):.2f})")
l_pnl = sum(t["pnl"] for t in losers)
print(f"  Total PnL: {l_pnl:.0f}")

if still_open:
    print()
    print("--- STILL OPEN ---")
    for t in still_open:
        print(fmt(t))
    o_strs = [t["strength"] for t in still_open]
    if o_strs:
        print(f"  Strength range: {min(o_strs):.2f} - {max(o_strs):.2f} (avg {sum(o_strs)/len(o_strs):.2f})")

print()
all_strs = [t["strength"] for t in trades]
if all_strs:
    print(f"=== OVERALL STRENGTH RANGE: {min(all_strs):.2f} - {max(all_strs):.2f} (avg {sum(all_strs)/len(all_strs):.2f}) ===")
    total_pnl = sum(t["pnl"] for t in trades if isinstance(t["pnl"], (int, float)))
    print(f"=== NET PnL: {total_pnl:+.0f} ===")
