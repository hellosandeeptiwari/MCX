#!/usr/bin/env python3
"""Quick OI_WATCHER trade review"""
import json, os
path = os.path.expanduser("~/titan/agentic_trader/trade_ledger")
if not os.path.isdir(path):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_ledger")

# Load all JSONL files
data = []
for fname in sorted(os.listdir(path)):
    if not fname.endswith(".jsonl"):
        continue
    with open(os.path.join(path, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
all_trades = [t for t in data if t.get("exit_time")]
oi = [t for t in data if t.get("setup_type") == "OI_WATCHER"]
oi_closed = [t for t in oi if t.get("exit_time")]
oi_open = [t for t in oi if not t.get("exit_time")]

print(f"=== OI_WATCHER REVIEW ===")
print(f"Total trades in ledger: {len(data)}")
print(f"OI_WATCHER trades: {len(oi)} (open={len(oi_open)}, closed={len(oi_closed)})")

# Friday trades
fri = [t for t in oi if t.get("entry_time","").startswith("2026-03-20")]
print(f"\nFriday (Mar 20) OI_WATCHER: {len(fri)}")
for t in fri:
    sym = t.get("symbol","?")
    pnl = t.get("pnl", 0) or 0
    st = t.get("status","?")
    et = (t.get("entry_time","") or "")[:19]
    xt = (t.get("exit_time","") or "")[:19]
    xr = t.get("exit_reason","")
    d = t.get("direction","?")
    print(f"  {sym}: {d} pnl=₹{pnl:+,.0f} {st} entry={et} exit={xt} reason={xr}")

# Weekly summary
from collections import defaultdict
by_date = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0, "losses": 0})
for t in oi_closed:
    date = (t.get("entry_time","") or "")[:10]
    pnl = t.get("pnl", 0) or 0
    by_date[date]["count"] += 1
    by_date[date]["pnl"] += pnl
    if pnl > 0:
        by_date[date]["wins"] += 1
    else:
        by_date[date]["losses"] += 1

print(f"\n=== OI_WATCHER DAILY SUMMARY (closed trades) ===")
total_pnl = 0
total_wins = 0
total_losses = 0
for date in sorted(by_date.keys())[-10:]:
    d = by_date[date]
    wr = d["wins"]/(d["wins"]+d["losses"])*100 if (d["wins"]+d["losses"]) > 0 else 0
    print(f"  {date}: {d['count']} trades, P&L=₹{d['pnl']:+,.0f}, W/L={d['wins']}/{d['losses']} ({wr:.0f}%)")
    total_pnl += d["pnl"]
    total_wins += d["wins"]
    total_losses += d["losses"]

total_wr = total_wins/(total_wins+total_losses)*100 if (total_wins+total_losses) > 0 else 0
print(f"\n  TOTAL: {total_wins+total_losses} closed, P&L=₹{total_pnl:+,.0f}, WR={total_wr:.0f}%")

# All setup types comparison
print(f"\n=== ALL SETUP TYPES COMPARISON (recent 10 days) ===")
by_type = defaultdict(lambda: {"count": 0, "pnl": 0, "wins": 0})
for t in all_trades:
    st = t.get("setup_type", "UNKNOWN")
    pnl = t.get("pnl", 0) or 0
    by_type[st]["count"] += 1
    by_type[st]["pnl"] += pnl
    if pnl > 0:
        by_type[st]["wins"] += 1
for st in sorted(by_type.keys(), key=lambda k: by_type[k]["pnl"], reverse=True):
    d = by_type[st]
    wr = d["wins"]/d["count"]*100 if d["count"] > 0 else 0
    print(f"  {st:20s}: {d['count']:3d} trades, P&L=₹{d['pnl']:+10,.0f}, WR={wr:.0f}%")
