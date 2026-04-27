#!/usr/bin/env python3
"""Analyze today's OI_WATCHER performance"""
import json
from collections import Counter

f = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-02.jsonl"
lines = open(f).readlines()
events = [json.loads(l) for l in lines if l.strip()]

entries = [e for e in events if e.get("event") == "ENTRY"]
exits = [e for e in events if e.get("event") == "EXIT"]
scans = [e for e in events if e.get("event") == "SCAN"]

print(f"TOTAL: {len(entries)} entries, {len(exits)} exits, {len(scans)} scans")
print()

# Outcome distribution
outcomes = Counter(s.get("outcome", "?") for s in scans)
print("=== SCAN OUTCOMES ===")
for k, v in outcomes.most_common():
    print(f"  {k}: {v}")
print()

# Entries
total_pnl = 0
print("=== ALL ENTRIES ===")
for e in entries:
    sym = e.get("symbol", "")
    src = e.get("source", "")[:40]
    direction = e.get("direction", "")
    score = e.get("smart_score", 0)
    t = e.get("ts", "")[11:16]
    ep = e.get("entry_price", 0)
    lots = e.get("lots", 0)
    prem = e.get("total_premium", 0)
    print(f"  {t} {direction:5s} {sym:40s} score={score:>4} src={src} lots={lots} prem={prem}")
print()

# Exits with P&L
print("=== ALL EXITS ===")
for x in exits:
    sym = x.get("symbol", "")
    pnl = x.get("pnl", 0)
    total_pnl += pnl
    pnl_pct = x.get("pnl_pct", 0)
    exit_type = x.get("exit_type", "")
    t = x.get("ts", "")[11:16]
    hold = x.get("hold_minutes", 0)
    reason = x.get("exit_reason", "")[:50]
    print(f"  {t} {sym:40s} pnl={pnl:+9.0f} ({pnl_pct:+.1f}%p) exit={exit_type} hold={hold:.0f}m | {reason}")

print(f"\nTOTAL P/L: {total_pnl:+,.0f}")
print()

# Watcher fired detail
fired = [s for s in scans if s.get("outcome") == "WATCHER_FIRED"]
print(f"=== WATCHER FIRED ({len(fired)}) ===")
for ff in fired:
    print(f"  {ff.get('ts', '')[11:16]} {ff.get('symbol', '')} score={ff.get('score', 0)} {ff.get('reason', '')[:90]}")
print()

# Gate blocks
blocks = [s for s in scans if "BLOCKED" in s.get("outcome", "") or s.get("outcome", "").startswith("WATCHER_LOW") or s.get("outcome", "").startswith("WATCHER_GRIND") or s.get("outcome", "").startswith("WATCHER_WEAK")]
block_outcomes = Counter(b.get("outcome", "?") for b in blocks)
print(f"=== GATE BLOCKS ({len(blocks)} total) ===")
for k, v in block_outcomes.most_common():
    print(f"  {k}: {v}")

# Which stocks actually moved big?
print()
print("=== SCORED_PASS stocks (candidates that passed scoring) ===")
scored = [s for s in scans if s.get("outcome") == "SCORED_PASS"]
seen = set()
for s in scored:
    sym = s.get("symbol", "")
    if sym not in seen:
        seen.add(sym)
        chg = s.get("extra", {}).get("change_pct", 0)
        sc = s.get("score", 0)
        print(f"  {sym:25s} chg={chg:+.2f}% score={sc}")

# Conviction blocks
conv_blocks = [s for s in scans if s.get("outcome") == "WATCHER_LOW_CONVICTION"]
if conv_blocks:
    print(f"\n=== LOW CONVICTION BLOCKS ({len(conv_blocks)}) ===")
    for c in conv_blocks:
        print(f"  {c.get('ts', '')[11:16]} {c.get('symbol', '')} score={c.get('score', 0)} {c.get('reason', '')[:90]}")

# OI anchor blocks
oi_blocks = [s for s in scans if "OI_ANCHOR" in s.get("outcome", "")]
if oi_blocks:
    print(f"\n=== OI ANCHOR BLOCKS ({len(oi_blocks)}) ===")
    for o in oi_blocks:
        print(f"  {o.get('ts', '')[11:16]} {o.get('symbol', '')} score={o.get('score', 0)} {o.get('reason', '')[:90]}")

# Trade failures  
failures = [s for s in scans if s.get("outcome") == "WATCHER_TRADE_FAILED"]
if failures:
    print(f"\n=== TRADE FAILURES ({len(failures)}) ===")
    for tf in failures:
        print(f"  {tf.get('ts', '')[11:16]} {tf.get('symbol', '')} {tf.get('reason', '')[:80]}")
