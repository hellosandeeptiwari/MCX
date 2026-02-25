"""Analyze today's sniper trades and full trade breakdown."""
import json, os
from collections import defaultdict

ledger_file = "trade_ledger/trade_ledger_2026-02-24.jsonl"
lines = open(ledger_file).readlines()

entries = [json.loads(l) for l in lines if json.loads(l).get("event") == "ENTRY"]
exits = [json.loads(l) for l in lines if json.loads(l).get("event") == "EXIT"]

print(f"=== TRADE BREAKDOWN: {len(exits)} closed trades ===\n")

# By source
by_source = defaultdict(list)
for t in exits:
    by_source[t["source"]].append(t)

for src, trades in sorted(by_source.items(), key=lambda x: sum(t["pnl"] for t in x[1])):
    total = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    print(f"\n--- {src} ({len(trades)} trades, PnL: {total:+,.2f}) ---")
    print(f"    Winners: {len(wins)}  |  Losers: {len(losses)}")
    for t in sorted(trades, key=lambda x: x["pnl"]):
        score = t.get("smart_score", "?")
        dr = t.get("dr_score", "?")
        print(f"    {t['underlying']:15s} {t.get('tradingsymbol','?'):25s} "
              f"PnL:{t['pnl']:>+10,.2f}  Entry:{t['entry_price']:>8.2f}  "
              f"Exit:{t['exit_price']:>8.2f}  Score:{score:>6}  "
              f"DR:{dr}  ExitType:{t['exit_type']}  Held:{t.get('hold_minutes','?')}min  "
              f"R:{t.get('r_multiple','?')}")

# Sniper detail
print("\n\n=== SNIPER TRADES DETAIL ===")
sniper_entries = [json.loads(l) for l in lines if "SNIPER" in json.loads(l).get("source", "").upper() or "SNIPER" in json.loads(l).get("source", "")]
for e in sniper_entries:
    print(json.dumps(e, indent=2))

# Overall summary
total_pnl = sum(t["pnl"] for t in exits)
wins = [t for t in exits if t["pnl"] > 0]
losses = [t for t in exits if t["pnl"] < 0]
print(f"\n\n=== OVERALL SUMMARY ===")
print(f"Total Trades: {len(exits)}")
print(f"Total PnL: {total_pnl:+,.2f}")
print(f"Winners: {len(wins)} ({sum(t['pnl'] for t in wins):+,.2f})")
print(f"Losers: {len(losses)} ({sum(t['pnl'] for t in losses):+,.2f})")
if wins:
    print(f"Avg Win: {sum(t['pnl'] for t in wins)/len(wins):+,.2f}")
if losses:
    print(f"Avg Loss: {sum(t['pnl'] for t in losses)/len(losses):+,.2f}")
print(f"Win Rate: {len(wins)/len(exits)*100:.1f}%")
