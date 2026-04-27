import json
from collections import Counter

with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-02.jsonl') as f:
    lines = [json.loads(l.strip()) for l in f if l.strip()]

events = Counter(t.get('event') for t in lines)
print(f"Event types: {dict(events)}")
print(f"Total records: {len(lines)}")

# Check trade_id field
entries = [t for t in lines if t.get('event') == 'ENTRY']
exits = [t for t in lines if t.get('event') == 'EXIT']

print(f"\nENTRIES: {len(entries)}")
if entries:
    print(f"  First entry trade_id: {entries[0].get('trade_id','')[:50]}")
    print(f"  First entry symbol: {entries[0].get('symbol','')}")
    print(f"  First entry underlying: {entries[0].get('underlying','')}")

print(f"\nEXITS: {len(exits)}")
if exits:
    print(f"  First exit trade_id: {exits[0].get('trade_id','')[:50]}")
    print(f"  First exit symbol: {exits[0].get('symbol','')}")

# Check if trade_ids match
entry_tids = set(t.get('trade_id','') for t in entries)
exit_tids = set(t.get('trade_id','') for t in exits)
matched = entry_tids & exit_tids
print(f"\nEntry trade_ids: {len(entry_tids)}")
print(f"Exit trade_ids: {len(exit_tids)}")
print(f"Matched: {len(matched)}")

if len(matched) < min(len(entry_tids), len(exit_tids)):
    print("\nSample unmatched exit trade_ids:")
    unmatched_exits = exit_tids - entry_tids
    for tid in list(unmatched_exits)[:5]:
        ex = [t for t in exits if t.get('trade_id') == tid][0]
        print(f"  {tid[:40]} -> {ex.get('symbol','')} pnl={ex.get('pnl',0)}")
    
    print("\nSample unmatched entry trade_ids:")
    unmatched_entries = entry_tids - exit_tids
    for tid in list(unmatched_entries)[:5]:
        en = [t for t in entries if t.get('trade_id') == tid][0]
        print(f"  {tid[:40]} -> {en.get('symbol','')} {en.get('underlying','')}")

# Check other sources of entries - maybe active_trades has them?
import os, glob
# Check if there are other ledger files
other_files = glob.glob('/home/ubuntu/titan/agentic_trader/trade_ledger/*2026-04-02*')
print(f"\nLedger files for today: {other_files}")

# Also check active_trades
try:
    with open('/home/ubuntu/titan/agentic_trader/active_trades.json') as f:
        active = json.load(f)
    print(f"\nActive trades: {len(active)}")
    if active:
        first = list(active.values())[0] if isinstance(active, dict) else active[0]
        print(f"  Sample keys: {list(first.keys())[:15]}")
except Exception as e:
    print(f"Error reading active_trades: {e}")
