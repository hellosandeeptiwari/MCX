import json, os, glob

# Check a trade's actual structure
for f in sorted(glob.glob('trade_ledger/trade_ledger_2026-03-17.jsonl')):
    for i, line in enumerate(open(f)):
        t = json.loads(line.strip())
        if 'TEST_GMM' in t.get('rationale', ''):
            print(f"=== ENTRY RECORD (line {i}) ===")
            print(json.dumps(t, indent=2))
            break

# Check active_trades.json
print("\n=== ACTIVE TRADES STRUCTURE ===")
if os.path.exists('active_trades.json'):
    data = json.load(open('active_trades.json'))
    if isinstance(data, list):
        for t in data[:2]:
            if 'TEST_GMM' in str(t.get('rationale', '')) or 'TEST_GMM' in str(t.get('setup_type', '')):
                print(json.dumps(t, indent=2)[:2000])
                break
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:20]}")
        for k, v in list(data.items())[:1]:
            print(json.dumps(v, indent=2)[:2000])

# Check paper positions
print("\n=== PAPER POSITIONS CHECK ===")
for pf in ['paper_positions.json', 'paper_trades.json', 'exit_manager_state.json']:
    if os.path.exists(pf):
        print(f"\n--- {pf} ---")
        data = json.load(open(pf))
        if isinstance(data, list):
            print(f"  {len(data)} entries")
            for t in data[:1]:
                print(json.dumps(t, indent=2)[:1500])
        elif isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:10]}")

# Check if there's a closed trades / exit log
print("\n=== LOOKING FOR EXIT/CLOSED FILES ===")
for pattern in ['closed_*.json*', 'exits_*.json*', '*exit*.json*', '*pnl*.json*', '*result*.json*']:
    matches = glob.glob(pattern)
    if matches:
        print(f"  {pattern}: {matches}")

# Check trade_ledger for fields with pnl != 0
print("\n=== NON-ZERO PNL ENTRIES ===")
count = 0
for f in sorted(glob.glob('trade_ledger/trade_ledger_2026-03-17.jsonl')):
    for line in open(f):
        t = json.loads(line.strip())
        pnl = t.get('pnl', 0)
        if pnl != 0:
            count += 1
            if count <= 3:
                print(json.dumps(t, indent=2)[:1500])
print(f"Total non-zero PNL entries: {count}")

# Check if trades have event_type or status field
print("\n=== DISTINCT EVENT TYPES / STATUSES ===")
from collections import Counter
events = Counter()
statuses = Counter()
for f in sorted(glob.glob('trade_ledger/trade_ledger_2026-03-17.jsonl')):
    for line in open(f):
        t = json.loads(line.strip())
        events[t.get('event_type', t.get('event', 'NONE'))] += 1
        statuses[t.get('status', 'NONE')] += 1
print(f"Events: {events}")
print(f"Statuses: {statuses}")
