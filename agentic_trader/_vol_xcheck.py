import json

# Check ALL DLF entries in the ledger and track any watcher order IDs
print("=== DLF ENTRIES IN LEDGER ===")
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if 'DLF' in t.get('underlying', '') and t.get('event') in ('ENTRY', 'EXIT'):
        print(f"  event={t['event']} oid={t.get('order_id','')} source={t.get('source','')} "
              f"pnl={t.get('pnl',0):+.0f} ts={t.get('ts','')[:19]}")
        if t.get('event') == 'ENTRY':
            print(f"    rat: {t.get('rationale','')[:120]}")

# Check if watcher_debug.log has SLOW_GRIND "TRADE PLACED" entries that DO match ledger
print("\n=== CROSS-CHECKING: watcher TRADE PLACED vs ledger ===")
import re

# Read watcher log placed trades
placed = []
with open('watcher_debug.log') as f:
    for line in f:
        if 'TRADE PLACED' in line:
            m = re.search(r'TRADE PLACED: (\S+) .* trigger=(\S+) .* order=(\S+)', line)
            if m:
                placed.append({
                    'sym': m.group(1),
                    'trigger': m.group(2),
                    'oid': m.group(3).split()[0],
                    'time': line[:15],
                })

print(f"Total TRADE PLACED in watcher log: {len(placed)}")

# Read ledger order IDs
ledger_oids = set()
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'ENTRY':
        ledger_oids.add(t.get('order_id', ''))

# Check overlap
in_ledger = [p for p in placed if p['oid'] in ledger_oids]
not_in_ledger = [p for p in placed if p['oid'] not in ledger_oids]

print(f"\nIn ledger: {len(in_ledger)}")
for p in in_ledger[:10]:
    print(f"  {p['time']} {p['sym']:<14} trigger={p['trigger']:<35} oid={p['oid']}")

print(f"\nNOT in ledger: {len(not_in_ledger)}")
# Group by trigger type
from collections import Counter
missing_by_trigger = Counter()
for p in not_in_ledger:
    # Extract just the trigger type (before the parenthesis)
    trig = p['trigger'].split('(')[0]
    missing_by_trigger[trig] += 1
print(f"Missing by trigger type: {dict(missing_by_trigger)}")
for p in not_in_ledger[:15]:
    print(f"  {p['time']} {p['sym']:<14} trigger={p['trigger']:<35} oid={p['oid']}")

# Check if those watcher placed entries are from BEFORE today's first ledger entry
print("\n=== TIMELINE CHECK ===")
first_ledger_ts = None
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'ENTRY':
        first_ledger_ts = t.get('ts', '')
        break
print(f"First ledger ENTRY timestamp: {first_ledger_ts}")

# Check watcher log for restart markers
print("\n=== WATCHER LOG RESTART MARKERS ===")
with open('watcher_debug.log') as f:
    for i, line in enumerate(f):
        if 'START' in line.upper() or 'INIT' in line.upper() or 'READY' in line.upper():
            if i < 20 or 'WATCHER' in line.upper():
                print(f"  line {i}: {line.strip()[:150]}")
        if i > 50000:
            break

# Check line count and if there are date indicators
print("\n=== LOG FILE STATS ===")
with open('watcher_debug.log') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First: {lines[0].strip()[:100]}")
    print(f"Last: {lines[-1].strip()[:100]}")
    
    # Find if there are any "gap" patterns (time going backwards = restart)
    prev_time = ""
    restarts = []
    for i, line in enumerate(lines):
        m = re.match(r'\[(\d{2}:\d{2}:\d{2})', line)
        if m:
            t = m.group(1)
            if prev_time and t < prev_time:
                restarts.append((i, prev_time, t))
            prev_time = t
    print(f"\nTime resets found (probable restarts): {len(restarts)}")
    for idx, pt, ct in restarts[:5]:
        print(f"  Line {idx}: time went from {pt} → {ct}")
