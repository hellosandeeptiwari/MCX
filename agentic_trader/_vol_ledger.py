import json

# Check what source/setup the VOLUME_SURGE-triggered trades have in the ledger
vol_entries = []
vol_exits = []
all_sources = set()

for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'ENTRY' and 'VOLUME_SURGE' in t.get('rationale', ''):
        vol_entries.append(t)
    if t.get('event') == 'EXIT':
        all_sources.add(t.get('source', '?') + '/' + t.get('setup', '?'))

print(f"VOLUME_SURGE entries in ledger: {len(vol_entries)}")
for e in vol_entries[:5]:
    print(f"  source={e.get('source','')} | sym={e.get('underlying','')}")
    print(f"    rationale: {e.get('rationale','')[:100]}")

print(f"\nAll exit source/setup combos: {sorted(all_sources)}")

# Check how many entries have WATCHER_VOLUME_SURGE as source
wvs_entries = [e for e in vol_entries if e.get('source') == 'WATCHER_VOLUME_SURGE']
w_entries = [e for e in vol_entries if e.get('source') == 'WATCHER']
oi_entries = [e for e in vol_entries if e.get('source') == 'OI_WATCHER']
print(f"\nSource breakdown:")
print(f"  'WATCHER_VOLUME_SURGE': {len(wvs_entries)}")
print(f"  'WATCHER': {len(w_entries)}")
print(f"  'OI_WATCHER': {len(oi_entries)}")

# Now check exits for these trades (match by order_id)
vol_oids = {e.get('order_id') for e in vol_entries}
vol_exit_pnl = 0
vol_exit_count = 0
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'EXIT' and t.get('order_id') in vol_oids:
        vol_exit_count += 1
        vol_exit_pnl += t.get('pnl', 0)

print(f"\nVOLUME_SURGE exits: {vol_exit_count}")
print(f"VOLUME_SURGE total PNL: {vol_exit_pnl:+.0f}")

# Show all today's source types for ENTRY events
from collections import Counter
entry_sources = Counter()
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'ENTRY':
        entry_sources[t.get('source', 'UNKNOWN')] += 1
print(f"\nAll ENTRY sources today: {dict(entry_sources)}")
