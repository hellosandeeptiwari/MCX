import json, os

ledger_path = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-05.jsonl'
entries = []
exits = []
scans = []

with open(ledger_path) as f:
    for line in f:
        rec = json.loads(line.strip())
        src = str(rec.get('source','') or rec.get('trade_source','') or '').upper()
        evt = rec.get('event','')
        if 'WATCHER' in src:
            if evt == 'ENTRY':
                entries.append(rec)
            elif evt == 'EXIT':
                exits.append(rec)
            elif evt == 'SCAN':
                scans.append(rec)

print("=" * 80)
print(f"WATCHER TRADES - FULL RAW DATA - March 5, 2026")
print(f"Entries: {len(entries)} | Exits: {len(exits)} | Blocked Scans: {len(scans)}")
print("=" * 80)

# Show ALL keys from first entry
if entries:
    print(f"\nAll fields in ENTRY record: {sorted(entries[0].keys())}")

print("\n" + "=" * 80)
print("PART 1: ALL ENTRIES (every field)")
print("=" * 80)
for i, e in enumerate(entries, 1):
    print(f"\n--- Entry #{i} ---")
    for k, v in sorted(e.items()):
        print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("PART 2: ALL EXITS (every field)")
print("=" * 80)
for i, x in enumerate(exits, 1):
    print(f"\n--- Exit #{i} ---")
    for k, v in sorted(x.items()):
        print(f"  {k}: {v}")

print("\n" + "=" * 80)
print("PART 3: ALL BLOCKED SCANS (every field)")
print("=" * 80)
for i, s in enumerate(scans, 1):
    print(f"\n--- Blocked Scan #{i} ---")
    for k, v in sorted(s.items()):
        print(f"  {k}: {v}")

# Now match entries to exits
print("\n" + "=" * 80)
print("PART 4: ENTRY-EXIT MATCHING SUMMARY")
print("=" * 80)
exit_map = {}
for x in exits:
    sym = x.get('symbol','')
    if sym not in exit_map:
        exit_map[sym] = []
    exit_map[sym].append(x)

total_pnl = 0
for e in entries:
    sym = e.get('symbol','')
    ts = e.get('timestamp','')[:19]
    strike = e.get('strike','')
    opt = e.get('option_type','')
    entry_prem = e.get('entry_premium','') or e.get('premium','')
    score = e.get('score','')
    
    matched_exit = None
    if sym in exit_map and exit_map[sym]:
        matched_exit = exit_map[sym].pop(0)
    
    if matched_exit:
        exit_ts = matched_exit.get('timestamp','')[:19]
        exit_reason = matched_exit.get('exit_reason','')
        pnl = matched_exit.get('pnl', matched_exit.get('realized_pnl', 'N/A'))
        exit_prem = matched_exit.get('exit_premium','')
        hold_time = ''
        try:
            from datetime import datetime
            t1 = datetime.fromisoformat(ts)
            t2 = datetime.fromisoformat(exit_ts)
            hold_time = str(t2 - t1)
        except:
            pass
        
        pnl_val = float(pnl) if pnl != 'N/A' else 0
        total_pnl += pnl_val
        print(f"\n{sym} {strike}{opt}")
        print(f"  Entry: {ts} @ Rs {entry_prem} | Score: {score}")
        print(f"  Exit:  {exit_ts} @ Rs {exit_prem} | Reason: {exit_reason}")
        print(f"  PnL: Rs {pnl} | Hold: {hold_time}")
    else:
        print(f"\n{sym} {strike}{opt}")
        print(f"  Entry: {ts} @ Rs {entry_prem} | Score: {score}")
        print(f"  Exit: *** NO EXIT FOUND *** (orphan or still open)")

print(f"\n{'='*80}")
print(f"WATCHER TOTAL PnL: Rs {total_pnl:,.0f}")
print(f"{'='*80}")

# Check for exits without entries
unmatched_exits = {sym: exs for sym, exs in exit_map.items() if exs}
if unmatched_exits:
    print(f"\nWARNING: Exits without matching entries:")
    for sym, exs in unmatched_exits.items():
        for x in exs:
            print(f"  {sym} - exit at {x.get('timestamp','')[:19]} reason={x.get('exit_reason','')} pnl={x.get('pnl','')}")
