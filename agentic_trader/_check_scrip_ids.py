#!/usr/bin/env python3
"""Check DHAN_SCRIP_MAP for duplicate/suspicious scrip IDs."""
import re

with open('/home/ubuntu/titan/agentic_trader/dhan_oi_fetcher.py') as f:
    src = f.read()

ids = {}
for line in src.split('\n'):
    if 'NSE_FNO' in line and 'scrip_id' in line:
        m = re.match(r"\s*'([^']+)'\s*:\s*\{'scrip_id':\s*(\d+)", line)
        if m:
            name, sid = m.group(1), int(m.group(2))
            ids.setdefault(sid, []).append(name)

print("=== DUPLICATE SCRIP IDs ===")
dups = 0
for sid, names in sorted(ids.items()):
    if len(names) > 1:
        print(f"  scrip_id={sid}: {names}")
        dups += 1

print(f"\n=== SUSPECT INDEX-COLLISION IDs ===")
# NIFTY=13, BANKNIFTY=25, FINNIFTY=27, MIDCPNIFTY=442
for sid in [13, 25, 27, 442]:
    if sid in ids:
        print(f"  scrip_id={sid} (INDEX ID!): {ids[sid]}")

total = sum(len(v) for v in ids.values())
print(f"\nTotal NSE_FNO entries: {total}, Unique IDs: {len(ids)}, Duplicates: {dups}")

# Also verify against dhan_instruments.csv if available
import os
csv_path = '/home/ubuntu/titan/agentic_trader/dhan_instruments.csv'
if os.path.exists(csv_path):
    import csv
    eq_map = {}
    with open(csv_path) as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            seg = row.get('SEM_EXM_EXCH_ID', '')
            inst = row.get('SEM_INSTRUMENT_NAME', '')
            sym = row.get('SEM_TRADING_SYMBOL', '')
            sid = row.get('SEM_SMST_SECURITY_ID', '')
            # NSE equity entries
            if seg == 'NSE' and inst in ('EQUITY', 'EQUITIES'):
                eq_map[sym] = int(sid)
    
    print(f"\n=== CROSS-CHECK vs dhan_instruments.csv ({len(eq_map)} NSE equities) ===")
    mismatches = 0
    for line in src.split('\n'):
        if 'NSE_FNO' in line and 'scrip_id' in line:
            m = re.match(r"\s*'([^']+)'\s*:\s*\{'scrip_id':\s*(\d+)", line)
            if m:
                name, sid = m.group(1), int(m.group(2))
                if name in eq_map and eq_map[name] != sid:
                    print(f"  MISMATCH: {name} map={sid} csv={eq_map[name]}")
                    mismatches += 1
                elif name not in eq_map:
                    # Try without hyphen etc
                    pass
    print(f"  Mismatches: {mismatches}")
else:
    print(f"\n  dhan_instruments.csv not found — skipping cross-check")
