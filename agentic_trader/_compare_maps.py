#!/usr/bin/env python3
"""Compare deployed DHAN_SCRIP_MAP vs CSV-derived correct map."""
import csv
import re
import sys

csv_path = 'dhan_instruments.csv'
fetcher_path = 'dhan_oi_fetcher.py'

# Build correct map from CSV
fno_underlyings = set()
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('SEM_EXM_EXCH_ID') == 'NSE' and row.get('SEM_INSTRUMENT_NAME') == 'OPTSTK':
            sym = row.get('SEM_TRADING_SYMBOL', '')
            und = sym.split('-')[0] if '-' in sym else ''
            if und:
                fno_underlyings.add(und)

equity_map = {}
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('SEM_EXM_EXCH_ID') == 'NSE' and row.get('SEM_INSTRUMENT_NAME') in ('EQUITY', 'EQUITIES'):
            sym = row.get('SEM_TRADING_SYMBOL', '')
            sid = row.get('SEM_SMST_SECURITY_ID', '')
            if sym and sid:
                equity_map[sym] = int(sid)

correct_map = {}
for und in fno_underlyings:
    if und in equity_map:
        correct_map[und] = equity_map[und]

# Parse deployed map from fetcher
deployed_map = {}
with open(fetcher_path, encoding='utf-8') as f:
    for line in f:
        if 'NSE_FNO' in line and 'scrip_id' in line:
            m = re.match(r"\s*'([^']+)'\s*:\s*\{'scrip_id':\s*(\d+)", line)
            if m:
                deployed_map[m.group(1)] = int(m.group(2))

# Compare
print(f"CSV F&O stocks: {len(correct_map)}")
print(f"Deployed stocks: {len(deployed_map)}")

mismatches = []
for sym, correct_sid in sorted(correct_map.items()):
    deployed_sid = deployed_map.get(sym)
    if deployed_sid is None:
        print(f"  MISSING in deployed: {sym} (correct sid={correct_sid})")
    elif deployed_sid != correct_sid:
        mismatches.append((sym, deployed_sid, correct_sid))
        print(f"  MISMATCH: {sym} deployed={deployed_sid} correct={correct_sid}")

# Stocks in deployed but not in CSV F&O list
extra = set(deployed_map.keys()) - set(correct_map.keys())
if extra:
    print(f"\nExtra in deployed (not in CSV F&O): {sorted(extra)}")
    # Check if these are aliases or renamed
    for e in sorted(extra):
        eq_sid = equity_map.get(e, 'NOT IN EQUITY')
        dep_sid = deployed_map[e]
        print(f"  {e}: deployed_sid={dep_sid}, equity_sid={eq_sid}")

print(f"\nTotal mismatches: {len(mismatches)}")
print(f"Missing from deployed: {len(correct_map) - len(set(correct_map.keys()) & set(deployed_map.keys()))}")
print(f"Extra in deployed: {len(extra)}")
