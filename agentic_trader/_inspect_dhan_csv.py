#!/usr/bin/env python3
"""Inspect DhanHQ instruments CSV to understand segment codes and find F&O stocks."""

import requests
import csv
import io

print("Fetching DhanHQ instruments CSV...")
url = 'https://images.dhan.co/api-data/api-scrip-master.csv'
r = requests.get(url, timeout=60)
print(f"Status: {r.status_code}, Size: {len(r.text)//1024}KB")

reader = csv.DictReader(io.StringIO(r.text))
all_rows = list(reader)
print(f"Total rows: {len(all_rows)}")

# Group by (Exchange, Segment, Instrument)
combos = {}
for row in all_rows:
    exch = row.get('SEM_EXM_EXCH_ID', '')
    seg = row.get('SEM_SEGMENT', '')
    inst = row.get('SEM_INSTRUMENT_NAME', '')
    key = (exch, seg, inst)
    if key not in combos:
        combos[key] = {'count': 0, 'sample': row}
    combos[key]['count'] += 1

print(f"\nUnique (Exchange, Segment, Instrument) combos: {len(combos)}")
for key in sorted(combos.keys()):
    exch, seg, inst = key
    cnt = combos[key]['count']
    sample = combos[key]['sample']
    sym = sample.get('SEM_TRADING_SYMBOL', '') or sample.get('SM_SYMBOL_NAME', '')
    print(f"  {exch:6s} {seg:3s} {inst:15s} count={cnt:6d}  sample={sym[:30]}")

# Find SBIN specifically to understand the format
print("\n\n=== SBIN entries ===")
for row in all_rows:
    sym = row.get('SEM_TRADING_SYMBOL', '') or ''
    name = row.get('SM_SYMBOL_NAME', '') or ''
    custom = row.get('SEM_CUSTOM_SYMBOL', '') or ''
    if 'SBIN' in sym and 'SBI' in sym and len(sym) < 8:
        print(f"  exch={row['SEM_EXM_EXCH_ID']} seg={row['SEM_SEGMENT']} inst={row['SEM_INSTRUMENT_NAME']} "
              f"scrip={row['SEM_SMST_SECURITY_ID']} sym={sym} name={name} custom={custom}")

# Find option contracts for SBIN
print("\n=== SBIN option/futures entries (first 5) ===")
count = 0
for row in all_rows:
    name = row.get('SM_SYMBOL_NAME', '') or ''
    sym = row.get('SEM_TRADING_SYMBOL', '') or ''
    if name == 'SBIN' and row.get('SEM_INSTRUMENT_NAME', '') in ('OPTSTK', 'FUTSTK'):
        print(f"  exch={row['SEM_EXM_EXCH_ID']} seg={row['SEM_SEGMENT']} inst={row['SEM_INSTRUMENT_NAME']} "
              f"scrip={row['SEM_SMST_SECURITY_ID']} sym={sym} name={name} "
              f"expiry={row.get('SEM_EXPIRY_DATE','')} strike={row.get('SEM_STRIKE_PRICE','')} "
              f"opt={row.get('SEM_OPTION_TYPE','')}")
        count += 1
        if count >= 5:
            break

# Find all unique F&O underlying names
print("\n=== All F&O underlying names ===")
fno_names = set()
for row in all_rows:
    inst = row.get('SEM_INSTRUMENT_NAME', '')
    if inst in ('OPTSTK', 'FUTSTK', 'OPTIDX', 'FUTIDX'):
        name = row.get('SM_SYMBOL_NAME', '')
        if name:
            fno_names.add(name)

print(f"Total F&O underlyings: {len(fno_names)}")
for name in sorted(fno_names):
    print(f"  {name}")
