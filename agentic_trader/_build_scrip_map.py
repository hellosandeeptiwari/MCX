#!/usr/bin/env python3
"""
Build correct DHAN_SCRIP_MAP from dhan_instruments.csv.
Approach: For each F&O stock, find its NSE equity scrip_id.
DhanHQ option chain API uses the EQUITY scrip_id, not the F&O contract ID.
"""
import csv
import re
import sys

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dhan_instruments.csv'

# Step 1: Get all unique F&O underlying names from OPTSTK entries
fno_underlyings = set()
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('SEM_EXM_EXCH_ID') == 'NSE' and row.get('SEM_INSTRUMENT_NAME') == 'OPTSTK':
            sym = row.get('SEM_TRADING_SYMBOL', '')
            # Extract underlying from e.g. "RELIANCE-Apr2026-1200-CE"
            und = sym.split('-')[0] if '-' in sym else ''
            if und:
                fno_underlyings.add(und)

print(f"F&O underlyings found: {len(fno_underlyings)}")

# Step 2: Build NSE equity scrip_id map
equity_map = {}  # symbol -> scrip_id
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        exch = row.get('SEM_EXM_EXCH_ID', '')
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        sym = row.get('SEM_TRADING_SYMBOL', '')
        sid = row.get('SEM_SMST_SECURITY_ID', '')
        if exch == 'NSE' and inst in ('EQUITY', 'EQUITIES') and sym and sid:
            equity_map[sym] = int(sid)

print(f"NSE equities: {len(equity_map)}")

# Step 3: Match F&O underlyings to equity scrip_ids
matched = {}
unmatched = []
for und in sorted(fno_underlyings):
    if und in equity_map:
        matched[und] = equity_map[und]
    else:
        unmatched.append(und)

print(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")
if unmatched:
    print(f"Unmatched: {unmatched}")
    # Try fuzzy match
    for um in unmatched:
        # TATAMOTORS -> TMPV etc
        for eq_sym, eq_sid in equity_map.items():
            if um in eq_sym or eq_sym in um:
                print(f"  Possible match: {um} -> {eq_sym} (sid={eq_sid})")

# Step 4: Also get NSE FUTSTK scrip_ids (alternative)
fut_map = {}
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('SEM_EXM_EXCH_ID') == 'NSE' and row.get('SEM_INSTRUMENT_NAME') == 'FUTSTK':
            sym = row.get('SEM_TRADING_SYMBOL', '')
            sid = row.get('SEM_SMST_SECURITY_ID', '')
            und = sym.split('-')[0] if '-' in sym else ''
            if und and sid and und not in fut_map:
                fut_map[und] = int(sid)

# Step 5: Output the corrected map
print(f"\n# === CORRECTED DHAN_SCRIP_MAP (NSE equity IDs) ===")
for sym in sorted(matched.keys()):
    sid = matched[sym]
    print(f"    '{sym}':{' ' * max(1, 15-len(sym))}{{'scrip_id': {sid}, 'segment': 'NSE_FNO'}},")

# Step 6: Show the specific wrong ones
print(f"\n# === VERIFICATION: ABB and ADANIENT ===")
for check in ['ABB', 'ADANIENT', 'RELIANCE', 'HINDUNILVR', 'INFY', 'KEI', 'BSE', 'CGPOWER', 'MARUTI']:
    eq_sid = equity_map.get(check, 'NOT FOUND')
    fut_sid = fut_map.get(check, 'NOT FOUND')
    print(f"  {check}: equity_sid={eq_sid}, fut_sid={fut_sid}")
