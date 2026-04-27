#!/usr/bin/env python3
"""Build complete DHAN_SCRIP_MAP from DhanHQ instruments CSV.
Cross-references NSE F&O underlyings with NSE equity scrip IDs."""

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

# Step 1: Get all NSE equity scrip IDs
equity_map = {}  # symbol -> scrip_id
for row in all_rows:
    if (row.get('SEM_EXM_EXCH_ID') == 'NSE' and 
        row.get('SEM_SEGMENT') == 'E' and 
        row.get('SEM_INSTRUMENT_NAME') == 'EQUITY'):
        sym = row.get('SEM_TRADING_SYMBOL', '').strip()
        scrip_id = row.get('SEM_SMST_SECURITY_ID', '').strip()
        if sym and scrip_id:
            equity_map[sym] = int(scrip_id)

print(f"NSE equity stocks: {len(equity_map)}")

# Step 2: Find all unique F&O underlyings from SEM_TRADING_SYMBOL
# Format: "CGPOWER-Jun2026-840-PE" → underlying = "CGPOWER"
fno_underlyings = set()
for row in all_rows:
    if (row.get('SEM_EXM_EXCH_ID') == 'NSE' and 
        row.get('SEM_SEGMENT') == 'D' and
        row.get('SEM_INSTRUMENT_NAME') in ('OPTSTK', 'FUTSTK')):
        sym = row.get('SEM_TRADING_SYMBOL', '')
        if '-' in sym:
            underlying = sym.split('-')[0].strip()
            if underlying:
                fno_underlyings.add(underlying)

print(f"NSE F&O underlyings: {len(fno_underlyings)}")

# Step 3: Cross-reference
complete = {}
missing = []
for sym in sorted(fno_underlyings):
    if sym in equity_map:
        complete[sym] = equity_map[sym]
    else:
        missing.append(sym)

print(f"Matched: {len(complete)}")
print(f"Missing equity entry: {missing}")

# Verify known entries match
known = {
    'SBIN': 3045, 'INFY': 1594, 'RELIANCE': 2885, 'HDFCBANK': 1333,
    'TATAMOTORS': 3456, 'MARUTI': 10999, 'HINDUNILVR': 1394,
    'WIPRO': 3787, 'TCS': 11536, 'ICICIBANK': 4963,
}
print("\nVerification:")
for sym, expected in known.items():
    actual = complete.get(sym)
    status = "✓" if actual == expected else f"✗ expected={expected}"
    print(f"  {sym}: scrip_id={actual} {status}")

# Output as Python dict
print(f"\n\n# === COMPLETE DHAN_SCRIP_MAP ({len(complete)} F&O stocks + indices) ===")
print("DHAN_SCRIP_MAP = {")
print("    # === INDICES ===")
print("    'NIFTY':       {'scrip_id': 13,    'segment': 'IDX_I'},")
print("    'NIFTY 50':    {'scrip_id': 13,    'segment': 'IDX_I'},")
print("    'BANKNIFTY':   {'scrip_id': 25,    'segment': 'IDX_I'},")
print("    'NIFTY BANK':  {'scrip_id': 25,    'segment': 'IDX_I'},")
print("    'FINNIFTY':    {'scrip_id': 27,    'segment': 'IDX_I'},")
print("    'MIDCPNIFTY':  {'scrip_id': 442,   'segment': 'IDX_I'},")
print(f"    # === F&O STOCKS ({len(complete)} auto-generated from DhanHQ instruments) ===")
for sym in sorted(complete.keys()):
    sid = complete[sym]
    print(f"    {repr(sym):20s}: {{'scrip_id': {sid}, 'segment': 'NSE_FNO'}},")
print("}")
