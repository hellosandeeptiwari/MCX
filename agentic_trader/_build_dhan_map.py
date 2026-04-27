#!/usr/bin/env python3
"""Build complete DHAN_SCRIP_MAP from DhanHQ instruments master CSV.
Outputs the mapping dict for all NSE F&O underlying stocks."""

import requests
import csv
import io
import json

print("Fetching DhanHQ instruments CSV...")
url = 'https://images.dhan.co/api-data/api-scrip-master.csv'
r = requests.get(url, timeout=60)
print(f"Status: {r.status_code}, Size: {len(r.text)//1024}KB")

reader = csv.DictReader(io.StringIO(r.text))
print(f"Headers: {reader.fieldnames}")

# Collect all rows
all_rows = list(reader)
print(f"Total rows: {len(all_rows)}")

# Find unique segments
segments = set()
for row in all_rows:
    seg = row.get('SEM_SEGMENT', '')
    if seg:
        segments.add(seg)
print(f"Segments: {sorted(segments)}")

# Find NSE F&O option contracts to extract underlying scrip IDs
# Options have instrument type OPTIDX or OPTSTK
fno_underlyings = {}
for row in all_rows:
    seg = row.get('SEM_SEGMENT', '')
    inst_type = row.get('SEM_INSTRUMENT_NAME', '')
    exch = row.get('SEM_EXM_EXCH_ID', '')
    
    # Look for NSE equity stocks that appear in F&O
    # The option chain API needs the UNDERLYING security ID
    if exch == 'NSE' and inst_type in ('EQUITY', 'EQ'):
        sym = row.get('SEM_TRADING_SYMBOL', '') or row.get('SM_SYMBOL_NAME', '')
        scrip_id = row.get('SEM_SMST_SECURITY_ID', '')
        if sym and scrip_id:
            # Store equity scrip ID — might be needed for option chain
            pass
    
    # F&O segment options
    if seg in ('NSE_FNO',) and inst_type in ('OPTSTK', 'OPTIDX', 'FUTSTK', 'FUTIDX'):
        sym = row.get('SM_SYMBOL_NAME', '') or row.get('SEM_CUSTOM_SYMBOL', '')
        scrip_id = row.get('SEM_SMST_SECURITY_ID', '')
        underlying = row.get('SEM_TRADING_SYMBOL', '')
        if not sym:
            continue
        # Extract underlying name from option symbol
        # e.g., "SBIN 24APR 780 CE" → underlying is "SBIN"
        parts = sym.split()
        if parts:
            base = parts[0]
            if base not in fno_underlyings:
                fno_underlyings[base] = {
                    'sample_scrip': scrip_id,
                    'segment': seg,
                    'inst': inst_type,
                    'sample_sym': sym,
                }

print(f"\nF&O underlyings found: {len(fno_underlyings)}")
for sym in sorted(fno_underlyings.keys())[:10]:
    print(f"  {sym}: {fno_underlyings[sym]}")

# Now find the EQUITY scrip IDs for these underlyings
equity_map = {}
for row in all_rows:
    exch = row.get('SEM_EXM_EXCH_ID', '')
    seg = row.get('SEM_SEGMENT', '')
    inst = row.get('SEM_INSTRUMENT_NAME', '')
    sym = row.get('SEM_TRADING_SYMBOL', '')
    scrip_id = row.get('SEM_SMST_SECURITY_ID', '')
    
    if not sym or not scrip_id:
        continue
    
    # NSE equity
    if exch == 'NSE' and seg in ('NSE_EQ',) and inst in ('EQUITY',):
        equity_map[sym] = scrip_id

print(f"\nNSE equity symbols: {len(equity_map)}")
print(f"Sample: {list(equity_map.items())[:5]}")

# Cross-reference: F&O underlyings that have equity scrip IDs
complete_map = {}
missing = []
for sym in sorted(fno_underlyings.keys()):
    if sym in equity_map:
        complete_map[sym] = {'scrip_id': int(equity_map[sym]), 'segment': 'NSE_FNO'}
    else:
        missing.append(sym)

print(f"\nComplete map: {len(complete_map)} stocks")
print(f"Missing equity scrip: {missing}")

# Also check: does DhanHQ option chain need equity scrip_id or something else?
# Let's look at index entries
for row in all_rows:
    sym = row.get('SEM_TRADING_SYMBOL', '')
    if sym in ('NIFTY 50', 'NIFTY BANK', 'NIFTY FIN SERVICE') and row.get('SEM_SEGMENT') == 'IDX_I':
        print(f"\nIndex: {sym} scrip_id={row.get('SEM_SMST_SECURITY_ID')} seg={row.get('SEM_SEGMENT')} inst={row.get('SEM_INSTRUMENT_NAME')}")

# Output the map
print("\n\n# === GENERATED DHAN_SCRIP_MAP ===")
print("DHAN_SCRIP_MAP = {")
# Indices first
print("    # === INDICES ===")
print("    'NIFTY':       {'scrip_id': 13,    'segment': 'IDX_I'},")
print("    'NIFTY 50':    {'scrip_id': 13,    'segment': 'IDX_I'},")
print("    'BANKNIFTY':   {'scrip_id': 25,    'segment': 'IDX_I'},")
print("    'NIFTY BANK':  {'scrip_id': 25,    'segment': 'IDX_I'},")
print("    'FINNIFTY':    {'scrip_id': 27,    'segment': 'IDX_I'},")
print("    'MIDCPNIFTY':  {'scrip_id': 442,   'segment': 'IDX_I'},")
print("    # === F&O STOCKS (auto-generated from DhanHQ instruments) ===")
for sym, info in sorted(complete_map.items()):
    print(f"    {repr(sym):15s}: {{'scrip_id': {info['scrip_id']}, 'segment': 'NSE_FNO'}},")
print("}")
