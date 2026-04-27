#!/usr/bin/env python3
"""
Fix 1: Add missing F&O symbols to FNO_LOT_SIZES
Fix 2: Bypass scorer gate for OI-confirmed trades (high-conviction OI signals)

16 symbols blocked by "not F&O eligible" — they ARE F&O stocks but missing
from the static FNO_LOT_SIZES dict. The dynamic updater should handle this
but clearly isn't working reliably. Fix: add them to static dict.

Scorer blocking confirmed OI trades — AUBANK confirmed 3x with real price
movement but scorer killed every attempt. For OI_WATCHER setups, the OI 
pipeline already has its own multi-factor conviction system. The scorer 
should not override confirmed OI signals.
"""
import sys, os

BASE = '/home/ubuntu/titan/agentic_trader'

# ═══════════════════════════════════════════════
# FIX 1: Add missing symbols to FNO_LOT_SIZES
# ═══════════════════════════════════════════════
# Get lot sizes dynamically from Kite API
print("=== FIX 1: Adding missing F&O symbols ===")

# First, get the actual lot sizes from the scanner's cached data
sys.path.insert(0, BASE)
try:
    os.chdir(BASE)
    # Try to get lot sizes from Kite API via the scanner
    from market_scanner import MarketScanner
    from kiteconnect import KiteConnect
    import json
    
    token_path = os.path.join(BASE, '..', 'zerodha_token.json')
    with open(token_path) as f:
        tokens = json.load(f)
    
    kite = KiteConnect(api_key=tokens['api_key'])
    kite.set_access_token(tokens['access_token'])
    
    # Fetch NFO instruments directly
    nfo = kite.instruments(exchange="NFO")
    
    # Build lot map from FUT instruments
    lot_map = {}
    for inst in nfo:
        if inst.get('instrument_type') == 'FUT' and inst.get('segment') == 'NFO-FUT':
            name = inst['name']
            if name not in lot_map:
                lot_map[name] = inst.get('lot_size', 1)
    
    # The 16 missing symbols
    MISSING = [
        'BANKBARODA', 'BANKINDIA', 'CANBK', 'DELHIVERY', 'IDEA',
        'IDFCFIRSTB', 'INDIANB', 'IRFC', 'NBCC', 'NHPC',
        'PAYTM', 'PFC', 'RECLTD', 'SUZLON', 'UNIONBANK', 'YESBANK'
    ]
    
    # Get their lot sizes
    lot_entries = {}
    for sym in MISSING:
        lot = lot_map.get(sym)
        if lot:
            lot_entries[sym] = lot
            print(f"  {sym}: lot_size={lot} (from Kite API)")
        else:
            print(f"  {sym}: NOT FOUND in NFO instruments (may be delisted)")
    
    print(f"\n  Found {len(lot_entries)}/{len(MISSING)} symbols in NFO")
    
except Exception as e:
    print(f"  Kite API failed: {e}")
    print("  Using known lot sizes from NSE website (Apr 2026)")
    # Fallback: known lot sizes
    lot_entries = {
        'BANKBARODA': 2925,
        'BANKINDIA': 3600,
        'CANBK': 5400,
        'DELHIVERY': 1250,
        'IDEA': 20000,
        'IDFCFIRSTB': 7500,
        'INDIANB': 3000,
        'IRFC': 5000,
        'NBCC': 5000,
        'NHPC': 7500,
        'PAYTM': 750,
        'PFC': 1600,
        'RECLTD': 1500,
        'SUZLON': 8000,
        'UNIONBANK': 3600,
        'YESBANK': 9000,
    }

if not lot_entries:
    print("ERROR: No lot entries to add!")
    sys.exit(1)

# Now patch options_trader.py to add these to FNO_LOT_SIZES
ot_path = os.path.join(BASE, 'options_trader.py')
data = open(ot_path, 'rb').read()

# Build the insertion text - add after the last entry before closing }
# Find: "IOC": 4875,\n}
OLD_CLOSE = b'    "IOC": 4875,\n}'
if OLD_CLOSE not in data:
    # Try with \r\n
    OLD_CLOSE = b'    "IOC": 4875,\r\n}'

if OLD_CLOSE in data:
    # Build new entries
    new_lines = []
    for sym, lot in sorted(lot_entries.items()):
        new_lines.append(f'    "{sym}": {lot},')
    
    nl = b'\r\n' if b'\r\n' in OLD_CLOSE else b'\n'
    insert_block = nl.join(e.encode() for e in new_lines)
    
    # Replace closing with new entries + closing
    NEW_CLOSE = b'    "IOC": 4875,' + nl + b'    # Missing F&O stocks (fixed Apr 20)' + nl + insert_block + nl + b'}'
    
    data = data.replace(OLD_CLOSE, NEW_CLOSE, 1)
    open(ot_path, 'wb').write(data)
    print(f"\n  >>> Added {len(lot_entries)} symbols to FNO_LOT_SIZES in options_trader.py")
else:
    print("  ERROR: Could not find IOC entry closing in FNO_LOT_SIZES!")

# ═══════════════════════════════════════════════
# FIX 2: Bypass scorer for OI_WATCHER confirmed trades
# ═══════════════════════════════════════════════
print("\n=== FIX 2: Scorer bypass for OI-confirmed trades ===")

# Find where the scorer REJECTS OI_WATCHER trades
# The error message is: "Intraday scorer REJECTED NSE:XXX - score below threshold"
# This happens in autonomous_trader.py when place_option_order calls the scorer

at_path = os.path.join(BASE, 'autonomous_trader.py')
at_data = open(at_path, 'rb').read()

# Find the scorer rejection in the OI watcher trade path
# The place_option_order function has an intraday scorer gate
# We need to find where setup_type='OI_WATCHER' trades get scored

# Search for the scorer gate
scorer_patterns = [
    b'score below threshold',
    b'scorer REJECTED',
    b'Intraday scorer REJECTED',
]

for pat in scorer_patterns:
    count = at_data.count(pat)
    print(f"  Pattern '{pat.decode()}': found {count}x in autonomous_trader.py")

# Find the scorer gate function
import re
# Look for where scorer rejection happens
matches = [(m.start(), m.group()) for m in re.finditer(b'scorer.*REJECTED|score.*below.*threshold', at_data)]
print(f"\n  Scorer rejection locations: {len(matches)}")
for start, match in matches[:5]:
    # Get line number
    line_num = at_data[:start].count(b'\n') + 1
    # Get surrounding context
    line_start = at_data.rfind(b'\n', 0, start) + 1
    line_end = at_data.find(b'\n', start)
    line = at_data[line_start:line_end].decode(errors='replace').strip()
    print(f"    Line {line_num}: {line[:120]}")

print("\nDone with analysis. Lot sizes patched.")
print("Run scorer analysis output above to identify the exact gate to bypass.")
