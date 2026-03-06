#!/usr/bin/env python3
"""Deep BLUESTARCO duplicate investigation."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from trade_ledger import get_trade_ledger
from datetime import datetime

tl = get_trade_ledger()
today = datetime.now().strftime('%Y-%m-%d')
entries = tl.get_entries(today)
exits = tl.get_exits(today)

print("=== ALL BLUESTARCO ENTRY records ===")
for i, en in enumerate(entries):
    sym = en.get('symbol', '')
    if 'BLUESTARCO' in sym:
        print(f"\nENTRY #{i}:")
        for k, v in sorted(en.items()):
            print(f"  {k:25s} = {v}")

print("\n\n=== ALL BLUESTARCO EXIT records ===")
for i, ex in enumerate(exits):
    sym = ex.get('symbol', '')
    if 'BLUESTARCO' in sym:
        print(f"\nEXIT #{i}:")
        for k, v in sorted(ex.items()):
            print(f"  {k:25s} = {v}")

# Also check the raw ledger dir
print("\n\n=== Raw ledger files for BLUESTARCO ===")
ledger_dir = os.path.join(os.path.dirname(__file__), 'trade_ledger')
if os.path.isdir(ledger_dir):
    for f in sorted(os.listdir(ledger_dir)):
        if 'BLUESTARCO' in f:
            fpath = os.path.join(ledger_dir, f)
            try:
                data = json.load(open(fpath))
                print(f"\n{f}:")
                print(json.dumps(data, indent=2, default=str)[:1000])
            except:
                print(f"\n{f}: (not JSON)")
