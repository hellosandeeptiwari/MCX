#!/usr/bin/env python3
"""Quick check: what expiry dates does Zerodha show for stock options?"""
import json
from kiteconnect import KiteConnect

with open('zerodha_token.json') as f:
    t = json.load(f)

kite = KiteConnect(api_key=t['api_key'])
kite.set_access_token(t['access_token'])
instruments = kite.instruments('NFO')

for stock in ['ETERNAL', 'RELIANCE', 'PNBHOUSING', 'HDFCBANK', 'TCS']:
    expiries = sorted(set(
        str(i['expiry']) for i in instruments
        if i['name'] == stock and i['instrument_type'] in ['CE', 'PE']
    ))
    print(f"{stock}: {expiries}")

# Also check a sample trading symbol format
for i in instruments:
    if i['name'] == 'ETERNAL' and i['instrument_type'] == 'PE' and '2026-03' in str(i['expiry']):
        print(f"  Sample: {i['tradingsymbol']} expiry={i['expiry']}")
        break
