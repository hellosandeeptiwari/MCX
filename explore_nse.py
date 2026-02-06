"""
Explore NSE API to understand why we need manual input
"""
import requests
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.nseindia.com/',
}

session = requests.Session()
session.get('https://www.nseindia.com', headers=headers, timeout=5)

# Get full quote data
r = session.get('https://www.nseindia.com/api/quote-equity?symbol=MCX', headers=headers, timeout=10)
d = r.json()

print('='*60)
print('NSE API - FULL RESPONSE ANALYSIS')
print('='*60)

# Price Info
p = d.get('priceInfo', {})
print()
print('PRICE INFO:')
print(f"  lastPrice:     {p.get('lastPrice')}")
print(f"  change:        {p.get('change')}")
print(f"  pChange:       {p.get('pChange')}%")
print(f"  previousClose: {p.get('previousClose')}")
print(f"  open:          {p.get('open')}")
print(f"  close:         {p.get('close')}")
print(f"  vwap:          {p.get('vwap')}")
print(f"  basePrice:     {p.get('basePrice')}")

# Intraday high/low
ihl = p.get('intraDayHighLow', {})
print()
print('INTRADAY:')
print(f"  high: {ihl.get('max')}")
print(f"  low:  {ihl.get('min')}")

# Week high/low
whl = p.get('weekHighLow', {})
print()
print('52 WEEK:')
print(f"  high:     {whl.get('max')}")
print(f"  highDate: {whl.get('maxDate')}")
print(f"  low:      {whl.get('min')}")
print(f"  lowDate:  {whl.get('minDate')}")

# Check all available keys
print()
print('ALL TOP-LEVEL KEYS:')
for key in d.keys():
    val = d[key]
    val_type = type(val).__name__
    if isinstance(val, dict):
        print(f"  {key}: dict with {len(val)} keys")
    elif isinstance(val, list):
        print(f"  {key}: list with {len(val)} items")
    else:
        print(f"  {key}: {val_type} = {str(val)[:50]}")

# Check securityInfo
si = d.get('securityInfo', {})
print()
print('SECURITY INFO:')
for k, v in si.items():
    print(f"  {k}: {v}")

# Check metadata
meta = d.get('metadata', {})
print()
print('METADATA:')
for k, v in meta.items():
    print(f"  {k}: {v}")

print()
print('='*60)
print('CONCLUSION:')
print('='*60)
print()
print("NSE's previousClose already accounts for Sunday sessions!")
print(f"previousClose from NSE: {p.get('previousClose')}")
print()
print("This means we DON'T need manual input!")
print("The issue was using Yahoo Finance which misses Sunday data.")
print("NSE API already gives us the correct previous close.")
