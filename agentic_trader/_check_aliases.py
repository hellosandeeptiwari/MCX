#!/usr/bin/env python3
"""Check specific symbol lookups in DhanHQ CSV"""
import requests, csv, io
r = requests.get('https://images.dhan.co/api-data/api-scrip-master.csv', timeout=60)
reader = csv.DictReader(io.StringIO(r.text))
targets = {'BAJAJ-AUTO', 'TATAMOTORS', 'TMPV', 'BAJFINANCE', 'NAM-INDIA', 'BAJAJ', 'NAM'}
for row in reader:
    sym = row.get('SEM_TRADING_SYMBOL', '')
    if sym in targets and row.get('SEM_EXM_EXCH_ID') == 'NSE':
        print(f"  {sym:15s} seg={row['SEM_SEGMENT']} inst={row['SEM_INSTRUMENT_NAME']} scrip={row['SEM_SMST_SECURITY_ID']}")
