#!/usr/bin/env python3
"""Test DhanHQ option chain API coverage for all F&O stocks."""
import sys, os, json, time
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from dhan_oi_fetcher import DhanOIFetcher

fetcher = DhanOIFetcher()

# Test stocks that were returning None earlier + some new ones
test_stocks = [
    'BSE', 'SIEMENS', 'PERSISTENT', 'RVNL', 'COLPAL',  # Previously None
    'KEI', 'HINDUNILVR', 'MARUTI', 'INFY', 'CGPOWER',   # Mixed results
    'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'TCS', 'SBIN',  # Major stocks
    'IRCTC', 'LTIM', 'SYNGENE',  # Missing from map
    'ADANIPORTS', 'BAJFINANCE', 'WIPRO', 'APOLLOHOSP',
    'CONCOR', 'CROMPTON', 'AXISBANK', 'KOTAKBANK',
    'ANGELONE', 'COCHINSHIP', 'POLYCAB', 'MCX',
]

results = {'ok': [], 'none': [], 'error': []}
for sym in test_stocks:
    try:
        data = fetcher.fetch(sym)
        if data and data.get('oi_buildup_strength', 0) > 0:
            str_val = data.get('oi_buildup_strength', 0)
            signal = data.get('oi_buildup', 'UNKNOWN')
            print(f"  OK   {sym:15s} str={str_val:.3f} signal={signal}")
            results['ok'].append(sym)
        elif data:
            print(f"  WEAK {sym:15s} str=0 data_keys={list(data.keys())[:5]}")
            results['ok'].append(sym)
        else:
            print(f"  NONE {sym:15s} — no data returned")
            results['none'].append(sym)
    except Exception as e:
        print(f"  ERR  {sym:15s} — {e}")
        results['error'].append(sym)
    time.sleep(0.3)  # Rate limit

print(f"\n=== SUMMARY ===")
print(f"OK: {len(results['ok'])}")
print(f"NONE: {len(results['none'])} — {results['none']}")
print(f"ERROR: {len(results['error'])} — {results['error']}")
