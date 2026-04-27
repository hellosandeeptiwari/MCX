#!/usr/bin/env python3
"""Quick re-test of previously-NONE stocks after fix."""
import sys, time
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from dhan_oi_fetcher import DhanOIFetcher
f = DhanOIFetcher()
for sym in ['IRCTC', 'LTIM', 'SYNGENE', 'MARUTI', 'ICICIBANK', 'WIPRO', 'AXISBANK', 'MCX']:
    d = f.fetch(sym)
    if d and d.get('oi_buildup_strength', 0) > 0:
        print(f"  OK   {sym:15s} str={d['oi_buildup_strength']:.3f}")
    elif d:
        print(f"  WEAK {sym:15s} str=0 strikes={len(d.get('strikes',[]))}")
    else:
        print(f"  NONE {sym:15s}")
    time.sleep(0.5)
