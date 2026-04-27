#!/usr/bin/env python3
"""Diagnose WHY specific stocks return None from DhanHQ."""
import sys, os, json, time, requests
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from dhan_oi_fetcher import DhanOIFetcher, DHAN_SCRIP_MAP

fetcher = DhanOIFetcher()

# The stocks that returned NONE
none_stocks = ['MARUTI', 'ICICIBANK', 'WIPRO', 'AXISBANK', 'MCX',
               'IRCTC', 'LTIM', 'SYNGENE']

for sym in none_stocks:
    print(f"\n{'='*50}")
    print(f"DIAGNOSING: {sym}")
    
    # Step 1: Check scrip map
    resolved = fetcher._resolve_symbol(sym)
    if not resolved:
        print(f"  [FAIL] Not in DHAN_SCRIP_MAP and no alias found")
        # Check if similar name exists
        for k in DHAN_SCRIP_MAP:
            if sym[:4] in k:
                print(f"    Similar: {k} -> {DHAN_SCRIP_MAP[k]}")
        continue
    
    scrip_id = resolved['scrip_id']
    segment = resolved['segment']
    print(f"  [OK] Resolved: scrip_id={scrip_id}, segment={segment}")
    
    # Step 2: Fetch expiries
    time.sleep(3)  # throttle
    payload = {
        'UnderlyingScrip': scrip_id,
        'UnderlyingSeg': segment,
    }
    try:
        r = requests.post(fetcher.EXPIRY_URL, headers=fetcher._headers(),
                         json=payload, timeout=15)
        print(f"  Expiry API: status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            expiries = data.get('data', [])
            print(f"  Expiries: {expiries[:3] if expiries else 'EMPTY'}")
        else:
            print(f"  Expiry response: {r.text[:200]}")
    except Exception as e:
        print(f"  Expiry error: {e}")
        continue
    
    if not expiries:
        print(f"  [FAIL] No expiries returned — DhanHQ has no option chain for this stock")
        continue
    
    # Step 3: Fetch chain with first expiry
    time.sleep(3)  # throttle
    chain_payload = {
        'UnderlyingScrip': scrip_id,
        'UnderlyingSeg': segment,
        'Expiry': expiries[0],
    }
    try:
        r = requests.post(fetcher.CHAIN_URL, headers=fetcher._headers(),
                         json=chain_payload, timeout=15)
        print(f"  Chain API: status={r.status_code}")
        if r.status_code == 200:
            raw = r.json()
            status = raw.get('status', 'unknown')
            rdata = raw.get('data', {})
            oc = rdata.get('oc', {})
            ltp = rdata.get('last_price', 0)
            print(f"  Status: {status}, LTP: {ltp}, Strikes: {len(oc)}")
            if oc:
                first_strike = list(oc.keys())[0]
                print(f"  Sample strike {first_strike}: {list(oc[first_strike].keys())}")
        else:
            print(f"  Chain response: {r.text[:300]}")
    except Exception as e:
        print(f"  Chain error: {e}")
