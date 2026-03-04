"""Deep diagnostic: Why does DhanHQ return data only up to Feb 26?"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhan_futures_oi import FuturesOIFetcher, DATA_DIR
from datetime import datetime, timedelta

fetcher = FuturesOIFetcher()

# Test 1: Raw API call to see the complete response
print("=== RAW API RESPONSE (SBIN Mar2026 contract) ===")
body = {
    'securityId': '52030',  # SBIN-Mar2026-FUT
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'oi': True,
    'fromDate': '2026-02-20',
    'toDate': '2026-03-02',
}
status, data = fetcher._request('POST', '/charts/historical', json_body=body)
print(f"  Status: {status}")
if data.get('timestamp'):
    import pandas as pd
    dates = pd.to_datetime(data['timestamp'], unit='s')
    print(f"  Timestamps returned: {len(dates)}")
    for d in dates:
        print(f"    {d}")
else:
    print(f"  Response keys: {list(data.keys())}")
    print(f"  Raw response (first 500 chars): {json.dumps(data)[:500]}")

# Test 2: Try the Feb contract (id might be different)
print("\n=== CHECKING FEB CONTRACT ===")
contracts = fetcher.load_instrument_master()
sbin_ctrs = contracts.get('SBIN', [])
for c in sbin_ctrs:
    print(f"  {c['sym']} id={c['id']} expiry={c['expiry']}")

# Find Feb contract
feb_ctr = None
for c in sbin_ctrs:
    if 'Feb' in c['sym']:
        feb_ctr = c
        break

if feb_ctr:
    print(f"\n=== RAW API RESPONSE (SBIN Feb contract: {feb_ctr['sym']}) ===")
    body2 = {
        'securityId': feb_ctr['id'],
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'FUTSTK',
        'oi': True,
        'fromDate': '2026-02-20',
        'toDate': '2026-02-28',
    }
    status2, data2 = fetcher._request('POST', '/charts/historical', json_body=body2)
    print(f"  Status: {status2}")
    if data2.get('timestamp'):
        import pandas as pd
        dates2 = pd.to_datetime(data2['timestamp'], unit='s')
        print(f"  Timestamps: {len(dates2)}")
        for d in dates2:
            print(f"    {d}")
    else:
        print(f"  Raw: {json.dumps(data2)[:500]}")
else:
    print("  No Feb contract found (already expired and removed from master?)")

# Test 3: Try equity data for SBIN to see if Feb 27 exists in spot market
print("\n=== SPOT EQUITY DATA (SBIN) — checking Feb 27 ===")
body3 = {
    'securityId': '3045',  # SBIN equity
    'exchangeSegment': 'NSE_EQ',
    'instrument': 'EQUITY',
    'fromDate': '2026-02-25',
    'toDate': '2026-03-02',
}
status3, data3 = fetcher._request('POST', '/charts/historical', json_body=body3)
print(f"  Status: {status3}")
if data3.get('timestamp'):
    import pandas as pd
    dates3 = pd.to_datetime(data3['timestamp'], unit='s')
    print(f"  Timestamps: {len(dates3)}")
    for d in dates3:
        print(f"    {d}")
else:
    print(f"  Raw: {json.dumps(data3)[:500]}")

# Test 4: Try RELIANCE too (different stock)
print("\n=== RAW FUTURES DATA (RELIANCE) ===")
rel_ctr = fetcher.get_nearest_contract('RELIANCE')
if rel_ctr:
    body4 = {
        'securityId': rel_ctr['id'],
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'FUTSTK',
        'oi': True,
        'fromDate': '2026-02-25',
        'toDate': '2026-03-02',
    }
    status4, data4 = fetcher._request('POST', '/charts/historical', json_body=body4)
    print(f"  Status: {status4}")
    if data4.get('timestamp'):
        import pandas as pd
        dates4 = pd.to_datetime(data4['timestamp'], unit='s')
        print(f"  Timestamps: {len(dates4)}")
        for d in dates4:
            print(f"    {d}")
    else:
        print(f"  Raw: {json.dumps(data4)[:500]}")

print("\n=== CONCLUSION ===")
print(f"Today: {datetime.now()} ({datetime.now().strftime('%A')})")
print(f"Feb 27, 2026: {datetime(2026,2,27).strftime('%A')}")
print(f"Feb 28, 2026: {datetime(2026,2,28).strftime('%A')}")
print("If Feb 27 (Friday) data is missing, it might be a market holiday (Maha Shivaratri?)")
print("If spot data also stops at Feb 26, the market was closed Feb 27+")
