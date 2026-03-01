"""Test multiple SBIN futures contract IDs against Dhan intraday API."""
import os, json, requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv('.env')

access_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
client_id = os.environ.get('DHAN_CLIENT_ID', '')

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'access-token': access_token,
    'client-id': client_id,
}

from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d 09:15:00')
to_date = datetime.now().strftime('%Y-%m-%d 15:30:00')

# Try all known SBIN futures IDs
test_cases = [
    ('828576', 'NSE_FNO', 'FUTSTK', 'SBIN Mar2026 - from CSV (NSE 15:30 exp)'),
    ('52030', 'NSE_FNO', 'FUTSTK', 'SBIN Mar2026 - from cache (BSE-style)'),
    ('52030', 'BSE_FNO', 'FUTSTK', 'SBIN Mar2026 - BSE_FNO'),
    ('828576', 'NSE_FNO', 'FUTSTK', 'SBIN Mar2026 - NSE_FNO retry'),
    ('1168554', 'NSE_FNO', 'FUTSTK', 'SBIN Feb2026 - from CSV expired'),
]

for sec_id, segment, instrument, label in test_cases:
    body = {
        'securityId': sec_id,
        'exchangeSegment': segment,
        'instrument': instrument,
        'interval': '5',
        'oi': True,
        'fromDate': from_date,
        'toDate': to_date,
    }
    
    r = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body, timeout=15)
    data = r.json()
    
    ts = data.get('timestamp', [])
    oi = data.get('open_interest', [])
    
    if ts:
        has_oi = len(oi) > 0 and any(x > 0 for x in oi)
        first_dt = datetime.fromtimestamp(ts[0])
        last_dt = datetime.fromtimestamp(ts[-1])
        print(f"OK  [{label}] ID={sec_id} seg={segment}: {len(ts)} candles, OI={has_oi}, {first_dt} to {last_dt}")
        if has_oi:
            print(f"      OI range: {min(oi):,} - {max(oi):,}")
            # Show a few
            for i in [0, len(ts)//2, -1]:
                idx = i if i >= 0 else len(ts) + i
                dt = datetime.fromtimestamp(ts[idx])
                print(f"      {dt.strftime('%Y-%m-%d %H:%M')} OI={oi[idx]:>12,}")
    else:
        err = data.get('errorMessage', data.get('errorCode', str(data)[:100]))
        print(f"FAIL [{label}] ID={sec_id} seg={segment}: {r.status_code} - {err}")

# Also test NIFTY index futures
print("\n=== NIFTY Index Futures ===")
# Search for NIFTY futures in cache
cache = json.load(open('ml_models/data/futures_oi/futures_contracts.json'))
nifty = cache.get('NIFTY', [])
print(f"NIFTY contracts: {len(nifty)}")
for c in nifty[:3]:
    print(f"  {c}")

if nifty:
    nf = nifty[0]  # nearest
    body = {
        'securityId': nf['id'],
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'FUTIDX',
        'interval': '5',
        'oi': True,
        'fromDate': from_date,
        'toDate': to_date,
    }
    r = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body, timeout=15)
    data = r.json()
    ts = data.get('timestamp', [])
    oi = data.get('open_interest', [])
    if ts:
        has_oi = len(oi) > 0 and any(x > 0 for x in oi)
        print(f"NIFTY: {len(ts)} candles, OI={has_oi}")
        if has_oi:
            print(f"OI range: {min(oi):,} - {max(oi):,}")
    else:
        print(f"NIFTY: {r.status_code} - {json.dumps(data)[:200]}")
