"""Test: How much historical data can DhanHQ actually provide for futures?"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

# Load DhanHQ config
config_path = os.path.join(os.path.dirname(__file__), 'dhan_config.json')
with open(config_path) as f:
    cfg = json.load(f)

TOKEN = cfg['access_token']
CLIENT_ID = cfg['client_id']
BASE = 'https://api.dhan.co/v2'

def dhan_post(endpoint, body):
    url = f"{BASE}{endpoint}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    req.add_header('access-token', TOKEN)
    req.add_header('Accept', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:
        if hasattr(e, 'read'):
            return getattr(e, 'code', 500), json.loads(e.read().decode())
        return 500, {'error': str(e)}

# SBIN NSE futures contract ID = 59466 (Feb 2026)
# Let's test with different lookback periods
print("=" * 60)
print("TEST 1: SBIN Feb futures — how far back can we go?")
print("=" * 60)

for months in [6, 12, 18, 24, 36]:
    from_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')
    body = {
        'securityId': '59466',
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'FUTSTK',
        'oi': True,
        'fromDate': from_date,
        'toDate': datetime.now().strftime('%Y-%m-%d'),
    }
    status, data = dhan_post('/charts/historical', body)
    if status == 200 and data.get('timestamp'):
        rows = len(data['timestamp'])
        first = datetime.fromtimestamp(data['timestamp'][0]).date()
        last = datetime.fromtimestamp(data['timestamp'][-1]).date()
        print(f"  months_back={months:2d}: {rows:4d} rows, {first} to {last}")
    else:
        print(f"  months_back={months:2d}: FAILED ({status})")
    time.sleep(0.5)

# Now let's look for OLDER contracts for SBIN
# Need to download instrument master
print("\n" + "=" * 60)
print("TEST 2: Find ALL SBIN futures contracts (past + current)")
print("=" * 60)

# Load cached instrument master
cache_file = 'ml_models/data/futures_oi/scrip_master_cache.csv'
contracts_file = 'ml_models/data/futures_oi/futures_contracts.json'

if os.path.exists(contracts_file):
    with open(contracts_file) as f:
        contracts_cache = json.load(f)
    sbin_contracts = contracts_cache.get('SBIN', [])
    print(f"  Found {len(sbin_contracts)} SBIN contracts in cache:")
    for c in sbin_contracts:
        print(f"    ID={c['id']}, Expiry={c['expiry']}, Name={c.get('name','')}")

# Now let's try fetching data for different months using those contracts
print("\n" + "=" * 60)
print("TEST 3: Fetch OI data for each contract separately")
print("=" * 60)

if sbin_contracts:
    for c in sbin_contracts:
        body = {
            'securityId': str(c['id']),
            'exchangeSegment': 'NSE_FNO',
            'instrument': 'FUTSTK',
            'oi': True,
            'fromDate': '2024-01-01',
            'toDate': datetime.now().strftime('%Y-%m-%d'),
        }
        status, data = dhan_post('/charts/historical', body)
        if status == 200 and data.get('timestamp') and len(data['timestamp']) > 0:
            rows = len(data['timestamp'])
            first = datetime.fromtimestamp(data['timestamp'][0]).date()
            last = datetime.fromtimestamp(data['timestamp'][-1]).date()
            avg_oi = sum(data.get('open_interest', [0])) / max(rows, 1)
            print(f"  Contract {c['id']} (exp={c['expiry']}): {rows} rows, {first} to {last}, avg_OI={avg_oi:,.0f}")
        else:
            print(f"  Contract {c['id']} (exp={c['expiry']}): NO DATA ({status})")
        time.sleep(0.5)

# Test 4: Try fetching with equity security ID but different months
# SBIN equity ID = 11536 (BSE), but we need NSE equity historical to compare
print("\n" + "=" * 60)
print("TEST 4: SBIN equity spot — how far back?")
print("=" * 60)

for months in [6, 12, 18, 24]:
    from_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')
    body = {
        'securityId': '11536',
        'exchangeSegment': 'NSE_EQ',
        'instrument': 'EQUITY',
        'fromDate': from_date,
        'toDate': datetime.now().strftime('%Y-%m-%d'),
    }
    status, data = dhan_post('/charts/historical', body)
    if status == 200 and data.get('timestamp') and len(data['timestamp']) > 0:
        rows = len(data['timestamp'])
        first = datetime.fromtimestamp(data['timestamp'][0]).date()
        last = datetime.fromtimestamp(data['timestamp'][-1]).date()
        print(f"  months_back={months:2d}: {rows} rows, {first} to {last}")
    else:
        print(f"  months_back={months:2d}: FAILED ({status})")
    time.sleep(0.5)

# Test 5: How far back can Kite 5-min data go?
print("\n" + "=" * 60)
print("TEST 5: Kite 5-min data limit check")
print("=" * 60)
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from kite_auth import get_kite
    kite = get_kite()
    
    # Try fetching SBIN 5-min for different lookbacks
    token = None
    for i in kite.instruments("NSE"):
        if i['tradingsymbol'] == 'SBIN':
            token = i['instrument_token']
            break
    
    if token:
        for days in [90, 180, 365, 730]:
            start = datetime.now() - timedelta(days=days)
            end_d = start + timedelta(days=55)  # just first chunk
            try:
                candles = kite.historical_data(
                    instrument_token=token,
                    from_date=start.strftime('%Y-%m-%d'),
                    to_date=end_d.strftime('%Y-%m-%d'),
                    interval='5minute'
                )
                if candles:
                    first_date = candles[0]['date']
                    last_date = candles[-1]['date']
                    print(f"  days_back={days:4d}: {len(candles)} candles from {first_date} to {last_date}")
                else:
                    print(f"  days_back={days:4d}: 0 candles returned")
            except Exception as e:
                print(f"  days_back={days:4d}: ERROR - {e}")
            time.sleep(0.5)
except Exception as e:
    print(f"  Kite check failed: {e}")
