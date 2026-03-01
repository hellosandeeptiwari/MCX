"""Test Dhan intraday 5-min chart API for futures OI."""
import csv, os, json, requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv('.env')

access_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
client_id = os.environ.get('DHAN_CLIENT_ID', '')
print(f'Dhan: client_id={bool(client_id)}, token len={len(access_token)}')

# Find SBIN current month futures contract from local instrument master
print('\n=== Finding SBIN futures from local instrument master ===')
contracts = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sym = row.get('SM_SYMBOL_NAME', '').strip()
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        exch = row.get('SEM_EXM_EXCH_ID', '')
        if sym == 'SBIN' and 'FUTSTK' in inst and 'NSE' in exch:
            expiry = row.get('SEM_EXPIRY_DATE', '')
            sec_id = row.get('SEM_SMST_SECURITY_ID', '')
            symbol = row.get('SEM_TRADING_SYMBOL', '')
            contracts.append({'id': sec_id, 'expiry': expiry, 'symbol': symbol})

contracts.sort(key=lambda x: x['expiry'])
now_str = datetime.now().strftime('%Y-%m-%d')
active = [c for c in contracts if c['expiry'] >= now_str]
print(f'Found {len(contracts)} total futures, {len(active)} active')
for c in active[:3]:
    print(f"  {c['symbol']} (ID={c['id']}, expiry={c['expiry']})")

if not active:
    print("No active contracts found!")
    exit(1)

contract = active[0]
print(f"\n=== Testing Dhan intraday API for {contract['symbol']} ===")

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'access-token': access_token,
    'client-id': client_id,
}

from_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d 09:15:00')
to_date = datetime.now().strftime('%Y-%m-%d 15:30:00')

body = {
    'securityId': contract['id'],
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'interval': '5',
    'oi': True,
    'fromDate': from_date,
    'toDate': to_date,
}
print(f'Request: {json.dumps(body)}')

r = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body, timeout=15)
print(f'Status: {r.status_code}')
data = r.json()
print(f'Keys: {list(data.keys())}')

if 'timestamp' in data and data['timestamp']:
    ts = data['timestamp']
    oi = data.get('open_interest', [])
    vol = data.get('volume', [])
    o = data.get('open', [])
    c = data.get('close', [])
    print(f'Candles: {len(ts)}')
    has_oi = len(oi) > 0 and any(x > 0 for x in oi)
    print(f'Has OI: {has_oi}')
    if ts:
        print(f'First: ts={datetime.fromtimestamp(ts[0])}, open={o[0] if o else None}, close={c[0] if c else None}, vol={vol[0] if vol else None}, oi={oi[0] if oi else None}')
        print(f'Last:  ts={datetime.fromtimestamp(ts[-1])}, open={o[-1] if o else None}, close={c[-1] if c else None}, vol={vol[-1] if vol else None}, oi={oi[-1] if oi else None}')
        if oi:
            print(f'OI range: min={min(oi):,}, max={max(oi):,}, mean={sum(oi)/len(oi):,.0f}')
        
        # Show 5-min candles for today
        today = datetime.now().date()
        for i, t in enumerate(ts):
            dt = datetime.fromtimestamp(t)
            if dt.date() == today or (i < 5 or i > len(ts) - 5):
                oi_val = oi[i] if i < len(oi) else 0
                print(f'  {dt.strftime("%Y-%m-%d %H:%M")} | O={o[i]:.1f} C={c[i]:.1f} V={vol[i]:>8,} OI={oi_val:>12,}')
else:
    print(f'Response: {json.dumps(data)[:500]}')

# Also test OPTIONS chain - can Dhan give 5-min OI for options?
print("\n=== Testing Dhan OPTIONS intraday API ===")
# Find SBIN CE option contract
opt_contracts = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sym = row.get('SM_SYMBOL_NAME', '').strip()
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        exch = row.get('SEM_EXM_EXCH_ID', '')
        otype = row.get('SEM_OPTION_TYPE', '').strip()
        if sym == 'SBIN' and 'OPTSTK' in inst and 'NSE' in exch and otype == 'CE':
            expiry = row.get('SEM_EXPIRY_DATE', '')
            if expiry >= now_str:
                sec_id = row.get('SEM_SMST_SECURITY_ID', '')
                strike = row.get('SEM_STRIKE_PRICE', '')
                symbol_name = row.get('SEM_TRADING_SYMBOL', '')
                opt_contracts.append({'id': sec_id, 'expiry': expiry, 'strike': float(strike), 'symbol': symbol_name})

opt_contracts.sort(key=lambda x: (x['expiry'], x['strike']))
# Find ATM - SBIN is ~750-800 range, pick nearest
atm_contracts = [c for c in opt_contracts if 700 <= c['strike'] <= 850][:5]
print(f'Found {len(opt_contracts)} CE options, {len(atm_contracts)} near ATM')
for c in atm_contracts[:3]:
    print(f"  {c['symbol']} strike={c['strike']} (ID={c['id']}, expiry={c['expiry']})")

if atm_contracts:
    oc = atm_contracts[0]
    body2 = {
        'securityId': oc['id'],
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'OPTSTK',
        'interval': '5',
        'oi': True,
        'fromDate': from_date,
        'toDate': to_date,
    }
    print(f'Request: {json.dumps(body2)}')
    r2 = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body2, timeout=15)
    print(f'Status: {r2.status_code}')
    data2 = r2.json()
    print(f'Keys: {list(data2.keys())}')
    if 'timestamp' in data2 and data2['timestamp']:
        ts2 = data2['timestamp']
        oi2 = data2.get('open_interest', [])
        print(f'Options 5-min candles: {len(ts2)}')
        print(f'Has OI: {len(oi2) > 0 and any(x > 0 for x in oi2)}')
        if oi2:
            print(f'OI range: min={min(oi2):,}, max={max(oi2):,}')
            # Show a few
            for i in [0, 1, len(ts2)//2, -2, -1]:
                idx = i if i >= 0 else len(ts2) + i
                if idx < len(ts2):
                    dt = datetime.fromtimestamp(ts2[idx])
                    print(f'  {dt.strftime("%Y-%m-%d %H:%M")} OI={oi2[idx]:>10,}')
    else:
        print(f'Response: {json.dumps(data2)[:500]}')
