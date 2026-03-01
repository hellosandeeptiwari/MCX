"""Test Dhan intraday 5-min futures OI with correct contract ID."""
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

# SBIN Mar 2026 futures: ID=828576
from_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d 09:15:00')
to_date = datetime.now().strftime('%Y-%m-%d 15:30:00')

# Test FUTURES intraday 5-min with OI
print("=== SBIN Mar2026 FUTURES - 5min with OI ===")
body = {
    'securityId': '828576',
    'exchangeSegment': 'NSE_FNO',
    'instrument': 'FUTSTK',
    'interval': '5',
    'oi': True,
    'fromDate': from_date,
    'toDate': to_date,
}
print(f"Request: {json.dumps(body)}")

r = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body, timeout=15)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Keys: {list(data.keys())}")

if 'timestamp' in data and data['timestamp']:
    ts = data['timestamp']
    oi = data.get('open_interest', [])
    vol = data.get('volume', [])
    o = data.get('open', [])
    c = data.get('close', [])
    
    print(f"Candles: {len(ts)}")
    has_oi = len(oi) > 0 and any(x > 0 for x in oi)
    print(f"Has OI: {has_oi}")
    
    if oi:
        print(f"OI range: min={min(oi):,}, max={max(oi):,}, mean={sum(oi)/len(oi):,.0f}")
    
    # Show some candles
    print("\nSample 5-min candles with OI:")
    for i in range(min(10, len(ts))):
        dt = datetime.fromtimestamp(ts[i])
        oi_val = oi[i] if i < len(oi) else 0
        v = vol[i] if i < len(vol) else 0
        print(f"  {dt.strftime('%Y-%m-%d %H:%M')} | O={o[i]:>8.2f} C={c[i]:>8.2f} V={v:>10,} OI={oi_val:>12,}")
    
    print("  ...")
    for i in range(max(0, len(ts)-5), len(ts)):
        dt = datetime.fromtimestamp(ts[i])
        oi_val = oi[i] if i < len(oi) else 0
        v = vol[i] if i < len(vol) else 0
        print(f"  {dt.strftime('%Y-%m-%d %H:%M')} | O={o[i]:>8.2f} C={c[i]:>8.2f} V={v:>10,} OI={oi_val:>12,}")

    # Show OI change pattern through the day
    print("\n5-min OI change pattern (last trading day):")
    last_date = datetime.fromtimestamp(ts[-1]).date()
    day_candles = [(datetime.fromtimestamp(ts[i]), oi[i] if i < len(oi) else 0) 
                   for i in range(len(ts)) if datetime.fromtimestamp(ts[i]).date() == last_date]
    if day_candles:
        prev_oi = day_candles[0][1]
        for dt, oi_val in day_candles:
            chg = oi_val - prev_oi
            pct = (chg / prev_oi * 100) if prev_oi > 0 else 0
            print(f"  {dt.strftime('%H:%M')} OI={oi_val:>12,}  chg={chg:>+10,} ({pct:>+.2f}%)")
            prev_oi = oi_val
else:
    print(f"Response: {json.dumps(data)[:500]}")

# Test OPTIONS 5-min with OI
print("\n\n=== SBIN CE Option (ATM) - 5min with OI ===")
# Find SBIN options from CSV
import csv
opt_contracts = []
now_str = datetime.now().strftime('%Y-%m-%d')
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trading_sym = row.get('SEM_TRADING_SYMBOL', '')
        seg = row.get('SEM_SEGMENT', '')
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        otype = row.get('SEM_OPTION_TYPE', '').strip()
        exp = row.get('SEM_EXPIRY_DATE', '')
        strike = row.get('SEM_STRIKE_PRICE', '0')
        if ('SBIN' in trading_sym and seg == 'D' and inst == 'OPTSTK' 
            and otype == 'CE' and exp >= now_str and '15:30' in exp):
            opt_contracts.append({
                'id': row['SEM_SMST_SECURITY_ID'],
                'symbol': trading_sym,
                'strike': float(strike),
                'expiry': exp,
            })

opt_contracts.sort(key=lambda x: (x['expiry'], x['strike']))
# Pick near 750-800
atm = [c for c in opt_contracts if 730 <= c['strike'] <= 800]
print(f"ATM CE options found: {len(atm)}")
for c in atm[:5]:
    print(f"  {c['symbol']} strike={c['strike']} ID={c['id']} exp={c['expiry']}")

if atm:
    oc = atm[0]
    body2 = {
        'securityId': oc['id'],
        'exchangeSegment': 'NSE_FNO',
        'instrument': 'OPTSTK',
        'interval': '5',
        'oi': True,
        'fromDate': from_date,
        'toDate': to_date,
    }
    r2 = requests.post('https://api.dhan.co/v2/charts/intraday', headers=headers, json=body2, timeout=15)
    print(f"Status: {r2.status_code}")
    data2 = r2.json()
    if 'timestamp' in data2 and data2['timestamp']:
        ts2 = data2['timestamp']
        oi2 = data2.get('open_interest', [])
        print(f"Options 5-min candles: {len(ts2)}")
        print(f"Has OI: {len(oi2) > 0 and any(x > 0 for x in oi2)}")
        if oi2:
            print(f"OI range: min={min(oi2):,}, max={max(oi2):,}")
            for i in [0, 1, len(ts2)//2, -2, -1]:
                idx = i if i >= 0 else len(ts2) + i
                if 0 <= idx < len(ts2):
                    dt = datetime.fromtimestamp(ts2[idx])
                    print(f"  {dt.strftime('%Y-%m-%d %H:%M')} OI={oi2[idx]:>10,}")
    else:
        print(f"Response: {json.dumps(data2)[:500]}")
