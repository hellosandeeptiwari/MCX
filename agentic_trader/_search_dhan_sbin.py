"""Search for real SBIN futures in Dhan CSV."""
import csv

# Search for SBIN anywhere in segment D (derivatives)
found = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trading_sym = row.get('SEM_TRADING_SYMBOL', '')
        sm_name = row.get('SM_SYMBOL_NAME', '')
        seg = row.get('SEM_SEGMENT', '')
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        if 'SBIN' in trading_sym and seg == 'D' and 'FUT' in inst:
            sid = row['SEM_SMST_SECURITY_ID']
            exp = row['SEM_EXPIRY_DATE']
            found.append(f"{trading_sym} | {sm_name} | {inst} | ID={sid} | exp={exp}")

print(f"SBIN FUT in segment D: {len(found)}")
for f_ in found[:10]:
    print(f"  {f_}")

# Check real FUTSTK (not test)
print()
real_fut = []
with open('dhan_instruments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        inst = row.get('SEM_INSTRUMENT_NAME', '')
        seg = row.get('SEM_SEGMENT', '')
        sym = row.get('SEM_TRADING_SYMBOL', '')
        if inst == 'FUTSTK' and seg == 'D' and 'TEST' not in sym.upper():
            real_fut.append(sym)
            
print(f"Real FUTSTK contracts (non-TEST): {len(real_fut)}")
for f_ in sorted(set(real_fut))[:20]:
    print(f"  {f_}")

# Check how dhan_oi_fetcher gets contracts
print("\n=== Trying Dhan API directly for SBIN FUT ===")
import os, requests, json
from dotenv import load_dotenv
load_dotenv('.env')

access_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
client_id = os.environ.get('DHAN_CLIENT_ID', '')

# Use the equity scrip ID and try futures
# SBIN equity = 3045, try lookup via option chain
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'access-token': access_token,
    'client-id': client_id,
}

# Try the option chain endpoint
body = {
    'UnderlyingScrip': 3045,
    'ExpiryDate': '2026-02-27',
}
r = requests.post('https://api.dhan.co/v2/optionchain', headers=headers, json=body, timeout=15)
print(f"Option chain status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        # Look for underlying info
        if 'data' in data:
            items = data['data']
            if isinstance(items, list) and items:
                print(f"Items: {len(items)}")
                sample = items[0]
                print(f"Sample keys: {list(sample.keys())}")
                # Find the futures contract
                for item in items[:3]:
                    print(f"  {json.dumps({k: item[k] for k in list(item.keys())[:8]})}")
    else:
        print(f"Response type: {type(data)}")
else:
    print(f"Body: {r.text[:300]}")

# Try expiry list
print("\n=== Expiry List ===")
r2 = requests.post('https://api.dhan.co/v2/optionchain/expirylist', headers=headers, 
                    json={'UnderlyingScrip': 3045, 'UnderlyingSeg': 'NSE_FNO'}, timeout=15)
print(f"Status: {r2.status_code}")
if r2.status_code == 200:
    data2 = r2.json()
    print(f"Keys: {list(data2.keys()) if isinstance(data2, dict) else type(data2)}")
    if isinstance(data2, dict) and 'data' in data2:
        print(f"Expiries: {data2['data'][:5]}")
