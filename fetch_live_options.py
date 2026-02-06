"""
Fetch REAL MCX Options Chain from NSE
"""
import requests
import time
import json

session = requests.Session()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

print('Step 1: Getting NSE homepage cookies...')
try:
    r1 = session.get('https://www.nseindia.com', headers=headers, timeout=10)
    print(f'  Homepage status: {r1.status_code}')
    print(f'  Cookies: {len(session.cookies)} received')
except Exception as e:
    print(f'  Error: {e}')

time.sleep(2)

print('\nStep 2: Visiting option chain page...')
try:
    r2 = session.get('https://www.nseindia.com/option-chain', headers=headers, timeout=10)
    print(f'  Option chain page status: {r2.status_code}')
except Exception as e:
    print(f'  Error: {e}')

time.sleep(2)

print('\nStep 3: Fetching MCX options API...')
api_headers = headers.copy()
api_headers['Accept'] = 'application/json'
api_headers['Referer'] = 'https://www.nseindia.com/option-chain'

try:
    url = 'https://www.nseindia.com/api/option-chain-equities?symbol=MCX'
    r3 = session.get(url, headers=api_headers, timeout=15)
    print(f'  API status: {r3.status_code}')
    
    if r3.status_code == 200:
        data = r3.json()
        records = data.get('records', {})
        spot = records.get('underlyingValue')
        timestamp = records.get('timestamp')
        expiries = records.get('expiryDates', [])
        chain = records.get('data', [])
        
        print(f'\n{"="*70}')
        print(f'MCX LIVE OPTIONS CHAIN')
        print(f'{"="*70}')
        print(f'Spot Price: Rs.{spot}')
        print(f'Timestamp: {timestamp}')
        print(f'Nearest Expiry: {expiries[0] if expiries else "N/A"}')
        
        # Filter for nearest expiry
        nearest = expiries[0] if expiries else None
        
        print(f'\n{"Strike":>8} | {"CE LTP":>10} {"CE OI":>12} {"CE IV":>8} | {"PE LTP":>10} {"PE OI":>12} {"PE IV":>8}')
        print('-' * 85)
        
        # Show strikes around ATM
        for item in chain:
            if item.get('expiryDate') != nearest:
                continue
            
            strike = item.get('strikePrice', 0)
            
            # Filter strikes around spot
            if abs(strike - spot) > 400:
                continue
            
            ce = item.get('CE', {})
            pe = item.get('PE', {})
            
            ce_ltp = ce.get('lastPrice', 0) or 0
            ce_oi = ce.get('openInterest', 0) or 0
            ce_iv = ce.get('impliedVolatility', 0) or 0
            
            pe_ltp = pe.get('lastPrice', 0) or 0
            pe_oi = pe.get('openInterest', 0) or 0
            pe_iv = pe.get('impliedVolatility', 0) or 0
            
            atm_marker = " <-- ATM" if abs(strike - spot) < 30 else ""
            
            print(f'{strike:>8.0f} | Rs.{ce_ltp:>8.2f} {ce_oi:>12,} {ce_iv:>7.1f}% | Rs.{pe_ltp:>8.2f} {pe_oi:>12,} {pe_iv:>7.1f}%{atm_marker}')
        
        # Save raw data
        with open('mcx_options_raw.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f'\n[Saved raw data to mcx_options_raw.json]')
        
    else:
        print(f'  Failed with status: {r3.status_code}')
        print(f'  Response: {r3.text[:500]}')
        
except Exception as e:
    print(f'  Error: {e}')
    import traceback
    traceback.print_exc()
