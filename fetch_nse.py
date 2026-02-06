"""Fetch MCX Options - Better NSE handling"""

import requests
import time
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

session = requests.Session()

# Step 1: Visit main page for cookies
print('Getting NSE cookies...')
r1 = session.get('https://www.nseindia.com', headers=headers, timeout=10)
print(f'Cookies: {len(session.cookies)} received')
time.sleep(2)

# Step 2: Get API data
print('Fetching MCX option chain...')
headers['Referer'] = 'https://www.nseindia.com/option-chain'
headers['Accept'] = 'application/json'

r = session.get('https://www.nseindia.com/api/option-chain-equities?symbol=MCX', headers=headers, timeout=15)
print(f'Status: {r.status_code}')

if r.status_code == 200:
    data = r.json()
    records = data.get('records', {})
    
    underlying = records.get('underlyingValue')
    expiries = records.get('expiryDates', [])
    chain = records.get('data', [])
    
    print(f'Underlying: {underlying}')
    print(f'Expiries: {expiries[:3] if expiries else "None"}')
    print(f'Data points: {len(chain)}')
    
    if chain:
        # Save for analysis
        with open('mcx_options.json', 'w') as f:
            json.dump(data, f, indent=2)
        print('Saved to mcx_options.json')
else:
    print(f'Failed: {r.text[:200]}')
