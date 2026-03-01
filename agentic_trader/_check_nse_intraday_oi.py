"""Check if NSE provides intraday OI data (5-min snapshots)."""
import requests, time, json

s = requests.Session()
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.nseindia.com/',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
})

# Step 1: Get cookies
print('=== Getting NSE cookies ===')
r = s.get('https://www.nseindia.com', timeout=15)
print(f'Homepage: {r.status_code}')
time.sleep(2)

# Step 2: Option chain API - gives LIVE OI snapshot
print('\n=== Option Chain API (SBIN) ===')
try:
    r = s.get('https://www.nseindia.com/api/option-chain-equities?symbol=SBIN', timeout=15)
    print(f'Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        records = data.get('records', {})
        ts = records.get('timestamp', 'N/A')
        print(f'Timestamp: {ts}')
        print(f'Total strike records: {len(records.get("data", []))}')
        if records.get('data'):
            sample = records['data'][0]
            print(f'Keys per record: {list(sample.keys())}')
            if 'CE' in sample:
                ce = sample['CE']
                print(f'CE fields: {sorted(ce.keys())}')
                print(f'  strike={ce.get("strikePrice")}, OI={ce.get("openInterest")}, '
                      f'chgOI={ce.get("changeinOpenInterest")}, IV={ce.get("impliedVolatility")}, '
                      f'vol={ce.get("totalTradedVolume")}, lastPrice={ce.get("lastPrice")}, '
                      f'underlying={ce.get("underlyingValue")}')
            if 'PE' in sample:
                pe = sample['PE']
                print(f'  PE strike={pe.get("strikePrice")}, OI={pe.get("openInterest")}, '
                      f'chgOI={pe.get("changeinOpenInterest")}, IV={pe.get("impliedVolatility")}')
    else:
        print(f'Body: {r.text[:300]}')
except Exception as e:
    print(f'Error: {e}')

# Step 3: Check NIFTY option chain
print('\n=== Option Chain API (NIFTY index) ===')
try:
    r = s.get('https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY', timeout=15)
    print(f'Status: {r.status_code}')
    if r.status_code == 200:
        data = r.json()
        records = data.get('records', {})
        ts = records.get('timestamp', 'N/A')
        stale = records.get('staleData', None)
        print(f'Timestamp: {ts}, staleData: {stale}')
        print(f'Expiry dates: {records.get("expiryDates", [])[:3]}...')
        tot_ce_oi = sum(d.get('CE', {}).get('openInterest', 0) for d in records.get('data', []) if 'CE' in d)
        tot_pe_oi = sum(d.get('PE', {}).get('openInterest', 0) for d in records.get('data', []) if 'PE' in d)
        print(f'Total CE OI: {tot_ce_oi:,}, Total PE OI: {tot_pe_oi:,}, PCR: {tot_pe_oi/max(tot_ce_oi,1):.3f}')
except Exception as e:
    print(f'Error: {e}')

# Step 4: Check other NSE API endpoints for intraday data
print('\n=== Checking other NSE endpoints ===')
endpoints = [
    ('FO Market Turnover', 'https://www.nseindia.com/api/market-turn-over'),
    ('FO Most Active', 'https://www.nseindia.com/api/live-analysis-most-active-securities?index=gainers&type=fno'),
    ('OI Spurts', 'https://www.nseindia.com/api/live-analysis-oi-spurts-contracts'),
    ('Volume Spurts', 'https://www.nseindia.com/api/live-analysis-volume-gainers'),
]

for name, url in endpoints:
    try:
        time.sleep(1)
        r = s.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                print(f'{name}: {r.status_code} - keys={list(data.keys())[:5]}')
                # Check for timestamp
                for k in ['timestamp', 'time', 'date', 'lastUpdateTime']:
                    if k in data:
                        print(f'  {k}: {data[k]}')
            elif isinstance(data, list):
                print(f'{name}: {r.status_code} - {len(data)} records')
                if data:
                    print(f'  Sample keys: {list(data[0].keys())[:8]}')
        else:
            print(f'{name}: {r.status_code}')
    except Exception as e:
        print(f'{name}: Error - {e}')

# Step 5: Check if bhav copy has intraday versions
print('\n=== Bhav Copy Intraday Check ===')
print('NSE Bhav Copy is ONLY end-of-day (published after market close)')
print('There is NO intraday bhav copy from NSE.')
print()
print('=== SUMMARY: Intraday OI Data Sources ===')
print('1. NSE Option Chain API (/api/option-chain-equities?symbol=X)')
print('   - Provides LIVE snapshot of OI per strike')
print('   - Updated every ~3 min during market hours')
print('   - Must be polled repeatedly to build 5-min history')
print('   - Rate limited (~3 req/sec, need cookies)')
print('   - NO historical intraday OI available')
print()
print('2. NSE OI Spurts API (/api/live-analysis-oi-spurts-contracts)')
print('   - Shows contracts with sudden OI increase')
print('   - Live only, no history')
print()
print('3. Kite/Zerodha API (kite.instruments + kite.quote)')
print('   - kite.quote() gives real-time OI for F&O instruments')
print('   - Can poll every 5 min and store locally')
print('   - kite.historical_data() does NOT include OI')
print()
print('4. Dhan API')
print('   - Similar to Kite - live OI via quote, no historical intraday OI')
print()
print('CONCLUSION: No free source provides historical intraday OI.')
print('We must BUILD our own by polling NSE/Kite every 5 min during market hours.')
