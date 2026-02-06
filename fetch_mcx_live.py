"""
Fetch REAL MCX Options Chain - Multiple Methods
"""
import requests
import time
import json
from datetime import datetime

print("=" * 70)
print("MCX OPTIONS CHAIN - LIVE FETCH ATTEMPT")
print(f"Time: {datetime.now()}")
print("=" * 70)

# Method 1: Try nsepython
print("\n[Method 1] Using nsepython library...")
try:
    from nsepython import option_chain, nse_optionchain_scrapper
    data = option_chain('MCX')
    if data and 'records' in data:
        print("  SUCCESS with nsepython!")
        records = data['records']
        print(f"  Spot: {records.get('underlyingValue')}")
    else:
        print("  Empty response from nsepython")
        data = None
except Exception as e:
    print(f"  Failed: {e}")
    data = None

# Method 2: Direct API with retries
if not data or not data.get('records'):
    print("\n[Method 2] Direct NSE API with retries...")
    
    session = requests.Session()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
    }
    
    for attempt in range(3):
        try:
            print(f"  Attempt {attempt + 1}...")
            
            # Get main page
            session.get('https://www.nseindia.com', headers=headers, timeout=10)
            time.sleep(1)
            
            # Get option chain API
            api_url = 'https://www.nseindia.com/api/option-chain-equities?symbol=MCX'
            r = session.get(api_url, headers=headers, timeout=15)
            
            if r.status_code == 200:
                data = r.json()
                if data and 'records' in data and data['records'].get('data'):
                    print("  SUCCESS!")
                    break
                else:
                    print(f"  Got 200 but empty data")
            else:
                print(f"  Status: {r.status_code}")
                
            time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

# Method 3: Check if we have any saved data
if not data or not data.get('records', {}).get('data'):
    print("\n[Method 3] Checking for saved data...")
    try:
        with open('mcx_options_saved.json', 'r') as f:
            data = json.load(f)
            if data.get('records', {}).get('data'):
                print("  Found saved data!")
    except:
        print("  No saved data found")

# If still no data, the issue is NSE blocking or market closed
if not data or not data.get('records', {}).get('data'):
    print("\n" + "=" * 70)
    print("UNABLE TO FETCH LIVE DATA")
    print("=" * 70)
    print("""
Possible reasons:
1. NSE API is blocking requests (rate limiting)
2. Market is closed (options data may not be available)
3. Network/firewall issues

Solutions:
1. Try again during market hours (9:15 AM - 3:30 PM IST)
2. Use a VPN or different network
3. Use broker's API (Zerodha, Upstox, etc.) for live data
4. Check https://www.nseindia.com/option-chain manually

For now, you can:
- Visit https://www.nseindia.com/option-chain
- Enter MCX in the symbol box
- Check the live options chain there
""")
else:
    # We have data - display it
    records = data['records']
    spot = records.get('underlyingValue', 0)
    expiries = records.get('expiryDates', [])
    chain = records.get('data', [])
    timestamp = records.get('timestamp', 'N/A')
    
    print("\n" + "=" * 70)
    print("MCX LIVE OPTIONS CHAIN")
    print("=" * 70)
    print(f"Spot Price: Rs.{spot}")
    print(f"Timestamp: {timestamp}")
    print(f"Expiries: {expiries[:3]}")
    
    nearest = expiries[0] if expiries else None
    print(f"\nOptions for: {nearest}")
    print(f"Lot Size: 625 shares")
    
    print(f"\n{'Strike':>8} | {'CE LTP':>10} {'CE OI':>12} | {'PE LTP':>10} {'PE OI':>12}")
    print("-" * 60)
    
    for item in sorted(chain, key=lambda x: x.get('strikePrice', 0)):
        if item.get('expiryDate') != nearest:
            continue
        
        strike = item.get('strikePrice', 0)
        if abs(strike - spot) > 400:
            continue
        
        ce = item.get('CE', {})
        pe = item.get('PE', {})
        
        ce_ltp = ce.get('lastPrice', 0) or 0
        ce_oi = ce.get('openInterest', 0) or 0
        pe_ltp = pe.get('lastPrice', 0) or 0
        pe_oi = pe.get('openInterest', 0) or 0
        
        atm = " <-- ATM" if abs(strike - spot) < 30 else ""
        print(f'{strike:>8.0f} | Rs.{ce_ltp:>8.2f} {ce_oi:>12,} | Rs.{pe_ltp:>8.2f} {pe_oi:>12,}{atm}')
    
    # Save for future use
    with open('mcx_options_saved.json', 'w') as f:
        json.dump(data, f)
    print("\n[Saved data for future reference]")
