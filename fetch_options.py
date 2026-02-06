"""Fetch MCX Option Chain from NSE"""

import requests
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Referer': 'https://www.nseindia.com/',
    'Accept-Language': 'en-US,en;q=0.9',
}

session = requests.Session()

# First get cookies
print("Getting NSE cookies...")
session.get('https://www.nseindia.com', headers=headers, timeout=10)

# Get option chain
print("Fetching MCX option chain...")
url = 'https://www.nseindia.com/api/option-chain-equities?symbol=MCX'
r = session.get(url, headers=headers, timeout=15)
print(f'Status: {r.status_code}')

if r.status_code == 200:
    data = r.json()
    records = data.get('records', {})
    spot = records.get('underlyingValue')
    expiries = records.get('expiryDates', [])
    
    print(f'\n{"="*60}')
    print(f'MCX OPTION CHAIN')
    print(f'{"="*60}')
    print(f'Spot Price: ₹{spot}')
    print(f'Nearest Expiry: {expiries[0] if expiries else "N/A"}')
    
    # Find relevant strikes
    chain = records.get('data', [])
    
    # Filter for nearest expiry
    nearest_expiry = expiries[0] if expiries else None
    
    print(f'\n{"="*60}')
    print(f'OPTIONS FOR EXPIRY: {nearest_expiry}')
    print(f'{"="*60}')
    
    # ATM is around 2400
    strikes_to_show = [2200, 2300, 2400, 2500, 2600]
    
    print(f'\n{"Strike":<10} {"CE LTP":<12} {"CE IV":<10} {"PE LTP":<12} {"PE IV":<10}')
    print('-' * 60)
    
    straddle_premium = 0
    ce_2400 = 0
    pe_2400 = 0
    
    for item in chain:
        strike = item.get('strikePrice')
        expiry = item.get('expiryDate')
        
        if strike in strikes_to_show and expiry == nearest_expiry:
            ce = item.get('CE', {})
            pe = item.get('PE', {})
            
            ce_ltp = ce.get('lastPrice', 0) or 0
            pe_ltp = pe.get('lastPrice', 0) or 0
            ce_iv = ce.get('impliedVolatility', 0) or 0
            pe_iv = pe.get('impliedVolatility', 0) or 0
            
            marker = " <-- ATM" if strike == 2400 else ""
            print(f'{strike:<10} ₹{ce_ltp:<10.2f} {ce_iv:<10.1f} ₹{pe_ltp:<10.2f} {pe_iv:<10.1f}{marker}')
            
            if strike == 2400:
                ce_2400 = ce_ltp
                pe_2400 = pe_ltp
                straddle_premium = ce_ltp + pe_ltp
    
    print(f'\n{"="*60}')
    print(f'2400 STRADDLE ANALYSIS')
    print(f'{"="*60}')
    print(f'2400 CE Premium: ₹{ce_2400:.2f}')
    print(f'2400 PE Premium: ₹{pe_2400:.2f}')
    print(f'Total Straddle Cost: ₹{straddle_premium:.2f} per share')
    print(f'Lot Size: 625 shares')
    print(f'Total Investment: ₹{straddle_premium * 625:,.0f}')
    
    # Breakeven
    upper_be = 2400 + straddle_premium
    lower_be = 2400 - straddle_premium
    be_pct = (straddle_premium / spot) * 100 if spot else 0
    
    print(f'\n{"="*60}')
    print(f'BREAKEVEN ANALYSIS')
    print(f'{"="*60}')
    print(f'Upper Breakeven: ₹{upper_be:.0f} (+{be_pct:.1f}%)')
    print(f'Lower Breakeven: ₹{lower_be:.0f} (-{be_pct:.1f}%)')
    print(f'Stock must move: >{be_pct:.1f}% to profit')
    
    # Compare with expected move
    expected_move = 9.3  # from our analysis
    print(f'\n{"="*60}')
    print(f'TRADE DECISION')
    print(f'{"="*60}')
    print(f'Expected Move (our system): {expected_move:.1f}%')
    print(f'Breakeven Required: {be_pct:.1f}%')
    
    if be_pct < expected_move * 0.7:
        print(f'\n✅ GOOD TRADE: Breakeven ({be_pct:.1f}%) < 70% of expected move ({expected_move*0.7:.1f}%)')
        print(f'   BUY THE 2400 STRADDLE')
    elif be_pct < expected_move:
        print(f'\n⚠️ MARGINAL: Breakeven ({be_pct:.1f}%) close to expected move ({expected_move:.1f}%)')
        print(f'   Consider smaller position or strangle')
    else:
        print(f'\n❌ AVOID: Breakeven ({be_pct:.1f}%) > expected move ({expected_move:.1f}%)')
        print(f'   Options too expensive, wait for IV to drop')
    
else:
    print(f'Failed to fetch option chain: {r.status_code}')
    print(r.text[:500])
