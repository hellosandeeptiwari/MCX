"""
=================================================================
BROKER API SETUP FOR LIVE OPTIONS DATA
=================================================================
This script helps you set up Zerodha Kite or Upstox API
for fetching live MCX options chain data.

STEP 1: Choose your broker and get API credentials
STEP 2: Configure the credentials below
STEP 3: Run this script to fetch live data
=================================================================
"""

# ========================
# CONFIGURATION - EDIT THIS
# ========================

# Choose your broker: "zerodha" or "upstox"
BROKER = "zerodha"

# Zerodha Kite Connect credentials
# Get from: https://developers.kite.trade/
ZERODHA_API_KEY = "your_api_key_here"
ZERODHA_API_SECRET = "your_api_secret_here"
ZERODHA_ACCESS_TOKEN = ""  # Will be generated after login

# Upstox credentials
# Get from: https://api.upstox.com/
UPSTOX_API_KEY = "your_api_key_here"
UPSTOX_API_SECRET = "your_api_secret_here"
UPSTOX_ACCESS_TOKEN = ""  # Will be generated after login

# Stock symbol
SYMBOL = "MCX"
LOT_SIZE = 625


import os
import json
import webbrowser
from datetime import datetime

print("=" * 70)
print("BROKER API SETUP FOR LIVE MCX OPTIONS DATA")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ========================================
# ZERODHA KITE CONNECT SETUP
# ========================================

def setup_zerodha():
    """Setup and fetch data from Zerodha Kite Connect"""
    
    print("\n" + "=" * 70)
    print("ZERODHA KITE CONNECT SETUP")
    print("=" * 70)
    
    print("""
PREREQUISITES:
--------------
1. Zerodha trading account
2. Kite Connect subscription (Rs. 2000/month)
3. API credentials from https://developers.kite.trade/

STEPS TO GET API CREDENTIALS:
-----------------------------
1. Go to https://developers.kite.trade/
2. Login with your Zerodha credentials
3. Create a new app
4. Note down your API Key and API Secret
5. Set the redirect URL to: http://127.0.0.1:5000/callback
""")
    
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("\n[!] Installing kiteconnect library...")
        os.system("pip install kiteconnect")
        from kiteconnect import KiteConnect
    
    if ZERODHA_API_KEY == "your_api_key_here":
        print("\n[ERROR] Please edit this file and add your Zerodha API credentials!")
        print("        Set ZERODHA_API_KEY and ZERODHA_API_SECRET at the top of this file.")
        return None
    
    kite = KiteConnect(api_key=ZERODHA_API_KEY)
    
    # Check if we have access token
    if not ZERODHA_ACCESS_TOKEN:
        print("\n[STEP 1] Generate Login URL...")
        login_url = kite.login_url()
        print(f"Login URL: {login_url}")
        print("\n[STEP 2] Opening browser for login...")
        webbrowser.open(login_url)
        
        print("\n[STEP 3] After login, you'll be redirected to a URL with 'request_token' parameter")
        request_token = input("Enter the request_token from the URL: ").strip()
        
        print("\n[STEP 4] Generating access token...")
        try:
            data = kite.generate_session(request_token, api_secret=ZERODHA_API_SECRET)
            access_token = data["access_token"]
            print(f"Access Token: {access_token}")
            print("\n[!] Save this access token in the ZERODHA_ACCESS_TOKEN variable above!")
            kite.set_access_token(access_token)
        except Exception as e:
            print(f"Error generating session: {e}")
            return None
    else:
        kite.set_access_token(ZERODHA_ACCESS_TOKEN)
    
    return kite


def fetch_zerodha_options(kite):
    """Fetch MCX options chain from Zerodha"""
    
    print("\n" + "=" * 70)
    print("FETCHING MCX OPTIONS CHAIN FROM ZERODHA")
    print("=" * 70)
    
    try:
        # Get instruments list
        instruments = kite.instruments("NFO")
        
        # Filter MCX options
        mcx_options = [i for i in instruments if i['name'] == 'MCX' and i['instrument_type'] in ['CE', 'PE']]
        
        if not mcx_options:
            print("No MCX options found!")
            return None
        
        # Get unique expiries
        expiries = sorted(set([i['expiry'] for i in mcx_options]))
        nearest_expiry = expiries[0]
        
        print(f"Available expiries: {expiries[:3]}")
        print(f"Using nearest expiry: {nearest_expiry}")
        
        # Filter for nearest expiry
        options = [i for i in mcx_options if i['expiry'] == nearest_expiry]
        
        # Get quotes for all options
        option_tokens = [i['instrument_token'] for i in options]
        quotes = kite.quote(option_tokens)
        
        # Get spot price
        spot_quote = kite.quote(["NSE:MCX"])
        spot = spot_quote["NSE:MCX"]["last_price"]
        
        print(f"\nSpot Price: Rs.{spot}")
        print(f"Lot Size: {LOT_SIZE}")
        
        # Build options chain
        chain_data = []
        for opt in options:
            token = opt['instrument_token']
            quote = quotes.get(str(token), {})
            
            chain_data.append({
                'strike': opt['strike'],
                'type': opt['instrument_type'],
                'ltp': quote.get('last_price', 0),
                'oi': quote.get('oi', 0),
                'volume': quote.get('volume', 0),
                'bid': quote.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                'ask': quote.get('depth', {}).get('sell', [{}])[0].get('price', 0),
            })
        
        return {
            'spot': spot,
            'expiry': nearest_expiry,
            'chain': chain_data
        }
        
    except Exception as e:
        print(f"Error fetching options: {e}")
        return None


# ========================================
# UPSTOX SETUP
# ========================================

def setup_upstox():
    """Setup and fetch data from Upstox API"""
    
    print("\n" + "=" * 70)
    print("UPSTOX API SETUP")
    print("=" * 70)
    
    print("""
PREREQUISITES:
--------------
1. Upstox trading account
2. API credentials from https://api.upstox.com/

STEPS TO GET API CREDENTIALS:
-----------------------------
1. Go to https://api.upstox.com/
2. Login with your Upstox credentials
3. Create a new app
4. Note down your API Key and API Secret
5. Set redirect URL to: http://127.0.0.1:5000/callback
""")
    
    try:
        import upstox_client
        from upstox_client.rest import ApiException
    except ImportError:
        print("\n[!] Installing upstox-python-sdk...")
        os.system("pip install upstox-python-sdk")
        import upstox_client
        from upstox_client.rest import ApiException
    
    if UPSTOX_API_KEY == "your_api_key_here":
        print("\n[ERROR] Please edit this file and add your Upstox API credentials!")
        print("        Set UPSTOX_API_KEY and UPSTOX_API_SECRET at the top of this file.")
        return None
    
    # Generate login URL
    if not UPSTOX_ACCESS_TOKEN:
        auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={UPSTOX_API_KEY}&redirect_uri=http://127.0.0.1:5000/callback"
        
        print(f"\n[STEP 1] Login URL: {auth_url}")
        print("\n[STEP 2] Opening browser for login...")
        webbrowser.open(auth_url)
        
        auth_code = input("\n[STEP 3] Enter the authorization code from redirect URL: ").strip()
        
        # Exchange for access token
        import requests
        token_url = "https://api.upstox.com/v2/login/authorization/token"
        
        payload = {
            'code': auth_code,
            'client_id': UPSTOX_API_KEY,
            'client_secret': UPSTOX_API_SECRET,
            'redirect_uri': 'http://127.0.0.1:5000/callback',
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(token_url, data=payload)
        if response.status_code == 200:
            access_token = response.json().get('access_token')
            print(f"\nAccess Token: {access_token}")
            print("\n[!] Save this access token in UPSTOX_ACCESS_TOKEN variable above!")
            return access_token
        else:
            print(f"Error getting token: {response.text}")
            return None
    
    return UPSTOX_ACCESS_TOKEN


def fetch_upstox_options(access_token):
    """Fetch MCX options chain from Upstox"""
    
    print("\n" + "=" * 70)
    print("FETCHING MCX OPTIONS CHAIN FROM UPSTOX")
    print("=" * 70)
    
    import requests
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    
    try:
        # Get option chain
        url = f"https://api.upstox.com/v2/option/chain?instrument_key=NSE_EQ|INE745G01035&expiry_date=2026-02-27"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


# ========================================
# DISPLAY OPTIONS CHAIN
# ========================================

def display_options_chain(data):
    """Display the options chain in a formatted table"""
    
    if not data:
        return
    
    spot = data.get('spot', 0)
    expiry = data.get('expiry', 'N/A')
    chain = data.get('chain', [])
    
    print(f"\n{'=' * 80}")
    print(f"MCX LIVE OPTIONS CHAIN")
    print(f"{'=' * 80}")
    print(f"Spot: Rs.{spot:,.2f}")
    print(f"Expiry: {expiry}")
    print(f"Lot Size: {LOT_SIZE}")
    
    # Separate CE and PE
    calls = {c['strike']: c for c in chain if c['type'] == 'CE'}
    puts = {p['strike']: p for p in chain if p['type'] == 'PE'}
    
    strikes = sorted(set(calls.keys()) | set(puts.keys()))
    
    # Filter around ATM
    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    strikes = [s for s in strikes if abs(s - spot) <= 400]
    
    print(f"\n{'Strike':>8} | {'CE LTP':>10} {'CE OI':>12} | {'PE LTP':>10} {'PE OI':>12}")
    print("-" * 65)
    
    for strike in strikes:
        ce = calls.get(strike, {})
        pe = puts.get(strike, {})
        
        ce_ltp = ce.get('ltp', 0)
        ce_oi = ce.get('oi', 0)
        pe_ltp = pe.get('ltp', 0)
        pe_oi = pe.get('oi', 0)
        
        atm = " <-- ATM" if strike == atm_strike else ""
        
        print(f"{strike:>8.0f} | Rs.{ce_ltp:>8.2f} {ce_oi:>12,} | Rs.{pe_ltp:>8.2f} {pe_oi:>12,}{atm}")
    
    return strikes, calls, puts, spot


def analyze_strategies(strikes, calls, puts, spot):
    """Analyze profitable strategies based on live data"""
    
    print(f"\n{'=' * 80}")
    print("STRATEGY ANALYSIS (Based on LIVE Data)")
    print(f"{'=' * 80}")
    
    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    
    # ATM Straddle
    atm_ce = calls.get(atm_strike, {}).get('ltp', 0)
    atm_pe = puts.get(atm_strike, {}).get('ltp', 0)
    straddle_premium = atm_ce + atm_pe
    
    print(f"\n1. ATM STRADDLE ({atm_strike})")
    print(f"   CE Premium: Rs.{atm_ce:.2f}")
    print(f"   PE Premium: Rs.{atm_pe:.2f}")
    print(f"   Total Premium: Rs.{straddle_premium:.2f} per share")
    print(f"   Total for 1 lot: Rs.{straddle_premium * LOT_SIZE:,.0f}")
    print(f"   Breakeven: Rs.{atm_strike - straddle_premium:.0f} - Rs.{atm_strike + straddle_premium:.0f}")
    
    # Iron Condor
    otm_strikes = [s for s in strikes if s > atm_strike][:2]
    itm_strikes = [s for s in sorted(strikes, reverse=True) if s < atm_strike][:2]
    
    if len(otm_strikes) >= 2 and len(itm_strikes) >= 2:
        sell_ce = otm_strikes[0]
        buy_ce = otm_strikes[1]
        sell_pe = itm_strikes[0]
        buy_pe = itm_strikes[1]
        
        sell_ce_prem = calls.get(sell_ce, {}).get('ltp', 0)
        buy_ce_prem = calls.get(buy_ce, {}).get('ltp', 0)
        sell_pe_prem = puts.get(sell_pe, {}).get('ltp', 0)
        buy_pe_prem = puts.get(buy_pe, {}).get('ltp', 0)
        
        ic_credit = sell_ce_prem - buy_ce_prem + sell_pe_prem - buy_pe_prem
        ic_total = ic_credit * LOT_SIZE
        wing_width = buy_ce - sell_ce
        max_loss = (wing_width - ic_credit) * LOT_SIZE
        
        print(f"\n2. IRON CONDOR")
        print(f"   Sell {sell_pe} PE @ Rs.{sell_pe_prem:.2f}")
        print(f"   Buy {buy_pe} PE @ Rs.{buy_pe_prem:.2f}")
        print(f"   Sell {sell_ce} CE @ Rs.{sell_ce_prem:.2f}")
        print(f"   Buy {buy_ce} CE @ Rs.{buy_ce_prem:.2f}")
        print(f"   Net Credit: Rs.{ic_credit:.2f} per share")
        print(f"   Total Premium: Rs.{ic_total:,.0f}")
        print(f"   Max Loss: Rs.{max_loss:,.0f}")
        print(f"   Risk/Reward: 1:{ic_total/max_loss:.2f}" if max_loss > 0 else "")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    
    if BROKER.lower() == "zerodha":
        kite = setup_zerodha()
        if kite:
            data = fetch_zerodha_options(kite)
            if data:
                strikes, calls, puts, spot = display_options_chain(data)
                analyze_strategies(strikes, calls, puts, spot)
    
    elif BROKER.lower() == "upstox":
        token = setup_upstox()
        if token:
            data = fetch_upstox_options(token)
            if data:
                display_options_chain(data)
    
    else:
        print(f"Unknown broker: {BROKER}")
        print("Please set BROKER to 'zerodha' or 'upstox'")
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
