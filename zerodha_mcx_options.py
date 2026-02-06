"""
MCX Options Chain - Zerodha Kite Connect
=========================================
Enter your API credentials below and run this script
"""

import webbrowser
import json
from datetime import datetime

# ============================================
# ENTER YOUR ZERODHA CREDENTIALS HERE
# ============================================

API_KEY = os.environ.get("ZERODHA_API_KEY", "")
API_SECRET = os.environ.get("ZERODHA_API_SECRET", "")

# ============================================

LOT_SIZE = 625
SYMBOL = "MCX"

def main():
    print("=" * 70)
    print("MCX OPTIONS CHAIN - ZERODHA KITE CONNECT")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check if credentials are entered
    global API_KEY, API_SECRET
    
    if not API_KEY:
        API_KEY = input("\nEnter your Zerodha API Key: ").strip()
    
    if not API_SECRET:
        API_SECRET = input("Enter your Zerodha API Secret: ").strip()
    
    # Install kiteconnect if needed
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("\n[Installing kiteconnect library...]")
        import subprocess
        subprocess.run(["pip", "install", "kiteconnect"], check=True)
        from kiteconnect import KiteConnect
    
    # Initialize Kite
    print("\n[1/4] Initializing Kite Connect...")
    kite = KiteConnect(api_key=API_KEY)
    
    # Generate login URL
    print("[2/4] Opening browser for login...")
    login_url = kite.login_url()
    print(f"\nLogin URL: {login_url}")
    webbrowser.open(login_url)
    
    print("\n" + "-" * 70)
    print("INSTRUCTIONS:")
    print("-" * 70)
    print("1. Login with your Zerodha credentials in the browser")
    print("2. After login, you'll be redirected to a URL like:")
    print("   http://127.0.0.1:5000/callback?request_token=XXXXXX&status=success")
    print("3. Copy the 'request_token' value (the XXXXXX part)")
    print("-" * 70)
    
    request_token = input("\nPaste the request_token here: ").strip()
    
    # Generate session
    print("\n[3/4] Generating access token...")
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        kite.set_access_token(access_token)
        print("SUCCESS! Connected to Zerodha.")
        
        # Save token for reuse
        with open("zerodha_token.json", "w") as f:
            json.dump({"access_token": access_token, "date": str(datetime.now().date())}, f)
        print("[Token saved for today's session]")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPossible issues:")
        print("- Request token expired (try again quickly after login)")
        print("- Wrong API Secret")
        print("- Request token copied incorrectly")
        return
    
    # Fetch options chain
    print("\n[4/4] Fetching MCX Options Chain...")
    
    try:
        # Get all NFO instruments
        instruments = kite.instruments("NFO")
        
        # Filter MCX options
        mcx_options = [i for i in instruments 
                       if i['name'] == 'MCX' 
                       and i['instrument_type'] in ['CE', 'PE']]
        
        if not mcx_options:
            print("No MCX options found!")
            return
        
        # Get unique expiries
        expiries = sorted(set([i['expiry'] for i in mcx_options]))
        nearest_expiry = expiries[0]
        
        print(f"\nExpiries available: {[str(e) for e in expiries[:3]]}")
        print(f"Using: {nearest_expiry}")
        
        # Filter for nearest expiry
        options = [i for i in mcx_options if i['expiry'] == nearest_expiry]
        
        # Get spot price
        spot_quote = kite.quote(["NSE:MCX"])
        spot = spot_quote["NSE:MCX"]["last_price"]
        
        print(f"\n{'=' * 70}")
        print(f"MCX LIVE OPTIONS CHAIN")
        print(f"{'=' * 70}")
        print(f"Spot Price: Rs.{spot:,.2f}")
        print(f"Expiry: {nearest_expiry}")
        print(f"Lot Size: {LOT_SIZE}")
        
        # Get quotes for options near ATM
        atm_strike = round(spot / 50) * 50  # Round to nearest 50
        
        relevant_options = [o for o in options 
                           if abs(o['strike'] - spot) <= 400]
        
        # Fetch quotes in batches
        tokens = [o['instrument_token'] for o in relevant_options]
        quotes = kite.quote(tokens)
        
        # Build chain
        chain = {}
        for opt in relevant_options:
            strike = opt['strike']
            if strike not in chain:
                chain[strike] = {'CE': {}, 'PE': {}}
            
            token = opt['instrument_token']
            q = quotes.get(str(token), {})
            
            chain[strike][opt['instrument_type']] = {
                'ltp': q.get('last_price', 0),
                'oi': q.get('oi', 0),
                'volume': q.get('volume', 0),
            }
        
        # Display
        print(f"\n{'Strike':>8} | {'CE LTP':>10} {'CE OI':>12} | {'PE LTP':>10} {'PE OI':>12}")
        print("-" * 65)
        
        for strike in sorted(chain.keys()):
            ce = chain[strike]['CE']
            pe = chain[strike]['PE']
            
            ce_ltp = ce.get('ltp', 0) or 0
            ce_oi = ce.get('oi', 0) or 0
            pe_ltp = pe.get('ltp', 0) or 0
            pe_oi = pe.get('oi', 0) or 0
            
            atm = " <-- ATM" if abs(strike - spot) < 30 else ""
            print(f"{strike:>8.0f} | Rs.{ce_ltp:>8.2f} {ce_oi:>12,} | Rs.{pe_ltp:>8.2f} {pe_oi:>12,}{atm}")
        
        # Strategy Analysis
        print(f"\n{'=' * 70}")
        print("STRATEGY ANALYSIS")
        print(f"{'=' * 70}")
        
        atm_strike = min(chain.keys(), key=lambda x: abs(x - spot))
        atm_ce = chain[atm_strike]['CE'].get('ltp', 0)
        atm_pe = chain[atm_strike]['PE'].get('ltp', 0)
        
        # 1. ATM Straddle
        straddle = atm_ce + atm_pe
        print(f"\n1. ATM STRADDLE ({atm_strike})")
        print(f"   Sell {atm_strike} CE @ Rs.{atm_ce:.2f}")
        print(f"   Sell {atm_strike} PE @ Rs.{atm_pe:.2f}")
        print(f"   Premium per share: Rs.{straddle:.2f}")
        print(f"   Total Premium (1 lot): Rs.{straddle * LOT_SIZE:,.0f}")
        print(f"   Breakeven: Rs.{atm_strike - straddle:.0f} to Rs.{atm_strike + straddle:.0f}")
        
        # 2. Iron Condor
        strikes_list = sorted(chain.keys())
        atm_idx = strikes_list.index(atm_strike)
        
        if atm_idx >= 2 and atm_idx < len(strikes_list) - 2:
            sell_pe = strikes_list[atm_idx - 1]
            buy_pe = strikes_list[atm_idx - 2]
            sell_ce = strikes_list[atm_idx + 1]
            buy_ce = strikes_list[atm_idx + 2]
            
            sell_pe_prem = chain[sell_pe]['PE'].get('ltp', 0)
            buy_pe_prem = chain[buy_pe]['PE'].get('ltp', 0)
            sell_ce_prem = chain[sell_ce]['CE'].get('ltp', 0)
            buy_ce_prem = chain[buy_ce]['CE'].get('ltp', 0)
            
            ic_credit = sell_pe_prem - buy_pe_prem + sell_ce_prem - buy_ce_prem
            wing = sell_ce - atm_strike
            max_loss = (wing - ic_credit)
            
            print(f"\n2. IRON CONDOR")
            print(f"   Sell {sell_pe} PE @ Rs.{sell_pe_prem:.2f}")
            print(f"   Buy {buy_pe} PE @ Rs.{buy_pe_prem:.2f}")
            print(f"   Sell {sell_ce} CE @ Rs.{sell_ce_prem:.2f}")
            print(f"   Buy {buy_ce} CE @ Rs.{buy_ce_prem:.2f}")
            print(f"   Net Credit: Rs.{ic_credit:.2f} per share")
            print(f"   Total Premium (1 lot): Rs.{ic_credit * LOT_SIZE:,.0f}")
            print(f"   Max Loss (1 lot): Rs.{max_loss * LOT_SIZE:,.0f}")
            if max_loss > 0:
                print(f"   Risk/Reward: 1:{ic_credit/max_loss:.2f}")
        
        # 3. Bull Call Spread
        if atm_idx < len(strikes_list) - 1:
            buy_strike = atm_strike
            sell_strike = strikes_list[atm_idx + 1]
            
            buy_prem = chain[buy_strike]['CE'].get('ltp', 0)
            sell_prem = chain[sell_strike]['CE'].get('ltp', 0)
            
            debit = buy_prem - sell_prem
            max_profit = (sell_strike - buy_strike) - debit
            
            print(f"\n3. BULL CALL SPREAD")
            print(f"   Buy {buy_strike} CE @ Rs.{buy_prem:.2f}")
            print(f"   Sell {sell_strike} CE @ Rs.{sell_prem:.2f}")
            print(f"   Net Debit: Rs.{debit:.2f} per share")
            print(f"   Cost (1 lot): Rs.{debit * LOT_SIZE:,.0f}")
            print(f"   Max Profit (1 lot): Rs.{max_profit * LOT_SIZE:,.0f}")
            if debit > 0:
                print(f"   Risk/Reward: {max_profit/debit:.2f}:1")
        
        print(f"\n{'=' * 70}")
        print("ANALYSIS COMPLETE - DATA IS LIVE FROM ZERODHA")
        print(f"{'=' * 70}")
        
        # Save data
        with open("mcx_options_live.json", "w") as f:
            json.dump({
                "spot": spot,
                "expiry": str(nearest_expiry),
                "timestamp": str(datetime.now()),
                "chain": {str(k): v for k, v in chain.items()}
            }, f, indent=2)
        print("\n[Data saved to mcx_options_live.json]")
        
    except Exception as e:
        print(f"\nError fetching options: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
