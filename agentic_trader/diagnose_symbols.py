"""
DIAGNOSTIC: Verify all APPROVED_UNIVERSE and F&O symbols fetch data correctly from Zerodha API
"""
import os, sys, json, time
from datetime import datetime, timedelta

# Load env
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_env_path)

from kiteconnect import KiteConnect
from config import APPROVED_UNIVERSE, FNO_CONFIG

# Connect
kite = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'zerodha_token.json')
with open(token_path, 'r') as f:
    token_data = json.load(f)
kite.set_access_token(token_data['access_token'])

print("=" * 80)
print("SYMBOL DIAGNOSTICS - Zerodha API Data Verification")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ===================== 1. QUOTE FETCH TEST =====================
print("\n\n=== 1. QUOTE FETCH TEST (all symbols at once) ===\n")

try:
    quotes = kite.quote(APPROVED_UNIVERSE)
    print(f"✅ Fetched quotes for {len(quotes)}/{len(APPROVED_UNIVERSE)} symbols\n")
    
    quote_ok = []
    quote_fail = []
    
    for sym in APPROVED_UNIVERSE:
        if sym in quotes:
            q = quotes[sym]
            ltp = q.get('last_price', 0)
            vol = q.get('volume', 0)
            chg = q.get('change', 0)
            
            # Check for suspicious data
            issues = []
            if ltp == 0:
                issues.append("LTP=0")
            if vol == 0:
                issues.append("VOL=0")
            
            status = "⚠️" if issues else "✅"
            fno = "[F&O]" if sym in FNO_CONFIG.get('prefer_options_for', []) else "     "
            print(f"  {status} {fno} {sym:25s} LTP=₹{ltp:>10.2f}  Vol={vol:>12,}  Chg={chg:>+6.2f}%  {'  '.join(issues)}")
            quote_ok.append(sym)
        else:
            print(f"  ❌       {sym:25s} NOT FOUND IN QUOTES!")
            quote_fail.append(sym)
    
    if quote_fail:
        print(f"\n❌ FAILED SYMBOLS: {quote_fail}")
    else:
        print(f"\n✅ All {len(quote_ok)} symbols returned valid quotes")
        
except Exception as e:
    print(f"❌ Quote fetch error: {e}")

# ===================== 2. INSTRUMENT TOKEN LOOKUP =====================
print("\n\n=== 2. INSTRUMENT TOKEN LOOKUP ===\n")

try:
    instruments = kite.instruments("NSE")
    inst_map = {inst['tradingsymbol']: inst for inst in instruments}
    
    token_ok = []
    token_fail = []
    
    for sym in APPROVED_UNIVERSE:
        _, tradingsymbol = sym.split(":")
        if tradingsymbol in inst_map:
            inst = inst_map[tradingsymbol]
            token_ok.append(sym)
            print(f"  ✅ {sym:25s} token={inst['instrument_token']}  name={inst.get('name', 'N/A')[:30]}")
        else:
            token_fail.append(sym)
            print(f"  ❌ {sym:25s} NO INSTRUMENT TOKEN FOUND!")
    
    if token_fail:
        print(f"\n❌ MISSING INSTRUMENTS: {token_fail}")
        # Try to find similar names
        for sym in token_fail:
            _, ts = sym.split(":")
            similar = [i['tradingsymbol'] for i in instruments if ts[:4] in i['tradingsymbol']][:5]
            if similar:
                print(f"   Did you mean: {similar}")
    else:
        print(f"\n✅ All {len(token_ok)} symbols have valid instrument tokens")
        
except Exception as e:
    print(f"❌ Instruments fetch error: {e}")

# ===================== 3. HISTORICAL DATA TEST (daily) =====================
print("\n\n=== 3. HISTORICAL DATA TEST (daily, last 60 days) ===\n")

hist_ok = []
hist_fail = []

for sym in APPROVED_UNIVERSE[:10]:  # Test first 10 to avoid rate limit
    _, tradingsymbol = sym.split(":")
    if tradingsymbol not in inst_map:
        hist_fail.append((sym, "No instrument token"))
        continue
    
    token = inst_map[tradingsymbol]['instrument_token']
    time.sleep(0.4)  # Rate limit
    
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=60)
        data = kite.historical_data(token, from_date, to_date, "day")
        
        if data and len(data) > 0:
            hist_ok.append(sym)
            print(f"  ✅ {sym:25s} {len(data)} daily candles  (latest: {data[-1]['date']}  close=₹{data[-1]['close']:.2f})")
        else:
            hist_fail.append((sym, "Empty data"))
            print(f"  ❌ {sym:25s} NO HISTORICAL DATA!")
    except Exception as e:
        hist_fail.append((sym, str(e)))
        print(f"  ❌ {sym:25s} ERROR: {e}")

if hist_fail:
    print(f"\n❌ HISTORICAL DATA FAILURES: {hist_fail}")
else:
    print(f"\n✅ All tested symbols have valid historical data")

# ===================== 4. INTRADAY 5-MIN DATA TEST =====================
print("\n\n=== 4. INTRADAY 5-MIN DATA TEST (today) ===\n")

intraday_ok = []
intraday_fail = []

for sym in APPROVED_UNIVERSE[:10]:  # Test first 10
    _, tradingsymbol = sym.split(":")
    if tradingsymbol not in inst_map:
        continue
    
    token = inst_map[tradingsymbol]['instrument_token']
    time.sleep(0.4)
    
    try:
        today = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        data = kite.historical_data(token, today, datetime.now(), "5minute")
        
        if data and len(data) > 0:
            intraday_ok.append(sym)
            print(f"  ✅ {sym:25s} {len(data)} 5-min candles  (latest: {str(data[-1]['date'])[-8:]}  close=₹{data[-1]['close']:.2f})")
        else:
            intraday_fail.append((sym, "Empty/no data"))
            print(f"  ⚠️ {sym:25s} No intraday data (market may be closed)")
    except Exception as e:
        intraday_fail.append((sym, str(e)))
        print(f"  ❌ {sym:25s} ERROR: {e}")

# ===================== 5. F&O INSTRUMENTS TEST =====================
print("\n\n=== 5. F&O INSTRUMENTS TEST (NFO exchange) ===\n")

fno_prefer = FNO_CONFIG.get('prefer_options_for', [])

try:
    nfo_instruments = kite.instruments("NFO")
    print(f"📊 Total NFO instruments: {len(nfo_instruments)}")
    
    fno_ok = []
    fno_fail = []
    
    for sym in fno_prefer:
        _, tradingsymbol = sym.split(":")
        
        # Find options for this underlying
        options = [i for i in nfo_instruments 
                   if i.get('name') == tradingsymbol 
                   and i.get('instrument_type') in ('CE', 'PE')
                   and i.get('expiry') is not None]
        
        # Find futures
        futures = [i for i in nfo_instruments 
                   if i.get('name') == tradingsymbol 
                   and i.get('instrument_type') == 'FUT']
        
        if options:
            # Get nearest expiry
            expiries = sorted(set(o['expiry'] for o in options))
            nearest = expiries[0] if expiries else None
            
            # Count strikes for nearest expiry
            nearest_options = [o for o in options if o['expiry'] == nearest]
            ce_count = len([o for o in nearest_options if o['instrument_type'] == 'CE'])
            pe_count = len([o for o in nearest_options if o['instrument_type'] == 'PE'])
            lot_size = nearest_options[0]['lot_size'] if nearest_options else 'N/A'
            
            fno_ok.append(sym)
            print(f"  ✅ {sym:25s} {len(options)} options  {len(futures)} futures  Nearest: {nearest}  CE:{ce_count} PE:{pe_count}  Lot:{lot_size}")
        else:
            fno_fail.append(sym)
            
            # Check if name pattern is different
            partial = [i for i in nfo_instruments if tradingsymbol in i.get('tradingsymbol', '')][:3]
            hint = f" (found similar: {[p['name'] for p in partial]})" if partial else ""
            print(f"  ❌ {sym:25s} NO OPTIONS FOUND!{hint}")
    
    if fno_fail:
        print(f"\n❌ F&O SYMBOLS WITHOUT OPTIONS: {fno_fail}")
    else:
        print(f"\n✅ All {len(fno_ok)} F&O symbols have valid options chains")
        
except Exception as e:
    print(f"❌ NFO instruments error: {e}")

# ===================== 6. LOT SIZE CROSS-CHECK =====================
print("\n\n=== 6. LOT SIZE CROSS-CHECK (config vs exchange) ===\n")

from options_trader import FNO_LOT_SIZES

lot_mismatches = []
for sym in fno_prefer:
    _, ts = sym.split(":")
    config_lot = FNO_LOT_SIZES.get(ts, None)
    
    # Find actual lot size from NFO
    actual_options = [i for i in nfo_instruments 
                      if i.get('name') == ts 
                      and i.get('instrument_type') in ('CE', 'PE')]
    actual_lot = actual_options[0]['lot_size'] if actual_options else None
    
    if config_lot and actual_lot:
        match = "✅" if config_lot == actual_lot else "⚠️ MISMATCH"
        if config_lot != actual_lot:
            lot_mismatches.append((ts, config_lot, actual_lot))
        print(f"  {match} {ts:20s} Config={config_lot:>6}  Exchange={actual_lot:>6}")
    elif not config_lot:
        print(f"  ❌ {ts:20s} NOT IN FNO_LOT_SIZES")
        lot_mismatches.append((ts, None, actual_lot))
    else:
        print(f"  ❌ {ts:20s} NOT FOUND ON EXCHANGE")
        lot_mismatches.append((ts, config_lot, None))

if lot_mismatches:
    print(f"\n⚠️ LOT SIZE MISMATCHES: {lot_mismatches}")
    print("   These should be updated in options_trader.py FNO_LOT_SIZES")
else:
    print(f"\n✅ All lot sizes match")

# ===================== SUMMARY =====================
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Symbols in universe: {len(APPROVED_UNIVERSE)}")
print(f"  Quotes OK: {len(quote_ok)}/{len(APPROVED_UNIVERSE)}")
print(f"  Instrument tokens OK: {len(token_ok)}/{len(APPROVED_UNIVERSE)}")
print(f"  Historical data OK: {len(hist_ok)}/10 (first 10 tested)")
print(f"  Intraday data OK: {len(intraday_ok)}/10 (first 10 tested)")
print(f"  F&O chains OK: {len(fno_ok)}/{len(fno_prefer)}")
print(f"  Lot size mismatches: {len(lot_mismatches)}")

if quote_fail or token_fail or fno_fail or lot_mismatches:
    print("\n⚠️ ISSUES FOUND - see details above")
else:
    print("\n✅ ALL CHECKS PASSED!")
