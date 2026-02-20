"""
Simulate Titan model-tracker trades with ACTUAL option premium data.
Uses Kite historical API to fetch ATM CE/PE 5min candle data.

Exit rules (from exit_manager.py):
  - Quick profit: +18% premium
  - Stop loss: -28% premium (SL = entry * 0.72)
  - Time stop: 10 candles (~50 min) if max R < 0.3R
  - Speed gate: 12 candles (~60 min) if max premium < +3%
  - Session cutoff: 15:15
"""
import sys, os, warnings, time, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

TODAY = date(2026, 2, 19)
ENTRY_CANDLE = 5  # index 5 = 09:40 candle close

# Exit thresholds
QUICK_PROFIT_PCT = 0.18    # +18%
STOPLOSS_PCT = -0.28       # -28%
TIME_STOP_CANDLES = 10     # 50 min
TIME_STOP_MIN_R = 0.30     # need +0.3R in 10 candles
SPEED_GATE_CANDLES = 12    # 60 min
SPEED_GATE_MIN_PCT = 0.03  # need +3% premium in 12 candles

print('='*70)
print('  TITAN OPTIONS SIMULATION (ACTUAL PREMIUMS) - ' + str(TODAY))
print('='*70)

# Get Kite client
from ml_models.data_fetcher import get_kite_client, get_instrument_token
kite = get_kite_client()
if not kite:
    print('ERROR: No Kite client')
    sys.exit(1)
print('Kite connected.')

# Get NFO instruments for option lookup
print('Loading NFO instruments...')
nfo = kite.instruments('NFO')
print('  ' + str(len(nfo)) + ' NFO instruments loaded')

# Our 7 CE picks + top 7 PE picks from the simulation
picks = [
    # CE BUY (UP signal)
    {'symbol': 'RECLTD',     'direction': 'BUY', 'option': 'CE', 'score': 73.2},
    {'symbol': 'PFC',        'direction': 'BUY', 'option': 'CE', 'score': 71.5},
    {'symbol': 'BEL',        'direction': 'BUY', 'option': 'CE', 'score': 70.8},
    {'symbol': 'HAL',        'direction': 'BUY', 'option': 'CE', 'score': 68.9},
    {'symbol': 'GODREJCP',   'direction': 'BUY', 'option': 'CE', 'score': 68.6},
    {'symbol': 'TATACONSUM', 'direction': 'BUY', 'option': 'CE', 'score': 68.6},
    {'symbol': 'DLF',        'direction': 'BUY', 'option': 'CE', 'score': 68.4},
]

# Also add top PE picks to see what would have happened  
pe_picks = [
    {'symbol': 'ADANIGREEN', 'direction': 'SELL', 'option': 'PE', 'score': 55.3},
    {'symbol': 'JUBLFOOD',   'direction': 'SELL', 'option': 'PE', 'score': 53.7},
    {'symbol': 'ASTRAL',     'direction': 'SELL', 'option': 'PE', 'score': 53.2},
    {'symbol': 'GLENMARK',   'direction': 'SELL', 'option': 'PE', 'score': 52.0},
    {'symbol': 'IEX',        'direction': 'SELL', 'option': 'PE', 'score': 50.2},
    {'symbol': 'KPITTECH',   'direction': 'SELL', 'option': 'PE', 'score': 49.3},
    {'symbol': 'CAMS',       'direction': 'SELL', 'option': 'PE', 'score': 49.2},
]

def find_atm_option(sym, option_type, spot_price, nfo_instruments):
    """Find ATM option instrument for nearest weekly expiry."""
    # Filter for this symbol's options
    opts = [i for i in nfo_instruments
            if i['name'] == sym 
            and i['instrument_type'] == option_type  # 'CE' or 'PE'
            and i['segment'] == 'NFO-OPT']
    
    if not opts:
        return None
    
    # Find nearest expiry >= today
    future_opts = [o for o in opts if o['expiry'] >= TODAY]
    if not future_opts:
        return None
    
    # Group by expiry, pick nearest
    expiries = sorted(set(o['expiry'] for o in future_opts))
    
    # Current week = nearest expiry within 7 days
    nearest_expiry = None
    for exp in expiries:
        if (exp - TODAY).days <= 7:
            nearest_expiry = exp
            break
    if nearest_expiry is None:
        nearest_expiry = expiries[0]
    
    # Filter to this expiry
    exp_opts = [o for o in future_opts if o['expiry'] == nearest_expiry]
    
    # Find ATM strike
    strikes = sorted(set(o['strike'] for o in exp_opts))
    if not strikes:
        return None
    
    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    
    # Get the instrument
    match = [o for o in exp_opts if o['strike'] == atm_strike]
    if not match:
        return None
    
    inst = match[0]
    return {
        'token': inst['instrument_token'],
        'tradingsymbol': inst['tradingsymbol'],
        'strike': atm_strike,
        'expiry': str(nearest_expiry),
        'lot_size': inst['lot_size'],
    }

def simulate_option_trade(kite, inst, entry_candle_idx, direction):
    """Simulate option trade using actual 5min premium data.
    
    Returns dict with entry, exit, P&L, exit reason, candles held.
    """
    token = inst['token']
    
    # Fetch option's 5min candles for today
    data = kite.historical_data(token, TODAY, TODAY, '5minute')
    if not data or len(data) <= entry_candle_idx:
        return {'error': 'No option data (' + str(len(data) if data else 0) + ' candles)'}
    
    # Entry at close of entry candle
    entry_price = data[entry_candle_idx]['close']
    if entry_price <= 0:
        return {'error': 'Zero entry price'}
    
    # SL and target
    sl_price = entry_price * (1 + STOPLOSS_PCT)   # -28% = * 0.72
    target_price = entry_price * (1 + QUICK_PROFIT_PCT)  # +18% = * 1.18
    
    # Walk through subsequent candles
    exit_price = None
    exit_reason = None
    exit_candle = None
    max_premium = entry_price
    max_pct = 0
    
    for j in range(entry_candle_idx + 1, len(data)):
        candle = data[j]
        high = candle['high']
        low = candle['low']
        close = candle['close']
        candle_time = candle['date']
        candles_held = j - entry_candle_idx
        
        # Track max premium seen
        if high > max_premium:
            max_premium = high
            max_pct = (max_premium - entry_price) / entry_price
        
        # Check session cutoff (15:15)
        if hasattr(candle_time, 'hour'):
            if candle_time.hour >= 15 and candle_time.minute >= 15:
                exit_price = close
                exit_reason = 'SESSION_CUTOFF'
                exit_candle = j
                break
        
        # Check stop loss hit (intra-candle)
        if low <= sl_price:
            exit_price = sl_price  # Assume SL fills at SL price
            exit_reason = 'STOP_LOSS (-28%)'
            exit_candle = j
            break
        
        # Check quick profit hit (intra-candle)
        if high >= target_price:
            exit_price = target_price  # Assume target fills at target
            exit_reason = 'QUICK_PROFIT (+18%)'
            exit_candle = j
            break
        
        # Time stop: 10 candles, need +0.3R (approximate as +9% for options)
        if candles_held >= TIME_STOP_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < 0.09:  # max premium never reached +9%
                exit_price = close
                exit_reason = 'TIME_STOP (10 candles, max +' + str(round(pct_gain*100, 1)) + '%)'
                exit_candle = j
                break
        
        # Speed gate: 12 candles, need +3% premium
        if candles_held >= SPEED_GATE_CANDLES:
            pct_gain = (max_premium - entry_price) / entry_price
            if pct_gain < SPEED_GATE_MIN_PCT:
                exit_price = close
                exit_reason = 'SPEED_GATE (12 candles, max +' + str(round(pct_gain*100, 1)) + '%)'
                exit_candle = j
                break
    
    if exit_price is None:
        # Didn't hit any exit — close at last candle (EOD)
        exit_price = data[-1]['close']
        exit_reason = 'EOD_CLOSE'
        exit_candle = len(data) - 1
    
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    candles_held = exit_candle - entry_candle_idx
    
    return {
        'entry_price': round(entry_price, 2),
        'exit_price': round(exit_price, 2),
        'pnl_pct': round(pnl_pct, 2),
        'pnl_abs': round(exit_price - entry_price, 2),
        'exit_reason': exit_reason,
        'candles_held': candles_held,
        'time_held_min': candles_held * 5,
        'max_premium': round(max_premium, 2),
        'max_pct': round((max_premium - entry_price) / entry_price * 100, 2),
        'total_candles': len(data),
    }


# ── Run simulation for all picks ──

def run_simulation(picks, label):
    print('\n' + '='*70)
    print('  ' + label)
    print('='*70)
    
    results = []
    
    for pick in picks:
        sym = pick['symbol']
        opt_type = pick['option']
        
        print('\n  ' + sym + ' ' + opt_type + ' BUY (score=' + str(pick['score']) + ')')
        
        # Get spot price at entry time
        spot_token = get_instrument_token(kite, sym)
        if spot_token == 0:
            print('    SKIP: No spot token')
            continue
        
        time.sleep(0.3)
        spot_data = kite.historical_data(spot_token, TODAY, TODAY, '5minute')
        if not spot_data or len(spot_data) <= ENTRY_CANDLE:
            print('    SKIP: No spot data')
            continue
        
        spot_at_entry = spot_data[ENTRY_CANDLE]['close']
        print('    Spot@09:40 = ' + str(round(spot_at_entry, 1)))
        
        # Find ATM option
        inst = find_atm_option(sym, opt_type, spot_at_entry, nfo)
        if not inst:
            print('    SKIP: No ATM ' + opt_type + ' option found')
            continue
        
        print('    Option: ' + inst['tradingsymbol'] + ' (strike=' + str(inst['strike']) + 
              ', exp=' + inst['expiry'] + ', lot=' + str(inst['lot_size']) + ')')
        
        # Simulate the trade
        time.sleep(0.3)
        result = simulate_option_trade(kite, inst, ENTRY_CANDLE, pick['direction'])
        
        if 'error' in result:
            print('    ERROR: ' + result['error'])
            continue
        
        # Position sizing: ~2% risk on 5L capital = 10K risk per trade
        # Premium per lot = entry_price * lot_size
        lot_size = inst['lot_size']
        premium_per_lot = result['entry_price'] * lot_size
        capital = 500000
        risk_per_trade = capital * 0.02  # 2% = 10K
        max_lots = max(1, int(risk_per_trade / (premium_per_lot * 0.28)))  # 28% SL
        lots = min(max_lots, 15)
        total_premium = result['entry_price'] * lot_size * lots
        pnl_rupees = result['pnl_abs'] * lot_size * lots
        
        tag = 'WIN' if result['pnl_pct'] > 0 else 'LOSS'
        
        print('    Entry: ' + str(result['entry_price']) + ' -> Exit: ' + str(result['exit_price']))
        print('    Premium P&L: ' + ('+' if result['pnl_pct'] >= 0 else '') + str(result['pnl_pct']) + '% ' + tag)
        print('    Exit reason: ' + result['exit_reason'])
        print('    Held: ' + str(result['candles_held']) + ' candles (' + str(result['time_held_min']) + ' min)')
        print('    Max premium: ' + str(result['max_premium']) + ' (+' + str(result['max_pct']) + '%)')
        print('    Size: ' + str(lots) + ' lots x ' + str(lot_size) + ' = ' + str(lots * lot_size) + ' qty')
        print('    P&L (Rs): ' + ('+' if pnl_rupees >= 0 else '') + str(round(pnl_rupees)) + 
              ' (invested: ' + str(round(total_premium)) + ')')
        
        results.append({
            'symbol': sym, 'option': opt_type, 'score': pick['score'],
            'strike': inst['strike'], 'expiry': inst['expiry'],
            'tradingsymbol': inst['tradingsymbol'],
            'entry': result['entry_price'], 'exit': result['exit_price'],
            'pnl_pct': result['pnl_pct'], 'pnl_rs': round(pnl_rupees),
            'exit_reason': result['exit_reason'],
            'candles_held': result['candles_held'],
            'max_pct': result['max_pct'], 'lots': lots, 'lot_size': lot_size,
            'invested': round(total_premium),
        })
    
    # Summary
    if results:
        print('\n  ' + '-'*60)
        print('  ' + label + ' SUMMARY')
        print('  ' + '-'*60)
        
        wins = sum(1 for r in results if r['pnl_pct'] > 0)
        losses = len(results) - wins
        total_rs = sum(r['pnl_rs'] for r in results)
        total_invested = sum(r['invested'] for r in results)
        avg_pnl = sum(r['pnl_pct'] for r in results) / len(results)
        avg_hold = sum(r['candles_held'] for r in results) / len(results)
        
        for r in results:
            tag = 'WIN' if r['pnl_pct'] > 0 else 'LOSS'
            print('    ' + r['symbol'] + ' ' + r['option'] + ' ' + r['tradingsymbol'] + 
                  ': ' + ('+' if r['pnl_pct'] >= 0 else '') + str(r['pnl_pct']) + '% = Rs ' + 
                  ('+' if r['pnl_rs'] >= 0 else '') + str(r['pnl_rs']) + 
                  ' (' + r['exit_reason'].split('(')[0].strip() + ', ' + str(r['candles_held']) + ' candles) ' + tag)
        
        print()
        print('    Win/Loss: ' + str(wins) + 'W / ' + str(losses) + 'L')
        print('    Total P&L: Rs ' + ('+' if total_rs >= 0 else '') + str(total_rs) + 
              ' on Rs ' + str(total_invested) + ' invested')
        print('    Avg P&L: ' + ('+' if avg_pnl >= 0 else '') + str(round(avg_pnl, 2)) + '% per trade')
        print('    Avg hold: ' + str(round(avg_hold, 1)) + ' candles (' + str(round(avg_hold * 5, 0)) + ' min)')
        
        by_reason = {}
        for r in results:
            reason = r['exit_reason'].split('(')[0].strip()
            by_reason[reason] = by_reason.get(reason, 0) + 1
        print('    Exit reasons: ' + str(by_reason))
    
    return results


# Run both groups
ce_results = run_simulation(picks, '7 CE BUY TRADES (Model Selected)')
pe_results = run_simulation(pe_picks, '7 PE BUY TRADES (What-If)')

# Combined comparison
print('\n' + '='*70)
print('  FINAL COMPARISON')
print('='*70)

if ce_results:
    ce_total = sum(r['pnl_rs'] for r in ce_results)
    ce_pct = sum(r['pnl_pct'] for r in ce_results) / len(ce_results)
    print('  CE (actual picks): Rs ' + ('+' if ce_total >= 0 else '') + str(ce_total) + 
          ' | avg ' + ('+' if ce_pct >= 0 else '') + str(round(ce_pct, 2)) + '%')

if pe_results:
    pe_total = sum(r['pnl_rs'] for r in pe_results)
    pe_pct = sum(r['pnl_pct'] for r in pe_results) / len(pe_results)
    print('  PE (what-if):      Rs ' + ('+' if pe_total >= 0 else '') + str(pe_total) + 
          ' | avg ' + ('+' if pe_pct >= 0 else '') + str(round(pe_pct, 2)) + '%')

if ce_results and pe_results:
    diff = pe_total - ce_total
    print('  Difference: Rs ' + ('+' if diff >= 0 else '') + str(diff) + ' in favor of ' + 
          ('PE' if diff > 0 else 'CE'))
