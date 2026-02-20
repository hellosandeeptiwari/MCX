"""
Simulate Titan's 7 model-tracker trades for today (Feb 19, 2026).
Uses parquet data (no Kite ticker needed), runs XGB + GMM, applies smart scoring.
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
os.environ['KITE_NO_TICKER'] = '1'  # prevent ticker from connecting

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from ml_models.predictor import MovePredictor
from ml_models.feature_engineering import get_sector_for_symbol, compute_features
from config import DOWN_RISK_GATING

# Config
TODAY = date(2026, 2, 19)
MAX_TRADES = 7
MAX_PER_SECTOR = 2
MAX_GMM_FLIPS = 2
ENTRY_CANDLE_IDX = 5  # ~09:40

print("=" * 70)
print("  TITAN MODEL-TRACKER SIMULATION -- " + str(TODAY))
print("=" * 70)

predictor = MovePredictor()
print("Predictor ready: " + str(predictor.ready) + ", type: " + str(predictor.model_type))

data_dir = 'ml_models/data/candles_5min'
foi_dir = 'ml_models/data/futures_oi'
parquets = [f for f in os.listdir(data_dir) 
            if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
symbols = sorted([f.replace('.parquet', '') for f in parquets])
print("Universe: " + str(len(symbols)) + " F&O stocks")

# Load NIFTY context
nifty_path = os.path.join(data_dir, 'NIFTY50.parquet')
nifty_df = pd.read_parquet(nifty_path) if os.path.exists(nifty_path) else None

# Run predictions
results = []
errors = []
skips = {'flat_unknown': 0, 'up_dr_block': 0, 'dir_prob_low': 0, 'conf_low': 0, 'predict_fail': 0, 'no_data': 0}
gmm_flip_count = 0
t0 = time.time()

print("\nRunning ML predictions (XGB + GMM)...")

for i, sym in enumerate(symbols):
    if (i+1) % 50 == 0:
        print("  [" + str(i+1) + "/" + str(len(symbols)) + "]...")
    
    try:
        path = os.path.join(data_dir, sym + '.parquet')
        df = pd.read_parquet(path)
        df = df[df['date'].dt.date <= TODAY]
        
        if len(df) < 100:
            skips['no_data'] += 1
            continue
        
        # Load futures OI if available
        foi_path = os.path.join(foi_dir, sym + '_futures_oi.parquet')
        foi_df = pd.read_parquet(foi_path) if os.path.exists(foi_path) else None
        
        # Get sector and sector index
        sector = get_sector_for_symbol(sym)
        sector_key = 'SECTOR_' + sector.upper().replace(' ', '_') if sector != 'Other' else None
        sector_df = None
        if sector_key:
            sp = os.path.join(data_dir, sector_key + '.parquet')
            if os.path.exists(sp):
                sector_df = pd.read_parquet(sp)
        
        pred = predictor.get_titan_signals(df, futures_oi_df=foi_df, 
                                  nifty_5min_df=nifty_df, 
                                  sector_5min_df=sector_df)
        if not pred or pred.get('ml_signal') == 'ERROR':
            skips['predict_fail'] += 1
            continue
        
        signal = pred.get('ml_signal', 'UNKNOWN')
        ml_conf = pred.get('ml_confidence', 0)
        dir_prob_up = pred.get('ml_prob_up', 0.5)
        dir_prob_down = pred.get('ml_prob_down', 0.5)
        move_prob = pred.get('ml_move_prob', 0.5)
        score_boost = pred.get('ml_score_boost', 0)
        dr_flag = pred.get('ml_down_risk_flag', False)
        dr_score = pred.get('ml_down_risk_score', 0)
        gmm_confirms = pred.get('ml_gmm_confirms_direction', False)
        gmm_regime = pred.get('ml_gmm_regime_used', None)
        
        # XGB + GMM Sync Decision Tree (use ml_signal for direction)
        trade_direction = None
        trade_type = 'MODEL_TRACKER'
        
        if signal == 'UP':
            if not dr_flag:
                trade_direction = 'BUY'
            elif dr_score >= 0.70 and gmm_flip_count < MAX_GMM_FLIPS:
                trade_direction = 'SELL'
                trade_type = 'GMM_FLIP'
            else:
                skips['up_dr_block'] += 1
                continue
        elif signal == 'DOWN':
            trade_direction = 'SELL'
        else:
            skips['flat_unknown'] += 1
            continue
        
        # Direction-specific probability
        if trade_type == 'GMM_FLIP':
            dir_prob = min(dr_score, 1.0)
        elif trade_direction == 'BUY':
            dir_prob = dir_prob_up
        else:
            dir_prob = dir_prob_down
        
        # Confidence floors (bypass for GMM_FLIP)
        if trade_type != 'GMM_FLIP':
            if dir_prob < 0.50:
                skips['dir_prob_low'] += 1
                continue
            if ml_conf < 0.40:
                skips['conf_low'] += 1
                continue
        
        # Smart Score
        if trade_type == 'GMM_FLIP':
            conviction = dr_score * 40
        else:
            conviction = ml_conf * dir_prob * 40
        
        if trade_direction == 'SELL':
            safety = dr_score * 20
            if signal == 'DOWN' and gmm_confirms:
                safety += 5
        else:
            safety = (1 - dr_score) * 20
        
        pre_score = max(0, score_boost + 50)
        technical = min(pre_score, 100) * 0.20
        
        model_agree = max(0, score_boost + 8) * (15 / 18)
        model_agree = min(model_agree, 15)
        
        move_component = move_prob * 5
        
        smart_score = conviction + safety + technical + model_agree + move_component
        
        # Soft adjustment
        safe_bonus = DOWN_RISK_GATING.get('safe_bonus', 5)
        risk_penalty = DOWN_RISK_GATING.get('risk_penalty', 5)
        
        if signal == 'DOWN':
            soft_adj = safe_bonus if gmm_confirms else -(risk_penalty // 2)
        elif signal in ('UP', 'FLAT'):
            soft_adj = -risk_penalty if dr_flag else safe_bonus
        else:
            soft_adj = 0
        
        # Today's price data for P&L
        today_df = df[df['date'].dt.date == TODAY].copy()
        entry_price = today_df.iloc[ENTRY_CANDLE_IDX]['close'] if len(today_df) > ENTRY_CANDLE_IDX else None
        eod_price = today_df.iloc[-1]['close'] if len(today_df) > 0 else None
        day_high = today_df['high'].max() if len(today_df) > 0 else None
        day_low = today_df['low'].min() if len(today_df) > 0 else None
        
        results.append({
            'symbol': sym, 'sector': sector, 'direction': trade_direction,
            'trade_type': trade_type, 'ml_signal': signal, 'ml_direction': signal,
            'ml_confidence': round(ml_conf, 3), 'dir_prob': round(dir_prob, 3),
            'move_prob': round(move_prob, 3), 'dr_flag': dr_flag,
            'dr_score': round(dr_score, 3), 'gmm_confirms': gmm_confirms,
            'gmm_regime': gmm_regime, 'score_boost': score_boost,
            'smart_score': round(smart_score, 1), 'soft_adj': soft_adj,
            'final_score': round(smart_score + soft_adj, 1),
            'conviction': round(conviction, 1), 'safety': round(safety, 1),
            'technical': round(technical, 1), 'model_agree': round(model_agree, 1),
            'move_comp': round(move_component, 1),
            'today_candles': len(today_df),
            'entry_price': entry_price, 'eod_price': eod_price,
            'day_high': day_high, 'day_low': day_low,
        })
        
        if trade_type == 'GMM_FLIP':
            gmm_flip_count += 1
            
    except Exception as e:
        errors.append(sym + ": " + str(e)[:50])

elapsed = time.time() - t0
print("Done in " + str(round(elapsed,1)) + "s. " + str(len(results)) + " candidates, " + str(len(errors)) + " errors")
print("Skip reasons: " + str(skips))
if errors:
    print("First 5 errors: " + str(errors[:5]))

if not results:
    print("\nNO CANDIDATES")
    sys.exit(1)

# Sort by final_score + sector diversification
candidates = sorted(results, key=lambda x: x['final_score'], reverse=True)
selected = []
sector_counts = {}

for c in candidates:
    if len(selected) >= MAX_TRADES:
        break
    sec = c['sector']
    if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
        continue
    selected.append(c)
    sector_counts[sec] = sector_counts.get(sec, 0) + 1

# Display
print("\n" + "=" * 70)
print("  TOP 7 TITAN PICKS (of " + str(len(candidates)) + " candidates)")
print("=" * 70)

for i, t in enumerate(selected, 1):
    opt = 'CE' if t['direction'] == 'BUY' else 'PE'
    flip_tag = ' [GMM_FLIP]' if t['trade_type'] == 'GMM_FLIP' else ''
    gmm_tag = ' GMM-YES' if t['gmm_confirms'] else ''
    
    print("\n  #" + str(i) + " " + t['symbol'] + " " + opt + " (" + t['direction'] + ")" + flip_tag)
    print("     Sector: " + t['sector'])
    print("     XGB: " + t['ml_signal'] + " -> " + t['ml_direction'] + 
          " (conf=" + str(round(t['ml_confidence']*100)) + "%, dir_prob=" + str(round(t['dir_prob']*100)) + 
          "%, move=" + str(round(t['move_prob']*100)) + "%)")
    print("     Down-risk: flag=" + str(t['dr_flag']) + " score=" + str(t['dr_score']) + 
          " regime=" + str(t['gmm_regime']) + gmm_tag)
    print("     Smart Score: " + str(t['final_score']) + " (raw=" + str(t['smart_score']) + 
          " soft=" + ("+" if t['soft_adj']>=0 else "") + str(t['soft_adj']) + ")")
    print("       Conv=" + str(t['conviction']) + " Safe=" + str(t['safety']) + 
          " Tech=" + str(t['technical']) + " Agree=" + str(t['model_agree']) + 
          " Move=" + str(t['move_comp']))
    
    if t['entry_price'] and t['eod_price']:
        entry = t['entry_price']
        eod = t['eod_price']
        if t['direction'] == 'BUY':
            pnl_pct = (eod - entry) / entry * 100
            best = (t['day_high'] - entry) / entry * 100
            worst = (t['day_low'] - entry) / entry * 100
        else:
            pnl_pct = (entry - eod) / entry * 100
            best = (entry - t['day_low']) / entry * 100
            worst = (entry - t['day_high']) / entry * 100
        
        win = 'WIN' if pnl_pct > 0 else 'LOSS'
        print("     Price: entry@09:40=" + str(round(entry,1)) + " -> EOD=" + str(round(eod,1)))
        print("     Stock P&L: " + ("+" if pnl_pct>=0 else "") + str(round(pnl_pct,2)) + "% " + win + 
              "  (best: +" + str(round(best,2)) + "%, worst: " + str(round(worst,2)) + "%)")
    else:
        print("     NO TODAY DATA for P&L simulation")

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

buys = [t for t in selected if t['direction'] == 'BUY']
sells = [t for t in selected if t['direction'] == 'SELL']
flips = [t for t in selected if t['trade_type'] == 'GMM_FLIP']
print("  CE trades (BUY):  " + str(len(buys)))
print("  PE trades (SELL): " + str(len(sells)) + " (GMM_FLIP: " + str(len(flips)) + ")")
print("  Sectors: " + str(dict(sector_counts)))

total_pnl = 0
wins = 0
losses = 0
for t in selected:
    if t['entry_price'] and t['eod_price']:
        e = t['entry_price']
        eod = t['eod_price']
        pnl = ((eod - e) / e * 100) if t['direction'] == 'BUY' else ((e - eod) / e * 100)
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1

print("\n  Stock-level P&L (directional):")
print("  Win/Loss: " + str(wins) + "W / " + str(losses) + "L")
print("  Avg per trade: " + ("+" if total_pnl/max(len(selected),1)>=0 else "") + 
      str(round(total_pnl/max(len(selected),1),2)) + "%")
print("  Total directional: " + ("+" if total_pnl>=0 else "") + str(round(total_pnl,2)) + "%")

# Runners-up
print("\n  NEXT 5 RUNNERS-UP:")
runners = [c for c in candidates if c not in selected][:5]
for i, t in enumerate(runners, 1):
    opt = 'CE' if t['direction'] == 'BUY' else 'PE'
    print("  " + str(i) + ". " + t['symbol'] + " " + opt + " score=" + str(t['final_score']) + 
          " sector=" + t['sector'] + " conf=" + str(round(t['ml_confidence']*100)) + 
          "% dr=" + str(t['dr_score']))

print("\n  Total candidates: " + str(len(candidates)))
print("  BUY/SELL split: " + str(sum(1 for c in candidates if c['direction']=='BUY')) + 
      " / " + str(sum(1 for c in candidates if c['direction']=='SELL')))
print("  GMM confirms: " + str(sum(1 for c in candidates if c['gmm_confirms'])))
print("  DR flagged: " + str(sum(1 for c in candidates if c['dr_flag'])))
