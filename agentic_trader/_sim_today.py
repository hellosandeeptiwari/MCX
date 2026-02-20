"""
Simulate Titan's 7 model-tracker trades for today (Feb 19, 2026).
Fetches fresh 5min data from Kite, runs XGB + GMM, applies smart scoring.
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from ml_models.data_fetcher import get_kite_client, fetch_candles
from ml_models.predictor import MovePredictor
from ml_models.feature_engineering import get_sector_for_symbol
from config import DOWN_RISK_GATING

# ── Config ──
TODAY = date(2026, 2, 19)
MAX_TRADES = 7
MAX_PER_SECTOR = 2
MAX_GMM_FLIPS = 2
ENTRY_CANDLE_IDX = 5  # simulate entry ~30 min after open (candle 6 = 09:40)

print("=" * 70)
print(f"  TITAN MODEL-TRACKER SIMULATION — {TODAY}")
print("=" * 70)

# ── Init ──
kite = get_kite_client()
predictor = MovePredictor()

# Get F&O universe
data_dir = 'ml_models/data/candles_5min'
parquets = [f for f in os.listdir(data_dir) 
            if f.endswith('.parquet') and not f.startswith('SECTOR_') and f != 'NIFTY50.parquet']
symbols = [f.replace('.parquet', '') for f in parquets]
print(f"\nUniverse: {len(symbols)} F&O stocks")

# ── Fetch fresh data + Run ML predictions ──
results = []
errors = []
gmm_flip_count = 0

print(f"\nRunning ML predictions (XGB + GMM)...")
t0 = time.time()

for i, sym in enumerate(sorted(symbols)):
    if (i+1) % 50 == 0:
        print(f"  [{i+1}/{len(symbols)}]...")
    
    try:
        # Use existing parquet + fetch any new data
        path = os.path.join(data_dir, f'{sym}.parquet')
        df = pd.read_parquet(path)
        
        # Only use data up to today (inclusive)
        df = df[df['date'].dt.date <= TODAY]
        
        if len(df) < 100:
            continue
        
        # Run prediction (uses last candle features)
        pred = predictor.predict(df, sym)
        if not pred or pred.get('ml_signal') == 'ERROR':
            continue
        
        signal = pred.get('ml_signal', 'UNKNOWN')
        direction = pred.get('ml_direction_bias', 'UNKNOWN')
        ml_conf = pred.get('ml_confidence', 0)
        dir_prob = pred.get('ml_prob_up', 0.5) if direction == 'UP' else pred.get('ml_prob_down', 0.5)
        move_prob = pred.get('ml_move_prob', 0.5)
        score_boost = pred.get('ml_score_boost', 0)
        dr_flag = pred.get('ml_down_risk_flag', False)
        dr_score = pred.get('ml_down_risk_score', 0)
        gmm_confirms = pred.get('ml_gmm_confirms_direction', False)
        gmm_regime = pred.get('ml_gmm_regime_used', None)
        
        # ── XGB + GMM Sync (Direction Decision Tree) ──
        trade_direction = None
        trade_type = 'MODEL_TRACKER'
        skip_reason = None
        
        if direction == 'UP':
            if not dr_flag:
                trade_direction = 'BUY'  # CE
            elif dr_score >= 0.70 and gmm_flip_count < MAX_GMM_FLIPS:
                trade_direction = 'SELL'  # PE via GMM_FLIP
                trade_type = 'GMM_FLIP'
            else:
                skip_reason = f"UP+dr_flag (dr_score={dr_score:.2f})"
                
        elif direction == 'DOWN':
            trade_direction = 'SELL'  # PE
            
        else:
            skip_reason = f"FLAT/UNKNOWN ({direction})"
        
        if skip_reason:
            continue
        
        # ── Confidence floors (skip for GMM_FLIP) ──
        if trade_type != 'GMM_FLIP':
            if dir_prob < 0.50:
                continue
            if ml_conf < 0.40:
                continue
        
        # ── Smart Score Calculation ──
        # Conviction (40%)
        if trade_type == 'GMM_FLIP':
            conviction = dr_score * 40
        else:
            conviction = ml_conf * dir_prob * 40
        
        # Safety (20%)
        if trade_direction == 'SELL':
            safety = dr_score * 20
            if direction == 'DOWN' and gmm_confirms:
                safety += 5  # GMM confirmation bonus
        else:
            safety = (1 - dr_score) * 20
        
        # Technical (20%) - use pre_score approximation from score_boost
        pre_score = max(0, score_boost + 50)  # normalize around 50
        technical = min(pre_score, 100) * 0.20
        
        # Model Agreement (15%)
        model_agree = max(0, score_boost + 8) * (15 / 18)
        model_agree = min(model_agree, 15)
        
        # Move Probability (5%)
        move_component = move_prob * 5
        
        smart_score = conviction + safety + technical + model_agree + move_component
        
        # ── Down-risk soft score adjustment ──
        soft_adj = 0
        safe_bonus = DOWN_RISK_GATING.get('safe_bonus', 5)
        risk_penalty = DOWN_RISK_GATING.get('risk_penalty', 5)
        
        if direction == 'DOWN':
            if gmm_confirms:
                soft_adj = +safe_bonus
            else:
                soft_adj = -(risk_penalty // 2)
        elif direction in ('UP', 'FLAT'):
            if dr_flag:
                soft_adj = -risk_penalty
            else:
                soft_adj = +safe_bonus
        
        sector = get_sector_for_symbol(sym)
        
        # Get today's price data for P&L simulation
        today_df = df[df['date'].dt.date == TODAY].copy()
        
        results.append({
            'symbol': sym,
            'sector': sector,
            'direction': trade_direction,
            'trade_type': trade_type,
            'ml_signal': signal,
            'ml_direction': direction,
            'ml_confidence': round(ml_conf, 3),
            'dir_prob': round(dir_prob, 3),
            'move_prob': round(move_prob, 3),
            'dr_flag': dr_flag,
            'dr_score': round(dr_score, 3),
            'gmm_confirms': gmm_confirms,
            'gmm_regime': gmm_regime,
            'score_boost': score_boost,
            'smart_score': round(smart_score, 1),
            'soft_adj': soft_adj,
            'final_score': round(smart_score + soft_adj, 1),
            'conviction': round(conviction, 1),
            'safety': round(safety, 1),
            'technical': round(technical, 1),
            'model_agree': round(model_agree, 1),
            'move_comp': round(move_component, 1),
            'today_candles': len(today_df),
            'entry_price': today_df.iloc[ENTRY_CANDLE_IDX]['close'] if len(today_df) > ENTRY_CANDLE_IDX else None,
            'eod_price': today_df.iloc[-1]['close'] if len(today_df) > 0 else None,
            'day_high': today_df['high'].max() if len(today_df) > 0 else None,
            'day_low': today_df['low'].min() if len(today_df) > 0 else None,
        })
        
        if trade_type == 'GMM_FLIP':
            gmm_flip_count += 1
            
    except Exception as e:
        errors.append(f"{sym}: {str(e)[:50]}")

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s. {len(results)} candidates, {len(errors)} errors")

if not results:
    print("NO CANDIDATES - check errors:")
    for e in errors[:10]:
        print(f"  {e}")
    sys.exit(1)

# ── Sort by final_score and apply sector diversification ──
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

# ── Display Results ──
print(f"\n{'='*70}")
print(f"  TOP 7 TITAN PICKS (of {len(candidates)} candidates)")
print(f"{'='*70}")

for i, t in enumerate(selected, 1):
    option_type = 'CE' if t['direction'] == 'BUY' else 'PE'
    gmm_tag = ' [GMM_FLIP]' if t['trade_type'] == 'GMM_FLIP' else ''
    gmm_conf_tag = ' GMM✓' if t['gmm_confirms'] else ''
    dr_tag = f" DR={t['dr_score']:.2f}" if t['dr_flag'] else ''
    
    print(f"\n  #{i} {t['symbol']:<12s} {option_type} ({t['direction']}){gmm_tag}")
    print(f"     Sector: {t['sector']}")
    print(f"     XGB Signal: {t['ml_signal']} → {t['ml_direction']} (conf={t['ml_confidence']:.1%}, dir_prob={t['dir_prob']:.1%})")
    print(f"     Move prob: {t['move_prob']:.1%}")
    print(f"     Down-risk: flag={t['dr_flag']}, score={t['dr_score']:.3f}{gmm_conf_tag}{dr_tag}")
    print(f"     GMM regime: {t['gmm_regime']}")
    print(f"     Smart Score: {t['final_score']:.1f} (raw={t['smart_score']:.1f}, soft_adj={t['soft_adj']:+d})")
    print(f"       Conv={t['conviction']:.1f} Safe={t['safety']:.1f} Tech={t['technical']:.1f} Agree={t['model_agree']:.1f} Move={t['move_comp']:.1f}")
    
    # P&L simulation
    if t['entry_price'] and t['eod_price']:
        entry = t['entry_price']
        eod = t['eod_price']
        
        if t['direction'] == 'BUY':
            pnl_pct = (eod - entry) / entry * 100
            best_pct = (t['day_high'] - entry) / entry * 100
            worst_pct = (t['day_low'] - entry) / entry * 100
        else:  # SELL
            pnl_pct = (entry - eod) / entry * 100
            best_pct = (entry - t['day_low']) / entry * 100
            worst_pct = (entry - t['day_high']) / entry * 100
        
        win = '✅' if pnl_pct > 0 else '❌'
        print(f"     Price: entry@09:40={entry:.1f} → EOD={eod:.1f}")
        print(f"     Stock P&L: {pnl_pct:+.2f}% {win}  (best: {best_pct:+.2f}%, worst: {worst_pct:+.2f}%)")

# ── Summary ──
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")

buys = [t for t in selected if t['direction'] == 'BUY']
sells = [t for t in selected if t['direction'] == 'SELL']
gmm_flips = [t for t in selected if t['trade_type'] == 'GMM_FLIP']

print(f"  CE trades (BUY):  {len(buys)}")
print(f"  PE trades (SELL): {len(sells)} (of which {len(gmm_flips)} GMM_FLIP)")
print(f"  Sectors: {dict(sector_counts)}")
print(f"  Score range: {selected[-1]['final_score']:.1f} - {selected[0]['final_score']:.1f}")

# P&L summary
total_pnl = 0
wins = 0
losses = 0
for t in selected:
    if t['entry_price'] and t['eod_price']:
        entry = t['entry_price']
        eod = t['eod_price']
        if t['direction'] == 'BUY':
            pnl = (eod - entry) / entry * 100
        else:
            pnl = (entry - eod) / entry * 100
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        else:
            losses += 1

print(f"\n  Stock-level P&L (directional, not options):")
print(f"  Win/Loss: {wins}W / {losses}L")
print(f"  Avg per trade: {total_pnl/max(len(selected),1):+.2f}%")
print(f"  Total directional: {total_pnl:+.2f}%")

# Also show rejected candidates that were close
print(f"\n{'='*70}")
print(f"  NEXT 5 RUNNERS-UP (not selected)")
print(f"{'='*70}")

runners_up = [c for c in candidates if c not in selected][:5]
for i, t in enumerate(runners_up, 1):
    option_type = 'CE' if t['direction'] == 'BUY' else 'PE'
    gmm_tag = ' [GMM_FLIP]' if t['trade_type'] == 'GMM_FLIP' else ''
    print(f"  #{i} {t['symbol']:<12s} {option_type}{gmm_tag} score={t['final_score']:.1f} "
          f"sector={t['sector']} conf={t['ml_confidence']:.1%} dir={t['dir_prob']:.1%} dr={t['dr_score']:.3f}")

# Full candidate stats
print(f"\n  Total candidates: {len(candidates)}")
print(f"  Direction split: {sum(1 for c in candidates if c['direction']=='BUY')} BUY / {sum(1 for c in candidates if c['direction']=='SELL')} SELL")
print(f"  GMM confirms: {sum(1 for c in candidates if c['gmm_confirms'])}")
print(f"  DR flagged: {sum(1 for c in candidates if c['dr_flag'])}")
