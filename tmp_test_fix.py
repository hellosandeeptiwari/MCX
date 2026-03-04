"""Test prediction with market regime adjustment."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from ml_models.predictor import MovePredictor
import logging

# Setup logger to see regime adjustment messages
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

pred = MovePredictor(logger=logger)

# Load data
candles_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data/candles_5min'
nifty_5m = pd.read_parquet(os.path.join(candles_dir, 'NIFTY50.parquet'))
nifty_5m['date'] = pd.to_datetime(nifty_5m['date'])

nifty_d_path = '/home/ubuntu/titan/agentic_trader/ml_models/data/candles_daily/NIFTY50.parquet'
nifty_daily = pd.read_parquet(nifty_d_path)
nifty_daily['date'] = pd.to_datetime(nifty_daily['date'])
print(f'NIFTY daily rows: {len(nifty_daily)} (need >=20 for adaptive EMA)')

oi_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data/futures_oi'
test_stocks = ['RELIANCE', 'TATASTEEL', 'SBIN', 'HDFCBANK', 'INFY', 'LT', 'ADANIENT']

up_count = 0
down_count = 0
flat_count = 0

for sym in test_stocks:
    try:
        candles = pd.read_parquet(os.path.join(candles_dir, f'{sym}.parquet'))
        candles['date'] = pd.to_datetime(candles['date'])
        
        oi_path = os.path.join(oi_dir, f'{sym}_futures_oi.parquet')
        oi_df = pd.read_parquet(oi_path) if os.path.exists(oi_path) else None
        if oi_df is not None:
            oi_df['date'] = pd.to_datetime(oi_df['date'])
        
        result = pred.predict(candles, nifty_5min_df=nifty_5m, nifty_daily_df=nifty_daily, futures_oi_df=oi_df)
        
        if result:
            sig = result.get('ml_signal', 'FLAT')
            if sig == 'UP': up_count += 1
            elif sig == 'DOWN': down_count += 1
            else: flat_count += 1
            
            print(f'\n{sym}: Signal={sig} | P(UP)={result.get("ml_prob_up",0):.4f} P(DOWN)={result.get("ml_prob_down",0):.4f} '
                  f'| P(UP|MOVE)={result.get("ml_p_up_given_move",0):.4f} | regime_adj={result.get("ml_regime_adj",0):+.4f}')
            
            # Show key features
            feat = result.get('_features_array')
            if feat is not None:
                for name in ['nifty_daily_trend', 'nifty_daily_rsi', 'nifty_roc_6', 'fut_oi_buildup']:
                    if name in pred.feature_names:
                        idx = pred.feature_names.index(name)
                        print(f'  {name}: {feat[0,idx]:.4f}')
        else:
            print(f'{sym}: No prediction')
    except Exception as e:
        print(f'{sym}: ERROR - {e}')

print(f'\n=== SUMMARY ===')
print(f'UP: {up_count}, DOWN: {down_count}, FLAT: {flat_count}')
print(f'BEFORE fix: 3/3 UP (RELIANCE=UP, TATASTEEL=UP, SBIN=UP)')
print(f'AFTER fix:  {up_count} UP, {down_count} DOWN, {flat_count} FLAT out of {len(test_stocks)}')
