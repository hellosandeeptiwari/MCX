"""Run a sample prediction and dump all feature values to see what the model sees."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')

from ml_models.predictor import MovePredictor

# Load the predictor
pred = MovePredictor()

# Load sample stock candles + NIFTY data  
candles_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data/candles_5min'
nifty_5m = pd.read_parquet(os.path.join(candles_dir, 'NIFTY50.parquet'))
nifty_5m['date'] = pd.to_datetime(nifty_5m['date'])

nifty_d_path = '/home/ubuntu/titan/agentic_trader/ml_models/data/candles_daily/NIFTY50.parquet'
nifty_daily = pd.read_parquet(nifty_d_path)
nifty_daily['date'] = pd.to_datetime(nifty_daily['date'])

print(f'NIFTY 5min: {len(nifty_5m)} candles, last={nifty_5m["date"].max()}, last close={nifty_5m["close"].iloc[-1]:.1f}')
print(f'NIFTY daily: {len(nifty_daily)} days, last={nifty_daily["date"].max()}')

# Test with a bearish stock
test_stocks = ['RELIANCE', 'TATASTEEL', 'SBIN']
oi_dir = '/home/ubuntu/titan/agentic_trader/ml_models/data/futures_oi'

for sym in test_stocks:
    try:
        candles = pd.read_parquet(os.path.join(candles_dir, f'{sym}.parquet'))
        candles['date'] = pd.to_datetime(candles['date'])
        
        oi_path = os.path.join(oi_dir, f'{sym}_futures_oi.parquet')
        oi_df = pd.read_parquet(oi_path) if os.path.exists(oi_path) else None
        if oi_df is not None:
            oi_df['date'] = pd.to_datetime(oi_df['date'])
        
        # Run prediction
        result = pred.predict(candles, nifty_5min_df=nifty_5m, nifty_daily_df=nifty_daily, futures_oi_df=oi_df)
        
        if result:
            print(f'\n=== {sym} ===')
            print(f'  Signal: {result.get("ml_signal")} | P(UP)={result.get("ml_prob_up",0):.4f} P(DOWN)={result.get("ml_prob_down",0):.4f}')
            print(f'  P(MOVE)={result.get("ml_p_move",0):.4f} P(UP|MOVE)={result.get("ml_p_up_given_move",0):.4f}')
            print(f'  Boost={result.get("ml_score_boost",0)}')
            
            # Extract feature values for the direction model features
            feat_array = result.get('_features_array')
            if feat_array is not None:
                # Show NIFTY and OI features
                key_features = ['fut_oi_buildup', 'oi_price_confirm', 'fut_basis_pct', 'fut_oi_5d_trend',
                               'fut_oi_change_pct', 'nifty_roc_6', 'nifty_rsi_14', 'nifty_bb_position',
                               'nifty_ema9_slope', 'nifty_atr_pct', 'nifty_daily_trend', 'nifty_daily_rsi',
                               'day_return_pct', 'atr_norm_day_return', 'rsi_14', 'relative_strength']
                for feat_name in key_features:
                    if feat_name in pred.feature_names:
                        idx = pred.feature_names.index(feat_name)
                        val = feat_array[0, idx]
                        print(f'  {feat_name:>25s} = {val:>10.4f}')
                    else:
                        print(f'  {feat_name:>25s} = NOT IN FEATURES')
        else:
            print(f'\n{sym}: No prediction returned')
    except Exception as e:
        print(f'\n{sym}: ERROR - {e}')
