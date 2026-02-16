"""Predict SBIN move probability for Monday using the trained ML model."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from ml_models.predictor import MovePredictor
from ml_models.data_fetcher import load_all_daily
from ml_models.feature_engineering import compute_features, get_feature_names
from pathlib import Path

# Load SBIN 5-min candles
candles = pd.read_parquet('ml_models/data/candles_5min/SBIN.parquet')
print(f'SBIN 5-min candles: {len(candles)} rows')
print(f'Date range: {candles["date"].min()} to {candles["date"].max()}')

# Last day's candles
last_date = candles['date'].dt.date.max()
last_day = candles[candles['date'].dt.date == last_date]
print(f'\nLast trading day: {last_date}')
print(f'Candles on last day: {len(last_day)}')
print(f'Open: {last_day.iloc[0]["open"]:.2f}  Close: {last_day.iloc[-1]["close"]:.2f}')
print(f'Day High: {last_day["high"].max():.2f}  Day Low: {last_day["low"].min():.2f}')
day_change = (last_day.iloc[-1]["close"] - last_day.iloc[0]["open"]) / last_day.iloc[0]["open"] * 100
print(f'Day change: {day_change:+.2f}%')

# Load daily candles for context
daily_data = load_all_daily(['SBIN'])
daily_df = daily_data.get('SBIN')
if daily_df is not None:
    print(f'\nDaily context: {len(daily_df)} days loaded')
    recent = daily_df.tail(5)
    print('Last 5 daily closes:', [round(x, 2) for x in recent["close"].tolist()])

# Run predictor
predictor = MovePredictor()

if predictor.ready:
    # Predict using all candles (features need warmup)
    result = predictor.predict(candles, daily_df=daily_df)
    
    print(f'\n{"="*60}')
    print(f'  SBIN ML PREDICTION')
    print(f'  (Based on last candle: {candles.iloc[-1]["date"]})')
    print(f'{"="*60}')
    for k, v in result.items():
        print(f'  {k}: {v}')
    
    prob = result['ml_move_prob']
    signal = result['ml_signal']
    boost = result['ml_score_boost']
    
    print(f'\n  INTERPRETATION:')
    if signal == 'MOVE':
        print(f'  -> {prob*100:.1f}% probability SBIN moves >= 0.5% in next 30 min')
        print(f'  -> Score boost: +{boost} points for Titan scoring')
        print(f'  -> FAVORABLE setup for options entry')
    else:
        print(f'  -> Only {prob*100:.1f}% probability of a >= 0.5% move')
        print(f'  -> Score boost: {boost} points (penalty for flat stock)')
        print(f'  -> SBIN likely to stay FLAT - avoid options entry')
    
    # Predict at each candle of the last trading day to show intraday pattern
    print(f'\n{"="*60}')
    print(f'  SBIN MOVE PROBABILITY THROUGHOUT {last_date}')
    print(f'{"="*60}')
    
    featured = compute_features(candles, daily_df=daily_df)
    last_day_feat = featured[featured['date'].dt.date == last_date].copy()
    
    if len(last_day_feat) > 0:
        fnames = get_feature_names()
        X = last_day_feat[fnames].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        probas = predictor.model.predict_proba(X)
        last_day_feat['move_prob'] = probas[:, 1]
        
        # Show every 6th candle (every 30 min)
        for idx in range(0, len(last_day_feat), 6):
            row = last_day_feat.iloc[idx]
            time_str = row['date'].strftime('%H:%M')
            p = row['move_prob']
            bar = '#' * int(p * 50)
            sig = 'MOVE' if p >= 0.30 else 'flat'
            print(f'  {time_str}  {p:.1%} {bar:50s} {sig}')
        
        # Summary stats
        avg_prob = last_day_feat['move_prob'].mean()
        max_prob = last_day_feat['move_prob'].max()
        max_time = last_day_feat.loc[last_day_feat['move_prob'].idxmax(), 'date'].strftime('%H:%M')
        move_candles = (last_day_feat['move_prob'] >= 0.30).sum()
        total = len(last_day_feat)
        
        print(f'\n  Summary:')
        print(f'  Avg move prob: {avg_prob:.1%}')
        print(f'  Max move prob: {max_prob:.1%} (at {max_time})')
        print(f'  MOVE signals (>=30%): {move_candles}/{total} candles ({move_candles/total*100:.0f}%)')
        
        # What this means for Monday
        print(f'\n{"="*60}')
        print(f'  WHAT THIS MEANS FOR MONDAY')
        print(f'{"="*60}')
        
        # Get daily features that carry over
        if daily_df is not None and len(daily_df) >= 20:
            last_close = daily_df.iloc[-1]['close']
            high_20d = daily_df.tail(20)['high'].max()
            low_20d = daily_df.tail(20)['low'].min()
            dist_from_high = (last_close - high_20d) / high_20d * 100
            dist_from_low = (last_close - low_20d) / low_20d * 100
            
            # Daily range and move frequency
            daily_df_recent = daily_df.tail(20).copy()
            daily_df_recent['daily_range'] = (daily_df_recent['high'] - daily_df_recent['low']) / daily_df_recent['close'] * 100
            daily_df_recent['daily_move'] = daily_df_recent['close'].pct_change().abs() * 100
            avg_range = daily_df_recent['daily_range'].mean()
            big_move_days = (daily_df_recent['daily_move'] >= 0.5).sum()
            
            print(f'  SBIN last close: Rs {last_close:.2f}')
            print(f'  Distance from 20-day high: {dist_from_high:+.1f}%')
            print(f'  Distance from 20-day low: {dist_from_low:+.1f}%')
            print(f'  Avg daily range (20d): {avg_range:.2f}%')
            print(f'  Days with >=0.5% move (last 20): {big_move_days}/20')
            
            if avg_prob >= 0.30:
                print(f'\n  VERDICT: SBIN is in a VOLATILE regime')
                print(f'  -> Good candidate for directional options trades')
                print(f'  -> Titan will get +{boost} score boost from ML')
            elif avg_prob >= 0.20:
                print(f'\n  VERDICT: SBIN has MODERATE move potential')
                print(f'  -> May see moves during opening 30 min and news events')
                print(f'  -> Titan will be neutral on ML signal')
            else:
                print(f'\n  VERDICT: SBIN is in a LOW-VOLATILITY regime')
                print(f'  -> Likely to chop sideways, avoid options premium decay')
                print(f'  -> Titan will penalize SBIN with {boost} score boost')
else:
    print("Model not ready. Run: python -m ml_models.trainer --threshold 0.5")
