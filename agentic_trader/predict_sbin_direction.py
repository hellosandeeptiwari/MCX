"""SBIN Direction Analysis — combines ML MOVE probability with directional indicators."""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from ml_models.predictor import MovePredictor
from ml_models.data_fetcher import load_all_daily
from ml_models.feature_engineering import compute_features, get_feature_names

# Load data
candles = pd.read_parquet('ml_models/data/candles_5min/SBIN.parquet')
daily_data = load_all_daily(['SBIN'])
daily_df = daily_data.get('SBIN')

predictor = MovePredictor()
featured = compute_features(candles, daily_df=daily_df)

last_date = featured['date'].dt.date.max()
last_row = featured.iloc[-1]

print("=" * 65)
print("  SBIN — DIRECTION ANALYSIS FOR MONDAY")
print("=" * 65)

# === 1. DAILY CONTEXT (what's the bigger picture?) ===
print("\n--- DAILY TIMEFRAME (bigger picture) ---")
if daily_df is not None and len(daily_df) >= 20:
    d = daily_df.tail(20).copy()
    last_close = d.iloc[-1]['close']
    
    # SMA trends
    sma_5 = d['close'].tail(5).mean()
    sma_10 = d['close'].tail(10).mean()
    sma_20 = d['close'].mean()
    
    print(f"  Last close:  Rs {last_close:.2f}")
    print(f"  SMA  5-day:  Rs {sma_5:.2f}  {'ABOVE' if last_close > sma_5 else 'BELOW'}")
    print(f"  SMA 10-day:  Rs {sma_10:.2f}  {'ABOVE' if last_close > sma_10 else 'BELOW'}")
    print(f"  SMA 20-day:  Rs {sma_20:.2f}  {'ABOVE' if last_close > sma_20 else 'BELOW'}")
    
    # Recent momentum
    ret_1d = (d.iloc[-1]['close'] / d.iloc[-2]['close'] - 1) * 100
    ret_3d = (d.iloc[-1]['close'] / d.iloc[-4]['close'] - 1) * 100
    ret_5d = (d.iloc[-1]['close'] / d.iloc[-6]['close'] - 1) * 100
    ret_10d = (d.iloc[-1]['close'] / d.iloc[-11]['close'] - 1) * 100
    
    print(f"\n  Returns:")
    print(f"    1-day:  {ret_1d:+.2f}%")
    print(f"    3-day:  {ret_3d:+.2f}%")
    print(f"    5-day:  {ret_5d:+.2f}%")
    print(f"   10-day:  {ret_10d:+.2f}%")
    
    # RSI
    delta = d['close'].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    print(f"\n  RSI (14):  {current_rsi:.1f}", end="")
    if current_rsi > 70:
        print("  -> OVERBOUGHT (pullback risk)")
    elif current_rsi > 60:
        print("  -> BULLISH momentum")
    elif current_rsi < 30:
        print("  -> OVERSOLD (bounce possible)")
    elif current_rsi < 40:
        print("  -> BEARISH momentum")
    else:
        print("  -> NEUTRAL")
    
    # Distance from highs/lows
    high_20d = d['high'].max()
    low_20d = d['low'].min()
    dist_high = (last_close - high_20d) / high_20d * 100
    dist_low = (last_close - low_20d) / low_20d * 100
    
    print(f"\n  20-day high: Rs {high_20d:.2f} ({dist_high:+.1f}% from current)")
    print(f"  20-day low:  Rs {low_20d:.2f} ({dist_low:+.1f}% from current)")
    
    # Support/Resistance levels
    pivots = d.tail(5)
    pivot = (pivots['high'].max() + pivots['low'].min() + last_close) / 3
    r1 = 2 * pivot - pivots['low'].min()
    s1 = 2 * pivot - pivots['high'].max()
    r2 = pivot + (pivots['high'].max() - pivots['low'].min())
    s2 = pivot - (pivots['high'].max() - pivots['low'].min())
    
    print(f"\n  Pivot levels (5-day):")
    print(f"    R2: Rs {r2:.2f}")
    print(f"    R1: Rs {r1:.2f}")
    print(f"    PP: Rs {pivot:.2f}")
    print(f"    S1: Rs {s1:.2f}")
    print(f"    S2: Rs {s2:.2f}")
    
    # Consecutive up/down days
    daily_returns = d['close'].pct_change().dropna()
    consecutive = 0
    direction = "UP" if daily_returns.iloc[-1] > 0 else "DOWN"
    for ret in reversed(daily_returns.values):
        if (direction == "UP" and ret > 0) or (direction == "DOWN" and ret < 0):
            consecutive += 1
        else:
            break
    print(f"\n  Consecutive {direction} days: {consecutive}")

# === 2. INTRADAY FEATURES (last candle signals) ===
print("\n--- INTRADAY FEATURES (from last 5-min candle) ---")

# Key directional features from the model
directional_features = {
    'roc_12': 'Rate of change (60min)',
    'price_vs_sma20': 'Price vs SMA20',
    'price_vs_ema9': 'Price vs EMA9',
    'rsi_14': 'RSI (14)',
    'macd_signal_dist': 'MACD signal distance',
}

for feat, label in directional_features.items():
    if feat in featured.columns:
        val = last_row[feat]
        arrow = "^" if val > 0 else "v" if val < 0 else "-"
        print(f"  {label:30s}: {val:+.4f}  {arrow}")

# Daily context features
daily_features = {
    'daily_trend_direction': 'Daily trend direction',
    'daily_trend_strength': 'Daily trend strength',
    'daily_rsi_14': 'Daily RSI',
    'daily_vol_regime': 'Volatility regime',
    'daily_dist_from_high': 'Dist from 20d high',
    'daily_dist_from_low': 'Dist from 20d low',
}

print("\n--- DAILY CONTEXT FEATURES ---")
for feat, label in daily_features.items():
    if feat in featured.columns:
        val = last_row[feat]
        print(f"  {label:30s}: {val:+.4f}")

# === 3. HISTORICAL PATTERN — When SBIN had MOVE signals before, which direction? ===
print("\n--- HISTORICAL DIRECTION WHEN MODEL SAID 'MOVE' ---")

fnames = get_feature_names()
X_all = featured[fnames].values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
probas = predictor.model.predict_proba(X_all)
featured = featured.copy()
featured['move_prob'] = probas[:, 1]

# Calculate actual next-30-min return for each candle
featured['future_return'] = featured['close'].shift(-6) / featured['close'] - 1

# Only look at candles where model predicted MOVE (prob >= 0.30)
move_signals = featured[featured['move_prob'] >= 0.30].dropna(subset=['future_return'])

if len(move_signals) > 0:
    up_moves = (move_signals['future_return'] > 0.005).sum()
    down_moves = (move_signals['future_return'] < -0.005).sum()
    flat_moves = len(move_signals) - up_moves - down_moves
    avg_return = move_signals['future_return'].mean() * 100
    median_return = move_signals['future_return'].median() * 100
    
    print(f"  Total MOVE signals in dataset: {len(move_signals)}")
    print(f"  Went UP   (>+0.5%): {up_moves}  ({up_moves/len(move_signals)*100:.0f}%)")
    print(f"  Went DOWN (<-0.5%): {down_moves}  ({down_moves/len(move_signals)*100:.0f}%)")
    print(f"  Stayed flat:        {flat_moves}  ({flat_moves/len(move_signals)*100:.0f}%)")
    print(f"  Avg return after MOVE signal: {avg_return:+.3f}%")
    print(f"  Median return:                {median_return:+.3f}%")
    
    # By time of day
    print(f"\n  Direction breakdown by time of day:")
    move_signals = move_signals.copy()
    move_signals['hour'] = move_signals['date'].dt.hour
    for hour in sorted(move_signals['hour'].unique()):
        hourly = move_signals[move_signals['hour'] == hour]
        avg_r = hourly['future_return'].mean() * 100
        up_pct = (hourly['future_return'] > 0).mean() * 100
        print(f"    {hour:02d}:xx  avg={avg_r:+.3f}%  up_bias={up_pct:.0f}%  ({len(hourly)} signals)")

# === 4. CURRENT DIRECTIONAL BIAS ===
print(f"\n{'='*65}")
print(f"  MONDAY DIRECTION VERDICT")
print(f"{'='*65}")

# Collect directional evidence
bullish = 0
bearish = 0
reasons_bull = []
reasons_bear = []

if daily_df is not None and len(daily_df) >= 20:
    d = daily_df.tail(20)
    last_close = d.iloc[-1]['close']
    sma_5 = d['close'].tail(5).mean()
    sma_20 = d['close'].mean()
    
    if last_close > sma_5:
        bullish += 1; reasons_bull.append(f"Price above 5-SMA ({last_close:.0f} > {sma_5:.0f})")
    else:
        bearish += 1; reasons_bear.append(f"Price below 5-SMA ({last_close:.0f} < {sma_5:.0f})")
    
    if last_close > sma_20:
        bullish += 1; reasons_bull.append(f"Price above 20-SMA ({last_close:.0f} > {sma_20:.0f})")
    else:
        bearish += 1; reasons_bear.append(f"Price below 20-SMA ({last_close:.0f} < {sma_20:.0f})")
    
    if ret_5d > 1.0:
        bullish += 1; reasons_bull.append(f"Strong 5-day rally ({ret_5d:+.1f}%)")
    elif ret_5d < -1.0:
        bearish += 1; reasons_bear.append(f"5-day decline ({ret_5d:+.1f}%)")
    
    if current_rsi > 65:
        bearish += 1; reasons_bear.append(f"RSI overbought ({current_rsi:.0f})")
    elif current_rsi < 35:
        bullish += 1; reasons_bull.append(f"RSI oversold ({current_rsi:.0f})")
    elif current_rsi > 50:
        bullish += 1; reasons_bull.append(f"RSI above 50 ({current_rsi:.0f})")
    else:
        bearish += 1; reasons_bear.append(f"RSI below 50 ({current_rsi:.0f})")
    
    if dist_high > -1:
        bearish += 1; reasons_bear.append(f"Near 20-day high (resistance, {dist_high:+.1f}%)")
    elif dist_low < 5:
        bullish += 1; reasons_bull.append(f"Near 20-day low (support, {dist_low:+.1f}%)")
    
    if consecutive >= 3 and direction == "UP":
        bearish += 1; reasons_bear.append(f"{consecutive} consecutive up days (mean reversion risk)")
    elif consecutive >= 3 and direction == "DOWN":
        bullish += 1; reasons_bull.append(f"{consecutive} consecutive down days (bounce possible)")

# Directional features from last candle
if 'roc_12' in featured.columns:
    roc = last_row['roc_12']
    if roc > 0.001:
        bullish += 1; reasons_bull.append(f"Positive 60-min momentum (ROC={roc:+.4f})")
    elif roc < -0.001:
        bearish += 1; reasons_bear.append(f"Negative 60-min momentum (ROC={roc:+.4f})")

if 'daily_trend_direction' in featured.columns:
    trend = last_row['daily_trend_direction']
    if trend > 0.3:
        bullish += 1; reasons_bull.append(f"Daily trend UP (direction={trend:+.2f})")
    elif trend < -0.3:
        bearish += 1; reasons_bear.append(f"Daily trend DOWN (direction={trend:+.2f})")

print(f"\n  BULLISH signals: {bullish}")
for r in reasons_bull:
    print(f"    + {r}")

print(f"\n  BEARISH signals: {bearish}")
for r in reasons_bear:
    print(f"    - {r}")

total = bullish + bearish
if total > 0:
    bull_pct = bullish / total * 100
    bear_pct = bearish / total * 100
    
    print(f"\n  Direction score: {bull_pct:.0f}% bullish / {bear_pct:.0f}% bearish")
    
    if bull_pct >= 70:
        print(f"  -> STRONG BULLISH BIAS for Monday")
        print(f"  -> If MOVE happens, likely UP")
        print(f"  -> Titan strategy: BUY CALL options / Bull Call Spread")
    elif bull_pct >= 55:
        print(f"  -> MILD BULLISH BIAS for Monday")
        print(f"  -> Slight edge toward upside")
        print(f"  -> Titan strategy: Favor CALLS, but keep position small")
    elif bear_pct >= 70:
        print(f"  -> STRONG BEARISH BIAS for Monday")
        print(f"  -> If MOVE happens, likely DOWN")
        print(f"  -> Titan strategy: BUY PUT options / Bear Put Spread")
    elif bear_pct >= 55:
        print(f"  -> MILD BEARISH BIAS for Monday")
        print(f"  -> Slight edge toward downside")
        print(f"  -> Titan strategy: Favor PUTS, but keep position small")
    else:
        print(f"  -> NO CLEAR DIRECTION — 50/50")
        print(f"  -> If MOVE happens, direction is uncertain")
        print(f"  -> Titan strategy: Iron Condor or Straddle (non-directional)")

print(f"\n  NOTE: This model predicts MOVE probability (74.8% at open).")
print(f"  Direction is inferred from technicals, not ML.")
print(f"  For a direction ML model, we'd need a separate classifier.")
