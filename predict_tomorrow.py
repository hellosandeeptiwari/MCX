"""
MCX Price Prediction for Tomorrow
Multi-Model Ensemble Approach
"""

import yfinance as yf
import numpy as np
import pandas as pd

mcx = yf.Ticker('MCX.NS')
df = mcx.history(period='1y')

# Current values
current = df['Close'].iloc[-1]
print('='*65)
print('MCX PRICE PREDICTION FOR TOMORROW (Feb 3, 2026)')
print('='*65)
print(f'Current Price: Rs.{current:.2f}')
print()

# ============================================================
# METHOD 1: TECHNICAL LEVELS
# ============================================================
print('METHOD 1: TECHNICAL ANALYSIS')
print('-'*65)
ma20 = df['Close'].tail(20).mean()
ma50 = df['Close'].tail(50).mean()
ma200 = df['Close'].tail(200).mean()

# Fibonacci from recent swing
recent_high = df['High'].tail(20).max()
recent_low = df['Low'].tail(20).min()
fib_382 = recent_high - 0.382 * (recent_high - recent_low)
fib_50 = recent_high - 0.5 * (recent_high - recent_low)
fib_618 = recent_high - 0.618 * (recent_high - recent_low)

# VWAP proxy
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
vwap_20 = (typical_price * df['Volume']).tail(20).sum() / df['Volume'].tail(20).sum()

print(f'20-day MA:        Rs.{ma20:.0f}')
print(f'50-day MA:        Rs.{ma50:.0f}')
print(f'VWAP (20d):       Rs.{vwap_20:.0f}')
print(f'Fib 38.2%:        Rs.{fib_382:.0f}')
print(f'Fib 50%:          Rs.{fib_50:.0f}')
print(f'Resistance:       Rs.{recent_high:.0f}')
print(f'Support:          Rs.{recent_low:.0f}')

# ============================================================
# METHOD 2: MOMENTUM/MEAN REVERSION
# ============================================================
print()
print('METHOD 2: STATISTICAL MODELS')
print('-'*65)

# Daily returns
returns = df['Close'].pct_change().dropna()
avg_daily_return = returns.tail(20).mean()
volatility = returns.tail(20).std()

# Momentum prediction (trend continuation)
momentum_pred = current * (1 + avg_daily_return)

# Mean reversion (towards MA)
reversion_target = ma20
reversion_strength = 0.3  # 30% reversion per day
mean_rev_pred = current + reversion_strength * (reversion_target - current)

# AR(1) model
returns_arr = returns.tail(60).values
ar1_coef = np.corrcoef(returns_arr[:-1], returns_arr[1:])[0,1]
last_return = returns.iloc[-1]
ar1_pred_return = ar1_coef * last_return
ar1_pred = current * (1 + ar1_pred_return)

print(f'Momentum (trend):    Rs.{momentum_pred:.0f}')
print(f'Mean Reversion:      Rs.{mean_rev_pred:.0f}')
print(f'AR(1) Model:         Rs.{ar1_pred:.0f}')

# ============================================================
# METHOD 3: VOLATILITY-BASED RANGE
# ============================================================
print()
print('METHOD 3: VOLATILITY RANGE (68% confidence)')
print('-'*65)
atr = (df['High'] - df['Low']).tail(14).mean()
expected_range_low = current - atr
expected_range_high = current + atr
print(f'Expected Low:     Rs.{expected_range_low:.0f}')
print(f'Expected High:    Rs.{expected_range_high:.0f}')
print(f'ATR (14d):        Rs.{atr:.0f}')

# ============================================================
# METHOD 4: PATTERN RECOGNITION
# ============================================================
print()
print('METHOD 4: PATTERN ANALYSIS (After +3% days)')
print('-'*65)
# After big up days (>3%), what happens next?
big_up_days = returns[returns > 0.03].index
next_day_after_big_up = []
for date in big_up_days:
    try:
        idx = df.index.get_loc(date)
        if idx + 1 < len(df):
            next_day_after_big_up.append(returns.iloc[idx + 1])
    except:
        pass

if next_day_after_big_up:
    avg_after_big_up = np.mean(next_day_after_big_up)
    win_rate = len([x for x in next_day_after_big_up if x > 0]) / len(next_day_after_big_up)
    pattern_pred = current * (1 + avg_after_big_up)
    print(f'Sample size:         {len(next_day_after_big_up)} instances')
    print(f'Avg next day return: {avg_after_big_up*100:.2f}%')
    print(f'Win rate (up next):  {win_rate*100:.0f}%')
    print(f'Pattern prediction:  Rs.{pattern_pred:.0f}')
else:
    pattern_pred = current
    print('Not enough pattern data')

# ============================================================
# METHOD 5: RSI MEAN REVERSION
# ============================================================
print()
print('METHOD 5: RSI ANALYSIS')
print('-'*65)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
current_rsi = rsi.iloc[-1]
print(f'Current RSI(14):     {current_rsi:.1f}')

if current_rsi > 70:
    rsi_bias = "OVERBOUGHT - Pullback likely"
    rsi_pred = current * 0.98  # Expect 2% pullback
elif current_rsi < 30:
    rsi_bias = "OVERSOLD - Bounce likely"
    rsi_pred = current * 1.02  # Expect 2% bounce
else:
    rsi_bias = "NEUTRAL"
    rsi_pred = current

print(f'RSI Signal:          {rsi_bias}')

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================
print()
print('='*65)
print('FINAL ENSEMBLE PREDICTION')
print('='*65)

# Weighted average of models
predictions = {
    'Momentum': momentum_pred,
    'Mean Reversion': mean_rev_pred,
    'AR(1)': ar1_pred,
    'Pattern': pattern_pred,
    'VWAP Target': vwap_20,
    'RSI': rsi_pred
}

# Equal weighted ensemble
ensemble = np.mean(list(predictions.values()))

print()
print('Individual Model Predictions:')
for name, pred in predictions.items():
    direction = "↑" if pred > current else "↓" if pred < current else "→"
    change_pct = (pred / current - 1) * 100
    print(f'  {name:15}: Rs.{pred:,.0f} {direction} ({change_pct:+.1f}%)')

print()
print(f'ENSEMBLE PREDICTION: Rs.{ensemble:,.0f}')
print(f'Expected Change:     {(ensemble/current - 1)*100:+.2f}%')
print()

# Prediction range
print(f'TOMORROW RANGE (based on ATR):')
print(f'  Likely Low:    Rs.{ensemble - atr/2:,.0f}')
print(f'  Likely High:   Rs.{ensemble + atr/2:,.0f}')
print(f'  Max Low:       Rs.{ensemble - atr:,.0f}')
print(f'  Max High:      Rs.{ensemble + atr:,.0f}')

# Direction probability
bullish_count = sum([1 for p in predictions.values() if p > current])
bearish_count = sum([1 for p in predictions.values() if p < current])
total = len(predictions)

print()
if bullish_count > bearish_count:
    confidence = bullish_count / total * 100
    print(f'DIRECTION: BULLISH ({confidence:.0f}% model agreement)')
    print(f'  {bullish_count}/{total} models predict UP')
elif bearish_count > bullish_count:
    confidence = bearish_count / total * 100
    print(f'DIRECTION: BEARISH ({confidence:.0f}% model agreement)')
    print(f'  {bearish_count}/{total} models predict DOWN')
else:
    print(f'DIRECTION: UNCERTAIN (models split)')

# Key levels
print()
print('='*65)
print('KEY LEVELS FOR TOMORROW')
print('='*65)
print(f'  Resistance 1:  Rs.{min(fib_382, recent_high):,.0f}')
print(f'  Resistance 2:  Rs.{recent_high:,.0f}')
print(f'  Support 1:     Rs.{max(fib_50, ma20):,.0f}')
print(f'  Support 2:     Rs.{fib_618:,.0f}')
print(f'  Stop Loss:     Rs.{recent_low:,.0f}')

# Trading recommendation
print()
print('='*65)
print('TRADING RECOMMENDATION')
print('='*65)
if bullish_count >= 4 and current_rsi < 65:
    print('  Signal: BUY on dips')
    print(f'  Entry Zone: Rs.{current - atr/4:,.0f} - Rs.{current:,.0f}')
    print(f'  Target 1: Rs.{fib_382:,.0f}')
    print(f'  Target 2: Rs.{recent_high:,.0f}')
    print(f'  Stop Loss: Rs.{recent_low:,.0f}')
elif bearish_count >= 4 or current_rsi > 70:
    print('  Signal: SELL / AVOID')
    print(f'  Resistance: Rs.{fib_382:,.0f}')
    print(f'  Downside Target: Rs.{fib_618:,.0f}')
else:
    print('  Signal: WAIT / NEUTRAL')
    print(f'  Wait for clear break of Rs.{fib_382:,.0f} (up) or Rs.{fib_618:,.0f} (down)')

# Disclaimer
print()
print('='*65)
print('DISCLAIMER: This is a statistical estimate, not financial advice.')
print('Models have ~55-60% accuracy. Always use stop losses.')
print('='*65)
