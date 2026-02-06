"""
Backtest: What would prediction have said on Sunday for Monday?
"""
import yfinance as yf
import numpy as np

# Get data
mcx = yf.Ticker('MCX.NS')
df = mcx.history(period='1y')

# Remove today (Monday Feb 2) to simulate Sunday prediction
df_sunday = df.iloc[:-1]  # All data except today

current = float(df_sunday['Close'].iloc[-1])  # Last close before Monday (Friday in Yahoo)
actual_monday = float(df['Close'].iloc[-1])  # Actual Monday close

# But user said Sunday close was around 2233.30 (from NSE)
# Let's use that as reference
sunday_close = 2233.30  # NSE previous close from earlier

print('='*60)
print('BACKTEST: What prediction said on SUNDAY for MONDAY')
print('='*60)
print(f'Sunday Close (NSE):       Rs.{sunday_close:.2f}')
print(f'Actual Monday Close:      Rs.{actual_monday:.2f}')
print(f'Actual Change:            {(actual_monday/sunday_close-1)*100:+.2f}%')
print()

# Use Sunday close as reference for predictions
current = sunday_close

# Run all models with data available on Sunday
ma20 = float(df_sunday['Close'].tail(20).mean())
ma50 = float(df_sunday['Close'].tail(50).mean())

# Returns
returns = df_sunday['Close'].pct_change().dropna()
avg_daily_return = float(returns.tail(20).mean())

# Models
momentum_pred = current * (1 + avg_daily_return)
mean_rev_pred = current + 0.3 * (ma20 - current)

# AR(1)
returns_arr = returns.tail(60).values
ar1_coef = float(np.corrcoef(returns_arr[:-1], returns_arr[1:])[0,1])
ar1_pred = current * (1 + ar1_coef * float(returns.iloc[-1]))

# Pattern after big down days (Sunday was a big down day from Friday)
# Let's check what happens after big down days
big_down_days = returns[returns < -0.03].index
next_day_returns = []
for date in big_down_days:
    try:
        idx = df_sunday.index.get_loc(date)
        if idx + 1 < len(df_sunday):
            next_day_returns.append(float(returns.iloc[idx + 1]))
    except:
        pass

if next_day_returns:
    avg_after_big_down = np.mean(next_day_returns)
    pattern_pred = current * (1 + avg_after_big_down)
    pattern_info = f"After big down days: avg {avg_after_big_down*100:.2f}%, {len(next_day_returns)} samples"
else:
    pattern_pred = current
    pattern_info = "No pattern data"

# VWAP
typical_price = (df_sunday['High'] + df_sunday['Low'] + df_sunday['Close']) / 3
vwap_20 = float((typical_price * df_sunday['Volume']).tail(20).sum() / df_sunday['Volume'].tail(20).sum())

# RSI
delta = df_sunday['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
current_rsi = float(rsi.iloc[-1])

if current_rsi > 70:
    rsi_pred = current * 0.98
    rsi_signal = "OVERBOUGHT"
elif current_rsi < 30:
    rsi_pred = current * 1.02
    rsi_signal = "OVERSOLD - bounce expected"
else:
    rsi_pred = current
    rsi_signal = "NEUTRAL"

predictions = {
    'Momentum': momentum_pred,
    'Mean Reversion': mean_rev_pred,
    'AR(1)': ar1_pred,
    'Pattern': pattern_pred,
    'VWAP Target': vwap_20,
    'RSI': rsi_pred
}

ensemble = np.mean(list(predictions.values()))

print('MODEL PREDICTIONS (as if made Sunday night):')
print('-'*60)
for name, pred in predictions.items():
    direction = "UP" if pred > current else "DOWN"
    pct_pred = (pred/current - 1) * 100
    error = pred - actual_monday
    accuracy = "CORRECT" if (pred > current and actual_monday > current) or (pred < current and actual_monday < current) else "WRONG"
    print(f'{name:15}: Rs.{pred:,.0f} ({pct_pred:+.1f}%) | Error: Rs.{error:+.0f} | {accuracy}')

print()
print('-'*60)
print(f'ENSEMBLE PREDICTION: Rs.{ensemble:,.0f} ({(ensemble/current-1)*100:+.1f}%)')
print(f'ACTUAL RESULT:       Rs.{actual_monday:,.0f} ({(actual_monday/current-1)*100:+.1f}%)')
print(f'PREDICTION ERROR:    Rs.{ensemble - actual_monday:+.0f} ({(ensemble/actual_monday-1)*100:+.1f}%)')
print()

bullish = sum([1 for p in predictions.values() if p > current])
bearish = 6 - bullish

print(f'Models predicting UP:   {bullish}/6')
print(f'Models predicting DOWN: {bearish}/6')
print()

predicted_direction = "BULLISH" if bullish > 3 else "BEARISH" if bearish > 3 else "NEUTRAL"
actual_direction = "UP" if actual_monday > current else "DOWN"

print(f'PREDICTED DIRECTION: {predicted_direction}')
print(f'ACTUAL DIRECTION:    {actual_direction} (+{(actual_monday/current-1)*100:.1f}%)')
print()

if (predicted_direction == "BULLISH" and actual_direction == "UP") or \
   (predicted_direction == "BEARISH" and actual_direction == "DOWN"):
    print('✅ DIRECTION PREDICTION: CORRECT!')
else:
    print('❌ DIRECTION PREDICTION: WRONG')

print()
print('='*60)
print('ANALYSIS:')
print('='*60)
print(f'RSI on Sunday: {current_rsi:.1f} ({rsi_signal})')
print(f'Pattern: {pattern_info}')
print(f'Sunday was a big down day from Friday (-11.5%)')
print(f'Historically after such drops, stocks tend to bounce')
