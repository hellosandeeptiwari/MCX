import pandas as pd
import numpy as np

nf = pd.read_parquet('/home/ubuntu/titan/agentic_trader/ml_models/data/candles_5min/NIFTY50.parquet')
nf['date'] = pd.to_datetime(nf['date'])
print('Total rows:', len(nf))
print('Date range:', nf['date'].min(), 'to', nf['date'].max())

print('\nLast 10 candles:')
for _, r in nf.tail(10).iterrows():
    print(f'  {r["date"]}  O={r["open"]:.1f}  H={r["high"]:.1f}  L={r["low"]:.1f}  C={r["close"]:.1f}  V={r["volume"]:.0f}')

# Check each day
for d in ['2026-02-28', '2026-03-02', '2026-03-03', '2026-03-04']:
    day_df = nf[nf['date'].dt.date == pd.Timestamp(d).date()]
    if len(day_df) > 0:
        day_ret = (day_df.iloc[-1]['close'] - day_df.iloc[0]['open']) / day_df.iloc[0]['open'] * 100
        print(f'\n{d}: {len(day_df)} candles, open={day_df.iloc[0]["open"]:.1f}, close={day_df.iloc[-1]["close"]:.1f}, day_ret={day_ret:.2f}%')
    else:
        print(f'\n{d}: NO DATA')

# Now compute the actual NIFTY features as the model would see them
# Using the LAST candle (what the model gets)
closes = nf['close'].values

# ROC-6 = (close[-1] / close[-7]) - 1) * 100
if len(closes) >= 7:
    roc_6 = (closes[-1] / closes[-7] - 1) * 100
    print(f'\nNIFTY roc_6 (last candle): {roc_6:.4f}%')

# RSI-14
def rsi(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

if len(closes) >= 15:
    r = rsi(closes)
    print(f'NIFTY rsi_14 (last candle): {r:.4f}')

# EMA-9 slope
def ema(data, span):
    alpha = 2 / (span + 1)
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

ema9 = ema(closes, 9)
ema9_slope = (ema9[-1] - ema9[-2]) / ema9[-2] * 100
print(f'NIFTY ema9_slope: {ema9_slope:.4f}%')

# BB position
sma20 = np.mean(closes[-20:])
std20 = np.std(closes[-20:])
bb_upper = sma20 + 2 * std20
bb_lower = sma20 - 2 * std20
bb_pos = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
print(f'NIFTY bb_position: {bb_pos:.4f}')

# ATR%
highs = nf['high'].values
lows = nf['low'].values
tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
atr14 = np.mean(tr[-14:])
atr_pct = atr14 / closes[-1] * 100
print(f'NIFTY atr_pct: {atr_pct:.4f}%')

# Also check NIFTY daily
print('\n\n=== NIFTY DAILY ===')
nd = pd.read_parquet('/home/ubuntu/titan/agentic_trader/ml_models/data/candles_daily/NIFTY50.parquet')
nd['date'] = pd.to_datetime(nd['date'])
print(f'Total rows: {len(nd)}, range: {nd["date"].min()} to {nd["date"].max()}')
print('\nLast 5 daily candles:')
for _, r in nd.tail(5).iterrows():
    print(f'  {r["date"].strftime("%Y-%m-%d")}  O={r["open"]:.1f}  H={r["high"]:.1f}  L={r["low"]:.1f}  C={r["close"]:.1f}')

# EMA-50 (needs 50+ rows)
if len(nd) >= 50:
    nd_closes = nd['close'].values
    ema50 = ema(nd_closes, 50)
    trend = (nd_closes[-2] - ema50[-2]) / ema50[-2] * 100  # prev day
    nd_rsi = rsi(nd_closes[-15:])
    print(f'\nnifty_daily_trend (prev day): {trend:.4f}%')
    print(f'nifty_daily_rsi (prev day): {nd_rsi:.4f}')
else:
    print(f'\nOnly {len(nd)} daily rows - NEED >= 60 for EMA-50! This is why nifty_daily_trend=0!')
