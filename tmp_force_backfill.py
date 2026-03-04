"""Force NIFTY daily backfill from Kite API — run on EC2"""
import sys, os
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
os.chdir('/home/ubuntu/titan/agentic_trader')

# Load .env for ZERODHA creds
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/titan/agentic_trader/.env')

from kiteconnect import KiteConnect
api_key = os.environ.get('ZERODHA_API_KEY')
token = os.environ.get('ZERODHA_ACCESS_TOKEN')
kite = KiteConnect(api_key=api_key)
kite.set_access_token(token)

# Verify token works
try:
    p = kite.profile()
    uname = p.get('user_name', 'Unknown')
    print(f"Kite OK: {uname}")
except Exception as e:
    print(f"Kite auth failed: {e}")
    sys.exit(1)

from ml_models.data_fetcher import fetch_candles
import pandas as pd

# Fetch 120 days of NIFTY daily
print("Fetching 120 days of NIFTY 50 daily candles...")
df = fetch_candles(kite, 'NIFTY 50', days=120, interval='day')
print(f"Got {len(df)} rows")
if len(df) == 0:
    print("FATAL: 0 rows returned")
    sys.exit(1)

df['date'] = pd.to_datetime(df['date'])
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)

# Load existing and merge
path = '/home/ubuntu/titan/agentic_trader/ml_models/data/candles_daily/NIFTY50.parquet'
if os.path.exists(path):
    old = pd.read_parquet(path)
    old['date'] = pd.to_datetime(old['date'])
    if old['date'].dt.tz is not None:
        old['date'] = old['date'].dt.tz_localize(None)
    df = pd.concat([old, df], ignore_index=True)

df = df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_parquet(path, index=False)

print(f"\nSaved: {len(df)} rows")
print(f"Range: {df['date'].min()} to {df['date'].max()}")
print(f"\nLast 5 days:")
for _, r in df.tail(5).iterrows():
    prev_close = df[df['date'] < r['date']]['close'].iloc[-1] if len(df[df['date'] < r['date']]) > 0 else r['open']
    daily_ret = (r['close'] - prev_close) / prev_close * 100
    print(f"  {r['date'].strftime('%Y-%m-%d')}  O={r['open']:.1f}  H={r['high']:.1f}  L={r['low']:.1f}  C={r['close']:.1f}  DayRet={daily_ret:+.2f}%")

# Test feature computation with the new data
print(f"\n=== Testing feature computation with {len(df)} daily rows ===")
ema50_ok = len(df) >= 60
ema20_ok = len(df) >= 25
print(f"EMA-50 warmup (≥60): {'YES' if ema50_ok else 'NO'} ({len(df)} rows)")
print(f"EMA-20 warmup (≥25): {'YES' if ema20_ok else 'NO'} ({len(df)} rows)")

# Compute what nifty_daily_trend and nifty_daily_rsi would be
import numpy as np
close = df['close'].values
ema_period = 50 if len(df) >= 60 else max(10, min(len(df) - 5, 20))
print(f"Using EMA period: {ema_period}")

# Simple EMA
alpha = 2 / (ema_period + 1)
ema = np.zeros(len(close))
ema[0] = close[0]
for i in range(1, len(close)):
    ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]

# RSI
deltas = np.diff(close)
gains = np.where(deltas > 0, deltas, 0)
losses = np.where(deltas < 0, -deltas, 0)
avg_gain = pd.Series(gains).rolling(14).mean().values
avg_loss = pd.Series(losses).rolling(14).mean().values
rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
rsi = 100 - (100 / (1 + rs))

last_idx = len(close) - 1
trend = (close[last_idx] - ema[last_idx]) / ema[last_idx] * 100
last_rsi = rsi[-1] if len(rsi) > 0 else 0

print(f"\nNIFTY daily features (prev-day = {df['date'].iloc[-1].strftime('%Y-%m-%d')}):")
print(f"  nifty_daily_trend = {trend:+.2f}%")
print(f"  nifty_daily_rsi   = {last_rsi:.1f}")
print(f"\n  (Before fix: both were 0.0 due to only 45 rows)")
