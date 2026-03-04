import pandas as pd
path = "/home/ubuntu/titan/agentic_trader/ml_models/data/candles_daily/NIFTY50.parquet"
df = pd.read_parquet(path)
print(f"ROWS: {len(df)}")
print(f"DATE RANGE: {df['date'].min()} to {df['date'].max()}")
print(f"\nLast 5 rows:")
for _, r in df.tail(5).iterrows():
    print(f"  {r['date']}  O={r['open']:.1f}  H={r['high']:.1f}  L={r['low']:.1f}  C={r['close']:.1f}")

# Check 5min too
path5 = "/home/ubuntu/titan/agentic_trader/ml_models/data/candles_5min/NIFTY50.parquet"
df5 = pd.read_parquet(path5)
print(f"\n5MIN ROWS: {len(df5)}")
print(f"5MIN RANGE: {df5['date'].min()} to {df5['date'].max()}")

# Check bot logs for NIFTY daily backfill message
import subprocess
result = subprocess.run(["grep", "-i", "nifty daily", "/home/ubuntu/titan/logs/titan-bot.log"], 
                       capture_output=True, text=True, timeout=5)
if result.stdout.strip():
    lines = result.stdout.strip().split("\n")
    print(f"\nBot log mentions of NIFTY daily ({len(lines)} lines):")
    for l in lines[-5:]:
        print(f"  {l}")
else:
    print("\nNo NIFTY daily mentions in bot log")
