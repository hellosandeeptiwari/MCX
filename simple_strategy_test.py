"""Quick strategy backtest"""
import yfinance as yf
import numpy as np
import pandas as pd

mcx = yf.Ticker('MCX.NS')
df = mcx.history(period='2y')
close = df['Close']
ret = close.pct_change()

results = pd.DataFrame(index=df.index[50:])
results['actual'] = (ret.shift(-1) > 0).iloc[50:].astype(int)

# RSI strategy
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
rsi_signal = np.where(rsi < 35, 1, np.where(rsi > 65, 0, np.nan))
results['rsi'] = pd.Series(rsi_signal, index=df.index).iloc[50:]

# 3-Day Reversal
down_3d = (ret.shift(1) < 0) & (ret.shift(2) < 0) & (ret.shift(3) < 0)
up_3d = (ret.shift(1) > 0) & (ret.shift(2) > 0) & (ret.shift(3) > 0)
rev_signal = np.where(down_3d, 1, np.where(up_3d, 0, np.nan))
results['reversal'] = pd.Series(rev_signal, index=df.index).iloc[50:]

# Bollinger
ma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
bb_lower = ma20 - 2*std20
bb_upper = ma20 + 2*std20
bb_signal = np.where(close < bb_lower, 1, np.where(close > bb_upper, 0, np.nan))
results['bollinger'] = pd.Series(bb_signal, index=df.index).iloc[50:]

# MA Crossover
ma5 = close.rolling(5).mean()
ma_signal = (ma5 > ma20).astype(int)
results['ma_cross'] = pd.Series(ma_signal, index=df.index).iloc[50:]

print("="*60)
print("SIMPLE STRATEGY BACKTEST RESULTS")
print("="*60)
print()

strategies = ['rsi', 'reversal', 'bollinger', 'ma_cross']
for strat in strategies:
    valid = results[strat].notna()
    if valid.sum() >= 5:
        correct = (results.loc[valid, strat] == results.loc[valid, 'actual'])
        acc = correct.mean() * 100
        edge = acc - 50
        star = " ⭐" if acc >= 60 else " ✓" if acc >= 55 else ""
        print(f"{strat:15}: {acc:5.1f}% accuracy | {valid.sum():3} signals | Edge: {edge:+5.1f}%{star}")
    else:
        print(f"{strat:15}: Not enough signals")

print()
print("="*60)
print("CURRENT STATE:")
print("="*60)
print(f"Price: Rs.{close.iloc[-1]:.2f}")
print(f"RSI: {rsi.iloc[-1]:.1f}")

bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
print(f"BB Position: {bb_pos:.2f} (0=lower band, 1=upper band)")

ma_trend = "BULLISH" if ma5.iloc[-1] > ma20.iloc[-1] else "BEARISH"
print(f"MA5 vs MA20: {ma_trend}")

last3 = ret.iloc[-3:].tolist()
print(f"Last 3 days: {['UP' if r > 0 else 'DOWN' for r in last3]}")

print()
print("="*60)
print("TODAY'S SIGNALS:")
print("="*60)

signals = []
if rsi.iloc[-1] < 35:
    signals.append("RSI OVERSOLD -> BUY")
elif rsi.iloc[-1] > 65:
    signals.append("RSI OVERBOUGHT -> SELL")

if bb_pos < 0.1:
    signals.append("BOLLINGER LOWER -> BUY")
elif bb_pos > 0.9:
    signals.append("BOLLINGER UPPER -> SELL")

if all(r < 0 for r in last3):
    signals.append("3-DAY DOWN STREAK -> BUY (reversal)")
elif all(r > 0 for r in last3):
    signals.append("3-DAY UP STREAK -> SELL (reversal)")

signals.append(f"MA TREND -> {ma_trend}")

for s in signals:
    print(f"  • {s}")

if not signals:
    print("  • No clear signals today")
