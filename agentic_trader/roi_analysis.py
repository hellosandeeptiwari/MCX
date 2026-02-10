import json
from datetime import datetime

h = json.load(open('trade_history.json'))
today = [t for t in h if '2026-02-10' in t.get('timestamp','')]

print("=" * 100)
print("DEEP ROI ANALYSIS - What's actually killing your money")
print("=" * 100)

total_friction = 0
for t in today:
    entry_t = datetime.fromisoformat(t['timestamp'])
    exit_t = datetime.fromisoformat(t.get('closed_at', t.get('exit_time', t['timestamp'])))
    mins = (exit_t - entry_t).total_seconds() / 60
    entry_p = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    pct = (exit_p - entry_p) / entry_p * 100 if entry_p > 0 else 0
    pnl = t.get('pnl', 0)
    qty = t.get('quantity', 0)
    status = t.get('status', '?')
    sym = t['symbol'][-28:]
    print(f"  {sym:28s} | held {mins:4.0f}min | {pct:+6.1f}% | PnL:{pnl:+8.0f} | {status}")
    # Estimate friction (brokerage + STT + stamp + GST for options)
    turnover = entry_p * qty + exit_p * qty
    friction = 40 + turnover * 0.0005  # flat ₹40 + 0.05% STT/stamps approx
    total_friction += friction

print()
print("PROBLEM #1: SPEED GATE IS DESTROYING VALUE")
speed_gate = [t for t in today if t.get('status')=='OPTION_SPEED_GATE']
time_stop = [t for t in today if t.get('status')=='TIME_STOP']
target_hit = [t for t in today if t.get('status')=='TARGET_HIT']

sg_pnl = sum(t.get('pnl', 0) for t in speed_gate)
ts_pnl = sum(t.get('pnl', 0) for t in time_stop)

print(f"  Speed gate: {len(speed_gate)} trades, PnL: {sg_pnl:+,.0f}")
print(f"  Time stop:  {len(time_stop)} trades, PnL: {ts_pnl:+,.0f}")
print(f"  Target hit: {len(target_hit)} trades (ZERO!)")

# Positive trades killed by speed gate
pos_killed = [t for t in speed_gate if t.get('pnl', 0) > 0]
print(f"\n  Profitable trades KILLED by speed gate: {len(pos_killed)}")
for t in pos_killed:
    entry_p = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    pct = (exit_p - entry_p) / entry_p * 100 if entry_p > 0 else 0
    print(f"    {t['symbol'][-25:]:25s} was at {pct:+.1f}% → PnL: {t.get('pnl',0):+,.0f}")

print()
print("PROBLEM #2: TRADE CHURNING")
print(f"  {len(today)} trades in one day")
print(f"  Estimated friction cost: ~₹{total_friction:,.0f}")
print(f"  That's {total_friction/len(today):,.0f} per trade in friction")

print()
print("PROBLEM #3: WIN RATE vs R:R MISMATCH")
winners = [t for t in today if t.get('pnl', 0) > 0]
losers = [t for t in today if t.get('pnl', 0) <= 0]
win_rate = len(winners) / len(today) * 100
avg_win = sum(t.get('pnl', 0) for t in winners) / len(winners) if winners else 0
avg_loss = abs(sum(t.get('pnl', 0) for t in losers) / len(losers)) if losers else 0
expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
print(f"  Win rate: {win_rate:.0f}%")
print(f"  Avg winner: ₹{avg_win:+,.0f}")
print(f"  Avg loser:  ₹{avg_loss:,.0f}")
print(f"  Expectancy per trade: ₹{expectancy:+,.0f}")
print(f"  Need win rate > {avg_loss/(avg_win+avg_loss)*100:.0f}% to break even at this R:R")

print()
print("PROBLEM #4: TIME OF ENTRY")
for t in today:
    entry_t = datetime.fromisoformat(t['timestamp'])
    hour = entry_t.hour
    pnl = t.get('pnl', 0)
    print(f"  {entry_t.strftime('%H:%M')} entry → PnL: {pnl:+,.0f}")
