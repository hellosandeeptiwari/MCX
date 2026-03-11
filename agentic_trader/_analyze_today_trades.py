"""Analyze today's trades from trade_history.json and logs"""
import json
from datetime import datetime

# Load trade history
with open('trade_history_today.json') as f:
    trades = json.load(f)

# Check what dates exist
dates = {}
for t in trades:
    et = str(t.get('entry_time', ''))[:10]
    dates[et] = dates.get(et, 0) + 1

print("=== TRADE HISTORY DATES ===")
for d in sorted(dates.keys())[-7:]:
    print(f"  {d}: {dates[d]} trades")

# Get today or most recent
target_date = '2026-03-09'
today_trades = [t for t in trades if str(t.get('entry_time', ''))[:10] == target_date]

if not today_trades:
    # Try the most recent date
    latest = sorted(dates.keys())[-1]
    print(f"\nNo trades for {target_date}, using latest: {latest}")
    today_trades = [t for t in trades if str(t.get('entry_time', ''))[:10] == latest]
    target_date = latest

print(f"\n{'='*80}")
print(f"TRADE ANALYSIS FOR {target_date}: {len(today_trades)} trades")
print(f"{'='*80}")

# Sort by entry time
today_trades.sort(key=lambda x: x.get('entry_time', ''))

wins = 0
losses = 0
total_pnl = 0
total_win_amt = 0
total_loss_amt = 0
by_setup = {}
by_exit = {}

for t in today_trades:
    sym = t.get('underlying', t.get('symbol', '?')).replace('NSE:', '')
    setup = t.get('setup_type', t.get('trade_type', '?'))
    direction = t.get('direction', '?')
    pnl = t.get('realized_pnl', t.get('pnl', 0)) or 0
    exit_r = t.get('exit_reason', '?')
    entry_t = str(t.get('entry_time', '?'))[11:19]
    exit_t = str(t.get('exit_time', '?'))[11:19] if t.get('exit_time') else 'OPEN'
    entry_price = t.get('entry_price', 0)
    
    total_pnl += pnl
    tag = '✅' if pnl > 0 else '❌' if pnl < 0 else '➖'
    if pnl > 0:
        wins += 1
        total_win_amt += pnl
    elif pnl < 0:
        losses += 1
        total_loss_amt += pnl
    
    # By setup
    if setup not in by_setup:
        by_setup[setup] = {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
    by_setup[setup]['count'] += 1
    by_setup[setup]['pnl'] += pnl
    if pnl > 0:
        by_setup[setup]['wins'] += 1
    elif pnl < 0:
        by_setup[setup]['losses'] += 1
    
    # By exit reason
    if exit_r not in by_exit:
        by_exit[exit_r] = {'count': 0, 'pnl': 0}
    by_exit[exit_r]['count'] += 1
    by_exit[exit_r]['pnl'] += pnl
    
    print(f"{tag} {sym:15s} {direction:4s} {setup:20s} {entry_t}-{exit_t}  P&L: ₹{pnl:>+9,.0f}  [{exit_r}]")

total = wins + losses
wr = (wins / total * 100) if total > 0 else 0
avg_win = (total_win_amt / wins) if wins > 0 else 0
avg_loss = (total_loss_amt / losses) if losses > 0 else 0
rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"  Total Trades: {total} | Wins: {wins} | Losses: {losses}")
print(f"  Win Rate: {wr:.0f}%")
print(f"  Net P&L: ₹{total_pnl:+,.0f}")
print(f"  Avg Win: ₹{avg_win:+,.0f} | Avg Loss: ₹{avg_loss:+,.0f}")
print(f"  Risk/Reward: {rr:.2f}")

print(f"\n--- BY SETUP TYPE ---")
for setup, data in sorted(by_setup.items(), key=lambda x: x[1]['pnl'], reverse=True):
    wr_s = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
    print(f"  {setup:20s}: {data['count']} trades | {data['wins']}W-{data['losses']}L ({wr_s:.0f}%) | P&L: ₹{data['pnl']:+,.0f}")

print(f"\n--- BY EXIT REASON ---")
for reason, data in sorted(by_exit.items(), key=lambda x: x[1]['pnl'], reverse=True):
    print(f"  {reason:30s}: {data['count']} trades | P&L: ₹{data['pnl']:+,.0f}")
