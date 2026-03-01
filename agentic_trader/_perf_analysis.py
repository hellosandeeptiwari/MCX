"""Quick performance analysis across all daily summaries."""
import json, os
from collections import defaultdict

d = 'daily_summaries'
all_days = []
for fname in sorted(os.listdir(d)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(d, fname)) as f:
        s = json.load(f)
    all_days.append(s)
    date = s['date']
    pnl = s['daily_pnl']
    roi = s['return_pct']
    wr = s['win_rate']
    trades = s['total_trades']
    wins = s['wins']
    losses = s['losses']
    avg_w = s.get('avg_winner', 0)
    avg_l = s.get('avg_loser', 0)
    
    print(f"===== {date} | PnL: {pnl:>+10,.0f} | ROI: {roi:>+6.2f}% | WR: {wr:.0f}% | {trades}T ({wins}W/{losses}L) =====")
    print(f"  Avg Winner: {avg_w:>+8,.0f} | Avg Loser: {avg_l:>+8,.0f} | Payoff: {s.get('payoff_ratio',0):.2f}")
    
    by_strat = s.get('by_strategy', {})
    for strat, info in sorted(by_strat.items(), key=lambda x: x[1].get('pnl', 0), reverse=True):
        sp = info.get('pnl', 0)
        sc = info.get('count', 0)
        sw = info.get('win_rate', 0)
        print(f"  {strat:22s}  {sc:2d} trades  PnL={sp:>+10,.0f}  WR={sw:.0f}%")
    print()

# Aggregate
total_pnl = sum(s['daily_pnl'] for s in all_days)
avg_roi = sum(s['return_pct'] for s in all_days) / len(all_days)
profit_days = sum(1 for s in all_days if s['daily_pnl'] > 0)
loss_days = sum(1 for s in all_days if s['daily_pnl'] < 0)
total_trades = sum(s['total_trades'] for s in all_days)
total_wins = sum(s['wins'] for s in all_days)
total_losses = sum(s['losses'] for s in all_days)

# Strategy aggregate
strat_agg = defaultdict(lambda: {'pnl': 0, 'count': 0, 'wins': 0})
for s in all_days:
    for strat, info in s.get('by_strategy', {}).items():
        strat_agg[strat]['pnl'] += info.get('pnl', 0)
        strat_agg[strat]['count'] += info.get('count', 0)
        w = info.get('wins', 0)
        if w == 0 and info.get('win_rate', 0) > 0:
            w = round(info.get('count', 0) * info.get('win_rate', 0) / 100)
        strat_agg[strat]['wins'] += w

print("=" * 70)
print(f"AGGREGATE ({len(all_days)} days, {total_trades} trades)")
print("=" * 70)
print(f"Total PnL:     {total_pnl:>+12,.0f}")
print(f"Avg daily ROI: {avg_roi:>+8.2f}%")
print(f"Profit days:   {profit_days} | Loss days: {loss_days} | Day WR: {profit_days/len(all_days)*100:.0f}%")
print(f"Trade W/L:     {total_wins}/{total_losses} | Trade WR: {total_wins/(total_wins+total_losses)*100:.1f}%")

best = max(all_days, key=lambda s: s['return_pct'])
worst = min(all_days, key=lambda s: s['return_pct'])
print(f"Best day:      {best['date']} ({best['return_pct']:+.2f}%)")
print(f"Worst day:     {worst['date']} ({worst['return_pct']:+.2f}%)")

print(f"\nBy Strategy (aggregate):")
for strat, info in sorted(strat_agg.items(), key=lambda x: x[1]['pnl'], reverse=True):
    cnt = info['count']
    pnl = info['pnl']
    w = info['wins']
    wr = (w/cnt*100) if cnt > 0 else 0
    avg = pnl / cnt if cnt > 0 else 0
    print(f"  {strat:22s}  {cnt:3d} trades  PnL={pnl:>+10,.0f}  WR={wr:.0f}%  Avg={avg:>+6,.0f}/trade")

# ROI consistency check
print(f"\nDaily ROI sequence: ", end="")
for s in all_days:
    print(f"{s['return_pct']:+.1f}% ", end="")
print()

# How many days >= 10%?
above_10 = sum(1 for s in all_days if s['return_pct'] >= 10)
print(f"Days >= 10% ROI: {above_10}/{len(all_days)} ({above_10/len(all_days)*100:.0f}%)")
above_5 = sum(1 for s in all_days if s['return_pct'] >= 5)
print(f"Days >= 5% ROI:  {above_5}/{len(all_days)} ({above_5/len(all_days)*100:.0f}%)")
above_0 = sum(1 for s in all_days if s['return_pct'] >= 0)
print(f"Days >= 0% ROI:  {above_0}/{len(all_days)} ({above_0/len(all_days)*100:.0f}%)")
