"""Impact analysis: What the 4 new fixes would have done on historical data"""
import json
from collections import defaultdict

trades = json.load(open('trade_history.json'))

daily = defaultdict(list)
for t in trades:
    ts = t.get('timestamp', '')
    if ts:
        daily[ts[:10]].append(t)

print('='*80)
print('  IMPACT ANALYSIS: WHAT THE 4 FIXES WOULD HAVE DONE')
print('='*80)

for d in sorted(daily.keys()):
    day = daily[d]
    elite = [t for t in day if (t.get('entry_score', 0) or 0) >= 78]
    near_elite = [t for t in day if 70 <= (t.get('entry_score', 0) or 0) < 78]
    all_scored = [t for t in day if (t.get('entry_score', 0) or 0) > 0]
    elite_pnl = sum(t.get('pnl', 0) or 0 for t in elite)
    near_elite_pnl = sum(t.get('pnl', 0) or 0 for t in near_elite)
    total_pnl = sum(t.get('pnl', 0) or 0 for t in day)
    
    print()
    print(f'  {d}: {len(day)} trades | Total PnL: {total_pnl:>+10,.0f}')
    
    if elite:
        print(f'    AUTO-FIRE ELIGIBLE (score>=78): {len(elite)} trades -> {elite_pnl:>+10,.0f}')
        for t in elite:
            sym = (t.get('symbol', '') or '')[:25]
            score = t.get('entry_score', 0) or 0
            pnl = t.get('pnl', 0) or 0
            print(f'      {sym:<25s} score={score:.0f} pnl={pnl:>+8,.0f}')
    else:
        print(f'    AUTO-FIRE ELIGIBLE: 0 (no scores >= 78)')
    
    if near_elite:
        print(f'    BONUS PICKS (score 70-77): {len(near_elite)} trades -> {near_elite_pnl:>+10,.0f}')
    
    score_70_plus = sum(1 for t in all_scored if (t.get('entry_score', 0) or 0) >= 70)
    if score_70_plus >= 3:
        print(f'    DYNAMIC PICKS: Would allow 5 picks (had {score_70_plus} stocks >= 70)')
    elif score_70_plus == 0:
        print(f'    DYNAMIC PICKS: Would restrict to 2 (choppy, {score_70_plus} stocks >= 70)')

print()
print('='*80)
print('  PROJECTED IMPACT (conservative)')
print('='*80)

all_elite = [t for t in trades if (t.get('entry_score', 0) or 0) >= 78]
if all_elite:
    e_pnl = sum(t.get('pnl', 0) or 0 for t in all_elite)
    e_wr = sum(1 for t in all_elite if (t.get('pnl', 0) or 0) > 0) / len(all_elite) * 100
    e_avg = e_pnl / len(all_elite)
    print(f'  Auto-Fire eligible (78+): {len(all_elite)} trades | WR: {e_wr:.0f}% | Avg P&L: {e_avg:>+,.0f}')
    print(f'  With Auto-Fire: These execute in ~2 sec (vs ~4-8 sec GPT wait time)')
    print(f'  Benefit: Better fill prices + never missed by GPT deliberation')

all_70_plus = [t for t in trades if (t.get('entry_score', 0) or 0) >= 70]
if all_70_plus:
    total_pnl_70 = sum(t.get('pnl', 0) or 0 for t in all_70_plus)
    avg_70 = total_pnl_70 / len(all_70_plus)
    print(f'  Bonus picks eligible (70+): {len(all_70_plus)} trades | Total P&L: {total_pnl_70:>+,.0f} | Avg: {avg_70:>+,.0f}')
    print(f'  With Dynamic 5 picks: More of these get through on trending days')
    print()
    projected_daily = avg_70 * 5
    total_pnl_all = sum(t.get('pnl', 0) or 0 for t in trades)
    current_avg = total_pnl_all / 6
    print(f'  PROJECTED: If Titan catches 5 score-70+ trades/day: {projected_daily:>+,.0f}/day ({projected_daily/500000*100:+.1f}%)')
    print(f'  vs current: {current_avg:>+,.0f}/day avg ({current_avg/500000*100:+.1f}%)')
    print()
    print(f'  KEY INSIGHT:')
    print(f'    Current: ~{current_avg/500000*100:.1f}%/day avg')
    print(f'    With fixes: ~{projected_daily/500000*100:.1f}%/day potential')
    print(f'    Improvement: {(projected_daily-current_avg)/current_avg*100:.0f}% more daily P&L')
