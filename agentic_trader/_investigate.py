"""Investigate: analyze_sizing.py recommendations vs today's actual performance"""
import json

with open('daily_summaries/daily_summary_2026-02-18.json') as f:
    ds = json.load(f)
with open('active_trades.json') as f:
    at = json.load(f)

capital = 500000

print('='*75)
print('POSITION SIZING INVESTIGATION: analyze_sizing.py RECS vs TODAY (Feb 18)')
print('='*75)

print('\n--- CURRENT CONFIG (config.py) ---')
print('  RISK_PER_TRADE:       3.5% base (tiered: 5%/3.5%/2%)')
print('  MAX_POSITIONS:        20 (mixed: 6, trending: 12)')
print('  MAX_PREMIUM/TRADE:    50% of capital')
print('  MAX_LOTS/TRADE:       15')
print(f'  CAPITAL:              Rs {capital:,}')

print('\n--- Feb 11 RECOMMENDATIONS (analyze_sizing.py) ---')
print('  risk_per_trade:     5% (Rs 25,000)')
print('  min_lots (premium): 2-3 lots')
print('  max_positions:      3-4 max')
print('  target %:           80-100% (runners)')
print('  SL %:               25% (tighter)')
print('  time_stop:          12+ candles')

trades = ds['individual_trades']
active = at.get('active_trades', [])
print(f'\n--- TODAY: {len(trades)} CLOSED TRADES + {len(active)} ACTIVE ---')
header = f"{'Symbol':<45} {'Premium':>10} {'%Cap':>6} {'PnL':>9} {'Return':>8} {'Exit':>15}"
print(header)
print('-'*95)

total_premium = 0
naked_premiums = []
for t in trades:
    sym = t['symbol']
    entry = t['entry']
    pnl = t['pnl']
    exit_type = t['result']
    
    if '|' not in sym:  # naked option
        exit_p = t['exit']
        if entry != exit_p:
            qty = abs(pnl / (exit_p - entry))
        else:
            qty = 0
        premium = qty * entry
        naked_premiums.append(premium)
    else:  # spread
        premium = abs(pnl) * 5  # rough estimate
    
    pct_cap = premium / capital * 100 if premium > 0 else 0
    ret = pnl / premium * 100 if premium > 0 else 0
    total_premium += premium
    print(f'{sym:<45} Rs{premium:>8,.0f} {pct_cap:>5.1f}% {pnl:>+9,.0f} {ret:>+6.1f}% {exit_type:<15}')

for a in active:
    qty = a['quantity']
    entry_p = a['avg_price']
    premium = qty * entry_p
    pct_cap = premium / capital * 100
    total_premium += premium
    lots = a.get('lots', 0)
    print(f"{a['symbol']:<45} Rs{premium:>8,.0f} {pct_cap:>5.1f}%   (OPEN)            ACTIVE ({lots} lots)")

print(f'\n{"="*75}')
print('DIAGNOSIS')
print('='*75)

# 1. Position sizing
if naked_premiums:
    avg_prem = sum(naked_premiums) / len(naked_premiums)
    max_prem = max(naked_premiums)
    min_prem = min(naked_premiums)
    print(f'\n1. NAKED OPTION SIZING:')
    print(f'   Avg premium/trade:  Rs {avg_prem:,.0f} ({avg_prem/capital*100:.1f}% of capital)')
    print(f'   Range:              Rs {min_prem:,.0f} - Rs {max_prem:,.0f}')
    print(f'   Feb 11 avg was:     Rs ~20,000 (3-6%) — PATHETICALLY SMALL')
    if avg_prem/capital*100 > 10:
        print(f'   STATUS: IMPROVED — now using meaningful position sizes')
    else:
        print(f'   STATUS: STILL TOO SMALL')

# 2. Win/loss quality
wins = [t for t in trades if t['pnl'] > 0]
losses = [t for t in trades if t['pnl'] <= 0]
print(f'\n2. PAYOFF RATIO:')
print(f'   Wins: {len(wins)}, Losses: {len(losses)}, Win Rate: {ds["win_rate"]:.0f}%')
print(f'   Avg winner: Rs {ds["avg_winner"]:+,.0f}')
print(f'   Avg loser:  Rs {ds["avg_loser"]:+,.0f}')
print(f'   Payoff ratio: {ds["payoff_ratio"]}x')
print(f'   TARGET: 2.0x+ — {"ACHIEVED" if ds["payoff_ratio"] >= 2.0 else "NOT MET (losers still bigger than winners)"}')

# 3. Strategy breakdown
print(f'\n3. STRATEGY P&L:')
for strat, data in ds['by_strategy'].items():
    emoji = '✅' if data['pnl'] > 0 else '❌'
    print(f'   {emoji} {strat:<15}: {data["wins"]}W/{data["losses"]}L = Rs {data["pnl"]:+,.0f}')

# 4. Position count
total_positions = ds['total_trades'] + len(active)
print(f'\n4. POSITION COUNT:')
print(f'   Today: {ds["total_trades"]} closed + {len(active)} active = {total_positions} total')
print(f'   Rec:   3-4 max concentrated')
print(f'   Config: MAX_POSITIONS=20, MIXED=6, TRENDING=12')
print(f'   STATUS: {"OK" if total_positions <= 6 else "STILL TOO MANY — scattered capital"}')

# 5. R-multiples and hold time
print(f'\n5. TRADE QUALITY:')
print(f'   Avg R-multiple:  {ds["exit_quality"]["avg_r_multiple"]}')
print(f'   Avg candles held: {ds["exit_quality"]["avg_candles_held"]}')
print(f'   Best R:          {max(t["r_multiple"] for t in trades)} (WAAREEENER QUICK_PROFIT)')
print(f'   Worst R:         {min(t["r_multiple"] for t in trades)}')

# 6. Exit type analysis
print(f'\n6. EXIT TYPES:')
for etype, count in ds['by_exit_type'].items():
    print(f'   {etype:<25}: {count}')
theta_exits = ds['by_exit_type'].get('THETA_DECAY_WARNING', 0)
if theta_exits > 0:
    theta_pnl = sum(t['pnl'] for t in trades if t['result'] == 'THETA_DECAY_WARNING')
    print(f'   ⚠️  THETA_DECAY exits lost Rs {abs(theta_pnl):,.0f} total')

# 7. Individual loss deep dive
print(f'\n7. LOSS DEEP DIVE:')
for t in sorted(trades, key=lambda x: x['pnl']):
    if t['pnl'] < 0:
        print(f'   {t["symbol"]:<45}')
        print(f'     PnL: Rs {t["pnl"]:+,.0f} | Score: {t["entry_score"]} | ADX: {t["adx_at_entry"]} | FT: {t["ft_at_entry"]}')
        print(f'     Held: {t["candles_held"]} candles | R: {t["r_multiple"]} | Exit: {t["result"]}')
        if 'INOXWIND' in t['symbol']:
            print(f'     >> NOW BLOCKED by ADX < 22 gate (ADX was {t["adx_at_entry"]})')

# 8. Active trade sizing  
if active:
    print(f'\n8. ACTIVE TRADE SIZING:')
    for a in active:
        qty = a['quantity']
        entry_p = a['avg_price']
        premium = qty * entry_p
        meta = a.get('entry_metadata', {})
        lots = a.get('lots', 0)
        orig = meta.get('original_lots', '?')
        ml_f = meta.get('ml_sizing_factor', '?')
        tier = meta.get('score_tier', '?') 
        score = meta.get('entry_score', a.get('entry_score', '?'))
        adx = meta.get('adx', '?')
        print(f'   {a["symbol"]}')
        print(f'     {lots} lots (orig {orig}, ML×{ml_f}) | Rs {premium:,.0f} ({premium/capital*100:.1f}%) | Tier: {tier} | Score: {score} | ADX: {adx}')

# VERDICT
print(f'\n{"="*75}')
print('VERDICT')
print('='*75)
print(f'  Daily P&L: Rs {ds["daily_pnl"]:+,.0f} ({ds["return_pct"]:+.2f}%)')
print()
print('  IMPLEMENTED vs analyze_sizing.py RECS:')
print('    ✅ Tiered risk (5%/3.5%/2%)    — active in config')
print('    ✅ Premium tier min 2 lots      — in position sizer')
print('    ✅ ADX < 22 hard block          — added today')
print('    ✅ ADX >= 30 bonus +3           — added today')
print('    ✅ 16% universal quick profit   — added prior session')
print('    ❌ Max positions 3-4            — still 20/6/12')
print('    ❌ Target 80-100%               — currently ~80% (SL is 28% via 0.72x)')
print('    ❌ SL tighten to 25%            — still 30% (0.7x multiplier)')
print('    ❓ Time stop 12+ candles        — need to check exit_manager config')
print()
print('  CRITICAL PROBLEM: NAKED OPTIONS ARE -Rs 15,865 (all 3 losses)')
print('  SPREADS ARE +Rs 13,681 (all profitable)') 
print('  => SPREADS ARE WORKING, NAKED OPTIONS ARE BLEEDING')
print()
print('  TOP PRIORITIES:')
print('  1. FIX PAYOFF RATIO (0.65x → 2.0x): Winners too small, losers too big')
print('  2. REDUCE MAX POSITIONS: 20→4-6 to concentrate capital')
print('  3. CONSIDER: Shift more capital toward spreads over naked options')
