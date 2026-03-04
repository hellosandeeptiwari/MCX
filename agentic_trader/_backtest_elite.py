#!/usr/bin/env python3
"""Backtest: would new ELITE gates have blocked today's losing trades?"""

trades = [
    {'sym': 'VBL', 'score': 88, 'ml_conf': 0.483, 'ml_move': 0.477, 'chg_pct': -3.11, 'pnl': -9915},
    {'sym': 'RVNL', 'score': 84, 'ml_conf': 0.500, 'ml_move': 0.496, 'chg_pct': -2.8, 'pnl': 305},
]

min_conf = 0.55
min_move = 0.52
max_existing_move = 2.5

print('ELITE Backtest (Mar 4, 2026)')
print('=' * 70)
for t in trades:
    blocked = []
    conf = t['ml_conf']
    move = t['ml_move']
    chg = abs(t['chg_pct'])
    
    if conf < min_conf:
        blocked.append(f'ML conf {conf:.3f} < {min_conf}')
    if move < min_move:
        blocked.append(f'ML move_prob {move:.3f} < {min_move}')
    if chg > max_existing_move:
        blocked.append(f'Move exhausted: {chg:.1f}% > {max_existing_move}%')
    
    status = '🚫 BLOCKED' if blocked else '✅ PASS'
    sym = t['sym']
    score = t['score']
    pnl = t['pnl']
    print(f'{sym:8} score={score:3} | P&L: ₹{pnl:+,} | {status}')
    for b in blocked:
        print(f'          → {b}')
    print()

total_saved = sum(abs(t['pnl']) for t in trades 
                  if any([t['ml_conf'] < min_conf, t['ml_move'] < min_move, abs(t['chg_pct']) > max_existing_move]) 
                  and t['pnl'] < 0)
print(f'Loss saved today: ₹+{total_saved:,}')
