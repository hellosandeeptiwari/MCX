import json
from collections import Counter

h = json.load(open('trade_history.json'))
closed = [t for t in h if t.get('status') not in ['OPEN', None]]

# Exit type breakdown
exits = Counter(t.get('status','') for t in closed)
print('EXIT TYPE BREAKDOWN:')
for k,v in exits.most_common():
    pnl = sum(t.get('pnl',0) for t in closed if t.get('status','')==k)
    wins = sum(1 for t in closed if t.get('status','')==k and t.get('pnl',0)>0)
    print(f'  {k:20s}: {v:3d} trades | W:{wins} L:{v-wins} | PnL: {pnl:>+12,.2f}')

print(f'\n--- WORST 10 TRADES ---')
losses = sorted(closed, key=lambda x: x.get('pnl',0))[:10]
for t in losses:
    sym = t.get('symbol','')[:42]
    print(f'  {sym:42s} {t.get("pnl",0):>+10,.2f} {t.get("status","")}')

print(f'\n--- BEST 10 TRADES ---')
best = sorted(closed, key=lambda x: x.get('pnl',0), reverse=True)[:10]
for t in best:
    sym = t.get('symbol','')[:42]
    print(f'  {sym:42s} {t.get("pnl",0):>+10,.2f} {t.get("status","")}')

# Direction analysis
print(f'\n--- DIRECTION ANALYSIS ---')
buys = [t for t in closed if t.get('side')=='BUY']
sells = [t for t in closed if t.get('side')=='SELL']
buy_wins = sum(1 for t in buys if t.get('pnl',0)>0)
sell_wins = sum(1 for t in sells if t.get('pnl',0)>0)
print(f'  BUY trades: {len(buys)} | Wins: {buy_wins} ({buy_wins/max(len(buys),1)*100:.0f}%) | PnL: {sum(t.get("pnl",0) for t in buys):+,.2f}')
print(f'  SELL trades: {len(sells)} | Wins: {sell_wins} ({sell_wins/max(len(sells),1)*100:.0f}%) | PnL: {sum(t.get("pnl",0) for t in sells):+,.2f}')

# Speed exit analysis
speed = [t for t in closed if 'SPEED' in t.get('status','')]
print(f'\n--- SPEED EXIT (premature exit) ---')
print(f'  Total: {len(speed)} | PnL: {sum(t.get("pnl",0) for t in speed):+,.2f}')
would_have = 0
for t in speed:
    if t.get('pnl',0) < 0:
        print(f'  LOSS: {t.get("symbol","")[:35]:35s} {t.get("pnl",0):>+8,.2f}')

# Credit spread analysis
spreads = [t for t in closed if t.get('is_credit_spread')]
print(f'\n--- CREDIT SPREADS ---')
print(f'  Total: {len(spreads)} | PnL: {sum(t.get("pnl",0) for t in spreads):+,.2f}')

# Naked options
naked = [t for t in closed if t.get('is_option') and not t.get('is_credit_spread')]
print(f'\n--- NAKED OPTIONS ---')
print(f'  Total: {len(naked)} | Wins: {sum(1 for t in naked if t.get("pnl",0)>0)} | PnL: {sum(t.get("pnl",0) for t in naked):+,.2f}')

# Time analysis - how long trades last
print(f'\n--- SESSION CUTOFF TRADES (forced exit at EOD) ---')
session = [t for t in closed if 'SESSION' in t.get('status','')]
for t in session:
    sym = t.get('symbol','')[:35]
    print(f'  {sym:35s} {t.get("pnl",0):>+8,.2f}')
