import json
from collections import Counter

with open('/home/ubuntu/titan/agentic_trader/active_trades.json') as f:
    trades = json.load(f)

setup_counts = Counter()
setup_pnl = {}
setup_wins = Counter()
setup_losses = Counter()
exit_reasons = Counter()

for t in trades:
    setup = t.get('setup_type', 'UNKNOWN')
    status = t.get('status', 'OPEN')
    pnl = t.get('pnl', 0)
    exit_detail = t.get('exit_detail', {})
    exit_type = exit_detail.get('exit_type', status) if isinstance(exit_detail, dict) else status
    
    if status != 'OPEN':
        setup_counts[setup] += 1
        setup_pnl[setup] = setup_pnl.get(setup, 0) + pnl
        if pnl > 0:
            setup_wins[setup] += 1
        else:
            setup_losses[setup] += 1
            exit_reasons[exit_type] += 1

print('=== TRADE BREAKDOWN BY SETUP TYPE ===')
for setup in sorted(setup_counts.keys()):
    w = setup_wins[setup]
    l = setup_losses[setup]
    pnl = setup_pnl[setup]
    wr = w/(w+l)*100 if (w+l) > 0 else 0
    print(f'{setup:25s} W:{w:2d} L:{l:2d} WR:{wr:5.1f}% P&L: Rs{pnl:+10,.0f}')

print(f'\nTotal closed: {sum(setup_counts.values())}')
print(f'Total P&L: Rs{sum(setup_pnl.values()):+,.0f}')

print('\n=== EXIT REASONS FOR LOSSES ===')
for reason, count in exit_reasons.most_common():
    print(f'  {reason:30s} x{count}')

# Show losing OI_AGGR trades with details
print('\n=== OI_AGGR LOSING TRADES DETAIL ===')
oi_aggr_losses = [t for t in trades if t.get('setup_type') == 'OI_AGGR' and t.get('status') != 'OPEN' and t.get('pnl', 0) <= 0]
for t in sorted(oi_aggr_losses, key=lambda x: x.get('pnl', 0)):
    ul = t.get('underlying', '?').replace('NSE:', '')
    pnl = t.get('pnl', 0)
    direction = t.get('direction', '?')
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    exit_detail = t.get('exit_detail', {})
    exit_type = exit_detail.get('exit_type', '?') if isinstance(exit_detail, dict) else '?'
    held = exit_detail.get('held_minutes', '?') if isinstance(exit_detail, dict) else '?'
    print(f'  {ul:15s} {direction:4s} entry={entry:7.2f} exit={exit_p:7.2f} P&L=Rs{pnl:+8,.0f} exit={exit_type} held={held}min')

# Open trades
open_trades = [t for t in trades if t.get('status') == 'OPEN']
print(f'\n=== OPEN TRADES ({len(open_trades)}) ===')
for t in open_trades:
    ul = t.get('underlying', '?').replace('NSE:', '')
    print(f'  {ul:15s} {t.get("direction","?"):4s} entry={t.get("avg_price",0):.2f} setup={t.get("setup_type","?")}')
