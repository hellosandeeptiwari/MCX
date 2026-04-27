#!/usr/bin/env python3
"""Quick P&L analysis for today's trades."""
import re

with open('/home/ubuntu/titan/logs/titan.log') as f:
    lines = f.readlines()

trades = []
for l in lines:
    if 'removed from memory' not in l or 'P&L:' not in l:
        continue
    m_sym = re.search(r'(NFO:\S+)', l)
    m_pnl = re.search(r'P&L:\s*₹([+-]?[\d,]+\.?\d*)', l)
    if m_sym and m_pnl:
        sym = m_sym.group(1).split('26MAR')[0].replace('NFO:', '')
        pnl = float(m_pnl.group(1).replace(',', ''))
        trades.append((sym, pnl))

wins = [(s, p) for s, p in trades if p > 0]
losses = [(s, p) for s, p in trades if p < 0]
total = sum(p for _, p in trades)

print(f'Closed: {len(trades)} | W:{len(wins)} L:{len(losses)}')
print(f'Realized P&L: Rs{total:+,.0f}')
print(f'Avg win: Rs{sum(p for _,p in wins)/len(wins):+,.0f}' if wins else '')
print(f'Avg loss: Rs{sum(p for _,p in losses)/len(losses):+,.0f}' if losses else '')
print()
print('BIGGEST LOSERS:')
for s, p in sorted(losses)[:7]:
    print(f'  {s}: Rs{p:+,.0f}')
print()
print('BIGGEST WINNERS:')
for s, p in sorted(wins, key=lambda x: -x[1])[:7]:
    print(f'  {s}: Rs{p:+,.0f}')
