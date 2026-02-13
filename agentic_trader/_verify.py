import json
h = json.load(open('trade_history.json'))
closed = [t for t in h if t.get('status') not in ['OPEN', None]]
total = sum(t.get('pnl', 0) for t in closed)

# Check suspicious big trades
for name in ['HINDCOPPER', 'BAJFINANCE', 'ITC', 'MCX', 'TITAN']:
    trades = [t for t in closed if name in t.get('symbol', '')]
    if trades:
        print(f'\n{name} trades ({len(trades)}):')
        for t in trades:
            sym = t.get('symbol', '')
            qty = t.get('quantity', 0)
            entry = t.get('avg_price', 0)
            exit_p = t.get('exit_price', 0)
            pnl = t.get('pnl', 0)
            side = t.get('side', '')
            status = t.get('status', '')
            is_opt = t.get('is_option', False)
            # Verify PnL math
            if side == 'BUY':
                calc_pnl = (exit_p - entry) * qty
            else:
                calc_pnl = (entry - exit_p) * qty
            match = abs(calc_pnl - pnl) < 1
            print(f'  {sym[:45]}')
            print(f'    {side} qty={qty} entry={entry} exit={exit_p}')
            print(f'    Reported PnL={pnl:+,.2f} | Calc PnL={calc_pnl:+,.2f} | Match={match}')
            print(f'    Status={status} | is_option={is_opt}')

print(f'\nTotal PnL: {total:+,.2f}')
print(f'Total closed: {len(closed)}')

# Verify overall
wins = sum(1 for t in closed if t.get('pnl',0) > 0)
losses = sum(1 for t in closed if t.get('pnl',0) < 0)
print(f'Wins: {wins} | Losses: {losses} | Win%: {wins/len(closed)*100:.1f}%')
