import json
from datetime import datetime

with open('trade_history.json', 'r') as f:
    history = json.load(f)

today = datetime.now().strftime('%Y-%m-%d')
today_trades = [t for t in history if t.get('timestamp', t.get('date', t.get('entry_time', '')))[:10] == today]

print(f'=== TODAY ({today}) TRADE ANALYSIS ===')
print(f'Total trades: {len(today_trades)}')

wins = [t for t in today_trades if t.get('pnl', 0) > 0]
losses = [t for t in today_trades if t.get('pnl', 0) < 0]
flat = [t for t in today_trades if t.get('pnl', 0) == 0]

total_pnl = sum(t.get('pnl', 0) for t in today_trades)
print(f'Wins: {len(wins)} | Losses: {len(losses)} | Flat: {len(flat)}')
if today_trades:
    print(f'Win Rate: {len(wins)/len(today_trades)*100:.0f}%')
print(f'Total P&L: Rs {total_pnl:,.0f}')
print()

print(f'{"Symbol":<28} {"Dir":<5} {"Scr":>4} {"Entry":>8} {"Exit":>8} {"P&L":>10} {"Exit Reason":<22} {"MLGt":>5} {"MLDr":>5} {"Time":<8}')
print('-' * 120)
for t in sorted(today_trades, key=lambda x: x.get('entry_time', '')):
    sym = t.get('symbol', t.get('instrument', ''))[:27]
    direction = t.get('direction', t.get('action', '?'))[:4]
    score = t.get('score', t.get('conviction_score', 0)) or 0
    entry = t.get('entry_price', t.get('avg_price', 0)) or 0
    exit_p = t.get('exit_price', 0) or 0
    pnl = t.get('pnl', 0) or 0
    reason = (t.get('exit_reason', t.get('status', '')) or '')[:21]
    ml_gate = t.get('ml_gate_confidence', t.get('entry_metadata', {}).get('ml_move_prob', 0)) or 0
    ml_dir = t.get('ml_direction_confidence', 0) or 0
    entry_time = (t.get('timestamp', t.get('entry_time', '')) or '')[-8:]
    ml_signal = (t.get('entry_metadata', {}) or {}).get('ml_signal', '?') or '?'
    marker = 'W' if pnl > 0 else 'L' if pnl < 0 else '-'
    mismatch = '*' if (direction[:3] == 'BUY' and ml_signal == 'DOWN') or (direction[:3] == 'SEL' and ml_signal == 'UP') else ' '
    print(f'{marker}{mismatch}{sym:<27} {direction:<5} {score:>4} {entry:>8.1f} {exit_p:>8.1f} {pnl:>+10,.0f} {reason:<22} {ml_gate:>5.2f} {ml_dir:>5.2f} ml:{ml_signal:<4} {entry_time}')

print()
print('=== LOSS DETAIL (worst first) ===')
for t in sorted(losses, key=lambda x: x.get('pnl', 0)):
    sym = t.get('symbol', '')[:25]
    pnl = t.get('pnl', 0) or 0
    reason = t.get('exit_reason', '') or ''
    score = t.get('score', t.get('conviction_score', 0)) or 0
    ml_gate = t.get('ml_gate_confidence', 0) or 0
    ml_dir = t.get('ml_direction_confidence', 0) or 0
    direction = t.get('direction', '?') or '?'
    entry_t = (t.get('timestamp', t.get('entry_time', '')) or '')[-8:]
    pct = t.get('pnl_pct', 0) or 0
    ml_signal = ((t.get('entry_metadata') or {}).get('ml_signal', '?')) or '?'
    htf = ((t.get('entry_metadata') or {}).get('htf_alignment', '?') or '?')[:10]
    mismatch = ' MISMATCH!' if (direction[:3] == 'BUY' and ml_signal == 'DOWN') or (direction[:3] == 'SEL' and ml_signal == 'UP') else ''
    print(f'  {pnl:>+9,.0f} ({pct:>+6.1f}%)  {sym:<25} {direction:<5} Score:{score:<4} Gate:{ml_gate:.2f} Dir:{ml_dir:.2f} ML:{ml_signal:<4} HTF:{htf}  {reason}  @{entry_t}{mismatch}')

print()
print('=== WIN DETAIL ===')
for t in sorted(wins, key=lambda x: x.get('pnl', 0), reverse=True):
    sym = t.get('symbol', '')[:25]
    pnl = t.get('pnl', 0) or 0
    reason = t.get('exit_reason', '') or ''
    score = t.get('score', t.get('conviction_score', 0)) or 0
    direction = t.get('direction', '?') or '?'
    entry_t = (t.get('timestamp', t.get('entry_time', '')) or '')[-8:]
    pct = t.get('pnl_pct', 0) or 0
    ml_signal = ((t.get('entry_metadata') or {}).get('ml_signal', '?')) or '?'
    print(f'  {pnl:>+9,.0f} ({pct:>+6.1f}%)  {sym:<25} {direction:<5} Score:{score:<4} ML:{ml_signal:<4}  {reason}  @{entry_t}')

# Summary stats
print()
print('=== PATTERN ANALYSIS ===')
buy_trades = [t for t in today_trades if t.get('direction', '') == 'BUY']
sell_trades = [t for t in today_trades if t.get('direction', '') == 'SELL']
buy_wins = len([t for t in buy_trades if t.get('pnl', 0) > 0])
sell_wins = len([t for t in sell_trades if t.get('pnl', 0) > 0])
print(f'BUY trades: {len(buy_trades)} (wins: {buy_wins}, WR: {buy_wins/len(buy_trades)*100:.0f}%)' if buy_trades else 'BUY: 0')
print(f'SELL trades: {len(sell_trades)} (wins: {sell_wins}, WR: {sell_wins/len(sell_trades)*100:.0f}%)' if sell_trades else 'SELL: 0')

# ML direction mismatch analysis
mismatches = []
for t in today_trades:
    d = t.get('direction', '')
    ml_sig = t.get('entry_metadata', {}).get('ml_signal', '')
    if (d == 'BUY' and ml_sig == 'DOWN') or (d == 'SELL' and ml_sig == 'UP'):
        mismatches.append(t)
mismatch_losses = [t for t in mismatches if t.get('pnl', 0) < 0]
print(f'\nML DIRECTION MISMATCHES: {len(mismatches)}/{len(today_trades)} trades went AGAINST ML signal!')
if mismatches:
    mm_pnl = sum(t.get('pnl', 0) for t in mismatches)
    print(f'  Mismatch P&L: Rs {mm_pnl:,.0f} ({len(mismatch_losses)} losses)')
    for t in mismatches:
        sym = t.get('symbol', '')[:20]
        d = t.get('direction', '')
        ml = t.get('entry_metadata', {}).get('ml_signal', '?')
        pnl = t.get('pnl', 0)
        print(f'  -> {sym} {d} vs ML:{ml} => Rs {pnl:+,.0f}')

# Avg score for wins vs losses
if wins:
    avg_win_score = sum(t.get('score', 0) for t in wins) / len(wins)
    print(f'Avg Score (wins): {avg_win_score:.0f}')
if losses:
    avg_loss_score = sum(t.get('score', 0) for t in losses) / len(losses)
    print(f'Avg Score (losses): {avg_loss_score:.0f}')

# Exit reason breakdown
from collections import Counter
exit_reasons = Counter(t.get('exit_reason', 'UNKNOWN') for t in today_trades)
print(f'\nExit Reasons: {dict(exit_reasons)}')

# ML confidence analysis
if today_trades:
    avg_gate_all = sum(t.get('ml_gate_confidence', 0) for t in today_trades) / len(today_trades)
    avg_dir_all = sum(t.get('ml_direction_confidence', 0) for t in today_trades) / len(today_trades)
    print(f'Avg ML Gate (all): {avg_gate_all:.2f}')
    print(f'Avg ML Direction (all): {avg_dir_all:.2f}')
    if wins:
        avg_gate_w = sum(t.get('ml_gate_confidence', 0) for t in wins) / len(wins)
        avg_dir_w = sum(t.get('ml_direction_confidence', 0) for t in wins) / len(wins)
        print(f'Avg ML Gate (wins): {avg_gate_w:.2f} | Dir: {avg_dir_w:.2f}')
    if losses:
        avg_gate_l = sum(t.get('ml_gate_confidence', 0) for t in losses) / len(losses)
        avg_dir_l = sum(t.get('ml_direction_confidence', 0) for t in losses) / len(losses)
        print(f'Avg ML Gate (losses): {avg_gate_l:.2f} | Dir: {avg_dir_l:.2f}')
