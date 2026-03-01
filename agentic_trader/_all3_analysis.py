"""Analyze ALL_AGREE (ALL3) trades across all ledger files."""
import json, glob, os

all_trades = []
for f in sorted(glob.glob('trade_ledger/*.jsonl')):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                all_trades.append(rec)
            except:
                pass

# Filter for ALL_AGREE EXIT trades
all3_exits = [t for t in all_trades if t.get('source') == 'ALL_AGREE' and t.get('event') == 'EXIT']
all3_entries = [t for t in all_trades if t.get('source') == 'ALL_AGREE' and t.get('event') == 'ENTRY']

print(f'=== ALL_AGREE TRADE ANALYSIS ===')
print(f'Entries: {len(all3_entries)} | Exits: {len(all3_exits)}')
print()

print('--- CLOSED TRADES ---')
total_pnl = 0
wins = 0
losses = 0
for t in all3_exits:
    sym = t.get('symbol', '?')
    underlying = t.get('underlying', '?')
    pnl = t.get('pnl', 0)
    total_pnl += pnl
    if pnl > 0:
        wins += 1
    else:
        losses += 1
    exit_type = t.get('exit_type', '?')
    entry = t.get('entry_price', 0)
    exit_p = t.get('exit_price', 0)
    dr = t.get('dr_score', '?')
    smart = t.get('smart_score', '?')
    direction = t.get('direction', '?')
    date = t.get('ts', '?')[:16]
    candles = t.get('candles_held', '?')
    regime = t.get('regime', '?')
    qty = t.get('quantity', '?')
    option_type = t.get('option_type', '?')
    sector = t.get('sector', '?')
    is_spread = '|' in sym
    spread_tag = ' [SPREAD]' if is_spread else ''
    pnl_pct = t.get('pnl_pct', 0)
    
    print(f'{date} | {underlying:20s} | {direction:4s} | {option_type} | dr={dr} | smart={smart}')
    print(f'  entry={entry} -> exit={exit_p} | candles={candles} | exit={exit_type} | pnl=Rs{pnl:+,.0f} ({pnl_pct:+.1f}%){spread_tag}')
    print()

print(f'SUMMARY: {wins}W-{losses}L | Total P&L: Rs{total_pnl:+,.0f}')
if wins > 0:
    avg_win = sum(t.get('pnl', 0) for t in all3_exits if t.get('pnl', 0) > 0) / wins
    print(f'  Avg win: Rs{avg_win:+,.0f}')
if losses > 0:
    avg_loss = sum(t.get('pnl', 0) for t in all3_exits if t.get('pnl', 0) <= 0) / losses
    print(f'  Avg loss: Rs{avg_loss:+,.0f}')

# Exit type breakdown
print('\n--- EXIT TYPE BREAKDOWN ---')
exit_types = {}
for t in all3_exits:
    et = t.get('exit_type', '?')
    if et not in exit_types:
        exit_types[et] = {'count': 0, 'pnl': 0}
    exit_types[et]['count'] += 1
    exit_types[et]['pnl'] += t.get('pnl', 0)
for et, info in sorted(exit_types.items(), key=lambda x: x[1]['pnl']):
    print(f'  {et:25s}: {info["count"]} trades, Rs{info["pnl"]:+,.0f}')

# ALL3 entries detail
print('\n--- ALL ENTRIES ---')
for t in all3_entries:
    sym = t.get('symbol', '?')
    underlying = t.get('underlying', '?')
    dr = t.get('dr_score', '?')
    smart = t.get('smart_score', '?')
    direction = t.get('direction', '?')
    date = t.get('ts', '?')[:16]
    xgb_signal = t.get('xgb_signal', '?')
    xgb_prob_down = t.get('xgb_prob_down', '?')
    xgb_prob_up = t.get('xgb_prob_up', '?')
    gmm_action = t.get('gmm_action', '?')
    gmm_regime = t.get('gmm_regime', '?')
    pre_score = t.get('pre_score', '?')
    option_type = t.get('option_type', '?')
    print(f'{date} | {underlying:20s} | {direction:4s} | {option_type} | pre={pre_score} | smart={smart} | dr={dr}')
    print(f'  xgb={xgb_signal} (up={xgb_prob_up}, dn={xgb_prob_down}) | gmm={gmm_action} regime={gmm_regime}')
