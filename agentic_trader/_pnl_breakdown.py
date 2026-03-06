import json

entries = {}
exits = []
with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-06.jsonl') as f:
    for line in f:
        try:
            r = json.loads(line.strip())
            if r.get('event') == 'ENTRY':
                key = r.get('symbol', '')
                entries[key] = r
            elif r.get('event') in ('EXIT', 'MANUAL_EXIT'):
                raw = r.get('symbol', '')
                base = raw.split('|')[0]
                e = entries.get(raw) or entries.get(base)
                if e:
                    r['_entry'] = e
                exits.append(r)
        except:
            pass

wins = [(x, x.get('pnl', 0)) for x in exits if x.get('pnl', 0) > 0]
losses = [(x, x.get('pnl', 0)) for x in exits if x.get('pnl', 0) < 0]

total_win = sum(p for _, p in wins)
total_loss = sum(p for _, p in losses)

print('=== WINS ===')
for x, p in sorted(wins, key=lambda x: x[1], reverse=True):
    src = (x.get('_entry') or x).get('source', '?')
    sym = x.get('symbol', '').replace('NFO:', '')[:28]
    print(f'  +Rs{p:>8,.0f}  {sym:<28} {src}')
print(f'  TOTAL WINS: Rs {total_win:+,.0f} ({len(wins)} trades)')

print()
print('=== LOSSES ===')
for x, p in sorted(losses, key=lambda x: x[1]):
    src = (x.get('_entry') or x).get('source', '?')
    sym = x.get('symbol', '').replace('NFO:', '')[:28]
    ex_reason = x.get('exit_reason', '')[:35]
    print(f'  Rs{p:>9,.0f}  {sym:<28} {src:<18} {ex_reason}')
print(f'  TOTAL LOSSES: Rs {total_loss:+,.0f} ({len(losses)} trades)')

print()
print(f'GROSS WIN:  Rs {total_win:+,.0f}')
print(f'GROSS LOSS: Rs {total_loss:+,.0f}')
print(f'NET P&L:    Rs {total_win + total_loss:+,.0f}')
print(f'Win/Loss ratio: {abs(total_win / total_loss) if total_loss else 0:.2f}')
if wins:
    print(f'Avg win:  Rs {total_win / len(wins):,.0f}')
if losses:
    print(f'Avg loss: Rs {total_loss / len(losses):,.0f}')

# Categorize losses by source
by_source_loss = {}
for x, p in losses:
    src = (x.get('_entry') or x).get('source', '?')
    by_source_loss.setdefault(src, 0)
    by_source_loss[src] += p

# Categorize losses by exit reason
orphan_loss = sum(p for x, p in losses if 'orphan' in x.get('exit_reason', '').lower())
manual_loss = sum(p for x, p in losses if 'manual' in x.get('exit_reason', '').lower())
sl_loss = sum(p for x, p in losses if 'stop' in x.get('exit_reason', '').lower() or 'bleed' in x.get('exit_reason', '').lower())
debit_loss = sum(p for x, p in losses if 'debit' in x.get('exit_reason', '').lower())
greeks_loss = sum(p for x, p in losses if 'greek' in x.get('exit_reason', '').lower() or 'speed' in x.get('exit_reason', '').lower())
tie_loss = sum(p for x, p in losses if 'tie:' in x.get('exit_reason', '').lower() or 'max pain' in x.get('exit_reason', '').lower() or 'never showed' in x.get('exit_reason', '').lower())

print()
print('=== LOSS BREAKDOWN BY STRATEGY ===')
for src, p in sorted(by_source_loss.items(), key=lambda x: x[1]):
    print(f'  {src:<20} Rs {p:+,.0f}')

print()
print('=== LOSS BREAKDOWN BY EXIT TYPE ===')
print(f'  EOD orphan (lost during restart):  Rs {orphan_loss:+,.0f}')
print(f'  Manual dashboard exits:            Rs {manual_loss:+,.0f}')
print(f'  Debit spread auto-exit:            Rs {debit_loss:+,.0f}')
print(f'  Greeks/Speed gate exits:           Rs {greeks_loss:+,.0f}')
print(f'  TIE exits (max pain/never showed): Rs {tie_loss:+,.0f}')
print(f'  SL/Bleed exits:                    Rs {sl_loss:+,.0f}')

# Hypothetical without GPT trades
gpt_sources = {'VWAP_TREND', 'VWAP_GRIND', 'WILDCARD_MOMENTUM', 'MOMENTUM'}
gpt_pnl = 0
for x in exits:
    src = (x.get('_entry') or x).get('source', '?')
    if src in gpt_sources:
        gpt_pnl += x.get('pnl', 0)

print()
print(f'=== HYPOTHETICAL WITHOUT GPT TRADES ===')
print(f'  GPT pipeline P&L:     Rs {gpt_pnl:+,.0f}')
print(f'  Without GPT, NET:     Rs {total_win + total_loss - gpt_pnl:+,.0f}')
