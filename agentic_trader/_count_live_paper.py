"""Count live vs paper trades in history and ledgers."""
import json, glob

live = []
paper = 0
for t in json.load(open('trade_history.json')):
    oid = str(t.get('order_id') or '')
    if oid and not oid.startswith('OPTION_PAPER') and not oid.startswith('PAPER_'):
        live.append(t)
    else:
        paper += 1
print('trade_history.json: live=', len(live), 'paper/empty=', paper)
for t in live[:5]:
    print(' ', t.get('symbol'), t.get('order_id'), t.get('exit_price'), t.get('status'))

live_led = 0
paper_led = 0
for f in sorted(glob.glob('trade_ledger/trade_ledger_*.jsonl')):
    for line in open(f):
        try:
            r = json.loads(line)
            if r.get('event') != 'EXIT':
                continue
            oid = str(r.get('order_id') or '')
            if oid and not oid.startswith('OPTION_PAPER') and not oid.startswith('PAPER_'):
                live_led += 1
            else:
                paper_led += 1
        except Exception:
            pass
print('Ledger EXITs: live=', live_led, 'paper/empty=', paper_led)
