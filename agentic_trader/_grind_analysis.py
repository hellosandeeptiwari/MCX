"""Analyze SLOW_GRIND P&L on Mar 17 to find what went wrong."""
import json, os
from collections import defaultdict

LEDGER = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-17.jsonl'

entries = {}  # order_id -> entry record
exits = {}   # order_id -> exit record

with open(LEDGER, 'r') as f:
    for line in f:
        try:
            rec = json.loads(line.strip())
            oid = rec.get('order_id', '')
            evt = rec.get('event', '')
            if evt == 'ENTRY':
                entries[oid] = rec
            elif evt == 'EXIT':
                exits[oid] = rec
        except:
            pass

# Match grind trades
grind_trades = []
for oid, entry in entries.items():
    src = entry.get('source', '') or entry.get('setup', '')
    if 'GRIND' in src.upper():
        exit_rec = exits.get(oid, {})
        pnl = exit_rec.get('pnl', 0)
        grind_trades.append({
            'order_id': oid,
            'symbol': entry.get('underlying', entry.get('symbol', '')),
            'direction': entry.get('direction', ''),
            'source': src,
            'smart_score': entry.get('smart_score', 0),
            'final_score': entry.get('final_score', 0),
            'move_prob': entry.get('ml_move_prob', 0),
            'entry_price': entry.get('entry_price', 0),
            'stop_loss': entry.get('stop_loss', 0),
            'target': entry.get('target', 0),
            'entry_time': entry.get('timestamp', ''),
            'exit_time': exit_rec.get('timestamp', ''),
            'exit_pnl': pnl,
            'exit_reason': exit_rec.get('exit_reason', exit_rec.get('reason', '')),
            'option_type': entry.get('option_type', ''),
            'iv': entry.get('iv', 0),
            'delta': entry.get('delta', 0),
            'dr_score': entry.get('dr_score', 0),
            'trigger_type': entry.get('trigger_type', ''),
            'rationale': entry.get('rationale', '')[:120],
        })

# Sort by P&L
grind_trades.sort(key=lambda x: x['exit_pnl'])

print(f"SLOW_GRIND TRADES ON MAR 17: {len(grind_trades)}")
print(f"{'='*100}")

total_pnl = 0
winners = 0
losers = 0
for t in grind_trades:
    pnl = t['exit_pnl']
    total_pnl += pnl
    if pnl > 0:
        winners += 1
    elif pnl < 0:
        losers += 1
    tag = '✅' if pnl > 0 else '❌' if pnl < 0 else '➖'
    print(f"\n{tag} {t['symbol']} ({t['source']}) {t['direction']} {t['option_type']}")
    print(f"   Score: smart={t['smart_score']}, final={t['final_score']}, DR={t['dr_score']:.3f}, P(move)={t['move_prob']:.2f}")
    print(f"   Entry: ₹{t['entry_price']:.2f} | SL: ₹{t['stop_loss']:.2f} | Target: ₹{t['target']:.2f}")
    print(f"   Time: {t['entry_time'][:19]} → {t['exit_time'][:19] if t['exit_time'] else 'OPEN'}")
    print(f"   P&L: ₹{pnl:,.0f} | Exit: {t['exit_reason'][:80]}")
    print(f"   IV={t['iv']:.3f} delta={t['delta']:.3f}")
    print(f"   Rationale: {t['rationale']}")

print(f"\n{'='*100}")
print(f"SUMMARY: {winners}W / {losers}L / {len(grind_trades) - winners - losers}BE")
print(f"Net P&L: ₹{total_pnl:,.0f}")
print(f"Avg P&L: ₹{total_pnl/len(grind_trades):,.0f}" if grind_trades else "No trades")

# Group by UP vs DOWN
up_pnl = sum(t['exit_pnl'] for t in grind_trades if 'UP' in t['source'])
dn_pnl = sum(t['exit_pnl'] for t in grind_trades if 'DOWN' in t['source'])
up_cnt = sum(1 for t in grind_trades if 'UP' in t['source'])
dn_cnt = sum(1 for t in grind_trades if 'DOWN' in t['source'])
print(f"\nGRIND_UP: {up_cnt} trades, P&L ₹{up_pnl:,.0f}")
print(f"GRIND_DOWN: {dn_cnt} trades, P&L ₹{dn_pnl:,.0f}")

# Common exit reasons
exit_reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
for t in grind_trades:
    r = t['exit_reason'][:40]
    exit_reasons[r]['count'] += 1
    exit_reasons[r]['pnl'] += t['exit_pnl']
print(f"\nExit reason breakdown:")
for r, v in sorted(exit_reasons.items(), key=lambda x: x[1]['pnl']):
    print(f"  {r}: {v['count']} trades, ₹{v['pnl']:,.0f}")

# Score distribution for losers vs winners  
print(f"\nScore analysis:")
losing = [t for t in grind_trades if t['exit_pnl'] < 0]
winning = [t for t in grind_trades if t['exit_pnl'] > 0]
if losing:
    print(f"  Losers avg smart_score: {sum(t['smart_score'] for t in losing)/len(losing):.0f}")
    print(f"  Losers avg P(move): {sum(t['move_prob'] for t in losing)/len(losing):.2f}")
if winning:
    print(f"  Winners avg smart_score: {sum(t['smart_score'] for t in winning)/len(winning):.0f}")
    print(f"  Winners avg P(move): {sum(t['move_prob'] for t in winning)/len(winning):.2f}")
