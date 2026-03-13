"""Check OI signal for all today's trades"""
import json, sys

ledger = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-12.jsonl'
entries = []
exits = {}

for line in open(ledger):
    d = json.loads(line.strip())
    if d.get('event') == 'ENTRY':
        entries.append(d)
    elif d.get('event') == 'EXIT':
        exits[d.get('option_symbol', d.get('symbol', ''))] = d

print(f"{'STOCK':15s} {'DIR':5s} {'SCORE':6s} {'OI_SIGNAL':20s} {'SRC':12s} {'PNL':>8s} {'EXIT':15s}")
print("-" * 90)
for e in entries:
    sym = e.get('underlying', '').replace('NSE:', '')
    oi = e.get('oi_signal', 'N/A')
    dr = e.get('direction', '?')
    sc = e.get('final_score', 0)
    src = e.get('source', '?')
    opt = e.get('option_symbol', '')
    ex = exits.get(opt, {})
    pnl = ex.get('pnl', '?')
    exit_reason = ex.get('exit_reason', 'OPEN')
    pnl_str = f"{pnl:,.0f}" if isinstance(pnl, (int, float)) else str(pnl)
    print(f"{sym:15s} {dr:5s} {sc:6.0f} {str(oi):20s} {src:12s} {pnl_str:>8s} {exit_reason:15s}")
