#!/usr/bin/env python3
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

ledger_file = os.path.join(os.path.dirname(__file__), 'trade_ledger', 'trade_ledger_2026-03-02.jsonl')
entries = []
exits = []
with open(ledger_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
            if ev.get('event') == 'ENTRY':
                entries.append(ev)
            elif ev.get('event') == 'EXIT':
                exits.append(ev)
        except:
            pass

print(f"ENTRY events: {len(entries)}")
for e in entries:
    sym = e.get('symbol', '?')
    d = e.get('direction', '?')
    ep = e.get('entry_price', 0)
    q = e.get('quantity', 0)
    sc = e.get('smart_score', 0)
    ts = str(e.get('timestamp', '?'))[:19]
    src = e.get('source', '?')
    print(f"  {sym} | {d} x{q} @ {ep} | score={sc} | {src} | {ts}")

print(f"\nEXIT events: {len(exits)}")
for x in exits:
    sym = x.get('symbol', '?')
    et = x.get('exit_type', '?')
    ep = x.get('entry_price', 0)
    xp = x.get('exit_price', 0)
    pnl = x.get('pnl', 0)
    ts = str(x.get('timestamp', '?'))[:19]
    print(f"  {sym} | {et} | entry@{ep} -> exit@{xp} | pnl={pnl:+,.2f} | {ts}")

print(f"\nTotal: {len(entries)} entries, {len(exits)} exits")
