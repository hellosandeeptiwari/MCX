import json, sys
path = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-16.jsonl'
entries = []
exits = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except:
        continue
    ev = d.get('event', '')
    if ev == 'ENTRY':
        entries.append(d)
    elif ev == 'EXIT':
        exits.append(d)

print(f"=== {len(entries)} ENTRIES, {len(exits)} EXITS ===\n")
for e in entries:
    sym = e.get('underlying', '').replace('NSE:', '')
    src = e.get('source', '?')
    d = e.get('direction', '?')
    sc = e.get('smart_score', 0)
    mp = e.get('ml_move_prob', 0)
    ep = e.get('entry_price', 0)
    rat = e.get('rationale', '')[:80]
    ts = e.get('ts', '?')[11:19]
    oi = e.get('oi_signal', '-')
    print(f"  {ts} {sym:15s} {d:4s} src={src:12s} score={sc:5.1f} P(m)={mp:.2f} entry={ep} OI={oi}")
    print(f"           rationale: {rat}")

print()
for x in exits:
    sym = x.get('underlying', '').replace('NSE:', '')
    pnl = x.get('pnl', 0)
    ext = x.get('exit_type', '?')
    mins = x.get('hold_minutes', 0)
    r = x.get('r_multiple', 0)
    src = x.get('source', '?')
    ts = x.get('ts', '?')[11:19]
    print(f"  {ts} {sym:15s} pnl={pnl:+10,.0f} exit={ext:25s} hold={mins:3.0f}m R={r:.1f} src={src}")

total_pnl = sum(x.get('pnl', 0) for x in exits)
print(f"\n  TOTAL P&L: {total_pnl:+,.0f}")
