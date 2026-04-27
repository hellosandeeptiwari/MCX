import json

with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-02.jsonl') as f:
    lines = [json.loads(l.strip()) for l in f if l.strip()]

entries = [t for t in lines if t.get('event') == 'ENTRY']
exits = [t for t in lines if t.get('event') == 'EXIT']

print("=== ENTRIES (47) ===")
for i, e in enumerate(entries):
    sym = e.get('symbol', '')
    und = e.get('underlying', '').replace('NSE:', '').replace('MCX:', '')
    src = e.get('source', '')
    sc = e.get('final_score', 0)
    pm = e.get('ml_move_prob', 0)
    ts = e.get('ts', '')[:19]
    strat = e.get('strategy_type', '')
    oid = e.get('order_id', '')
    tid = e.get('trade_id', '')
    extra = e.get('extra', {}) or {}
    trig = extra.get('trigger_type', '')
    b2 = extra.get('b2_conviction', 0)
    print(f"  E{i+1:2d} | {ts} | {und:15s} | {sym[:30]:30s} | src={src:12s} | score={sc:5.0f} | pm={pm:.2f} | trig={trig:20s} | b2={b2} | strat={strat} | oid={oid[:20]} | tid={tid[:20]}")

print(f"\n=== EXITS (44) ===")
for i, e in enumerate(exits):
    sym = e.get('symbol', '')
    und = e.get('underlying', '').replace('NSE:', '').replace('MCX:', '')
    src = e.get('source', '')
    pnl = e.get('pnl', 0)
    ts = e.get('ts', '')[:19]
    ext = e.get('exit_type', '')
    hold = e.get('hold_minutes', 0)
    sc = e.get('final_score', 0)
    oid = e.get('order_id', '')
    tid = e.get('trade_id', '')
    ets = e.get('entry_time', '')[:19]
    print(f"  X{i+1:2d} | {ts} | {und:15s} | {sym[:30]:30s} | src={src:12s} | pnl={pnl:>8.0f} | hold={hold:>5.0f}m | exit={ext:25s} | score={sc:5.0f} | ets={ets} | oid={oid[:20]} | tid={tid[:20]}")

# Try matching by option symbol
print(f"\n=== MATCHING BY SYMBOL ===")
entry_syms = {}
for e in entries:
    sym = e.get('symbol', '')
    if sym not in entry_syms:
        entry_syms[sym] = []
    entry_syms[sym].append(e)

matched = 0
unmatched_exits = 0
for e in exits:
    sym = e.get('symbol', '')
    if sym in entry_syms and entry_syms[sym]:
        en = entry_syms[sym].pop(0)  # FIFO matching
        matched += 1
    else:
        unmatched_exits += 1
        print(f"  UNMATCHED EXIT: {sym} pnl={e.get('pnl',0)}")

print(f"Matched: {matched}, Unmatched exits: {unmatched_exits}")
