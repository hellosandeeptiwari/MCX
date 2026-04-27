import json, sys, re

entries = {}
exits = []
ledger = sys.argv[1] if len(sys.argv) > 1 else 'trade_ledger/trade_ledger_2026-03-20.jsonl'

with open(ledger) as f:
    for line in f:
        line = line.strip()
        if not line or 'OI_WATCHER' not in line:
            continue
        r = json.loads(line)
        if r.get('source') != 'OI_WATCHER' and r.get('setup') != 'OI_WATCHER':
            continue
        oid = r.get('order_id', '')
        if r['event'] == 'ENTRY':
            # Parse strength from rationale string: "strength=0.85"
            rat = r.get('rationale', '')
            m = re.search(r'strength=([\d.]+)', rat)
            strength = float(m.group(1)) if m else 0
            sig = r.get('oi_signal', '?')
            entries[oid] = {'str': strength, 'sig': sig}
        elif r['event'] == 'EXIT':
            exits.append(r)

hdr = f"{'Symbol':<30} {'Dir':<5} {'Signal':<20} {'Str':>5} {'PnL':>10} {'PnL%':>7} {'Exit Type':<32} {'Hold':>5}"
print(hdr)
print('-' * len(hdr))
total = 0
for e in exits:
    sym = e['symbol'].replace('NFO:', '')
    oid = e.get('order_id', '')
    ent = entries.get(oid, {})
    s = ent.get('str', 0)
    sig = ent.get('sig', '?')
    pnl = e.get('pnl', 0)
    pnl_pct = e.get('pnl_pct', 0)
    d = e.get('direction', '')
    et = e.get('exit_type', '')
    hm = e.get('hold_minutes', 0)
    total += pnl
    print(f"{sym:<30} {d:<5} {sig:<20} {s:>5.2f} {pnl:>10.0f} {pnl_pct:>6.1f}% {et:<32} {hm:>4}m")
print('-' * len(hdr))
print(f"{'TOTAL':<30} {'':5} {'':20} {'':>5} {total:>10.0f}")
print(f"\nTrades: {len(exits)} | Winners: {sum(1 for e in exits if e['pnl'] > 0)} | Losers: {sum(1 for e in exits if e['pnl'] <= 0)}")

blocked = sum(1 for e in exits if entries.get(e.get('order_id', ''), {}).get('str', 1) < 0.65)
blocked_pnl = sum(e['pnl'] for e in exits if entries.get(e.get('order_id', ''), {}).get('str', 1) < 0.65)
print(f"Would be BLOCKED by str>=0.65: {blocked} trades, PnL impact={blocked_pnl:+.0f}")

# Optimal threshold analysis
print("\n=== OPTIMAL STRENGTH THRESHOLD ANALYSIS ===")
print(f"{'Threshold':>10} {'Trades':>7} {'Win':>5} {'Loss':>5} {'Win%':>6} {'Net PnL':>10} {'Avg PnL':>9} {'Saved vs All':>12}")
print("-" * 75)
all_pnl = sum(e['pnl'] for e in exits)
for thresh in [0.0, 0.40, 0.45, 0.50, 0.55, 0.57, 0.59, 0.60, 0.63, 0.65, 0.67, 0.69, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    kept = [e for e in exits if entries.get(e.get('order_id', ''), {}).get('str', 0) >= thresh]
    if not kept:
        continue
    w = sum(1 for e in kept if e['pnl'] > 0)
    l = len(kept) - w
    net = sum(e['pnl'] for e in kept)
    avg = net / len(kept) if kept else 0
    wr = w / len(kept) * 100 if kept else 0
    saved = net - all_pnl
    print(f"{thresh:>10.2f} {len(kept):>7} {w:>5} {l:>5} {wr:>5.1f}% {net:>10.0f} {avg:>9.0f} {saved:>+12.0f}")
