"""Analyze today's trades from the centralized trade ledger."""
import json, os
from datetime import datetime

today = '2026-02-26'
ledger_file = f'trade_ledger/trade_ledger_{today}.jsonl'

if not os.path.exists(ledger_file):
    print(f'No ledger file: {ledger_file}')
    exit()

records = []
with open(ledger_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

entries = [r for r in records if r.get('event') == 'ENTRY']
exits = [r for r in records if r.get('event') == 'EXIT']
scans = [r for r in records if r.get('event') == 'SCAN']

print(f"{'='*80}")
print(f"  DAILY TRADE REPORT: {today}")
print(f"{'='*80}")
print(f"Total records: {len(records)} | Entries: {len(entries)} | Exits: {len(exits)} | Scans: {len(scans)}")
print()

# ── ENTRIES ──
print(f"{'─'*80}")
print("  ENTRIES")
print(f"{'─'*80}")
by_source = {}
for e in entries:
    src = e.get('source', 'UNKNOWN')
    by_source.setdefault(src, []).append(e)
    ts = e.get('ts', '')[:19]
    sym = e.get('symbol', '')
    dirn = e.get('direction', '')
    score = e.get('smart_score', 0)
    dr = e.get('dr_score', 0)
    price = e.get('entry_price', 0)
    sector = e.get('sector', '')
    print(f"  {ts} | {sym:<30s} | {dirn:<4s} | {src:<15s} | smart={score:>5.1f} | dr={dr:.3f} | ₹{price:.2f} | {sector}")

print(f"\n  Entry breakdown by source:")
for src, trades in sorted(by_source.items()):
    print(f"    {src}: {len(trades)} trades")

# ── EXITS ──
print(f"\n{'─'*80}")
print("  EXITS")
print(f"{'─'*80}")
total_pnl = 0
wins = 0; losses = 0; breakeven = 0
exit_by_type = {}
pnl_by_source = {}

for e in exits:
    pnl = e.get('pnl', 0)
    total_pnl += pnl
    if pnl > 50: wins += 1
    elif pnl < -50: losses += 1
    else: breakeven += 1
    
    exit_type = e.get('exit_type', 'UNKNOWN')
    exit_by_type.setdefault(exit_type, {'count': 0, 'pnl': 0})
    exit_by_type[exit_type]['count'] += 1
    exit_by_type[exit_type]['pnl'] += pnl
    
    src = e.get('source', 'UNKNOWN')
    pnl_by_source.setdefault(src, {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0})
    pnl_by_source[src]['count'] += 1
    pnl_by_source[src]['pnl'] += pnl
    if pnl > 50: pnl_by_source[src]['wins'] += 1
    elif pnl < -50: pnl_by_source[src]['losses'] += 1
    
    ts = e.get('ts', '')[:19]
    sym = e.get('symbol', '')
    hold = e.get('hold_minutes', 0)
    candles = e.get('candles_held', 0)
    entry_p = e.get('entry_price', 0)
    exit_p = e.get('exit_price', 0)
    reason = e.get('exit_reason', exit_type)
    
    marker = '✅' if pnl > 50 else ('❌' if pnl < -50 else '➖')
    print(f"  {marker} {ts} | {sym:<30s} | {exit_type:<25s} | ₹{pnl:>+10,.2f} | {candles}c/{hold}min | entry=₹{entry_p:.2f} exit=₹{exit_p:.2f}")

# ── SUMMARY ──
print(f"\n{'='*80}")
print("  DAILY SUMMARY")
print(f"{'='*80}")

total_trades = wins + losses + breakeven
if total_trades > 0:
    print(f"  Wins: {wins} | Losses: {losses} | Breakeven: {breakeven} | Total: {total_trades}")
    print(f"  Win Rate: {wins/total_trades*100:.0f}%")
else:
    print("  No closed trades today")

print(f"  TOTAL REALIZED P&L: ₹{total_pnl:+,.2f}")
print()

# By exit type
print("  P&L by Exit Type:")
for et, data in sorted(exit_by_type.items(), key=lambda x: x[1]['pnl']):
    print(f"    {et:<30s}: {data['count']} trades | ₹{data['pnl']:>+10,.2f}")

print()
print("  P&L by Source/Strategy:")
for src, data in sorted(pnl_by_source.items(), key=lambda x: x[1]['pnl']):
    wr = f"{data['wins']}/{data['count']}" if data['count'] > 0 else "0/0"
    print(f"    {src:<15s}: {data['count']} trades | W/L={wr} | ₹{data['pnl']:>+10,.2f}")

# ── STILL OPEN (from SQLite DB) ──
print(f"\n{'─'*80}")
print("  POSITIONS STILL OPEN AT EOD")
print(f"{'─'*80}")
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from state_db import get_state_db
    db = get_state_db()
    open_trades_list, _, _ = db.load_active_trades(today)
    open_trades = [t for t in open_trades_list if t.get('status', 'OPEN') == 'OPEN']
    if open_trades:
        for t in open_trades:
            sym = t.get('symbol', '')
            dirn = t.get('direction', '')
            src = t.get('setup_type', '')
            entry = t.get('avg_price', 0)
            print(f"  ⚠️  {sym:<30s} | {dirn} | {src} | entry=₹{entry:.2f}")
        print(f"  Total still open: {len(open_trades)}")
    else:
        print("  All positions closed ✅")
except Exception as ex:
    print(f"  Could not read active trades from DB: {ex}")

# ── PROFIT TARGET EVENTS ──
pt_scans = [s for s in scans if 'PROFIT' in str(s.get('action', ''))]
if pt_scans:
    print(f"\n{'─'*80}")
    print("  PROFIT TARGET EVENTS")
    print(f"{'─'*80}")
    for s in pt_scans:
        print(f"  {s.get('ts', '')} | {s.get('action', '')} | {s.get('reason', '')}")
else:
    print(f"\n  ⚠️  NO PROFIT TARGET KILL-ALL TRIGGERED TODAY")

# ── MULTI-DAY CONTEXT ──
print(f"\n{'─'*80}")
print("  MULTI-DAY P&L (last 3 days)")
print(f"{'─'*80}")
for d in ['2026-02-24', '2026-02-25', '2026-02-26']:
    lf = f'trade_ledger/trade_ledger_{d}.jsonl'
    if not os.path.exists(lf):
        continue
    day_exits = []
    with open(lf, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get('event') == 'EXIT':
                    day_exits.append(r)
    day_pnl = sum(e.get('pnl', 0) for e in day_exits)
    day_wins = sum(1 for e in day_exits if e.get('pnl', 0) > 50)
    day_losses = sum(1 for e in day_exits if e.get('pnl', 0) < -50)
    day_total = len(day_exits)
    marker = '🟢' if day_pnl > 0 else '🔴'
    print(f"  {marker} {d}: {day_total} exits | W={day_wins} L={day_losses} | ₹{day_pnl:>+10,.2f}")

print(f"\n{'='*80}")
