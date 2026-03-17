#!/usr/bin/env python3
import json

ledger = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-17.jsonl'
entries = []
exits = []
conversions = []

with open(ledger) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        ev = d.get('event','')
        if ev == 'ENTRY': entries.append(d)
        elif ev == 'EXIT': exits.append(d)
        elif ev == 'CONVERSION': conversions.append(d)

print("=" * 100)
print("TITAN v5 - TRADE ANALYSIS - March 17, 2026")
print("=" * 100)

# ENTRIES
print(f"\nTOTAL ENTRIES: {len(entries)}")
print("-" * 100)
print(f"{'Time':<12} {'Underlying':<20} {'Dir':<5} {'Source':<28} {'Score':>6} {'Entry':>8} {'Qty':>6} {'Premium':>10}")
print("-" * 100)
for d in sorted(entries, key=lambda x: x['ts']):
    ts = d['ts'][11:19]
    und = d.get('underlying', d.get('symbol','?'))
    if und.startswith('NSE:'): und = und[4:]
    print(f"{ts:<12} {und:<20} {d.get('direction','?'):<5} {d.get('source','?'):<28} {d.get('smart_score',0):>6.1f} {d.get('entry_price',0):>8.2f} {d.get('quantity',0):>6} {d.get('total_premium',0):>10,.0f}")

# EXITS
print(f"\nTOTAL EXITS: {len(exits)}")
print("-" * 100)
print(f"{'Time':<12} {'Underlying':<20} {'Dir':<5} {'Source':<28} {'Score':>6} {'P&L':>10} {'P&L%':>7} {'Exit Reason'}")
print("-" * 100)

total_pnl = 0
winners = 0
losers = 0
for d in sorted(exits, key=lambda x: x['ts']):
    ts = d['ts'][11:19]
    und = d.get('underlying', d.get('symbol','?'))
    if und.startswith('NSE:'): und = und[4:]
    pnl = d.get('pnl', 0) or 0
    pnl_pct = d.get('pnl_pct', 0) or 0
    total_pnl += pnl
    if pnl > 0: winners += 1
    elif pnl < 0: losers += 1
    icon = 'WIN ' if pnl > 0 else 'LOSS' if pnl < 0 else 'BE  '
    reason = d.get('exit_reason', '?')[:50]
    print(f"{ts:<12} {und:<20} {d.get('direction','?'):<5} {d.get('source','?'):<28} {d.get('smart_score',0):>6.1f} {pnl:>10,.0f} {pnl_pct:>+6.1f}% {icon} {reason}")

n = len(exits)
be = n - winners - losers
print(f"\n{'=' * 100}")
print(f"P&L SUMMARY: {winners}W / {losers}L / {be}BE out of {n} trades")
print(f"Net P&L: Rs {total_pnl:,.0f}")
if n > 0:
    print(f"Win Rate: {winners/n*100:.0f}% | Avg P&L: Rs {total_pnl/n:,.0f}")
    winner_pnl = sum(d.get('pnl',0) or 0 for d in exits if (d.get('pnl',0) or 0) > 0)
    loser_pnl = sum(d.get('pnl',0) or 0 for d in exits if (d.get('pnl',0) or 0) < 0)
    print(f"Total Wins: Rs {winner_pnl:,.0f} | Total Losses: Rs {loser_pnl:,.0f}")
    if winners > 0: print(f"Avg Win: Rs {winner_pnl/winners:,.0f}")
    if losers > 0: print(f"Avg Loss: Rs {loser_pnl/losers:,.0f}")

# CONVERSIONS 
print(f"\n\nCONVERSIONS (Naked -> Spread): {len(conversions)}")
print("-" * 100)
for d in sorted(conversions, key=lambda x: x['ts']):
    ts = d['ts'][11:19]
    und = d.get('underlying','?')
    if und.startswith('NSE:'): und = und[4:]
    print(f"  {ts} {und:<20} {d.get('conversion_type','?')} [{d.get('tie_check','?')}] net_debit={d.get('net_debit',0):.2f}")

# BY TRADE TYPE
print(f"\n\nBREAKDOWN BY SOURCE:")
print("-" * 100)
source_stats = {}
for d in exits:
    src = d.get('source', d.get('setup', '?'))
    if src not in source_stats:
        source_stats[src] = {'count': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
    source_stats[src]['count'] += 1
    pnl = d.get('pnl', 0) or 0
    source_stats[src]['pnl'] += pnl
    if pnl > 0: source_stats[src]['wins'] += 1
    elif pnl < 0: source_stats[src]['losses'] += 1

for src, s in sorted(source_stats.items(), key=lambda x: -x[1]['pnl']):
    wr = s['wins']/s['count']*100 if s['count'] > 0 else 0
    print(f"  {src:<30} {s['count']} trades | {s['wins']}W/{s['losses']}L | WR={wr:.0f}% | P&L: Rs {s['pnl']:,.0f}")

# BY EXIT TYPE
print(f"\n\nBREAKDOWN BY EXIT REASON:")
print("-" * 100)
exit_stats = {}
for d in exits:
    etype = d.get('exit_type', '?')
    if etype not in exit_stats:
        exit_stats[etype] = {'count': 0, 'pnl': 0}
    exit_stats[etype]['count'] += 1
    exit_stats[etype]['pnl'] += (d.get('pnl', 0) or 0)

for etype, s in sorted(exit_stats.items(), key=lambda x: -x[1]['count']):
    print(f"  {etype:<40} {s['count']} trades | P&L: Rs {s['pnl']:,.0f}")

# SCORE VS OUTCOME
print(f"\n\nSCORE vs OUTCOME (sorted by score):")
print("-" * 100)
for d in sorted(exits, key=lambda x: x.get('smart_score', 0) or 0, reverse=True):
    und = d.get('underlying','?')
    if und.startswith('NSE:'): und = und[4:]
    pnl = d.get('pnl', 0) or 0
    icon = 'WIN ' if pnl > 0 else 'LOSS' if pnl < 0 else 'BE  '
    print(f"  Score {d.get('smart_score',0):>5.1f} | {und:<20} | {d.get('source','?'):<28} | Rs {pnl:>+10,.0f} ({d.get('pnl_pct',0):>+6.1f}%) {icon}")

# BLOCKED TRADES ANALYSIS
print(f"\n\nBLOCKED/MISSED TRADES:")
print("-" * 100)
blocked = {}
with open(ledger) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        d = json.loads(line)
        if d.get('event') != 'SCAN': continue
        outcome = d.get('outcome', '')
        if 'FAILED' in outcome or 'BLOCK' in outcome or 'CONFLICT' in outcome or 'VETO' in outcome or 'FILTERED' in outcome or 'EXHAUSTED' in outcome:
            reason_short = d.get('reason', '')[:60]
            key = f"{d.get('symbol','?')} ({d.get('direction','?')}) - {outcome}"
            if key not in blocked:
                blocked[key] = {'count': 0, 'score': d.get('score', 0), 'reason': reason_short}
            blocked[key]['count'] += 1
            blocked[key]['score'] = max(blocked[key]['score'], d.get('score', 0))

for k, v in sorted(blocked.items(), key=lambda x: -x[1]['score']):
    if v['score'] >= 40:  # Only show meaningful ones
        print(f"  {k}")
        print(f"    Attempts: {v['count']}x | Best Score: {v['score']} | {v['reason']}")

print(f"\n{'=' * 100}")
print("END OF ANALYSIS")
