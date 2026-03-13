#!/usr/bin/env python3
"""Analyze OI signal accuracy vs trade outcomes for March 12, 2026."""
import json, re

entries = {}
exits = {}
with open('/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-12.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except:
            continue
        ev = r.get('event', '')
        sym = r.get('underlying', r.get('symbol', ''))
        if ev == 'ENTRY':
            entries[sym] = r
        elif ev == 'EXIT':
            exits[sym] = r

# Get OI penalties from AUDIT in GATE CHECK log lines
audits = {}
# Also get raw OI signal names from OI cross-val lines
oi_signals = {}
with open('/home/ubuntu/titan/logs/titan.log') as f:
    for line in f:
        if 'GATE CHECK' in line:
            m = re.search(r'(\w+)\s+score=.*OI:([+-]?\d+)', line)
            if m:
                stock = m.group(1)
                oi_pts = int(m.group(2))
                audits[stock] = oi_pts
        if 'OI cross' in line:
            # Extract stock name and OI signal
            m2 = re.search(r'(\w+).*OI cross.*?(\w+_\w+|NEUTRAL)', line)
            if m2:
                oi_signals[m2.group(1)] = m2.group(2)
        # Also look for OI= patterns in watcher logs
        if 'OI=' in line and ('BUILDUP' in line or 'UNWINDING' in line or 'COVERING' in line):
            m3 = re.search(r'(\w+).*OI=(\w+)', line)
            if m3:
                oi_signals.setdefault(m3.group(1), m3.group(2))

# Also get OI signals from the scorer AUDIT strings
with open('/home/ubuntu/titan/logs/titan.log') as f:
    for line in f:
        if 'OI cross-val' in line or 'oi_xval' in line.lower():
            m = re.search(r'(\w+).*?(LONG_BUILDUP|SHORT_BUILDUP|LONG_UNWINDING|SHORT_COVERING|NEUTRAL)', line)
            if m:
                oi_signals[m.group(1)] = m.group(2)

print("=" * 100)
print("OI SIGNAL ACCURACY REPORT — March 12, 2026")
print("=" * 100)
print()

print("OI PENALTIES FROM AUDIT STRINGS:")
print("-" * 50)
for stock, pts in sorted(audits.items()):
    signal = oi_signals.get(stock, '?')
    print(f"  {stock:15s}: OI penalty = {pts:+d} pts  (signal: {signal})")

print()
print("TRADE RESULTS vs OI SIGNAL:")
print("-" * 100)
print(f"{'Stock':15s} {'Dir':5s} {'Score':6s} {'Source':14s} {'OI pts':7s} {'OI Status':10s} {'P&L':>10s} {'Result':6s} {'OI Verdict':15s}")
print("-" * 100)

oi_right = 0
oi_wrong = 0
oi_neutral = 0
total_conflict_pnl = 0
total_aligned_pnl = 0
total_neutral_pnl = 0

for sym in sorted(entries.keys()):
    e = entries[sym]
    x = exits.get(sym)
    direction = e.get('direction', '')
    score = e.get('final_score', 0)
    src = e.get('source', '')
    pnl = x.get('total_pnl', 0) if x else None
    stock = sym.replace('NSE:', '')
    oi_pts = audits.get(stock, 0)
    
    if oi_pts < 0:
        oi_status = 'CONFLICT'
    elif oi_pts > 0:
        oi_status = 'ALIGNED'
    else:
        oi_status = 'NEUTRAL'
    
    if pnl is not None:
        result = 'WIN' if pnl > 0 else 'LOSS'
        pnl_str = f"{pnl:+,.0f}"
    else:
        result = 'OPEN'
        pnl_str = 'OPEN'
    
    oi_verdict = ''
    if oi_pts < 0:  # OI conflicted with trade direction
        if result == 'LOSS':
            oi_verdict = 'OI WAS RIGHT ✓'
            oi_right += 1
            total_conflict_pnl += pnl if pnl else 0
        elif result == 'WIN':
            oi_verdict = 'OI WAS WRONG ✗'
            oi_wrong += 1
            total_conflict_pnl += pnl if pnl else 0
        else:
            oi_verdict = '(pending)'
    elif oi_pts > 0:
        total_aligned_pnl += pnl if pnl else 0
    else:
        total_neutral_pnl += pnl if pnl else 0
        oi_neutral += 1
    
    signal = oi_signals.get(stock, '')
    print(f"  {stock:15s} {direction:5s} {score:6.1f} {src:14s} {oi_pts:+4d}    {oi_status:10s} {pnl_str:>10s} {result:6s} {oi_verdict}")

print("-" * 100)
print()
print("SUMMARY:")
print(f"  Trades with OI CONFLICT (OI said opposite): {oi_right + oi_wrong}")
print(f"    OI was RIGHT (trade lost):  {oi_right}")
print(f"    OI was WRONG (trade won):   {oi_wrong}")
if oi_right + oi_wrong > 0:
    accuracy = oi_right / (oi_right + oi_wrong) * 100
    print(f"    OI ACCURACY:                {accuracy:.0f}%")
    print(f"    P&L on conflicted trades:   ₹{total_conflict_pnl:+,.0f}")
print()
print(f"  Trades with OI ALIGNED:      P&L = ₹{total_aligned_pnl:+,.0f}")
print(f"  Trades with OI NEUTRAL:      P&L = ₹{total_neutral_pnl:+,.0f} ({oi_neutral} trades)")
