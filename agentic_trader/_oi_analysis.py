#!/usr/bin/env python3
"""Analyze why no LONG_BUILDUP / BUY signals appeared today."""
import json, os

ledger = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-16.jsonl'
records = []
with open(ledger) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        records.append(json.loads(line))

# 1) Count OI overrides
oi_overrides = [r for r in records if r.get('outcome') == 'WATCHER_OI_OVERRIDE']
print(f"=== OI OVERRIDES (BUY flipped to SELL): {len(oi_overrides)} ===")
for o in oi_overrides:
    ts = o['ts'][11:16]
    sym = o['symbol'].replace('NSE:', '')
    reason = o.get('reason', '')
    score = o.get('score', 0)
    print(f"  {ts} {sym:20s} score={score:5.1f}  {reason}")

# 2) Entry direction breakdown
entries = [r for r in records if r.get('event') == 'ENTRY']
buy_entries = [e for e in entries if e.get('direction') == 'BUY']
sell_entries = [e for e in entries if e.get('direction') == 'SELL']
print(f"\n=== ENTRIES: {len(entries)} total, {len(buy_entries)} BUY, {len(sell_entries)} SELL ===")

# 3) OI signal distribution in entries
oi_signals = {}
for e in entries:
    sig = e.get('oi_signal', '') or 'NONE'
    oi_signals[sig] = oi_signals.get(sig, 0) + 1
print(f"\n=== OI SIGNALS IN ENTRIES ===")
for sig, cnt in sorted(oi_signals.items(), key=lambda x: -x[1]):
    print(f"  {sig}: {cnt}")

# 4) All watcher scan outcomes
scan_outcomes = {}
scans = [r for r in records if r.get('event') == 'SCAN']
for s in scans:
    outcome = s.get('outcome', 'UNKNOWN')
    scan_outcomes[outcome] = scan_outcomes.get(outcome, 0) + 1
print(f"\n=== SCAN OUTCOMES ({len(scans)} total) ===")
for outcome, cnt in sorted(scan_outcomes.items(), key=lambda x: -x[1]):
    print(f"  {outcome}: {cnt}")

# 5) Check watcher_debug.log for LONG_BUILDUP mentions
wlog = '/home/ubuntu/titan/agentic_trader/watcher_debug.log'
if os.path.exists(wlog):
    long_lines = []
    with open(wlog) as f:
        for line in f:
            if 'LONG_BUILDUP' in line or 'LONG_UNWINDING' in line:
                long_lines.append(line.strip()[:200])
    print(f"\n=== LONG_BUILDUP/LONG_UNWINDING in watcher_debug.log: {len(long_lines)} lines ===")
    for ll in long_lines[:20]:
        print(f"  {ll}")
    if len(long_lines) > 20:
        print(f"  ... and {len(long_lines)-20} more")

# 6) Check bot_debug.log for OI results
blog = '/home/ubuntu/titan/agentic_trader/bot_debug.log'
if os.path.exists(blog):
    oi_lines = []
    with open(blog) as f:
        for line in f:
            if 'LONG_BUILDUP' in line or 'oi_signal' in line.lower() or 'buildup' in line.lower():
                oi_lines.append(line.strip()[:200])
    print(f"\n=== OI-related in bot_debug.log: {len(oi_lines)} lines ===")
    for ll in oi_lines[:20]:
        print(f"  {ll}")

# 7) Check if Gate F was still overriding directions
gate_f_lines = []
for r in records:
    if r.get('event') == 'SCAN':
        reason = r.get('reason', '')
        if 'OI=' in reason or 'oi_signal' in reason or 'SHORT_BUILDUP' in reason or 'LONG_BUILDUP' in reason:
            gate_f_lines.append(r)
print(f"\n=== SCAN events with OI mention: {len(gate_f_lines)} ===")

# 8) Specifically: watcher triggers that were BUY direction
buy_scans = [s for s in scans if 'BUY' in s.get('reason', '') or s.get('direction') == 'BUY']
print(f"\n=== Watcher BUY-direction scans: {len(buy_scans)} ===")
for bs in buy_scans[:20]:
    ts = bs['ts'][11:16]
    sym = bs['symbol'].replace('NSE:', '')
    outcome = bs.get('outcome', '')
    reason = bs.get('reason', '')[:100]
    print(f"  {ts} {sym:20s} outcome={outcome}  reason={reason}")
