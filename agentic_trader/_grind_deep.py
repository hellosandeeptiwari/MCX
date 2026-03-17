"""Deep analysis: WHERE in the move did grind enter vs where the move ended?
For each grind trade, check:
1. Total intraday move at entry time vs at close
2. Was entry near the TOP/BOTTOM of the move (end of slope)?
3. How much move was LEFT after entry?
4. Velocity at entry vs velocity at close (was grind decelerating?)
"""
import json, os
from datetime import datetime

LEDGER = '/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-17.jsonl'
WATCHER = '/home/ubuntu/titan/agentic_trader/watcher_debug.log'

entries = {}
exits = {}

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

# Get grind trades
grind_trades = []
for oid, e in entries.items():
    src = e.get('source', '') or e.get('setup', '')
    if 'GRIND' not in src.upper():
        continue
    ex = exits.get(oid, {})
    sym = (e.get('underlying', '') or e.get('symbol', '')).replace('NSE:', '')
    grind_trades.append({
        'order_id': oid,
        'symbol': sym,
        'direction': e.get('direction', ''),
        'source': src,
        'smart_score': e.get('smart_score', 0),
        'entry_price': e.get('entry_price', 0),
        'pnl': ex.get('pnl', 0),
        'exit_reason': ex.get('exit_reason', ex.get('reason', '')),
        'rationale': e.get('rationale', ''),
        'option_type': e.get('option_type', ''),
        'trigger_type': e.get('trigger_type', ''),
    })

# Session 9 starts at line 13730
start_line = 13730

# Read watcher log session 9
watcher_lines = []
with open(WATCHER, 'r') as f:
    for i, line in enumerate(f, 1):
        if i >= start_line:
            watcher_lines.append(line.rstrip())

print("="*80)
print("GRIND LATE-ENTRY / END-OF-SLOPE ANALYSIS")
print("="*80)

import re

for t in sorted(grind_trades, key=lambda x: x['pnl']):
    sym = t['symbol']
    pnl = t['pnl']
    tag = '❌' if pnl < 0 else '✅' if pnl > 0 else '➖'
    
    print(f"\n{tag} {sym} ({t['source']}) PNL=₹{pnl:,.0f}")
    
    # Extract trigger move from rationale
    trig_match = re.search(r'([+-]?\d+\.\d+%)', t['rationale'])
    trig_move = trig_match.group(1) if trig_match else '?'
    print(f"   Trigger: {trig_move} | Score: {t['smart_score']:.0f}")
    
    # Find this symbol's GATE CHECK line in watcher to get full context
    gate_lines = []
    for l in watcher_lines:
        if f'GATE CHECK: {sym}' in l and 'GRIND' in l:
            gate_lines.append(l)
    
    # Find LATE-DECAY lines
    decay_lines = []
    for l in watcher_lines:
        if 'LATE-DECAY' in l and sym in l:
            decay_lines.append(l)
    
    # Find EXHAUST lines
    exhaust_lines = []
    for l in watcher_lines:
        if 'EXHAUST' in l and sym in l:
            exhaust_lines.append(l)
    
    # Find TRADE PLACED/FAILED lines
    trade_lines = []
    for l in watcher_lines:
        if ('TRADE PLACED' in l or 'TRADE FAIL' in l) and sym in l:
            trade_lines.append(l)
    
    # Extract intraday move from last gate check
    if gate_lines:
        last_gate = gate_lines[-1]
        # Extract peak from gate check
        peak_match = re.search(r'peak=([0-9.]+)', last_gate)
        held_match = re.search(r'held=([0-9.]+)', last_gate)
        score_match = re.search(r'score=(\d+)', last_gate)
        trigger_match = re.search(r'trigger=(\w+)\(([+-][0-9.]+%)\)', last_gate)
        vwap_match = re.search(r'VWAP=(\w+)', last_gate)
        adx_match = re.search(r'ADX=(\d+)', last_gate)
        vol_match = re.search(r'VOL=(\w+)', last_gate)
        rsi_match = re.search(r'RSI=(\d+)', last_gate)
        
        print(f"   Gate Check: score={score_match.group(1) if score_match else '?'} "
              f"trigger={trigger_match.group(1) if trigger_match else '?'}({trigger_match.group(2) if trigger_match else '?'}) "
              f"peak={peak_match.group(1) if peak_match else '?'}% "
              f"held={held_match.group(1) if held_match else '?'}% "
              f"VWAP={vwap_match.group(1) if vwap_match else '?'} "
              f"ADX={adx_match.group(1) if adx_match else '?'} "
              f"VOL={vol_match.group(1) if vol_match else '?'} "
              f"RSI={rsi_match.group(1) if rsi_match else '?'}")
    
    if decay_lines:
        last_decay = decay_lines[-1]
        # Extract intraday move from LATE-DECAY
        decay_match = re.search(r'moved ([+-]?\d+\.\d+)% intraday', last_decay)
        penalty_match = re.search(r'→ (-?\d+) →', last_decay)
        if decay_match:
            print(f"   LATE-DECAY: Already moved {decay_match.group(1)}% intraday, penalty={penalty_match.group(1) if penalty_match else '?'}")
    
    if exhaust_lines:
        last_ex = exhaust_lines[-1]
        ei_match = re.search(r'EI=([0-9.]+)', last_ex)
        intra_match = re.search(r'intra=([+-][0-9.]+)%', last_ex)
        range_match = re.search(r'range=([0-9]+)%', last_ex)
        if ei_match:
            print(f"   EXHAUST: EI={ei_match.group(1)} intra={intra_match.group(1) if intra_match else '?'}% range={range_match.group(1) if range_match else '?'}%")
    
    if trade_lines:
        for tl in trade_lines[-2:]:  # Last 2 trade lines
            print(f"   {tl[:150]}")
    
    # Count how many times this symbol appeared in gate checks for grind
    grind_count = len([l for l in gate_lines])
    print(f"   Appeared in {grind_count} GRIND gate checks today")
    
    print(f"   Exit: {t['exit_reason'][:100]}")

# Summary: the key metric is ratio of trigger move to total intraday move
print(f"\n{'='*80}")
print("KEY INSIGHT: TRIGGER FRESHNESS RATIO")
print("="*80)
print("When the grind trigger (+0.5-1.1%) is a TINY fraction of the total")
print("intraday move (+2-4%), you're catching the END of the slope.")
print("The stock has already moved most of its daily range.")
print()

# Also check: how many grind triggers appeared per unique symbol
print("="*80)
print("GRIND TRIGGER FREQUENCY (re-fires)")
print("="*80)
grind_syms = {}
for l in watcher_lines:
    m = re.search(r'GATE CHECK: (\w+).*trigger=(SLOW_GRIND_\w+)', l)
    if m:
        sym = m.group(1)
        grind_syms[sym] = grind_syms.get(sym, 0) + 1

for sym, cnt in sorted(grind_syms.items(), key=lambda x: -x[1])[:15]:
    traded = any(t['symbol'] == sym for t in grind_trades)
    tag = " ← TRADED" if traded else ""
    print(f"  {sym}: {cnt} grind gate checks{tag}")

# Check grind config
print(f"\n{'='*80}")
print("GRIND CONFIG")
print("="*80)
# Read config
import importlib.util
spec = importlib.util.spec_from_file_location("config", "/home/ubuntu/titan/agentic_trader/config.py")
assert spec is not None and spec.loader is not None
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
bw = cfg.BREAKOUT_WATCHER
print(f"  slow_grind_pct: {bw.get('slow_grind_pct', 1.0)}%")
print(f"  vol_confirmed threshold lowers to: {bw.get('slow_grind_pct', 1.0) * 0.65:.2f}%")
print(f"  sustain_seconds: {bw.get('sustain_seconds', 60)}s")
print(f"  sustain_seconds_volume: {bw.get('sustain_seconds_volume', 45)}s")
print(f"  cooldown_seconds: {bw.get('cooldown_seconds', 120)}s")
