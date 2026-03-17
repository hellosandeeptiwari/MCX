"""Analyze why VOLUME_SURGE was rare on Mar 17.
Check: 
1. How many times volume surge conditions were met but preempted by higher-priority triggers
2. Sustain phase failures
3. Cooldown blocks
4. The actual trigger flow
"""
import re
from collections import Counter, defaultdict

LOG = '/home/ubuntu/titan/agentic_trader/watcher_debug.log'
START_LINE = 13730  # Session 9 (Mar 17)

lines = []
with open(LOG, 'r') as f:
    for i, line in enumerate(f, 1):
        if i >= START_LINE:
            lines.append(line.rstrip())

print(f"Session 9 lines: {len(lines)}")

# 1. Count ALL trigger types that reached the watcher
print("\n" + "="*60)
print("ALL TRIGGERS RECEIVED BY WATCHER (from kite_ticker)")
print("="*60)

# Triggers come as "TRIGGER — N breakout(s) detected: SYM1, SYM2"
trigger_pat = re.compile(r'TRIGGER.*?(\d+) breakout')
trigger_count = 0
for l in lines:
    m = trigger_pat.search(l)
    if m:
        trigger_count += int(m.group(1))
print(f"Total individual triggers received: {trigger_count}")

# 2. Count triggers by TYPE in pipeline/gate check lines
print("\n" + "="*60)
print("GATE CHECKS BY TRIGGER TYPE")
print("="*60)

gate_pat = re.compile(r'GATE CHECK:.*?trigger=(\w+)')
gate_types = Counter()
for l in lines:
    m = gate_pat.search(l)
    if m:
        gate_types[m.group(1)] += 1
for t, c in gate_types.most_common():
    print(f"  {t}: {c}")
print(f"  TOTAL: {sum(gate_types.values())}")

# 3. Count TRADE PLACED by trigger type
print("\n" + "="*60)
print("TRADES PLACED BY TRIGGER TYPE")
print("="*60)

placed_pat = re.compile(r'TRADE PLACED:.*?trigger=(\w+)')
placed_types = Counter()
for l in lines:
    m = placed_pat.search(l)
    if m:
        placed_types[m.group(1)] += 1
for t, c in placed_types.most_common():
    print(f"  {t}: {c}")

# 4. Check pipeline send lines for trigger types
print("\n" + "="*60)
print("PIPELINE SENDS (trigger types entering full pipeline)")
print("="*60)

# Format: "PIPELINE: Sending N stocks → [SYM1(TRIG_TYPE +X.X%), ...]"
pipe_pat = re.compile(r'(\w+)\((\w+)')
pipe_types = Counter()
pipe_lines = [l for l in lines if 'PIPELINE: Sending' in l]
for l in pipe_lines:
    # Extract all stock(trigger pairs
    bracket_content = l.split('→')[-1] if '→' in l else ''
    for m in re.finditer(r'(\w+)\((\w{4,})', bracket_content):
        trig_abbrev = m.group(2)
        if 'VOLUME' in trig_abbrev or 'VOL' in trig_abbrev:
            pipe_types['VOLUME_SURGE'] += 1
        elif 'SPIKE' in trig_abbrev or 'PRICE' in trig_abbrev:
            pipe_types['PRICE_SPIKE'] += 1
        elif 'GRIND' in trig_abbrev or 'SLOW' in trig_abbrev:
            pipe_types['SLOW_GRIND'] += 1
        elif 'HIGH' in trig_abbrev or 'LOW' in trig_abbrev or 'DAY' in trig_abbrev:
            pipe_types['NEW_DAY_EXT'] += 1
        else:
            pipe_types[trig_abbrev] += 1

print(f"Pipeline sends: {len(pipe_lines)} batches")
for t, c in pipe_types.most_common():
    print(f"  {t}: {c} stocks")

# 5. Check OI Layer-1 blocks for VOLUME_SURGE
print("\n" + "="*60)
print("LAYER-1 OI FILTER BLOCKS")
print("="*60)
oi_block_lines = [l for l in lines if 'LAYER-1 OI FILTER' in l and 'blocked' in l]
print(f"Total OI Layer-1 blocks: {len(oi_block_lines)}")
# Show first 10
for l in oi_block_lines[:10]:
    print(f"  {l[:180]}")

# 6. Count how many triggers from kite_ticker were VOLUME_SURGE
# kite_ticker prints: "Trigger N: SYM VOLUME_SURGE"
# or it's in the trigger queue. Let's check bot_debug.log for volume detection stats
print("\n" + "="*60)
print("VOLUME-RELATED LOG LINES (kite_ticker level)")
print("="*60)
# Check bot debug log for ticker stats
import os
bot_log = '/home/ubuntu/titan/agentic_trader/bot_debug.log'
if os.path.exists(bot_log):
    with open(bot_log, 'r') as f:
        bot_lines = f.readlines()
    vol_bot = [l.strip() for l in bot_lines if 'vol_surge' in l.lower() or 'VOLUME_SURGE' in l]
    print(f"bot_debug.log vol_surge lines: {len(vol_bot)}")
    for l in vol_bot[:10]:
        print(f"  {l[:200]}")

# 7. Actually check if VOLUME_SURGE was preempted by priority in kite_ticker
# The logic is: `if not trigger_type and _vol_delta > 0` — meaning VOLUME_SURGE 
# is ONLY checked if no other trigger has fired. Check for SLOW_GRIND/SPIKE/DAY_EXT
# triggers on the same symbols that had VOLUME_SURGE gate checks
print("\n" + "="*60)
print("VOLUME_SURGE PREEMPTION ANALYSIS")
print("="*60)
print("In kite_ticker, VOLUME_SURGE is checked ONLY if no SPIKE/GRIND/DAY_EXT fired.")
print("So symbols that triggered SPIKE/GRIND likely also had volume surge conditions")
print("but VOLUME_SURGE was never evaluated because higher-priority triggers fired first.")
print()

# Count unique symbols per trigger type in gate checks
sym_pat = re.compile(r'GATE CHECK: (\w+).*?trigger=(\w+)')
sym_by_type = defaultdict(set)
for l in lines:
    m = sym_pat.search(l)
    if m:
        sym_by_type[m.group(2)].add(m.group(1))

for t in sorted(sym_by_type.keys()):
    syms = sym_by_type[t]
    print(f"  {t}: {len(syms)} unique symbols — {', '.join(sorted(syms)[:8])}{'...' if len(syms) > 8 else ''}")

# 8. Check VOLUME_SURGE specific failures after gate check
print("\n" + "="*60)
print("WHAT HAPPENED TO EACH VOLUME_SURGE GATE CHECK")
print("="*60)
vol_gate_lines = []
for i, l in enumerate(lines):
    if 'GATE CHECK' in l and 'VOLUME_SURGE' in l:
        sym = re.search(r'GATE CHECK: (\w+)', l)
        score = re.search(r'score=(\d+)', l)
        # Look at next 10 lines for result
        context = lines[i:i+12]
        result = "UNKNOWN"
        for cl in context[1:]:
            if 'ALL GATES PASSED' in cl:
                result = "PASSED GATES"
                break
            elif 'BLOCKED' in cl:
                result = re.search(r'BLOCKED\(([^)]+)\)', cl)
                result = f"BLOCKED: {result.group(1)}" if result else "BLOCKED"
                break
            elif 'TRADE FAILED' in cl:
                result = re.search(r'TRADE FAILED:.*?—\s*(.+)', cl)
                result = f"FAILED: {result.group(1)[:60]}" if result else "FAILED"
                break
            elif 'GATE CHECK' in cl and cl != l:
                result = "NO EXPLICIT BLOCK (next gate check started)"
                break
        
        _sym = sym.group(1) if sym else "?"
        _score = score.group(1) if score else "?"
        print(f"  {_sym} score={_score} → {result}")
