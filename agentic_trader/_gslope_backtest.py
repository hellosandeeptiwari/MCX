"""Backtest G-SLOPE gate against Mar 17 grind trades.
For each trade, compute every signal and show PASS/BLOCK + which condition triggered.
Then show: would NET PNL improve? Any good trades lost?
"""
import json, os, re, glob
from collections import defaultdict

LEDGER = 'trade_ledger/trade_ledger_2026-03-17.jsonl'
WATCHER = 'watcher_debug.log'

# --- Load trades ---
entries = {}
exit_groups = defaultdict(list)
with open(LEDGER) as f:
    for line in f:
        t = json.loads(line.strip())
        oid = t.get('order_id', '')
        ev = t.get('event', '')
        src = t.get('source', '') or t.get('setup', '')
        if ev == 'ENTRY' and 'GRIND' in src.upper():
            entries[oid] = t
        elif ev == 'EXIT' and 'GRIND' in src.upper():
            exit_groups[oid].append(t)

trades = []
for oid, e in entries.items():
    if oid not in exit_groups: continue
    total_pnl = sum(x.get('pnl', 0) for x in exit_groups[oid])
    trades.append({**e, 'pnl': total_pnl, 'order_id': oid,
                   'sym': (e.get('underlying', '') or e.get('symbol', '')).replace('NSE:', '')})

# --- Load watcher log session 9 (Mar 17) ---
watcher_lines = []
with open(WATCHER) as f:
    for i, line in enumerate(f, 1):
        if i >= 13730:  # Session 9 start
            watcher_lines.append(line.rstrip())

def find_gate_data(sym, watcher_lines):
    """Extract last GATE CHECK + EXHAUST + LATE-DECAY for this symbol's grind."""
    gate = {}
    for l in watcher_lines:
        if f'GATE CHECK: {sym}' in l and 'GRIND' in l:
            # Extract fields
            m = re.search(r'score=(\d+)', l)
            if m: gate['score'] = int(m.group(1))
            m = re.search(r'trigger=SLOW_GRIND_(\w+)\(([+-][\d.]+)%\)', l)
            if m:
                gate['direction'] = m.group(1)
                gate['trigger_pct'] = float(m.group(2))
            m = re.search(r'peak=([\d.]+)%', l)
            if m: gate['peak'] = float(m.group(1))
            m = re.search(r'held=([\d.]+)%', l)
            if m: gate['held'] = float(m.group(1))
            m = re.search(r'ADX=(\d+)', l)
            if m: gate['adx'] = int(m.group(1))
            m = re.search(r'VOL=(\w+)', l)
            if m: gate['vol'] = m.group(1)
            m = re.search(r'RSI=(\d+)', l)
            if m: gate['rsi'] = int(m.group(1))
            m = re.search(r'VWAP=(\w+)', l)
            if m: gate['vwap'] = m.group(1)
            m = re.search(r'vc=(\w+)', l)
            if m: gate['vc'] = m.group(1) == 'True'
        if f'EXHAUST' in l and sym in l:
            m = re.search(r'EI=([\d.]+)', l)
            if m: gate['ei'] = float(m.group(1))
            m = re.search(r'intra=([+-][\d.]+)%', l)
            if m: gate['intraday_pct'] = float(m.group(1))
            m = re.search(r'range=(\d+)%', l)
            if m: gate['range_pos'] = int(m.group(1)) / 100.0
        if 'LATE-DECAY' in l and sym in l:
            m = re.search(r'moved ([+-]?[\d.]+)% intraday', l)
            if m: gate['late_decay_move'] = float(m.group(1))
    return gate

print("=" * 120)
print("G-SLOPE BACKTEST: Mar 17 Grind Trades — Signal-by-Signal Analysis")
print("=" * 120)
print(f"\n{'SYM':<14} {'PNL':>8} {'DIR':<6} {'SCORE':>5} {'ADX':>4} {'VC':>5} {'INTRA%':>7} {'TRIG%':>7} {'FRESH':>6} {'RANGE':>6} {'VEL-D':>6} {'SB-S':>5} {'OI':>8} | {'GATE RESULT':<40}")
print("-" * 120)

total_passed_pnl = 0
total_blocked_pnl = 0
passed_trades = []
blocked_trades = []

for t in sorted(trades, key=lambda x: x['pnl']):
    sym = t['sym']
    pnl = t['pnl']
    g = find_gate_data(sym, watcher_lines)
    
    score = t.get('smart_score', g.get('score', 50))
    adx = g.get('adx', 50)
    vc = g.get('vc', False)
    intraday = abs(g.get('intraday_pct', g.get('late_decay_move', 0)))
    trig_pct = abs(g.get('trigger_pct', 0))
    range_pos = g.get('range_pos', 0.5)
    direction = g.get('direction', 'UP')
    trigger_type = f'SLOW_GRIND_{direction}'
    
    # Compute freshness
    freshness = min(trig_pct / intraday, 1.0) if intraday > 0.3 else 1.0
    
    # --- Simulate G-SLOPE signals ---
    # We don't have velocity/SB data from logs (these are new fields added today)
    # So mark them as N/A and check what the gate WOULD do with just the available signals
    
    oi_signal = ''  # Not in today's watcher logs for grinds
    oi_strength = 0
    
    # Signal A: OI unwinding (not available in today's data — mark N/A)
    oi_unwinding = False
    sig_a = 'N/A'
    
    # Signal B: Velocity deceleration (not available — mark N/A)
    vel_decel = False
    sig_b = 'N/A'
    
    # Signal C: SB stall (not available — mark N/A) 
    sb_stall = False
    sig_c = 'N/A'
    
    # Signal D: Weak trend
    sig_d = 'BLOCK' if (adx < 20 and not vc) else 'ok'
    
    # Signal E: Chasing crumbs
    sig_e = 'BLOCK' if (freshness < 0.25 and score < 55) else 'ok'
    
    # Compound: count warnings
    warnings = sum([oi_unwinding, vel_decel, sb_stall, freshness < 0.50, adx < 25])
    
    # Determine gate result
    blocked = False
    reason = 'PASS'
    if sig_d == 'BLOCK':
        blocked = True
        reason = f'weak-trend ADX={adx}<20 no-vol'
    elif sig_e == 'BLOCK':
        blocked = True
        reason = f'chasing fresh={freshness:.2f}<0.25 sc={score}<55'
    elif warnings >= 3 and score < 70:
        blocked = True
        reason = f'compound({warnings} warns) sc={score}<70'
    
    tag = '❌' if pnl < 0 else '✅' if pnl > 0 else '➖'
    gate_tag = '🚫 BLOCK' if blocked else '✅ PASS'
    
    if blocked:
        total_blocked_pnl += pnl
        blocked_trades.append((sym, pnl))
    else:
        total_passed_pnl += pnl
        passed_trades.append((sym, pnl))
    
    print(f"{tag} {sym:<12} {pnl:>+8.0f} {direction:<6} {score:>5.0f} {adx:>4} {str(vc):>5} {intraday:>+7.1f} {trig_pct:>+7.1f} {freshness:>6.2f} {range_pos:>5.0%} {sig_b:>6} {sig_c:>5} {sig_a:>8} | {gate_tag} {reason}")

print(f"\n{'=' * 120}")
print("SUMMARY")
print(f"{'=' * 120}")
print(f"\nPASSED trades ({len(passed_trades)}):")
for sym, pnl in sorted(passed_trades, key=lambda x: x[1]):
    tag = '✅' if pnl > 0 else '❌' if pnl < 0 else '➖'
    print(f"  {tag} {sym}: ₹{pnl:+,.0f}")
if passed_trades:
    print(f"  → Net PNL: ₹{total_passed_pnl:+,.0f}")
    pw = sum(1 for _, p in passed_trades if p > 0)
    pl = sum(1 for _, p in passed_trades if p <= 0)
    print(f"  → WR: {pw}/{pw+pl} = {pw/(pw+pl)*100:.0f}%")

print(f"\nBLOCKED trades ({len(blocked_trades)}):")
for sym, pnl in sorted(blocked_trades, key=lambda x: x[1]):
    tag = '✅' if pnl > 0 else '❌' if pnl < 0 else '➖'
    print(f"  {tag} {sym}: ₹{pnl:+,.0f}")
if blocked_trades:
    print(f"  → Avoided PNL: ₹{total_blocked_pnl:+,.0f}")
    bw = sum(1 for _, p in blocked_trades if p > 0)
    bl = sum(1 for _, p in blocked_trades if p <= 0)
    print(f"  → Would-block WR: {bw}/{bw+bl} = {bw/(bw+bl)*100:.0f}%")

print(f"\nIMPACT:")
original_pnl = total_passed_pnl + total_blocked_pnl
print(f"  Original grind PNL: ₹{original_pnl:+,.0f}")
print(f"  After G-SLOPE PNL:  ₹{total_passed_pnl:+,.0f}")
print(f"  Improvement:        ₹{total_passed_pnl - original_pnl:+,.0f}")
print(f"  False positives blocked (winners): {sum(1 for _, p in blocked_trades if p > 0)}")

# --- Also check: what about VELOCITY + SB signals tomorrow? ---
print(f"\n{'=' * 120}")
print("SIGNALS NOT AVAILABLE TODAY (new — will fire tomorrow)")
print(f"{'=' * 120}")
print("""
Signal B (Velocity Decel): Blocks when current vel < 40% of peak vel + freshness<65% + score<60
  → Detects: grind was moving fast, now stalled (slope flattening)
  → Score<60 exemption: high-conviction grinds still pass
  → Vol-confirmed grinds typically maintain velocity → won't trigger

Signal C (SB Stall): Blocks when last-60s move < 15% of total grind + freshness<65% + score<60
  → Detects: grind has stopped in the last minute
  → Only blocks when ALSO low freshness (not standalone)

Signal A (OI Unwinding): Blocks when OI shows LONG_UNWINDING on grind-up (or SHORT_COVERING on grind-down)
  → Detects: smart money closing positions in grind direction
  → Only blocks with OI strength >= 0.40 + score<65
  → If OI shows LONG_BUILDUP → actually CONFIRMS grind → won't block

These signals have score exemptions (60-65-70) so high-conviction grinds always pass.
Strong grinds (score >= 65-70) are NEVER blocked by any signal.
""")

# --- Check today's winners: would they survive with hypothetical vel/SB data? ---
print(f"{'=' * 120}")
print("WINNER SAFETY CHECK (would new signals hurt good trades?)")
print(f"{'=' * 120}")
for t in sorted(trades, key=lambda x: -x['pnl']):
    if t['pnl'] <= 0: continue
    sym = t['sym']
    g = find_gate_data(sym, watcher_lines)
    score = t.get('smart_score', g.get('score', 50))
    adx = g.get('adx', 50)
    vc = g.get('vc', False)
    intraday = abs(g.get('intraday_pct', g.get('late_decay_move', 0)))
    trig_pct = abs(g.get('trigger_pct', 0))
    freshness = min(trig_pct / intraday, 1.0) if intraday > 0.3 else 1.0
    
    # Would velocity decel block? Only if score < 60
    vel_safe = 'SAFE (score≥60)' if score >= 60 else f'AT RISK (score={score}<60)'
    # Would SB stall block? Only if freshness < 65% AND score < 60
    sb_safe = 'SAFE' if score >= 60 or freshness >= 0.65 else f'AT RISK'
    # Would OI unwind block? Only if score < 65
    oi_safe = 'SAFE (score≥65)' if score >= 65 else f'DEPENDS on OI data (score={score}<65)'
    
    print(f"\n  ✅ {sym} (PNL=₹{t['pnl']:+,.0f}, score={score}, ADX={adx}, fresh={freshness:.2f})")
    print(f"     Vel decel: {vel_safe}")
    print(f"     SB stall:  {sb_safe}")
    print(f"     OI unwind: {oi_safe}")
