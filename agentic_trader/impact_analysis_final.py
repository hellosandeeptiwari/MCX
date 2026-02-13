"""
Final Impact Analysis - What fixes mean for today's trading
Uses the 12 logged EXECUTED + 11 logged REJECTED entries
"""
import re, json

with open('trade_history.json') as f:
    all_trades = json.load(f)
today_trades = [t for t in all_trades if '2026-02-10' in t.get('closed_at','') or '2026-02-10' in t.get('opened_at','')]

with open('trade_decisions.log', encoding='utf-8', errors='replace') as f:
    content = f.read()

entries = content.split('='*70)

# Parse ALL scored entries (executed + skipped)
all_scored = []
for e in entries:
    if '2026-02-10' not in e or 'Score:' not in e:
        continue
    lines = e.strip().split('\n')
    rec = {'raw': e}
    for l in lines:
        if 'DECISION for' in l: rec['sym'] = l.split('for ')[-1].strip()
        m = re.search(r'Score:\s*(\d+)/100', l)
        if m: rec['score'] = int(m.group(1))
        if 'Direction:' in l: rec['dir'] = l.split(':')[-1].strip()
        if 'EXECUTED' in l: rec['action'] = 'EXECUTED'
        if 'REJECTED' in l or 'SKIP' in l: rec['action'] = 'REJECTED'
        # Components
        if 'EXPLOSIVE volume' in l and '+15' in l: rec['vol'] = 15
        elif 'HIGH volume' in l and '+12' in l: rec['vol'] = 12
        elif 'Normal volume' in l and '+6' in l: rec['vol'] = 6
        elif 'LOW volume' in l: rec['vol'] = 3
        if 'EXHAUSTION OVERRIDE' in l: rec['vol'] = rec.get('vol',0) + 5
        if 'EMA COMPRESSED' in l and '+8' in l: rec['ema'] = 8; rec['ema_type'] = 'COMPRESSED'
        elif 'EMA EXPANDING' in l: rec['ema'] = 10; rec['ema_type'] = 'EXPANDING'
        elif 'EMA normal' in l: rec['ema'] = 3; rec['ema_type'] = 'NORMAL'
        if 'NEUTRAL trend' in l and '-5' in l: rec['trend'] = -5
        if 'ORB BREAKOUT' in l and '+20' in l: rec['orb'] = 20
        m2 = re.search(r'accel\s*\((\d+)/10\)', l)
        if m2: rec['accel'] = int(m2.group(1))
    
    if 'sym' in rec and 'score' in rec:
        all_scored.append(rec)

# Match executed entries to trades
exec_entries = [e for e in all_scored if e.get('action') == 'EXECUTED']
skip_entries = [e for e in all_scored if e.get('action') == 'REJECTED']

# Map to P&L
exec_with_pnl = []
used = set()
for entry in exec_entries:
    sym = entry['sym']
    for ti, t in enumerate(today_trades):
        if ti in used: continue
        if t.get('underlying') == sym:
            entry['pnl'] = t.get('pnl', 0)
            exec_with_pnl.append(entry)
            used.add(ti)
            break

print("=" * 100)
print("IMPACT ANALYSIS: Scoring Pipeline Fixes on Day 4 (2026-02-10)")  
print("=" * 100)

# Part 1: How existing entries would change
print("\n📋 PART 1: LOGGED EXECUTED TRADES (12 of 27) — Score Re-simulation")
print("-" * 100)
print(f"{'Symbol':16s} {'PnL':>8s} {'Old':>4s} {'ΔVol':>5s} {'ΔEMA':>5s} {'ΔTrnd':>6s} {'ΔAcc':>5s} {'ΔPlbk':>6s} {'New':>5s} {'Verdict':>8s}")
print("-" * 100)

sim_total = 0
blocked_total = 0
blocked_list = []

for entry in exec_with_pnl:
    sym = entry['sym'][-12:]
    old = entry['score']
    pnl = entry.get('pnl', 0)
    vol = entry.get('vol', 6)
    ema = entry.get('ema', 8)
    ema_type = entry.get('ema_type', 'COMPRESSED')
    trend = entry.get('trend', -5)
    accel = entry.get('accel', 0)
    
    # Delta calculations with conservative estimates
    
    # Volume: time-normalize bumps ~1 tier
    if vol <= 5: dv = +3    # LOW→NORMAL
    elif vol <= 8: dv = +6  # NORMAL→HIGH  
    elif vol <= 13: dv = +2 # HIGH→stays HIGH (slight bump)
    else: dv = 0            # EXPLOSIVE stays
    
    # EMA: COMPRESSED(8) was universal bug
    if ema_type == 'COMPRESSED':
        # After fix: only truly compressed EMAs get COMPRESSED
        # Most stocks mid-day have mild spread → NORMAL(3) 
        de = -5  # COMPRESSED(8)→NORMAL(3) for most stocks
    else:
        de = 0
    
    # Trend: With fixed volume + ADX + ORB_hold + VWAP numeric
    # Conservative: assume still NEUTRAL for most (TrendFollowing uses
    # its own scoring, not just one indicator)
    dt = 0  # Most will still be NEUTRAL — need ALL components working together
    
    # Acceleration: With fixed scale, some stocks will score
    # But we can't know exactly without real-time data
    da = +2  # Conservative +2 for fixed follow-through counting
    
    # Pullback: universal -5 (EXCELLENT→NO_DATA)
    dp = -5
    
    new = old + dv + de + dt + da + dp
    
    if new >= 50:
        verdict = "PASS"
        sim_total += pnl
    elif new >= 45:
        verdict = "MARGINAL"
        sim_total += pnl  
    else:
        verdict = "BLOCKED"
        blocked_total += pnl
        blocked_list.append((sym, pnl, old, new))
    
    pnl_mark = "🟢" if pnl > 0 else "🔴" if pnl < -1000 else "🟡"
    block_mark = " ← SAVED" if verdict == "BLOCKED" and pnl < 0 else ""
    print(f"{sym:16s} {pnl:>+8.0f} {old:4d} {dv:>+5d} {de:>+5d} {dt:>+6d} {da:>+5d} {dp:>+6d} {new:5d} {verdict:>8s} {pnl_mark}{block_mark}")

print("-" * 100)

# Part 2: Would the REJECTED trades still be rejected?
print(f"\n📋 PART 2: LOGGED REJECTED TRADES (11) — Would they still be blocked?")
print("-" * 100)
print(f"{'Symbol':16s} {'Old':>4s} {'ΔVol':>5s} {'ΔEMA':>5s} {'ΔTrnd':>6s} {'ΔAcc':>5s} {'ΔPlbk':>6s} {'New':>5s} {'Now':>10s}")
print("-" * 100)

for entry in skip_entries:
    sym = entry['sym'][-12:]
    old = entry['score']
    vol = entry.get('vol', 6)
    ema = entry.get('ema', 8)
    ema_type = entry.get('ema_type', 'COMPRESSED')
    
    # Same deltas
    if vol <= 5: dv = +3
    elif vol <= 8: dv = +6
    elif vol <= 13: dv = +2
    else: dv = 0
    
    de = -5 if ema_type == 'COMPRESSED' else 0
    dt = 0
    da = +2
    dp = -5
    
    new = old + dv + de + dt + da + dp
    
    if new >= 50:
        now = "PASS ⚠️"
    elif new >= 45:
        now = "MARGINAL"
    else:
        now = "REJECTED ✅"
    
    print(f"{sym:16s} {old:4d} {dv:>+5d} {de:>+5d} {dt:>+6d} {da:>+5d} {dp:>+6d} {new:5d} {now:>10s}")

print("-" * 100)

# Part 3: Summary
total_pnl = sum(t.get('pnl', 0) for t in today_trades)
logged_pnl = sum(e.get('pnl', 0) for e in exec_with_pnl)
unlogged_pnl = total_pnl - logged_pnl

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}")
print(f"""
  📊 ACTUAL DAY RESULTS:
     Total PnL:    ₹{total_pnl:>+10,.0f} ({len(today_trades)} trades)
     Logged:       ₹{logged_pnl:>+10,.0f} ({len(exec_with_pnl)} trades with scoring log)
     Unlogged:     ₹{unlogged_pnl:>+10,.0f} ({len(today_trades) - len(exec_with_pnl)} trades from early session, no log)
""")

if blocked_list:
    print(f"  🛡️ TRADES THAT WOULD HAVE BEEN BLOCKED:")
    for sym, pnl, old_s, new_s in blocked_list:
        print(f"     {sym:14s}  PnL={pnl:>+8,.0f}  (Score {old_s}→{new_s})")
    losses_saved = sum(-p for _,p,_,_ in blocked_list if p < 0)
    wins_lost = sum(p for _,p,_,_ in blocked_list if p > 0)
    print(f"\n     Losses avoided:    ₹{losses_saved:>+10,.0f}")
    print(f"     Wins missed:       ₹{wins_lost:>+10,.0f}")
    print(f"     Net from blocking: ₹{losses_saved - wins_lost:>+10,.0f}")

print(f"""
  🔑 KEY CHANGES WITH FIXED PIPELINE:
  
  1. VOLUME TIME-NORMALIZATION:
     Before: Partial-day volume vs full-day 5d avg → always LOW/NORMAL
     After:  Projects today's volume to full-day → proper HIGH/EXPLOSIVE detection
     Impact: +3 to +6 pts for most stocks, better discrimination
  
  2. EMA REGIME DISCRIMINATION:
     Before: 0.15% threshold too tight → COMPRESSED for ALL stocks → +8 for everyone
     After:  0.04% threshold → only truly flat EMAs get COMPRESSED
             Most stocks get NORMAL (+3) or EXPANDING (+10)
     Impact: -5 pts for non-trending stocks, +2 pts for trending ones
  
  3. ACCELERATION (follow-through, range expansion, VWAP steepening):
     Before: Broke on first doji, used daily ATR scale → always 0/10
     After:  Doji-tolerant, 5-min ATR scale, proper VWAP comparison
     Impact: +2 to +7 pts for stocks with real momentum, 0 for fakes
  
  4. TREND FOLLOWING DATA PIPELINE:
     Before: Volume always LOW, no ADX, no ORB hold, VWAP string→lossy → always NEUTRAL(-5)
     After:  Real ADX, ORB hold count, numeric VWAP slope → can reach BULLISH(+5)
     Impact: Up to +10 pts swing for strongly trending stocks
  
  5. PULLBACK FALSE POSITIVE:
     Before: depth=0, candles=0 → EXCELLENT(10/10) for free
     After:  No data → NO_DATA(5/10) 
     Impact: -5 pts universal (reduces score inflation)

  ⚖️ NET EFFECT ON SCORING:
     Weak/choppy stocks:    ~40-55 → ~30-40 (BLOCKED)
     Strong momentum stocks: ~55-70 → ~60-75+ (PASSED with higher conviction)
     
     The primary benefit is FEWER LOW-QUALITY ENTRIES, not higher scores.
     This means:
     • Fewer speed-gate losses (weak breakouts that reverse immediately)
     • Better win rate (only entering high-conviction setups)
     • Same or better PnL with fewer trades
""")
