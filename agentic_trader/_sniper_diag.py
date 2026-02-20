"""Diagnose why sniper is not picking trades"""
import json

decisions = json.load(open('scan_decisions.json'))
latest_time = decisions[-1].get('timestamp', '')
latest = [d for d in decisions if d.get('timestamp', '') == latest_time]
if not latest:
    latest = decisions[-50:]

print(f"Latest cycle: {latest_time} ({len(latest)} entries)")
print()

# Check all stocks' dr_score distribution
all_dr = []
sniper_pass = []
for d in latest:
    extra = d.get('extra', {})
    dr = extra.get('dr_score', d.get('dr_score'))
    dr_flag = extra.get('dr_flag', d.get('dr_flag'))
    ml_move = extra.get('ml_move_prob', d.get('ml_move_prob'))
    score = d.get('score', 0)
    sym = d.get('symbol', '?')
    direction = extra.get('direction', d.get('direction', '?'))
    outcome = d.get('outcome', '?')
    
    if dr is not None:
        all_dr.append(dr)
        if dr <= 0.10 and dr_flag == False:
            sniper_pass.append({
                'sym': sym, 'dr': dr, 'dr_flag': dr_flag,
                'ml_move': ml_move or 0, 'score': score,
                'direction': direction, 'outcome': outcome
            })

# DR distribution
if all_dr:
    all_dr.sort()
    print(f"DR score distribution ({len(all_dr)} stocks with DR):")
    print(f"  Min: {min(all_dr):.4f}")
    print(f"  Max: {max(all_dr):.4f}")
    print(f"  Median: {all_dr[len(all_dr)//2]:.4f}")
    print(f"  <= 0.05: {sum(1 for d in all_dr if d <= 0.05)}")
    print(f"  <= 0.10: {sum(1 for d in all_dr if d <= 0.10)}")
    print(f"  <= 0.15: {sum(1 for d in all_dr if d <= 0.15)}")
    print(f"  <= 0.20: {sum(1 for d in all_dr if d <= 0.20)}")
    print()

# Sniper candidates
print(f"Stocks passing sniper GMM gate (dr<=0.10, dr_flag=False): {len(sniper_pass)}")
for s in sorted(sniper_pass, key=lambda x: x['dr']):
    gate_tag = "PASS" if s['ml_move'] >= 0.55 else "FAIL"
    score_tag = "PASS" if s['score'] >= 55 else "FAIL"
    print(f"  {s['sym']:20} dr={s['dr']:.4f} gate={s['ml_move']:.2f}({gate_tag}) "
          f"score={s['score']:.0f}({score_tag}) dir={s['direction']} outcome={s['outcome']}")

# Also check: how many pass ALL sniper gates?
full_pass = [s for s in sniper_pass if s['ml_move'] >= 0.55 and s['score'] >= 55]
print(f"\nFully qualified sniper candidates (all gates): {len(full_pass)}")
for s in full_pass:
    print(f"  {s['sym']:20} dr={s['dr']:.4f} gate={s['ml_move']:.2f} score={s['score']:.0f} dir={s['direction']}")

# Check active trades to see what's already taken
try:
    active = json.load(open('active_trades.json'))
    active_syms = set()
    for t in active:
        if t.get('status', 'OPEN') == 'OPEN':
            ul = t.get('underlying', '')
            active_syms.add(ul)
    print(f"\nActive positions ({len(active_syms)}): {', '.join(s.replace('NSE:','') for s in active_syms)}")
    
    # Check overlap
    blocked = [s for s in full_pass if f"NSE:{s['sym'].replace('NSE:','')}" in active_syms or s['sym'] in active_syms]
    if blocked:
        print(f"Already in portfolio: {', '.join(s['sym'] for s in blocked)}")
except Exception as e:
    print(f"Could not check active trades: {e}")
