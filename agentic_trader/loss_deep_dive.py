"""Deep loss analysis — WHY did 17 trades lose despite technical signals?"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
from collections import defaultdict

trades = json.load(open('trade_history.json'))
data = json.load(open('active_trades.json'))
active = data.get('active_trades', []) if isinstance(data, dict) else data
all_today = [t for t in trades if '2026-02-16' in t.get('timestamp', '')] + \
            [t for t in active if '2026-02-16' in t.get('timestamp', '')]

losers = [t for t in all_today if (t.get('pnl', 0) or 0) < 0]
winners = [t for t in all_today if (t.get('pnl', 0) or 0) > 0]

print("=" * 80)
print("  WHY DID 17 TRADES LOSE? — ROOT CAUSE ANALYSIS")
print("=" * 80)

# Load scan decisions for context
decisions = json.load(open('scan_decisions.json'))
today_decisions = [d for d in decisions if '2026-02-16' in d.get('timestamp', '')]

print(f"\n{'='*60}")
print(f"  LOSER DEEP DIVE — Each trade examined")
print(f"{'='*60}")

for t in sorted(losers, key=lambda x: x.get('pnl', 0)):
    sym = t.get('symbol', '?')
    short_sym = sym.replace('NFO:', '').split('|')[0]
    underlying = t.get('underlying', '')
    pnl = t.get('pnl', 0) or 0
    status = t.get('status', '?')
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0) or 0
    score = t.get('entry_score', t.get('score', '?'))
    side = t.get('side', '?')
    qty = t.get('quantity', 0)
    ts = t.get('timestamp', '')
    exit_time = t.get('exit_time', '')
    rationale = t.get('rationale', '')
    
    # Calculate hold time
    hold_mins = 0
    try:
        entry_dt = datetime.fromisoformat(ts)
        exit_dt = datetime.fromisoformat(exit_time)
        hold_mins = (exit_dt - entry_dt).total_seconds() / 60
    except:
        pass
    
    # Entry metadata
    meta = t.get('entry_metadata', {})
    setup = meta.get('setup_type', meta.get('strategy_type', '?'))
    ml_gate = meta.get('ml_gate_prob', '?')
    ml_dir = meta.get('ml_direction_prob', '?') 
    ml_hint = meta.get('ml_direction_hint', '?')
    conviction = meta.get('directional_strength', meta.get('conviction', '?'))
    
    # Exit detail
    exit_detail = t.get('exit_detail', {})
    candles = exit_detail.get('candles_held', '?')
    max_r = exit_detail.get('r_multiple_achieved', '?')
    breakeven = exit_detail.get('breakeven_applied', '?')
    trailing = exit_detail.get('trailing_active', '?')
    partial = exit_detail.get('partial_booked', '?')
    
    # Premium decay  
    if entry > 0 and exit_p > 0:
        prem_decay_pct = (exit_p - entry) / entry * 100
    else:
        prem_decay_pct = 0
    
    is_spread = t.get('is_credit_spread', False) or t.get('is_debit_spread', False)
    
    print(f"\n  [X] {short_sym} | Loss: Rs{pnl:,.0f}")
    print(f"     Score: {score} | Setup: {setup} | Side: {side}")
    print(f"     Entry: Rs{entry:.2f} -> Exit: Rs{exit_p:.2f} ({prem_decay_pct:+.1f}%)")
    print(f"     Held: {hold_mins:.0f} min | Exit: {status}")
    print(f"     Candles: {candles} | MaxR: {max_r} | BE: {breakeven} | Trail: {trailing}")
    print(f"     ML: gate={ml_gate} dir={ml_dir} hint={ml_hint}")
    print(f"     Rationale: {rationale[:120]}")
    
    # Classify the failure mode
    failure_mode = "UNKNOWN"
    if status == 'TIME_STOP':
        failure_mode = "DIRECTION_RIGHT_MOMENTUM_DIED"
        if hold_mins < 20:
            failure_mode = "ENTERED_TOO_LATE_IN_MOVE"
    elif status == 'STOPLOSS_HIT':
        if hold_mins < 15:
            failure_mode = "IMMEDIATE_REVERSAL"
        else:
            failure_mode = "TREND_REVERSAL_AGAINST"
    elif status == 'THETA_DECAY_WARNING':
        failure_mode = "PREMIUM_TIME_DECAY"
    elif 'SPREAD' in status:
        failure_mode = "SPREAD_WENT_AGAINST"
    elif status == 'SESSION_CUTOFF':
        failure_mode = "DIRECTION_WRONG_OR_FLAT"
    
    print(f"     >> FAILURE MODE: {failure_mode}")

# === PATTERN ANALYSIS ===
print(f"\n{'='*60}")
print(f"  FAILURE MODE SUMMARY")
print(f"{'='*60}")

# Categorize all losses
failure_categories = defaultdict(lambda: {'count': 0, 'pnl': 0, 'trades': []})

for t in losers:
    status = t.get('status', '?')
    hold_mins = 0
    try:
        entry_dt = datetime.fromisoformat(t.get('timestamp', ''))
        exit_dt = datetime.fromisoformat(t.get('exit_time', ''))
        hold_mins = (exit_dt - entry_dt).total_seconds() / 60
    except:
        pass
    
    if status == 'TIME_STOP':
        mode = "TIME_STOP (momentum died / entered late)"
    elif status == 'STOPLOSS_HIT' and hold_mins < 15:
        mode = "IMMEDIATE_REVERSAL (<15 min)"
    elif status == 'STOPLOSS_HIT':
        mode = "SL_HIT (wrong direction)"
    elif status == 'THETA_DECAY_WARNING':
        mode = "THETA_BLEED (re-entry / bad timing)"
    elif 'SPREAD' in status:
        mode = "SPREAD_LOSS"
    elif status == 'SESSION_CUTOFF':
        mode = "SESSION_CUTOFF (too slow / wrong dir)"
    else:
        mode = status
    
    failure_categories[mode]['count'] += 1
    failure_categories[mode]['pnl'] += (t.get('pnl', 0) or 0)
    sym = t.get('symbol', '?').replace('NFO:', '').split('|')[0]
    failure_categories[mode]['trades'].append(sym)

for mode, info in sorted(failure_categories.items(), key=lambda x: x[1]['pnl']):
    print(f"  {mode}")
    print(f"    Count: {info['count']} | Loss: Rs{info['pnl']:,.0f}")
    print(f"    Trades: {', '.join(info['trades'])}")

# === COMPARE WINNERS vs LOSERS ===
print(f"\n{'='*60}")
print(f"  WINNERS vs LOSERS — WHAT'S DIFFERENT?")
print(f"{'='*60}")

def avg_safe(lst):
    return sum(lst) / len(lst) if lst else 0

# Score comparison
w_scores = [t.get('entry_score', 0) or 0 for t in winners if t.get('entry_score')]
l_scores = [t.get('entry_score', 0) or 0 for t in losers if t.get('entry_score')]

print(f"\n  Average Score:")
print(f"    Winners: {avg_safe(w_scores):.1f}")
print(f"    Losers:  {avg_safe(l_scores):.1f}")

# Hold time comparison
w_holds = []
l_holds = []
for t in winners:
    try:
        entry_dt = datetime.fromisoformat(t.get('timestamp', ''))
        exit_dt = datetime.fromisoformat(t.get('exit_time', ''))
        w_holds.append((exit_dt - entry_dt).total_seconds() / 60)
    except:
        pass
for t in losers:
    try:
        entry_dt = datetime.fromisoformat(t.get('timestamp', ''))
        exit_dt = datetime.fromisoformat(t.get('exit_time', ''))
        l_holds.append((exit_dt - entry_dt).total_seconds() / 60)
    except:
        pass

print(f"\n  Average Hold Time:")
print(f"    Winners: {avg_safe(w_holds):.0f} min")
print(f"    Losers:  {avg_safe(l_holds):.0f} min")

# Entry time comparison  
w_hours = []
l_hours = []
for t in winners:
    try:
        dt = datetime.fromisoformat(t.get('timestamp', ''))
        w_hours.append(dt.hour + dt.minute/60)
    except:
        pass
for t in losers:
    try:
        dt = datetime.fromisoformat(t.get('timestamp', ''))
        l_hours.append(dt.hour + dt.minute/60)
    except:
        pass

print(f"\n  Average Entry Hour:")
print(f"    Winners: {avg_safe(w_hours):.1f} ({int(avg_safe(w_hours))}:{int((avg_safe(w_hours)%1)*60):02d})")
print(f"    Losers:  {avg_safe(l_hours):.1f} ({int(avg_safe(l_hours))}:{int((avg_safe(l_hours)%1)*60):02d})")

# ML metadata comparison
w_ml_gates = [t.get('entry_metadata', {}).get('ml_gate_prob', 0) for t in winners if t.get('entry_metadata', {}).get('ml_gate_prob')]
l_ml_gates = [t.get('entry_metadata', {}).get('ml_gate_prob', 0) for t in losers if t.get('entry_metadata', {}).get('ml_gate_prob')]

print(f"\n  ML Gate Probability:")
print(f"    Winners: {avg_safe(w_ml_gates):.3f} ({len(w_ml_gates)} with data)")
print(f"    Losers:  {avg_safe(l_ml_gates):.3f} ({len(l_ml_gates)} with data)")

# Conviction comparison
w_conv = [t.get('entry_metadata', {}).get('directional_strength', 0) for t in winners if t.get('entry_metadata', {}).get('directional_strength')]
l_conv = [t.get('entry_metadata', {}).get('directional_strength', 0) for t in losers if t.get('entry_metadata', {}).get('directional_strength')]

print(f"\n  Directional Conviction:")
print(f"    Winners: {avg_safe(w_conv):.1f}/8 ({len(w_conv)} with data)")
print(f"    Losers:  {avg_safe(l_conv):.1f}/8 ({len(l_conv)} with data)")

# Setup type comparison
print(f"\n  Setup Type Distribution:")
w_setups = defaultdict(int)
l_setups = defaultdict(int)
for t in winners:
    setup = t.get('entry_metadata', {}).get('setup_type', 'UNKNOWN')
    w_setups[setup] += 1
for t in losers:
    setup = t.get('entry_metadata', {}).get('setup_type', 'UNKNOWN')
    l_setups[setup] += 1

all_setups = set(list(w_setups.keys()) + list(l_setups.keys()))
for s in sorted(all_setups):
    w = w_setups.get(s, 0)
    l = l_setups.get(s, 0)
    total = w + l
    wr = w/total*100 if total > 0 else 0
    print(f"    {s:25s}: {w}W/{l}L ({wr:.0f}% WR)")

# === RE-ENTRY ANALYSIS ===
print(f"\n{'='*60}")
print(f"  RE-ENTRY / SAME-STOCK ANALYSIS")
print(f"{'='*60}")

stock_trades = defaultdict(list)
for t in all_today:
    underlying = t.get('underlying', t.get('symbol', '').replace('NFO:', '').split('|')[0][:20])
    stock_trades[underlying].append(t)

multi_entry = {k: v for k, v in stock_trades.items() if len(v) > 1}
for stock, group in sorted(multi_entry.items(), key=lambda x: -len(x[1])):
    total_pnl = sum(t.get('pnl', 0) or 0 for t in group)
    print(f"\n  {stock}: {len(group)} trades | Net: Rs{total_pnl:+,.0f}")
    for t in group:
        sym = t.get('symbol', '?').replace('NFO:', '').split('|')[0]
        pnl = t.get('pnl', 0) or 0
        status = t.get('status', '?')
        spread = '[SPREAD]' if t.get('is_credit_spread') else ''
        ts = t.get('timestamp', '')[:16]
        emoji = '[W]' if pnl > 0 else '[L]' if pnl < 0 else '[-]'
        print(f"    {emoji} {ts} | {sym} {spread} | {status} | Rs{pnl:+,.0f}")

print(f"\n{'='*80}")
print(f"  END OF ROOT CAUSE ANALYSIS")
print(f"{'='*80}")
