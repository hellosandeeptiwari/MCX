import json, os
from datetime import datetime

ledger_file = os.path.join(os.path.dirname(__file__), 'trade_ledger_2026-02-24.jsonl')

trades = []
with open(ledger_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

entries = {t.get('symbol',''): t for t in trades if t.get('event') == 'ENTRY'}
exits = [t for t in trades if t.get('event') == 'EXIT']
exits_sorted = sorted(exits, key=lambda e: e.get('ts', ''))

print("=" * 100)
print("DEEP LOSS ANALYSIS — Feb 24, 2026")
print("=" * 100)

# 1. PREMIUM DECAY / THETA BLEED ANALYSIS
print("\n🔍 THETA BLEED & PREMIUM DECAY ANALYSIS")
print("-" * 80)
total_theta_bleed = 0
for e in exits_sorted:
    entry_px = e.get('entry_price', 0)
    exit_px = e.get('exit_price', 0)
    pnl = e.get('pnl', 0)
    qty = e.get('quantity', 0)
    sym = e.get('underlying', '?')
    symbol = e.get('symbol', '')
    
    # Check if penny premium (very cheap option)
    is_penny = entry_px < 5.0
    
    # % drop in premium
    pct_drop = ((exit_px - entry_px) / entry_px * 100) if entry_px > 0 else 0
    
    # Effective SL distance
    # Find matching entry for theta/IV info
    entry_match = entries.get(symbol, {})
    entry_theta = entry_match.get('extra', {}).get('theta', None)
    entry_iv = entry_match.get('extra', {}).get('iv', None)
    
    if pnl < 0:
        category = ""
        if is_penny and entry_px <= 1.0:
            category = "PENNY_PREMIUM"
        elif pct_drop < -40:
            category = "PREMIUM_CRUSH"
        elif pct_drop < -20:
            category = "STEEP_DECAY"
        else:
            category = "NORMAL_SL"
        
        # Estimate theta bleed per hour
        hold_min = e.get('hold_minutes', 0)
        if hold_min > 0 and entry_px > 0:
            decay_per_hour = abs(pnl) / (hold_min / 60.0)
        else:
            decay_per_hour = 0
        
        print(f"  {category:16s} | {sym:14s} | entry={entry_px:7.2f} exit={exit_px:7.2f} "
              f"| drop={pct_drop:+6.1f}% | P&L=Rs {pnl:+9,.0f} | qty={qty}")

# 2. STOP LOSS SIZE ANALYSIS
print(f"\n🎯 STOP LOSS EFFICIENCY ANALYSIS")
print("-" * 80)
sl_ratios = []
for e in exits_sorted:
    entry_px = e.get('entry_price', 0)
    exit_px = e.get('exit_price', 0)
    pnl = e.get('pnl', 0)
    qty = e.get('quantity', 0)
    sym = e.get('underlying', '?')
    
    if entry_px > 0 and pnl < 0:
        sl_pct = abs(exit_px - entry_px) / entry_px * 100
        loss_per_lot = abs(pnl) / max(qty, 1) if qty > 0 else abs(pnl)
        sl_ratios.append({
            'sym': sym, 'entry': entry_px, 'exit': exit_px, 
            'sl_pct': sl_pct, 'pnl': pnl, 'qty': qty,
            'loss_per_unit': abs(entry_px - exit_px)
        })

sl_ratios.sort(key=lambda x: x['pnl'])
print(f"  {'Symbol':14s} | {'Entry':>8s} {'Exit':>8s} | {'SL %':>6s} | {'Loss/unit':>9s} | {'Total P&L':>10s} | {'Qty':>5s}")
for s in sl_ratios:
    print(f"  {s['sym']:14s} | {s['entry']:8.2f} {s['exit']:8.2f} | {s['sl_pct']:5.1f}% | Rs {s['loss_per_unit']:7.2f} | Rs {s['pnl']:+9,.0f} | {s['qty']:5d}")

# 3. CHEAP vs EXPENSIVE OPTIONS
print(f"\n💰 CHEAP vs EXPENSIVE OPTION ANALYSIS")
print("-" * 80)
buckets = {
    'Rs 0-1 (penny)': {'entries': [], 'pnl': 0},
    'Rs 1-5 (cheap)': {'entries': [], 'pnl': 0},
    'Rs 5-15 (mid)': {'entries': [], 'pnl': 0},
    'Rs 15-50 (rich)': {'entries': [], 'pnl': 0},
    'Rs 50+ (deep)': {'entries': [], 'pnl': 0},
}
for e in exits_sorted:
    px = e.get('entry_price', 0)
    pnl = e.get('pnl', 0)
    sym = e.get('underlying', '?')
    if px < 1:
        b = 'Rs 0-1 (penny)'
    elif px < 5:
        b = 'Rs 1-5 (cheap)'
    elif px < 15:
        b = 'Rs 5-15 (mid)'
    elif px < 50:
        b = 'Rs 15-50 (rich)'
    else:
        b = 'Rs 50+ (deep)'
    buckets[b]['entries'].append({'sym': sym, 'pnl': pnl, 'px': px})
    buckets[b]['pnl'] += pnl

for b, v in buckets.items():
    n = len(v['entries'])
    wins = sum(1 for e in v['entries'] if e['pnl'] > 0)
    if n > 0:
        wr = wins / n * 100
        syms = ', '.join(f"{e['sym']}({e['pnl']:+,.0f})" for e in v['entries'])
        print(f"  {b:18s}: {n} trades, WR={wr:.0f}%, P&L=Rs {v['pnl']:+,.0f}")
        print(f"    {syms}")

# 4. RE-ENTRY ANALYSIS (same symbol multiple times)
print(f"\n🔄 RE-ENTRY ON SAME UNDERLYING (REPEAT TRADES)")
print("-" * 80)
by_underlying = {}
for e in exits_sorted:
    sym = e.get('underlying', '?')
    if sym not in by_underlying:
        by_underlying[sym] = []
    by_underlying[sym].append(e)

for sym, trades_list in sorted(by_underlying.items(), key=lambda x: sum(t.get('pnl',0) for t in x[1])):
    if len(trades_list) > 1:
        total = sum(t.get('pnl',0) for t in trades_list)
        wins = sum(1 for t in trades_list if t.get('pnl',0) > 0)
        print(f"  {sym}: {len(trades_list)} trades, {wins}W/{len(trades_list)-wins}L, Net P&L=Rs {total:+,.0f}")
        for i, t in enumerate(trades_list, 1):
            reason = t.get('exit_reason', '?')
            print(f"    #{i} entry={t.get('entry_price',0):.2f} exit={t.get('exit_price',0):.2f} "
                  f"P&L={t.get('pnl',0):+,.0f} | {t.get('direction','?')} | dr={t.get('dr_score',0):.3f} "
                  f"smart={t.get('smart_score',0):.1f} | {reason[:40]}")

# 5. TIME-OF-DAY ANALYSIS
print(f"\n⏰ TIME-OF-DAY P&L")
print("-" * 80)
# Group exits by approximate entry time from entry records
for e in exits_sorted:
    entry_time = e.get('entry_time', e.get('ts', ''))
    try:
        if entry_time:
            dt = datetime.fromisoformat(entry_time)
            e['_hour'] = dt.hour
        else:
            e['_hour'] = -1
    except:
        e['_hour'] = -1

hourly = {}
for e in exits_sorted:
    h = e.get('_hour', -1)
    if h not in hourly:
        hourly[h] = {'count': 0, 'pnl': 0, 'wins': 0}
    hourly[h]['count'] += 1
    hourly[h]['pnl'] += e.get('pnl', 0)
    if e.get('pnl', 0) > 0:
        hourly[h]['wins'] += 1

for h in sorted(hourly.keys()):
    v = hourly[h]
    wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
    label = f"{h}:00-{h}:59" if h >= 0 else "Unknown"
    print(f"  {label}: {v['count']} trades, WR={wr:.0f}%, P&L=Rs {v['pnl']:+,.0f}")

# 6. DRAMATIC LOSS DEEP DIVE
print(f"\n💥 DRAMATIC LOSSES (> Rs 10,000)")  
print("-" * 80)
big_losses = [e for e in exits_sorted if e.get('pnl',0) < -10000]
big_losses.sort(key=lambda e: e.get('pnl', 0))
for e in big_losses:
    sym = e.get('underlying', '?')
    pnl = e.get('pnl', 0)
    entry_px = e.get('entry_price', 0)
    exit_px = e.get('exit_price', 0)
    qty = e.get('quantity', 0)
    direction = e.get('direction', '?')
    dr = e.get('dr_score', 0)
    smart = e.get('smart_score', 0)
    reason = e.get('exit_reason', '?')
    pct_drop = ((exit_px - entry_px) / entry_px * 100) if entry_px > 0 else 0
    
    # Was this a 0DTE? (today is expiry)
    # Check option type from symbol name
    option_info = e.get('symbol', '')
    
    print(f"  Rs {pnl:+,.0f} | {sym} {direction}")
    print(f"    Option: {option_info}")
    print(f"    Premium: {entry_px:.2f} → {exit_px:.2f} ({pct_drop:+.1f}%)")
    print(f"    Qty: {qty} | Loss/unit: Rs {abs(entry_px-exit_px):.2f}")
    print(f"    Scores: dr={dr:.4f} smart={smart:.1f}")
    print(f"    Exit: {reason}")
    
    # Diagnose
    issues = []
    if pct_drop < -30:
        issues.append("PREMIUM CRUSHED >30%")
    if entry_px < 3:
        issues.append(f"CHEAP OPTION ({entry_px:.2f}) — high gamma risk, fast decay")
    if dr > 0.10:
        issues.append(f"HIGH DR SCORE ({dr:.3f}) — should NOT have been taken")
    if direction == 'SELL':
        issues.append("SELL direction — PE bought, if underlying rallied = premium crush")
    if qty > 1000:
        issues.append(f"LARGE POSITION ({qty} units)")
    
    print(f"    DIAGNOSIS: {' | '.join(issues) if issues else 'Normal SL hit'}")
    print()

# SUMMARY
print("=" * 100)
print("SUMMARY OF WHAT WENT WRONG")
print("=" * 100)
total_loss = sum(e.get('pnl',0) for e in exits_sorted if e.get('pnl',0) < 0)
total_win = sum(e.get('pnl',0) for e in exits_sorted if e.get('pnl',0) > 0)
print(f"  Total Wins:   Rs {total_win:+,.0f} (from {sum(1 for e in exits_sorted if e.get('pnl',0)>0)} trades)")
print(f"  Total Losses: Rs {total_loss:+,.0f} (from {sum(1 for e in exits_sorted if e.get('pnl',0)<0)} trades)")
print(f"  Net P&L:      Rs {total_win+total_loss:+,.0f}")
print(f"  Risk/Reward:  Avg loss Rs {total_loss/max(1,sum(1 for e in exits_sorted if e.get('pnl',0)<0)):,.0f} vs Avg win Rs {total_win/max(1,sum(1 for e in exits_sorted if e.get('pnl',0)>0)):,.0f}")
