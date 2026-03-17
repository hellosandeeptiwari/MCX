#!/usr/bin/env python3
"""Analyze ALL TEST_GMM trades across all days to find winner vs loser profiles."""
import json, os, re, glob

ledger_dir = os.path.join(os.path.dirname(__file__), 'trade_ledger')
ledger_files = sorted(glob.glob(os.path.join(ledger_dir, 'trade_ledger_*.jsonl')))

all_entries = []
all_exits = {}

for lf in ledger_files:
    day = os.path.basename(lf).replace('trade_ledger_', '').replace('.jsonl', '')
    with open(lf) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except:
                continue
            src = rec.get('source', rec.get('setup_type', ''))
            if 'TEST_GMM' not in src:
                continue
            if rec.get('event') == 'ENTRY':
                rec['_day'] = day
                all_entries.append(rec)
            elif rec.get('event') == 'EXIT':
                key = f"{day}_{rec.get('underlying', rec.get('symbol', ''))}"
                all_exits[key] = rec

print(f"Total TEST_GMM entries across all days: {len(all_entries)}")
print(f"Total TEST_GMM exits: {len(all_exits)}")
print()

winners = []
losers = []

for e in all_entries:
    sym = e.get('underlying', e.get('symbol', ''))
    day = e.get('_day', '')
    rationale = e.get('rationale', '')
    ts = e.get('ts', '')
    direction = e.get('direction', 'UNKNOWN')
    smart = e.get('smart_score', e.get('pre_score', 0))
    
    # Parse UP/DN/gap from rationale
    up_m = re.search(r'UP=([\d.]+)', rationale)
    dn_m = re.search(r'DN=([\d.]+)', rationale)
    gap_m = re.search(r'gap=([\d.]+)', rationale)
    
    up_val = float(up_m.group(1)) if up_m else 0
    dn_val = float(dn_m.group(1)) if dn_m else 0
    gap_val = float(gap_m.group(1)) if gap_m else 0
    
    # Get exit P&L
    exit_key = f"{day}_{sym}"
    exit_rec = all_exits.get(exit_key, {})
    pnl = exit_rec.get('pnl', exit_rec.get('realized_pnl', 0))
    
    time_str = ts[11:16] if len(ts) > 11 else ts
    
    row = {
        'sym': sym.replace('NSE:', ''),
        'day': day,
        'time': time_str,
        'dir': direction,
        'up': up_val,
        'dn': dn_val,
        'gap': gap_val,
        'pnl': pnl,
        'smart': smart,
    }
    
    if pnl >= 0:
        winners.append(row)
    else:
        losers.append(row)

# Print all trades
print("ALL TEST_GMM TRADES:")
print(f"{'STATUS':5s} {'DAY':12s} {'TIME':6s} {'SYMBOL':20s} {'DIR':5s} {'UP':7s} {'DN':7s} {'GAP':7s} {'SMART':6s} {'PNL':>8s}")
print("-" * 90)
for e in all_entries:
    sym = e.get('underlying', '').replace('NSE:', '')
    day = e.get('_day', '')
    ts = e.get('ts', '')
    time_str = ts[11:16] if len(ts) > 11 else ts
    rationale = e.get('rationale', '')
    up_m = re.search(r'UP=([\d.]+)', rationale)
    dn_m = re.search(r'DN=([\d.]+)', rationale)
    gap_m = re.search(r'gap=([\d.]+)', rationale)
    up_val = float(up_m.group(1)) if up_m else 0
    dn_val = float(dn_m.group(1)) if dn_m else 0
    gap_val = float(gap_m.group(1)) if gap_m else 0
    smart = e.get('smart_score', e.get('pre_score', 0))
    exit_key = f"{day}_{e.get('underlying', '')}"
    exit_rec = all_exits.get(exit_key, {})
    pnl = exit_rec.get('pnl', exit_rec.get('realized_pnl', 0))
    status = "WIN" if pnl >= 0 else "LOSS"
    print(f"{status:5s} {day:12s} {time_str:6s} {sym:20s} {e.get('direction',''):5s} "
          f"{up_val:7.3f} {dn_val:7.3f} {gap_val:7.3f} {smart:6.1f} {pnl:+8.0f}")

# Profiles
print(f"\n{'='*60}")
print(f"WINNER PROFILE ({len(winners)} trades):")
if winners:
    avg_up = sum(w['up'] for w in winners) / len(winners)
    avg_dn = sum(w['dn'] for w in winners) / len(winners)
    avg_gap = sum(w['gap'] for w in winners) / len(winners)
    avg_smart = sum(w['smart'] for w in winners) / len(winners)
    total_pnl = sum(w['pnl'] for w in winners)
    print(f"  Avg UP={avg_up:.4f}, Avg DN={avg_dn:.4f}, Avg gap={avg_gap:.4f}, Avg smart={avg_smart:.1f}")
    print(f"  UP range: {min(w['up'] for w in winners):.4f} - {max(w['up'] for w in winners):.4f}")
    print(f"  DN range: {min(w['dn'] for w in winners):.4f} - {max(w['dn'] for w in winners):.4f}")
    print(f"  Gap range: {min(w['gap'] for w in winners):.4f} - {max(w['gap'] for w in winners):.4f}")
    print(f"  Smart range: {min(w['smart'] for w in winners):.1f} - {max(w['smart'] for w in winners):.1f}")
    print(f"  Total PnL: {total_pnl:+.0f}")

print(f"\nLOSER PROFILE ({len(losers)} trades):")
if losers:
    avg_up = sum(w['up'] for w in losers) / len(losers)
    avg_dn = sum(w['dn'] for w in losers) / len(losers)
    avg_gap = sum(w['gap'] for w in losers) / len(losers)
    avg_smart = sum(w['smart'] for w in losers) / len(losers)
    total_pnl = sum(w['pnl'] for w in losers)
    print(f"  Avg UP={avg_up:.4f}, Avg DN={avg_dn:.4f}, Avg gap={avg_gap:.4f}, Avg smart={avg_smart:.1f}")
    print(f"  UP range: {min(w['up'] for w in losers):.4f} - {max(w['up'] for w in losers):.4f}")
    print(f"  DN range: {min(w['dn'] for w in losers):.4f} - {max(w['dn'] for w in losers):.4f}")
    print(f"  Gap range: {min(w['gap'] for w in losers):.4f} - {max(w['gap'] for w in losers):.4f}")
    print(f"  Smart range: {min(w['smart'] for w in losers):.1f} - {max(w['smart'] for w in losers):.1f}")
    print(f"  Total PnL: {total_pnl:+.0f}")

# Threshold sweep
print(f"\n{'='*60}")
print("THRESHOLD SWEEP (DN cap — block over-confident signals):")
all_trades = winners + losers
for dn_cap in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.45, 0.50]:
    w_kept = [w for w in winners if w['dn'] <= dn_cap]
    l_kept = [w for w in losers if w['dn'] <= dn_cap]
    w_pnl = sum(w['pnl'] for w in w_kept)
    l_pnl = sum(w['pnl'] for w in l_kept)
    net = w_pnl + l_pnl
    print(f"  DN≤{dn_cap:.2f}: keep {len(w_kept)}W/{len(l_kept)}L = {len(w_kept)+len(l_kept)} trades, net PnL={net:+.0f}")

print("\nTHRESHOLD SWEEP (min smart score):")
for ms in [0, 10, 15, 20, 25, 30, 35, 40, 45]:
    w_kept = [w for w in winners if w['smart'] >= ms]
    l_kept = [w for w in losers if w['smart'] >= ms]
    w_pnl = sum(w['pnl'] for w in w_kept)
    l_pnl = sum(w['pnl'] for w in l_kept)
    net = w_pnl + l_pnl
    print(f"  smart≥{ms}: keep {len(w_kept)}W/{len(l_kept)}L = {len(w_kept)+len(l_kept)} trades, net PnL={net:+.0f}")

print("\nTHRESHOLD SWEEP (min gap):")
for mg in [0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
    w_kept = [w for w in winners if w['gap'] >= mg]
    l_kept = [w for w in losers if w['gap'] >= mg]
    w_pnl = sum(w['pnl'] for w in w_kept)
    l_pnl = sum(w['pnl'] for w in l_kept)
    net = w_pnl + l_pnl
    print(f"  gap≥{mg:.2f}: keep {len(w_kept)}W/{len(l_kept)}L = {len(w_kept)+len(l_kept)} trades, net PnL={net:+.0f}")

print("\nCOMBINED SWEEP (DN cap + smart floor):")
for dn_cap in [0.34, 0.35, 0.36, 0.38, 0.40]:
    for ms in [15, 20, 25, 30]:
        w_kept = [w for w in winners if w['dn'] <= dn_cap and w['smart'] >= ms]
        l_kept = [w for w in losers if w['dn'] <= dn_cap and w['smart'] >= ms]
        w_pnl = sum(w['pnl'] for w in w_kept)
        l_pnl = sum(w['pnl'] for w in l_kept)
        net = w_pnl + l_pnl
        if len(w_kept) + len(l_kept) > 0:
            wr = len(w_kept)/(len(w_kept)+len(l_kept))*100
        else:
            wr = 0
        print(f"  DN≤{dn_cap:.2f} + smart≥{ms}: {len(w_kept)}W/{len(l_kept)}L ({wr:.0f}% WR), net={net:+.0f}")
