#!/usr/bin/env python3
"""Check live TEST_GMM config and analyze today's trades."""
import json, os, re, sys
from datetime import datetime

# 1) Load titan_settings.json overrides
settings_path = os.path.join(os.path.dirname(__file__), 'titan_settings.json')
settings = {}
if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)

# 2) Load config.py defaults
sys.path.insert(0, os.path.dirname(__file__))
import config
cfg_defaults = getattr(config, 'TEST_GMM', {})

print("=" * 60)
print("config.py TEST_GMM defaults:")
for k, v in sorted(cfg_defaults.items()):
    print(f"  {k} = {v}")

print("\ntitan_settings.json TEST_GMM overrides:")
test_gmm_settings = settings.get('TEST_GMM', {})
if test_gmm_settings:
    for k, v in sorted(test_gmm_settings.items()):
        print(f"  {k} = {v}")
else:
    for k, v in sorted(settings.items()):
        if 'gmm' in k.lower():
            print(f"  {k} = {v}")
    if not any('gmm' in k.lower() for k in settings):
        print("  (no GMM overrides found)")

# 3) Effective config (settings override config)
effective = dict(cfg_defaults)
if test_gmm_settings:
    effective.update(test_gmm_settings)
print("\nEFFECTIVE TEST_GMM config:")
for k, v in sorted(effective.items()):
    print(f"  {k} = {v}")

# 4) Analyze today's TEST_GMM trades from ledger
print("\n" + "=" * 60)
print("TODAY'S TEST_GMM TRADES:")
ledger_path = os.path.join(os.path.dirname(__file__), 'trade_ledger', 'trade_ledger_2026-03-16.jsonl')
if not os.path.exists(ledger_path):
    print(f"No ledger at {ledger_path}")
    sys.exit(0)

entries = []
exits = {}
with open(ledger_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get('setup_type') == 'TEST_GMM':
            if rec.get('event') == 'ENTRY':
                entries.append(rec)
            elif rec.get('event') == 'EXIT':
                exits[rec.get('underlying', rec.get('symbol', ''))] = rec

print(f"\nEntries: {len(entries)}, Exits: {len(exits)}")

winners = []
losers = []
for e in entries:
    sym = e.get('underlying', e.get('symbol', ''))
    rationale = e.get('rationale', '')
    ts = e.get('timestamp', '')
    
    # Parse UP/DN/gap from rationale
    up_m = re.search(r'UP=([\d.]+)', rationale)
    dn_m = re.search(r'DN=([\d.]+)', rationale)
    gap_m = re.search(r'gap=([\d.]+)', rationale)
    
    up_val = float(up_m.group(1)) if up_m else 0
    dn_val = float(dn_m.group(1)) if dn_m else 0
    gap_val = float(gap_m.group(1)) if gap_m else 0
    
    # Get exit P&L
    exit_rec = exits.get(sym, {})
    pnl = exit_rec.get('pnl', exit_rec.get('realized_pnl', 0))
    
    direction = e.get('direction', 'UNKNOWN')
    
    row = {
        'sym': sym.replace('NSE:', ''),
        'time': ts[11:16] if len(ts) > 11 else ts,
        'dir': direction,
        'up': up_val,
        'dn': dn_val,
        'gap': gap_val,
        'pnl': pnl,
        'rationale': rationale[:80],
    }
    
    if pnl >= 0:
        winners.append(row)
    else:
        losers.append(row)
    
    status = "WIN" if pnl >= 0 else "LOSS"
    print(f"  {status:4s} {row['sym']:20s} {row['time']} {row['dir']:4s} "
          f"UP={row['up']:.3f} DN={row['dn']:.3f} gap={row['gap']:.3f} "
          f"PnL={pnl:+.0f}")

# 5) Winner vs Loser profile
print(f"\n{'='*60}")
print("WINNER PROFILE (TEST_GMM):")
if winners:
    avg_up = sum(w['up'] for w in winners) / len(winners)
    avg_dn = sum(w['dn'] for w in winners) / len(winners)
    avg_gap = sum(w['gap'] for w in winners) / len(winners)
    min_gap = min(w['gap'] for w in winners)
    max_gap = max(w['gap'] for w in winners)
    total_pnl = sum(w['pnl'] for w in winners)
    print(f"  Count: {len(winners)}")
    print(f"  Avg UP={avg_up:.4f}, Avg DN={avg_dn:.4f}, Avg gap={avg_gap:.4f}")
    print(f"  Gap range: {min_gap:.4f} - {max_gap:.4f}")
    print(f"  Total PnL: {total_pnl:+.0f}")
    times = [w['time'] for w in winners]
    print(f"  Times: {', '.join(times)}")

print("\nLOSER PROFILE (TEST_GMM):")
if losers:
    avg_up = sum(w['up'] for w in losers) / len(losers)
    avg_dn = sum(w['dn'] for w in losers) / len(losers)
    avg_gap = sum(w['gap'] for w in losers) / len(losers)
    min_gap = min(w['gap'] for w in losers)
    max_gap = max(w['gap'] for w in losers)
    total_pnl = sum(w['pnl'] for w in losers)
    print(f"  Count: {len(losers)}")
    print(f"  Avg UP={avg_up:.4f}, Avg DN={avg_dn:.4f}, Avg gap={avg_gap:.4f}")
    print(f"  Gap range: {min_gap:.4f} - {max_gap:.4f}")
    print(f"  Total PnL: {total_pnl:+.0f}")
    times = [w['time'] for w in losers]
    print(f"  Times: {', '.join(times)}")

# 6) What thresholds would have filtered out losers but kept winners?
print(f"\n{'='*60}")
print("THRESHOLD ANALYSIS:")
if winners and losers:
    w_max_dn = max(w['dn'] for w in winners)
    w_min_gap = min(w['gap'] for w in winners)
    l_min_dn = min(w['dn'] for w in losers)
    l_max_gap = max(w['gap'] for w in losers)
    
    print(f"  Winner DN range: {min(w['dn'] for w in winners):.4f} - {w_max_dn:.4f}")
    print(f"  Loser  DN range: {l_min_dn:.4f} - {max(w['dn'] for w in losers):.4f}")
    print(f"  Winner gap range: {w_min_gap:.4f} - {max(w['gap'] for w in winners):.4f}")
    print(f"  Loser  gap range: {min(w['gap'] for w in losers):.4f} - {l_max_gap:.4f}")
    
    # Can we separate by DN threshold?
    for dn_cap in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
        w_kept = [w for w in winners if w['dn'] <= dn_cap]
        l_kept = [w for w in losers if w['dn'] <= dn_cap]
        print(f"  DN≤{dn_cap}: keep {len(w_kept)}/{len(winners)} winners, {len(l_kept)}/{len(losers)} losers")
    
    # Can we separate by gap threshold?
    for gap_cap in [0.20, 0.25, 0.28, 0.30, 0.32, 0.35]:
        w_kept = [w for w in winners if w['gap'] <= gap_cap]
        l_kept = [w for w in losers if w['gap'] <= gap_cap]
        print(f"  gap≤{gap_cap}: keep {len(w_kept)}/{len(winners)} winners, {len(l_kept)}/{len(losers)} losers")
    
    # Time analysis
    print(f"\n  TIME ANALYSIS:")
    for hr in ['09', '10', '11', '12', '13', '14', '15']:
        w_in_hr = [w for w in winners if w['time'].startswith(hr)]
        l_in_hr = [w for w in losers if w['time'].startswith(hr)]
        if w_in_hr or l_in_hr:
            print(f"  Hour {hr}: {len(w_in_hr)} winners, {len(l_in_hr)} losers")
