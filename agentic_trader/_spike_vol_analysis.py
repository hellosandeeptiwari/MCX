import json, os, glob, re
from collections import Counter, defaultdict

print("=" * 70)
print("SPIKE & VOLUME WATCHER ANALYSIS — March 17, 2026")
print("=" * 70)

# Load all events for today
f = 'trade_ledger/trade_ledger_2026-03-17.jsonl'
entries = []
exits = []
scans = []
all_events = []

for line in open(f):
    t = json.loads(line.strip())
    all_events.append(t)
    ev = t.get('event', '')
    src = t.get('source', t.get('setup', ''))
    if ev == 'ENTRY':
        entries.append(t)
    elif ev == 'EXIT':
        exits.append(t)
    elif ev == 'SCAN':
        scans.append(t)

# === SPIKE TRADES ===
print("\n" + "=" * 70)
print("1. WATCHER_PRICE_SPIKE TRADES")
print("=" * 70)

spike_entries = [e for e in entries if 'SPIKE' in str(e.get('source', '')) or 'SPIKE' in str(e.get('rationale', ''))]
spike_exits = [e for e in exits if 'SPIKE' in str(e.get('source', '')) or 'SPIKE' in str(e.get('setup', ''))]

print(f"\nSpike entries: {len(spike_entries)}")
for e in spike_entries:
    sym = e.get('underlying', '').replace('NSE:', '')
    rat = e.get('rationale', '')[:100]
    ts = e.get('ts', '')[:19]
    print(f"  {ts} {sym:<14} | {rat}")

print(f"\nSpike exits: {len(spike_exits)}")
total_spike_pnl = 0
for e in spike_exits:
    sym = e.get('underlying', '').replace('NSE:', '')
    pnl = e.get('pnl', 0)
    total_spike_pnl += pnl
    exit_type = e.get('exit_type', '')
    ts = e.get('ts', '')[:19]
    hold = e.get('hold_minutes', 0)
    print(f"  {ts} {sym:<14} PNL={pnl:>+8.0f} exit={exit_type:<20} hold={hold}min")

print(f"\nTotal Spike PNL: {total_spike_pnl:+.0f}")

# === VOLUME WATCHER ===
print("\n" + "=" * 70)
print("2. VOLUME_WATCHER / OI_VOLUME TRADES")
print("=" * 70)

vol_entries = [e for e in entries if 'VOLUME' in str(e.get('source', '')).upper() or 'VOLUME' in str(e.get('rationale', '')).upper()]
vol_exits = [e for e in exits if 'VOLUME' in str(e.get('source', '')).upper() or 'VOLUME' in str(e.get('setup', '')).upper()]

print(f"\nVolume entries: {len(vol_entries)}")
for e in vol_entries:
    sym = e.get('underlying', '').replace('NSE:', '')
    rat = e.get('rationale', '')[:100]
    ts = e.get('ts', '')[:19]
    print(f"  {ts} {sym:<14} | {rat}")

print(f"\nVolume exits: {len(vol_exits)}")
for e in vol_exits:
    sym = e.get('underlying', '').replace('NSE:', '')
    pnl = e.get('pnl', 0)
    exit_type = e.get('exit_type', '')
    ts = e.get('ts', '')[:19]
    print(f"  {ts} {sym:<14} PNL={pnl:>+8.0f} exit={exit_type}")

# === ALL SOURCES TODAY ===
print("\n" + "=" * 70)
print("3. ALL ENTRY SOURCES TODAY")
print("=" * 70)
src_count = Counter()
src_pnl = defaultdict(float)
for e in entries:
    src = e.get('source', 'UNKNOWN')
    src_count[src] += 1

# Match exits by source
exit_src_count = Counter()
exit_src_pnl = defaultdict(float)
for e in exits:
    src = e.get('source', e.get('setup', 'UNKNOWN'))
    exit_src_count[src] += 1
    exit_src_pnl[src] += e.get('pnl', 0)

print(f"\n{'SOURCE':<25} {'ENTRIES':>8} {'EXITS':>8} {'PNL':>10}")
print("-" * 55)
all_sources = set(list(src_count.keys()) + list(exit_src_count.keys()))
for src in sorted(all_sources):
    print(f"{src:<25} {src_count.get(src, 0):>8} {exit_src_count.get(src, 0):>8} {exit_src_pnl.get(src, 0):>+10.0f}")

# === SCAN EVENTS - check if VOLUME/SPIKE scans happened ===
print("\n" + "=" * 70)
print("4. SCAN EVENTS (SPIKE / VOLUME related)")
print("=" * 70)
spike_scans = [s for s in scans if 'SPIKE' in str(s.get('source', '')).upper() or 'SPIKE' in str(s.get('rationale', '')).upper()]
vol_scans = [s for s in scans if 'VOLUME' in str(s.get('source', '')).upper() or 'VOLUME' in str(s.get('rationale', '')).upper()]
print(f"Spike scan events: {len(spike_scans)}")
print(f"Volume scan events: {len(vol_scans)}")

# Show scan sources breakdown
scan_sources = Counter()
for s in scans:
    scan_sources[s.get('source', 'UNKNOWN')] += 1
print(f"\nAll scan sources: {dict(scan_sources)}")

# === Check bot logs for VOLUME_WATCHER mentions ===
print("\n" + "=" * 70)
print("5. CHECKING BOT LOGS FOR VOLUME/SPIKE MENTIONS")
print("=" * 70)

import subprocess
try:
    result = subprocess.run(
        ['journalctl', '-u', 'titan-bot', '--since', '2026-03-17 09:00', '--until', '2026-03-17 16:00', '--no-pager'],
        capture_output=True, text=True, timeout=10
    )
    logs = result.stdout
    
    # Count SPIKE mentions
    spike_lines = [l for l in logs.split('\n') if 'SPIKE' in l.upper() or 'spike' in l]
    print(f"SPIKE mentions in logs: {len(spike_lines)}")
    for l in spike_lines[:10]:
        print(f"  {l.strip()[:150]}")
    
    # Count VOLUME mentions
    vol_lines = [l for l in logs.split('\n') if 'VOLUME' in l.upper() and 'WATCHER' in l.upper()]
    print(f"\nVOLUME_WATCHER mentions in logs: {len(vol_lines)}")
    for l in vol_lines[:10]:
        print(f"  {l.strip()[:150]}")
    
    # Check if volume watcher is enabled
    vol_enabled = [l for l in logs.split('\n') if 'volume' in l.lower() and ('enable' in l.lower() or 'disable' in l.lower() or 'skip' in l.lower())]
    print(f"\nVolume enable/disable mentions: {len(vol_enabled)}")
    for l in vol_enabled[:5]:
        print(f"  {l.strip()[:150]}")

except Exception as e:
    print(f"Log check failed: {e}")
