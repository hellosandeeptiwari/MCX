import json
from config import BREAKOUT_WATCHER

print("=== BREAKOUT_WATCHER volume/surge settings ===")
for k, v in sorted(BREAKOUT_WATCHER.items()):
    if 'vol' in k.lower() or 'surge' in k.lower():
        print(f"  {k}: {v}")

# Also check titan_settings override
print("\n=== titan_settings.json overrides ===")
try:
    settings = json.load(open('titan_settings.json'))
    for k, v in sorted(settings.items()):
        if 'vol' in k.lower() or 'surge' in k.lower():
            print(f"  {k}: {v}")
except:
    print("  (no titan_settings.json)")

# Check kite_ticker stats
print("\n=== Checking today's journal logs for volume surge ===")
import subprocess
result = subprocess.run(
    ['journalctl', '-u', 'titan-bot', '--since', '2026-03-17 09:00', '--until', '2026-03-17 16:00', '--no-pager', '-o', 'cat'],
    capture_output=True, text=True, timeout=30
)
logs = result.stdout

# Volume surge detections
vol_lines = [l for l in logs.split('\n') if 'VOLUME_SURGE' in l or 'vol_surge' in l or 'SURGE' in l]
print(f"VOLUME_SURGE log lines: {len(vol_lines)}")
for l in vol_lines[:15]:
    print(f"  {l.strip()[:160]}")

# Price spike detections
spike_lines = [l for l in logs.split('\n') if 'SPIKE' in l]
print(f"\nSPIKE log lines: {len(spike_lines)}")
for l in spike_lines[:15]:
    print(f"  {l.strip()[:160]}")

# Watcher stats summary
stat_lines = [l for l in logs.split('\n') if 'stats' in l.lower() and 'watcher' in l.lower()]
print(f"\nWatcher stats lines: {len(stat_lines)}")
for l in stat_lines[:10]:
    print(f"  {l.strip()[:160]}")

# Check for "trigger" mentions
trig_lines = [l for l in logs.split('\n') if 'trigger' in l.lower() and ('volume' in l.lower() or 'surge' in l.lower())]
print(f"\nVolume trigger lines: {len(trig_lines)}")
for l in trig_lines[:10]:
    print(f"  {l.strip()[:160]}")

# Check for watcher summary line
summary_lines = [l for l in logs.split('\n') if 'watcher' in l.lower() and ('fired' in l.lower() or 'trigger' in l.lower() or 'detected' in l.lower())]
print(f"\nWatcher fired/detected lines: {len(summary_lines)}")
for l in summary_lines[:15]:
    print(f"  {l.strip()[:160]}")
