#!/usr/bin/env python3
"""Quick diagnostic: why is watcher not triggering?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import BREAKOUT_WATCHER
from datetime import datetime

now = datetime.now()
ts = now.strftime("%H:%M:%S")

print("=== BREAKOUT WATCHER CONFIG ===")
for k, v in sorted(BREAKOUT_WATCHER.items()):
    print(f"  {k}: {v}")

active_after = BREAKOUT_WATCHER.get("active_after", "09:16")
active_until = BREAKOUT_WATCHER.get("active_until", "15:15")
h1, m1 = map(int, active_after.split(":"))
h2, m2 = map(int, active_until.split(":"))
start = now.replace(hour=h1, minute=m1, second=0)
end = now.replace(hour=h2, minute=m2, second=0)
in_window = start <= now <= end
print(f"\nNow: {ts} | Active window: {active_after}-{active_until} | In window: {in_window}")

# Check if WebSocket is connected by reading titan.log for recent tick counters
import subprocess
r = subprocess.run(
    ["grep", "-c", "Ticker error\|403\|REST.*fallback\|Max reconnect", "/home/ubuntu/titan/logs/titan.log"],
    capture_output=True, text=True
)
err_count = r.stdout.strip()
print(f"\nTicker errors in titan.log: {err_count}")

# Check for sustain/spike events in last section of titan log
r2 = subprocess.run(
    ["tail", "-500", "/home/ubuntu/titan/logs/titan.log"],
    capture_output=True, text=True
)
lines = r2.stdout.split("\n")
watcher_lines = [l for l in lines if "Watcher:" in l or "watcher" in l.lower() and "SUSTAINED" in l]
sustain_lines = [l for l in lines if "SUSTAINED" in l or "sustain" in l.lower()]
spike_lines = [l for l in lines if "SPIKE" in l or "NEW_DAY" in l]
print(f"Watcher log lines (last 500): {len(watcher_lines)}")
print(f"Sustain events: {len(sustain_lines)}")
print(f"Spike/DayHigh events: {len(spike_lines)}")
for l in (watcher_lines + sustain_lines + spike_lines)[-10:]:
    print(f"  {l.strip()[:150]}")

# Check the actual tick flow — look for ticks/TPS in recent heartbeats
print("\n=== TICK FLOW CHECK ===")
r3 = subprocess.run(
    ["grep", "-i", "ticks\|tps\|tick_count\|_on_ticks", "/home/ubuntu/titan/logs/titan.log"],
    capture_output=True, text=True
)
tick_log = r3.stdout.strip().split("\n") if r3.stdout.strip() else []
print(f"Tick-related log lines: {len(tick_log)}")
for l in tick_log[-5:]:
    print(f"  {l.strip()[:150]}")

# Check how many ticks the KiteTicker processes prints
print("\n=== TICKER CONNECTED CHECK ===")
r4 = subprocess.run(
    ["grep", "connected\|Connected\|subscribing\|Subscribed\|subscribe", "/home/ubuntu/titan/logs/titan.log"],
    capture_output=True, text=True
)
conn_lines = r4.stdout.strip().split("\n") if r4.stdout.strip() else []
print(f"Connection lines: {len(conn_lines)}")
for l in conn_lines[-5:]:
    print(f"  {l.strip()[:150]}")
