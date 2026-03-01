#!/usr/bin/env python3
"""
TITAN v5 — Market Hours Log Watchdog
======================================
Silently tails titan.log during market hours (9:10 AM – 3:35 PM).
Catches errors, exceptions, token failures, order rejections, etc.
Writes alerts to /logs/watchdog.log and exposes via dashboard API.

Run as: systemd service (auto-starts, always on)
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

# ── Config ────────────────────────────────────────────
MARKET_OPEN = (9, 10)   # Start watching 5 min before 9:15
MARKET_CLOSE = (15, 35)  # Stop 5 min after 3:30
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
BOT_LOG = LOG_DIR / 'titan.log'
WATCHDOG_LOG = LOG_DIR / 'watchdog.log'
ALERTS_FILE = LOG_DIR / 'watchdog_alerts.json'

# Patterns to watch for (compiled for speed)
CRITICAL_PATTERNS = [
    (re.compile(r'(FATAL|CRITICAL)', re.I), 'CRITICAL'),
    (re.compile(r'(token.*invalid|token.*expired|TokenException|InputException)', re.I), 'TOKEN_FAIL'),
    (re.compile(r'(EOFError|OSError.*Broken pipe)', re.I), 'CRASH_RISK'),
    (re.compile(r'order.*reject|order.*failed|order.*error', re.I), 'ORDER_FAIL'),
    (re.compile(r'NetworkException|ConnectionError|TimeoutError|ConnectionReset', re.I), 'NETWORK'),
    (re.compile(r'kill.?switch.*activ', re.I), 'KILL_SWITCH'),
    (re.compile(r'max.*daily.*loss|circuit.*break|daily.*loss.*hit', re.I), 'RISK_BREACH'),
    (re.compile(r'MemoryError|OOM|Cannot allocate', re.I), 'OOM'),
]

WARNING_PATTERNS = [
    (re.compile(r'⚠|WARNING', re.I), 'WARNING'),
    (re.compile(r'Traceback \(most recent', re.I), 'TRACEBACK'),
    (re.compile(r'Exception|Error(?!.*log)', re.I), 'EXCEPTION'),
    (re.compile(r'stale.*data|data.*stale|halted', re.I), 'DATA_STALE'),
    (re.compile(r'slippage.*>\s*[5-9]|slippage.*>\s*\d{2,}', re.I), 'HIGH_SLIPPAGE'),
    (re.compile(r'retry|retrying|attempt \d+', re.I), 'RETRY'),
]

# Ignore noise
IGNORE_PATTERNS = [
    re.compile(r'HEARTBEAT.*system_state=ACTIVE', re.I),
    re.compile(r'SCAN.*EXIT.*outside trading hours', re.I),
    re.compile(r'DeprecationWarning', re.I),
    re.compile(r'UserWarning', re.I),
]


def is_market_hours():
    """Check if current time is within market monitoring window."""
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


def log(msg: str):
    """Write to watchdog log with timestamp."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}\n"
    with open(WATCHDOG_LOG, 'a') as f:
        f.write(line)


def save_alerts(alerts: list):
    """Save alerts to JSON (consumed by dashboard)."""
    try:
        with open(ALERTS_FILE, 'w') as f:
            json.dump({
                'updated': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'count': len(alerts),
                'critical': sum(1 for a in alerts if a['level'] == 'CRITICAL'),
                'warnings': sum(1 for a in alerts if a['level'] == 'WARNING'),
                'alerts': alerts[-200:],  # Keep last 200
            }, f, indent=2)
    except Exception:
        pass


def classify_line(line: str):
    """Classify a log line. Returns (level, tag) or None."""
    # Skip noise
    for pat in IGNORE_PATTERNS:
        if pat.search(line):
            return None

    # Check critical first
    for pat, tag in CRITICAL_PATTERNS:
        if pat.search(line):
            return ('CRITICAL', tag)

    # Then warnings
    for pat, tag in WARNING_PATTERNS:
        if pat.search(line):
            return ('WARNING', tag)

    return None


def tail_and_watch():
    """Main watchdog loop — tail the log file and classify lines."""
    alerts_today = []
    traceback_buffer = []
    in_traceback = False
    consecutive_errors = 0
    last_heartbeat = None

    log("WATCHDOG: Starting market-hours monitoring")

    try:
        with open(BOT_LOG, 'r', encoding='utf-8', errors='replace') as f:
            # Seek to end — only watch NEW lines
            f.seek(0, 2)

            while True:
                # Check if still market hours
                if not is_market_hours():
                    if alerts_today:
                        save_alerts(alerts_today)
                        log(f"WATCHDOG: Market closed — {len(alerts_today)} alerts today "
                            f"({sum(1 for a in alerts_today if a['level']=='CRITICAL')} critical)")
                    return alerts_today

                where = f.tell()
                line = f.readline()

                if not line:
                    # No new data — check for heartbeat staleness
                    if last_heartbeat:
                        gap = (datetime.now() - last_heartbeat).total_seconds()
                        if gap > 120:  # No heartbeat for 2 min
                            alert = {
                                'time': datetime.now().isoformat(),
                                'level': 'CRITICAL',
                                'tag': 'BOT_HUNG',
                                'msg': f'No heartbeat for {gap:.0f}s — bot may be hung/crashed',
                            }
                            alerts_today.append(alert)
                            log(f"CRITICAL: {alert['msg']}")
                            last_heartbeat = datetime.now()  # Reset to avoid spam

                    # Check for log rotation
                    try:
                        if os.path.getsize(BOT_LOG) < where:
                            f.seek(0)
                            continue
                    except OSError:
                        pass

                    time.sleep(0.2)
                    continue

                line = line.strip()
                if not line:
                    continue

                # Track heartbeat
                if 'HEARTBEAT' in line:
                    last_heartbeat = datetime.now()
                    consecutive_errors = 0
                    continue

                # Track tracebacks (multi-line)
                if 'Traceback (most recent' in line:
                    in_traceback = True
                    traceback_buffer = [line]
                    continue
                if in_traceback:
                    traceback_buffer.append(line)
                    if not line.startswith(' ') and not line.startswith('File'):
                        # End of traceback — this line is the exception
                        in_traceback = False
                        full_tb = '\n'.join(traceback_buffer[-5:])  # Last 5 lines
                        alert = {
                            'time': datetime.now().isoformat(),
                            'level': 'CRITICAL',
                            'tag': 'TRACEBACK',
                            'msg': line[:200],
                            'traceback': full_tb[:500],
                        }
                        alerts_today.append(alert)
                        log(f"CRITICAL TRACEBACK: {line[:200]}")
                        consecutive_errors += 1
                        traceback_buffer = []
                    continue

                # Classify the line
                result = classify_line(line)
                if result:
                    level, tag = result
                    alert = {
                        'time': datetime.now().isoformat(),
                        'level': level,
                        'tag': tag,
                        'msg': line[:300],
                    }
                    alerts_today.append(alert)
                    log(f"{level} [{tag}]: {line[:200]}")

                    if level == 'CRITICAL':
                        consecutive_errors += 1

                    # Escalation: 5+ critical errors in a row = something very wrong
                    if consecutive_errors >= 5:
                        alert = {
                            'time': datetime.now().isoformat(),
                            'level': 'CRITICAL',
                            'tag': 'ERROR_STORM',
                            'msg': f'{consecutive_errors} consecutive critical errors — possible crash loop',
                        }
                        alerts_today.append(alert)
                        log(f"CRITICAL ERROR_STORM: {consecutive_errors} consecutive errors")
                        consecutive_errors = 0  # Reset

                # Save periodically
                if len(alerts_today) % 10 == 0 and alerts_today:
                    save_alerts(alerts_today)

    except FileNotFoundError:
        log("WATCHDOG ERROR: titan.log not found")
    except Exception as e:
        log(f"WATCHDOG ERROR: {e}")


def main():
    """Run watchdog — loops through market hours, sleeps outside."""
    log("=" * 50)
    log("WATCHDOG SERVICE STARTED")
    log("=" * 50)

    while True:
        if is_market_hours():
            alerts = tail_and_watch()
            # Brief pause after market close before next check
            time.sleep(60)
        else:
            # Outside market hours — sleep and check every 30s
            time.sleep(30)


if __name__ == '__main__':
    main()
