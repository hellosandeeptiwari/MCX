#!/bin/bash
# ──────────────────────────────────────────────────────────────
# DHAN TOKEN AUTO-RENEW CRON
# Renews DhanHQ access token before it expires (24h validity)
# Runs BEFORE oi_backfill cron (8:00 AM) so token is fresh
#
# Crontab: 0 8 * * 1-5 /home/ubuntu/titan/agentic_trader/cron_dhan_token.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/home/ubuntu/titan/venv/bin/activate"
LOG="/home/ubuntu/titan/logs/dhan_token_cron.log"

mkdir -p "$(dirname "$LOG")"

# Skip weekends (Sat=6, Sun=0)
DOW=$(date +%w)
if [ "$DOW" = "0" ] || [ "$DOW" = "6" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [SKIP] Weekend — no renewal needed" >> "$LOG"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [START] Dhan token renewal" >> "$LOG"

# Activate venv and load .env
source "$VENV"
cd "$SCRIPT_DIR"

# Run token manager
python dhan_token_manager.py >> "$LOG" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [OK] Token renewed successfully" >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL] Token renewal failed (exit=$EXIT_CODE)" >> "$LOG"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ALERT] Manual renewal needed at https://web.dhan.co/" >> "$LOG"
fi

# Keep log file manageable (last 500 lines)
if [ -f "$LOG" ]; then
    tail -500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [END] Dhan token renewal cron" >> "$LOG"
