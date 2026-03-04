#!/bin/bash
# ============================================================
# OI Backfill Cron Script
# Runs daily at 8:45 AM IST (Mon-Fri) to refresh futures OI data
# before market opens at 9:15 AM
# ============================================================

TITAN_DIR="/home/ubuntu/titan/agentic_trader"
VENV="/home/ubuntu/titan/venv/bin/python3"
LOG="/home/ubuntu/titan/logs/oi_backfill_cron.log"

# Skip weekends (just in case cron fires somehow)
DOW=$(date +%u)  # 1=Mon, 7=Sun
if [ "$DOW" -ge 6 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP: weekend (day=$DOW)" >> "$LOG"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') START: OI backfill cron" >> "$LOG"

# Delete stale marker so backfill actually runs
rm -f "$TITAN_DIR/ml_models/data/futures_oi/.oi_backfill_done_"* 2>/dev/null

# Run the backfill (months_back=1 is sufficient for daily refresh)
cd "$TITAN_DIR"
"$VENV" -c "
import sys, os
sys.path.insert(0, '.')
from dhan_futures_oi import FuturesOIFetcher

f = FuturesOIFetcher()
if not f.ready:
    print('FATAL: DhanHQ not ready')
    sys.exit(1)

r = f.backfill_all(months_back=1)
ok = sum(1 for v in r.values() if v['status'] == 'ok')
failed = sum(1 for v in r.values() if v['status'] == 'failed')
print(f'RESULT: {ok} OK, {failed} failed out of {len(r)} total')
" >> "$LOG" 2>&1

EXIT_CODE=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') END: exit_code=$EXIT_CODE" >> "$LOG"
echo "---" >> "$LOG"

# Keep log file manageable (last 500 lines)
tail -500 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
