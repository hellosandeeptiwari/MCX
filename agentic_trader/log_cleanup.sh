#!/bin/bash
# ============================================================
# Titan Bot — Log Cleanup Script
# Runs on weekdays (after close & before open) and weekends
# KEEPS: OI snapshots (for retraining), trade_ledger (60 days), state files
# ============================================================
set -e

TITAN_DIR=/home/ubuntu/titan
APP_DIR=$TITAN_DIR/agentic_trader
LOG_DIR=$TITAN_DIR/logs
NOW=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$NOW] === Titan Log Cleanup Starting ==="

# --- 1. scan_decisions.json: trim to last 5000 lines ---
SCAN_FILE=$APP_DIR/scan_decisions.json
if [ -f "$SCAN_FILE" ]; then
    LINES=$(wc -l < "$SCAN_FILE")
    if [ "$LINES" -gt 5000 ]; then
        echo "  Trimming scan_decisions.json: $LINES -> 5000 lines"
        tail -n 5000 "$SCAN_FILE" > "${SCAN_FILE}.tmp" && mv "${SCAN_FILE}.tmp" "$SCAN_FILE"
    else
        echo "  scan_decisions.json OK ($LINES lines)"
    fi
fi

# --- 2. watchdog_alerts.json: trim to last 500 lines ---
WD_FILE=$LOG_DIR/watchdog_alerts.json
if [ -f "$WD_FILE" ]; then
    LINES=$(wc -l < "$WD_FILE")
    if [ "$LINES" -gt 500 ]; then
        echo "  Trimming watchdog_alerts.json: $LINES -> 500 lines"
        tail -n 500 "$WD_FILE" > "${WD_FILE}.tmp" && mv "${WD_FILE}.tmp" "$WD_FILE"
    else
        echo "  watchdog_alerts.json OK ($LINES lines)"
    fi
fi

# --- 3. watchdog.log: trim to last 2000 lines ---
WD_LOG=$LOG_DIR/watchdog.log
if [ -f "$WD_LOG" ]; then
    LINES=$(wc -l < "$WD_LOG")
    if [ "$LINES" -gt 2000 ]; then
        echo "  Trimming watchdog.log: $LINES -> 2000 lines"
        tail -n 2000 "$WD_LOG" > "${WD_LOG}.tmp" && mv "${WD_LOG}.tmp" "$WD_LOG"
    fi
fi

# --- 4. Old GMM cache files: delete if older than 5 days ---
GMM_COUNT=$(find $APP_DIR -maxdepth 1 -name '_gmm_cache_*.json' -mtime +5 2>/dev/null | wc -l)
if [ "$GMM_COUNT" -gt 0 ]; then
    echo "  Removing $GMM_COUNT old GMM cache files (>5 days)"
    find $APP_DIR -maxdepth 1 -name '_gmm_cache_*.json' -mtime +5 -delete
else
    echo "  GMM cache files OK"
fi

# --- 5. Old trade_ledger files: keep last 60 days, compress old ones ---
LEDGER_DIR=$APP_DIR/trade_ledger
if [ -d "$LEDGER_DIR" ]; then
    OLD_LEDGER=$(find $LEDGER_DIR -name 'trade_ledger_*.jsonl' -mtime +60 2>/dev/null | wc -l)
    if [ "$OLD_LEDGER" -gt 0 ]; then
        echo "  Archiving $OLD_LEDGER old trade_ledger files (>60 days)"
        find $LEDGER_DIR -name 'trade_ledger_*.jsonl' -mtime +60 -exec gzip {} \;
    fi
    # Delete compressed ledgers older than 120 days
    OLD_GZ=$(find $LEDGER_DIR -name '*.jsonl.gz' -mtime +120 2>/dev/null | wc -l)
    if [ "$OLD_GZ" -gt 0 ]; then
        echo "  Deleting $OLD_GZ ancient trade_ledger archives (>120 days)"
        find $LEDGER_DIR -name '*.jsonl.gz' -mtime +120 -delete
    fi
fi

# --- 6. Old training/analysis output files: truncate if > 7 days old ---
for F in train_v51_log.txt loss_output.txt backtest_output.txt; do
    FPATH=$APP_DIR/$F
    if [ -f "$FPATH" ]; then
        OLD=$(find "$FPATH" -mtime +7 2>/dev/null | wc -l)
        if [ "$OLD" -gt 0 ]; then
            SIZE=$(du -h "$FPATH" | cut -f1)
            echo "  Clearing old $F ($SIZE, >7 days)"
            > "$FPATH"
        fi
    fi
done

# --- 7. Vacuum systemd journal: keep max 50M ---
echo "  Vacuuming systemd journal to 50M"
sudo journalctl --vacuum-size=50M 2>/dev/null || true

# --- 8. Force logrotate (catches anything over maxsize) ---
sudo logrotate -f /etc/logrotate.d/titan 2>/dev/null || true

# --- 9. Clear old compressed logrotate files ---
find $LOG_DIR -name '*.gz' -mtime +30 -delete 2>/dev/null || true
find $APP_DIR -name '*.log.*.gz' -mtime +14 -delete 2>/dev/null || true
find $APP_DIR -name '*.jsonl.*.gz' -mtime +14 -delete 2>/dev/null || true

# --- 10. health-check.log: trim to last 500 lines ---
HC_LOG=$LOG_DIR/health-check.log
if [ -f "$HC_LOG" ]; then
    LINES=$(wc -l < "$HC_LOG")
    if [ "$LINES" -gt 500 ]; then
        echo "  Trimming health-check.log: $LINES -> 500 lines"
        tail -n 500 "$HC_LOG" > "${HC_LOG}.tmp" && mv "${HC_LOG}.tmp" "$HC_LOG"
    fi
fi

# --- Summary ---
echo "  Disk after cleanup:"
df -h / | tail -1 | awk '{print "    Used: " $3 " / " $2 " (" $5 ")"}'
echo "[$NOW] === Cleanup Complete ==="
