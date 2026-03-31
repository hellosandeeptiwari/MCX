#!/bin/bash
# ============================================================
# Titan Bot — Daily EC2 Maintenance
# Runs daily at 00:30 IST (after market + after-hours settle)
#
# SAFE: Never deletes trade_ledger, ML models, OI snapshots,
#        state JSONs, or config. Only manages logs, caches,
#        temp files, and creates backups.
# ============================================================
set -euo pipefail

TITAN_DIR=/home/ubuntu/titan
APP_DIR=$TITAN_DIR/agentic_trader
LOG_DIR=$TITAN_DIR/logs
BACKUP_DIR=/home/ubuntu/backups
MAINT_LOG=$LOG_DIR/maintenance.log
TODAY=$(date '+%Y-%m-%d')
NOW=$(date '+%Y-%m-%d %H:%M:%S')
HOSTNAME=$(hostname)

# --- Logging helper ---
log() { echo "[$NOW] $1" >> "$MAINT_LOG"; }
log_section() { echo "" >> "$MAINT_LOG"; echo "[$NOW] === $1 ===" >> "$MAINT_LOG"; }

log_section "DAILY MAINTENANCE STARTING"

# ============================================================
# 1. BACKUP CRITICAL FILES (config, state, trade data, models)
# ============================================================
log_section "BACKUPS"

BACKUP_TODAY=$BACKUP_DIR/$TODAY
mkdir -p "$BACKUP_TODAY"

# Config + environment (NEVER lose these)
cp -f $APP_DIR/config.py "$BACKUP_TODAY/" 2>/dev/null && log "  ✓ config.py" || log "  ✗ config.py missing"
cp -f $APP_DIR/.env "$BACKUP_TODAY/" 2>/dev/null && log "  ✓ .env" || log "  ✗ .env missing"
cp -f $APP_DIR/titan_settings.json "$BACKUP_TODAY/" 2>/dev/null && log "  ✓ titan_settings.json" || true
cp -f $APP_DIR/dhan_config.json "$BACKUP_TODAY/" 2>/dev/null && log "  ✓ dhan_config.json" || true

# Active trade state (critical — losing this = orphaned positions)
for F in active_trades.json exit_manager_state.json orb_trades_state.json data_health_state.json order_idempotency.json; do
    if [ -f "$APP_DIR/$F" ]; then
        cp -f "$APP_DIR/$F" "$BACKUP_TODAY/"
        log "  ✓ $F"
    fi
done

# Log state files
for F in gmm_pending_tracks.json watchdog_alerts.json log_bookmarks.json; do
    if [ -f "$LOG_DIR/$F" ]; then
        cp -f "$LOG_DIR/$F" "$BACKUP_TODAY/"
        log "  ✓ logs/$F"
    fi
done

# Trade ledger — copy today's (small, critical)
if [ -d "$APP_DIR/trade_ledger" ]; then
    mkdir -p "$BACKUP_TODAY/trade_ledger"
    # Copy last 7 days of ledger files
    find "$APP_DIR/trade_ledger" -name 'trade_ledger_*.jsonl' -mtime -7 -exec cp {} "$BACKUP_TODAY/trade_ledger/" \; 2>/dev/null
    LEDGER_COUNT=$(ls "$BACKUP_TODAY/trade_ledger/" 2>/dev/null | wc -l)
    log "  ✓ trade_ledger: $LEDGER_COUNT files (last 7 days)"
fi

# ML models — weekly full backup (Sunday only), daily meta-only
DOW=$(date '+%u')  # 1=Mon, 7=Sun
if [ "$DOW" -eq 7 ]; then
    if [ -d "$APP_DIR/ml_models/saved_models" ]; then
        tar czf "$BACKUP_TODAY/ml_models_saved.tar.gz" -C "$APP_DIR/ml_models" saved_models/ 2>/dev/null
        log "  ✓ ML models full backup (Sunday)"
    fi
else
    # Daily: just back up model metadata JSONs
    mkdir -p "$BACKUP_TODAY/ml_meta"
    find "$APP_DIR/ml_models/saved_models" -name '*.json' -exec cp {} "$BACKUP_TODAY/ml_meta/" \; 2>/dev/null
    META_COUNT=$(ls "$BACKUP_TODAY/ml_meta/" 2>/dev/null | wc -l)
    log "  ✓ ML model metadata: $META_COUNT files"
fi

# Zerodha token backup
cp -f $TITAN_DIR/zerodha_token.json "$BACKUP_TODAY/" 2>/dev/null && log "  ✓ zerodha_token.json" || true

# Delete backups older than 14 days
OLD_BACKUPS=$(find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +14 2>/dev/null | wc -l)
if [ "$OLD_BACKUPS" -gt 0 ]; then
    find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +14 -exec rm -rf {} + 2>/dev/null
    log "  Pruned $OLD_BACKUPS old backup dirs (>14 days)"
fi

# ============================================================
# 2. LOG MANAGEMENT (beyond what logrotate handles)
# ============================================================
log_section "LOG MANAGEMENT"

# Force logrotate for titan logs (handles root-owned titan.log)
sudo logrotate -f /etc/logrotate.d/titan 2>/dev/null && log "  ✓ logrotate forced" || log "  ✗ logrotate failed"

# watcher_debug.log — not in logrotate, trim to 5MB
WD_LOG=$APP_DIR/watcher_debug.log
if [ -f "$WD_LOG" ]; then
    WD_SIZE=$(stat -c%s "$WD_LOG" 2>/dev/null || echo 0)
    if [ "$WD_SIZE" -gt 5242880 ]; then
        tail -c 2097152 "$WD_LOG" > "${WD_LOG}.tmp" && mv "${WD_LOG}.tmp" "$WD_LOG"
        log "  ✓ watcher_debug.log trimmed (was $(numfmt --to=iec $WD_SIZE))"
    else
        log "  watcher_debug.log OK ($(numfmt --to=iec $WD_SIZE))"
    fi
fi

# bot_debug.log — safety trim (logrotate should handle, but just in case)
BD_LOG=$APP_DIR/bot_debug.log
if [ -f "$BD_LOG" ]; then
    BD_SIZE=$(stat -c%s "$BD_LOG" 2>/dev/null || echo 0)
    if [ "$BD_SIZE" -gt 10485760 ]; then
        tail -c 2097152 "$BD_LOG" > "${BD_LOG}.tmp" && mv "${BD_LOG}.tmp" "$BD_LOG"
        log "  ✓ bot_debug.log trimmed (was $(numfmt --to=iec $BD_SIZE))"
    fi
fi

# dashboard-error.log — trim to 500KB
DE_LOG=$LOG_DIR/dashboard-error.log
if [ -f "$DE_LOG" ]; then
    DE_SIZE=$(stat -c%s "$DE_LOG" 2>/dev/null || echo 0)
    if [ "$DE_SIZE" -gt 524288 ]; then
        tail -c 262144 "$DE_LOG" > "${DE_LOG}.tmp" && mv "${DE_LOG}.tmp" "$DE_LOG"
        log "  ✓ dashboard-error.log trimmed (was $(numfmt --to=iec $DE_SIZE))"
    fi
fi

# Delete old compressed logs (>30 days)
OLD_GZ=$(find "$LOG_DIR" -name '*.gz' -mtime +30 2>/dev/null | wc -l)
if [ "$OLD_GZ" -gt 0 ]; then
    find "$LOG_DIR" -name '*.gz' -mtime +30 -delete 2>/dev/null
    log "  Deleted $OLD_GZ old compressed logs (>30 days)"
fi

# Maintenance log self-trim (keep last 500 lines)
if [ -f "$MAINT_LOG" ]; then
    ML_LINES=$(wc -l < "$MAINT_LOG")
    if [ "$ML_LINES" -gt 500 ]; then
        tail -n 500 "$MAINT_LOG" > "${MAINT_LOG}.tmp" && mv "${MAINT_LOG}.tmp" "$MAINT_LOG"
    fi
fi

# ============================================================
# 3. CACHE & TEMP CLEANUP
# ============================================================
log_section "CACHE CLEANUP"

# __pycache__ (also done on restart, but clean stale ones daily)
PYCACHE_COUNT=$(find "$TITAN_DIR" -type d -name __pycache__ 2>/dev/null | wc -l)
if [ "$PYCACHE_COUNT" -gt 0 ]; then
    find "$TITAN_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    log "  ✓ Cleared $PYCACHE_COUNT __pycache__ dirs"
fi

# Old GMM cache files (>7 days)
GMM_OLD=$(find "$APP_DIR" -maxdepth 1 -name '_gmm_cache_*.json' -mtime +7 2>/dev/null | wc -l)
if [ "$GMM_OLD" -gt 0 ]; then
    find "$APP_DIR" -maxdepth 1 -name '_gmm_cache_*.json' -mtime +7 -delete 2>/dev/null
    log "  ✓ Deleted $GMM_OLD old GMM cache files"
fi

# APT cache cleanup (safe — just cached .deb packages)
APT_SIZE=$(du -sm /var/cache/apt/archives/ 2>/dev/null | cut -f1)
if [ "${APT_SIZE:-0}" -gt 50 ]; then
    sudo apt-get clean -y 2>/dev/null && log "  ✓ APT cache cleaned (was ${APT_SIZE}MB)" || log "  ✗ APT clean failed"
fi

# Systemd journal — keep max 50MB
sudo journalctl --vacuum-size=50M 2>/dev/null && log "  ✓ Journal vacuumed" || true

# Old .tmp files in titan directory (>2 days)
TMP_OLD=$(find "$TITAN_DIR" -name '*.tmp' -mtime +2 2>/dev/null | wc -l)
if [ "$TMP_OLD" -gt 0 ]; then
    find "$TITAN_DIR" -name '*.tmp' -mtime +2 -delete 2>/dev/null
    log "  ✓ Deleted $TMP_OLD old .tmp files"
fi

# ============================================================
# 4. DISK SPACE CHECK
# ============================================================
log_section "DISK HEALTH"

DISK_PCT=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
DISK_AVAIL=$(df -h / | tail -1 | awk '{print $4}')
log "  Disk usage: ${DISK_PCT}% (${DISK_AVAIL} available)"

if [ "$DISK_PCT" -gt 85 ]; then
    log "  ⚠️ WARNING: Disk usage ${DISK_PCT}% > 85% threshold!"
    # Emergency: clear more aggressively
    find "$LOG_DIR" -name '*.gz' -mtime +7 -delete 2>/dev/null
    sudo apt-get autoremove -y 2>/dev/null || true
    sudo apt-get clean -y 2>/dev/null || true
    log "  Emergency cleanup performed"
elif [ "$DISK_PCT" -gt 75 ]; then
    log "  ℹ️ Disk above 75% — monitor closely"
fi

# ============================================================
# 5. SYSTEM HEALTH
# ============================================================
log_section "SYSTEM HEALTH"

# Memory
MEM_TOTAL=$(free -m | awk '/Mem:/ {print $2}')
MEM_USED=$(free -m | awk '/Mem:/ {print $3}')
MEM_PCT=$((MEM_USED * 100 / MEM_TOTAL))
log "  Memory: ${MEM_USED}MB / ${MEM_TOTAL}MB (${MEM_PCT}%)"

# Swap
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
SWAP_USED=$(free -m | awk '/Swap:/ {print $3}')
if [ "$SWAP_TOTAL" -gt 0 ]; then
    SWAP_PCT=$((SWAP_USED * 100 / SWAP_TOTAL))
    log "  Swap: ${SWAP_USED}MB / ${SWAP_TOTAL}MB (${SWAP_PCT}%)"
    if [ "$SWAP_PCT" -gt 50 ]; then
        log "  ⚠️ High swap usage — bot may be memory-constrained"
    fi
else
    log "  Swap: not configured"
fi

# Titan bot status
BOT_STATUS=$(systemctl is-active titan-bot 2>/dev/null || echo "unknown")
log "  titan-bot: $BOT_STATUS"

# Uptime
UPTIME=$(uptime -p 2>/dev/null || uptime)
log "  System: $UPTIME"

# Check for zombie/orphan python processes (not the main bot)
MAIN_PID=$(systemctl show titan-bot -p MainPID --value 2>/dev/null || echo "0")
ORPHAN_PY=$(ps aux | grep '[p]ython' | grep -v "$MAIN_PID" | grep -v 'logrotate\|cron\|apt' | wc -l)
if [ "$ORPHAN_PY" -gt 0 ]; then
    log "  ℹ️ Found $ORPHAN_PY python processes outside titan-bot PID $MAIN_PID"
fi

# ============================================================
# 6. OI DATA MAINTENANCE
# ============================================================
log_section "OI DATA"

OI_DIR=$APP_DIR/ml_models/data/futures_oi
if [ -d "$OI_DIR" ]; then
    OI_SIZE=$(du -sm "$OI_DIR" 2>/dev/null | cut -f1)
    log "  OI data size: ${OI_SIZE}MB"
    # Compress OI snapshots older than 14 days (keep for retraining, just compress)
    OI_OLD=$(find "$OI_DIR" -name '*.csv' -mtime +14 -not -name '*contracts*' -not -name '*scrip*' 2>/dev/null | wc -l)
    if [ "$OI_OLD" -gt 0 ]; then
        find "$OI_DIR" -name '*.csv' -mtime +14 -not -name '*contracts*' -not -name '*scrip*' -exec gzip {} \; 2>/dev/null
        log "  ✓ Compressed $OI_OLD old OI CSVs (>14 days)"
    fi
    # Delete compressed OI older than 90 days
    OI_ANCIENT=$(find "$OI_DIR" -name '*.csv.gz' -mtime +90 2>/dev/null | wc -l)
    if [ "$OI_ANCIENT" -gt 0 ]; then
        find "$OI_DIR" -name '*.csv.gz' -mtime +90 -delete 2>/dev/null
        log "  Deleted $OI_ANCIENT ancient OI archives (>90 days)"
    fi
fi

# ============================================================
# 7. TRADE LEDGER ARCHIVAL
# ============================================================
log_section "TRADE LEDGER"

LEDGER_DIR=$APP_DIR/trade_ledger
if [ -d "$LEDGER_DIR" ]; then
    LEDGER_SIZE=$(du -sm "$LEDGER_DIR" 2>/dev/null | cut -f1)
    LEDGER_FILES=$(find "$LEDGER_DIR" -name '*.jsonl' 2>/dev/null | wc -l)
    log "  Trade ledger: ${LEDGER_SIZE}MB, $LEDGER_FILES active files"
    
    # Compress ledgers older than 60 days
    OLD_LEDGER=$(find "$LEDGER_DIR" -name 'trade_ledger_*.jsonl' -mtime +60 2>/dev/null | wc -l)
    if [ "$OLD_LEDGER" -gt 0 ]; then
        find "$LEDGER_DIR" -name 'trade_ledger_*.jsonl' -mtime +60 -exec gzip {} \; 2>/dev/null
        log "  ✓ Compressed $OLD_LEDGER old ledger files (>60 days)"
    fi
    
    # Delete compressed ledgers older than 180 days
    ANCIENT_LEDGER=$(find "$LEDGER_DIR" -name '*.jsonl.gz' -mtime +180 2>/dev/null | wc -l)
    if [ "$ANCIENT_LEDGER" -gt 0 ]; then
        find "$LEDGER_DIR" -name '*.jsonl.gz' -mtime +180 -delete 2>/dev/null
        log "  Deleted $ANCIENT_LEDGER ancient ledger archives (>180 days)"
    fi
fi

# ============================================================
# SUMMARY
# ============================================================
log_section "MAINTENANCE COMPLETE"

BACKUP_SIZE=$(du -sh "$BACKUP_TODAY" 2>/dev/null | cut -f1)
DISK_AFTER=$(df -h / | tail -1 | awk '{print $3 " / " $2 " (" $5 ")"}')
log "  Backup size: $BACKUP_SIZE"
log "  Disk: $DISK_AFTER"
log "  Duration: $(date '+%H:%M:%S') (started $NOW)"

echo "[$NOW] Maintenance complete — backup: $BACKUP_SIZE, disk: $DISK_AFTER"
