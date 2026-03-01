#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Titan Bot - Server-side rollback script
# Usage: ./rollback.sh [backup_file]
# ═══════════════════════════════════════════════════════════
set -e

DEPLOY_DIR=/home/ubuntu/titan
BACKUP_DIR=/home/ubuntu/backups

echo "Available backups:"
ls -lht $BACKUP_DIR/pre-deploy-*.tar.gz 2>/dev/null || { echo "No backups found!"; exit 1; }
echo ""

if [ -n "$1" ]; then
    BACKUP="$1"
else
    BACKUP=$(ls -t $BACKUP_DIR/pre-deploy-*.tar.gz | head -1)
    echo "Using latest: $BACKUP"
fi

if [ ! -f "$BACKUP" ]; then
    echo "Error: Backup file not found: $BACKUP"
    exit 1
fi

echo ""
read -p "Rollback to $BACKUP? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo "Stopping bot..."
sudo systemctl stop titan-bot || true

echo "Preserving .env..."
cp $DEPLOY_DIR/agentic_trader/.env /tmp/.env.backup

echo "Restoring backup..."
cd $DEPLOY_DIR
rm -rf agentic_trader/
tar xzf "$BACKUP"

echo "Restoring .env..."
cp /tmp/.env.backup $DEPLOY_DIR/agentic_trader/.env

echo "Restarting bot..."
sudo systemctl start titan-bot
sleep 2
sudo systemctl status titan-bot --no-pager | head -5

echo ""
echo "✓ Rollback complete!"
