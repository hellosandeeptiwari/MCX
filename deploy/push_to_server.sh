#!/usr/bin/env bash
# ============================================================
#  Quick deploy to AWS — run this from your LOCAL machine
#  
#  Usage:
#    bash deploy/push_to_server.sh <server-ip> [ssh-key-path]
#
#  Example:
#    bash deploy/push_to_server.sh 13.235.42.100 ~/.ssh/titan-key.pem
# ============================================================
set -euo pipefail

SERVER_IP="${1:?Usage: $0 <server-ip> [ssh-key-path]}"
KEY_ARG=""
if [ -n "${2:-}" ]; then
    KEY_ARG="-i $2"
fi

SSH="ssh $KEY_ARG titan@$SERVER_IP"
SCP="scp $KEY_ARG"

echo "=== Deploying to titan@$SERVER_IP ==="

# 1. Sync code (excludes secrets, DB, caches)
echo "[1/4] Syncing code..."
rsync -avz --delete \
    --exclude '.env' \
    --exclude '*.db' \
    --exclude '*.db-wal' \
    --exclude '*.db-shm' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'logs/' \
    --exclude '*.json.migrated' \
    --exclude '.git' \
    --exclude 'node_modules' \
    -e "ssh $KEY_ARG" \
    ./ titan@$SERVER_IP:~/app/

# 2. Install/update deps
echo "[2/4] Installing dependencies..."
$SSH "cd ~/app && source .venv/bin/activate && pip install -q -r requirements.txt"

# 3. Run preflight check
echo "[3/4] Running preflight check..."
$SSH "cd ~/app/agentic_trader && source ~/app/.venv/bin/activate && python _preflight.py"

# 4. Restart service
echo "[4/4] Restarting titan-bot..."
$SSH "sudo systemctl restart titan-bot"

echo ""
echo "=== Deploy complete! ==="
echo "  View logs: ssh $KEY_ARG titan@$SERVER_IP 'journalctl -u titan-bot -f'"
echo "  Check status: ssh $KEY_ARG titan@$SERVER_IP 'sudo systemctl status titan-bot'"
