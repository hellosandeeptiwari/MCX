#!/bin/bash
# =====================================================
# TITAN — One-Command Upload to EC2
# =====================================================
# Usage: ./upload.sh <EC2-PUBLIC-IP> <PATH-TO-PEM>
# Example: ./upload.sh 13.233.45.67 ~/titan-key.pem
# =====================================================

set -e

EC2_IP=${1:?"Usage: ./upload.sh <EC2-IP> <PEM-PATH>"}
PEM=${2:?"Usage: ./upload.sh <EC2-IP> <PEM-PATH>"}

echo "🚀 Uploading Titan to $EC2_IP..."

# Exclude unnecessary files
EXCLUDE="--exclude __pycache__ --exclude .pytest_cache --exclude *.pyc --exclude loss_output.txt --exclude candle_replay_results.json"

# Upload agentic_trader (core code + models)
echo "📦 Uploading code..."
rsync -avz --progress -e "ssh -i $PEM" $EXCLUDE \
  ./agentic_trader/ ubuntu@$EC2_IP:/opt/titan/agentic_trader/

# Upload zerodha_token.json
echo "🔑 Uploading token..."
scp -i $PEM ./zerodha_token.json ubuntu@$EC2_IP:/opt/titan/zerodha_token.json 2>/dev/null || echo "  (no token file, will auth on VM)"

# Install systemd services
echo "⚙️ Installing systemd services..."
ssh -i $PEM ubuntu@$EC2_IP << 'EOF'
  sudo cp /opt/titan/agentic_trader/deploy/titan.service /etc/systemd/system/
  sudo cp /opt/titan/agentic_trader/deploy/titan-start.service /etc/systemd/system/
  sudo cp /opt/titan/agentic_trader/deploy/titan-start.timer /etc/systemd/system/
  sudo cp /opt/titan/agentic_trader/deploy/titan-stop.service /etc/systemd/system/
  sudo cp /opt/titan/agentic_trader/deploy/titan-stop.timer /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable titan-start.timer titan-stop.timer
  sudo systemctl start titan-start.timer titan-stop.timer
  sudo timedatectl set-timezone Asia/Kolkata
  echo "✅ Systemd timers installed (auto start 9:00, stop 15:45 IST)"
EOF

echo ""
echo "✅ Upload complete!"
echo ""
echo "To start NOW:    ssh -i $PEM ubuntu@$EC2_IP 'sudo systemctl start titan'"
echo "To view logs:    ssh -i $PEM ubuntu@$EC2_IP 'journalctl -u titan -f'"
echo "To stop:         ssh -i $PEM ubuntu@$EC2_IP 'sudo systemctl stop titan'"
echo "To check status: ssh -i $PEM ubuntu@$EC2_IP 'sudo systemctl status titan'"
