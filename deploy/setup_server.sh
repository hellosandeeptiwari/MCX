#!/usr/bin/env bash
# ============================================================
#  Titan Trading Bot — AWS Ubuntu Server Setup
#  Run once on a fresh Ubuntu 22.04 t3.small (ap-south-1)
#
#  Usage:
#    chmod +x setup_server.sh
#    sudo ./setup_server.sh
# ============================================================
set -euo pipefail

echo "============================================"
echo "  Titan Server Setup — AWS Mumbai"
echo "============================================"

# ── 1. System packages ──
apt-get update -y
apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git nginx certbot python3-certbot-nginx \
    htop tmux curl unzip jq sqlite3 fail2ban ufw

# ── 2. Firewall: SSH + HTTP/HTTPS only ──
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP  (dashboard)
ufw allow 443/tcp   # HTTPS (dashboard)
ufw --force enable
echo "  [OK] Firewall configured"

# ── 3. Create app user (non-root) ──
if ! id -u titan &>/dev/null; then
    useradd -m -s /bin/bash titan
    echo "  [OK] Created user 'titan'"
else
    echo "  [OK] User 'titan' already exists"
fi

# ── 4. App directory ──
APP_DIR="/home/titan/app"
mkdir -p "$APP_DIR"
chown titan:titan "$APP_DIR"

# ── 5. Clone repo (or copy manually later) ──
echo ""
echo "  Next steps (run as 'titan' user):"
echo "    sudo su - titan"
echo "    cd ~/app"
echo "    git clone <your-repo-url> ."
echo "    # OR: scp from your Windows machine"
echo ""

# ── 6. Python virtual environment ──
sudo -u titan bash -c "
    cd $APP_DIR
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip wheel setuptools
    echo '  [OK] Python venv created at $APP_DIR/.venv'
"

# ── 7. Install systemd services ──
# Copy service files (should be in deploy/ after git clone)
if [ -f "$APP_DIR/deploy/titan-bot.service" ]; then
    cp "$APP_DIR/deploy/titan-bot.service" /etc/systemd/system/
    cp "$APP_DIR/deploy/titan-dashboard.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable titan-bot titan-dashboard
    echo "  [OK] Systemd services installed"
else
    echo "  [SKIP] Service files not found — copy repo first, then run:"
    echo "    sudo cp ~/app/deploy/titan-bot.service /etc/systemd/system/"
    echo "    sudo cp ~/app/deploy/titan-dashboard.service /etc/systemd/system/"
    echo "    sudo systemctl daemon-reload"
    echo "    sudo systemctl enable titan-bot titan-dashboard"
fi

# ── 8. Nginx reverse proxy (optional, for dashboard) ──
if [ -f "$APP_DIR/deploy/nginx-titan.conf" ]; then
    cp "$APP_DIR/deploy/nginx-titan.conf" /etc/nginx/sites-available/titan
    ln -sf /etc/nginx/sites-available/titan /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl restart nginx
    echo "  [OK] Nginx configured"
fi

# ── 9. Log rotation ──
cat > /etc/logrotate.d/titan <<'EOF'
/home/titan/app/agentic_trader/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
echo "  [OK] Log rotation configured"

# ── 10. Timezone (IST for market hours) ──
timedatectl set-timezone Asia/Kolkata
echo "  [OK] Timezone set to IST"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  NEXT STEPS:"
echo "  1. Copy code:  scp -r agentic_trader/ titan@<ip>:~/app/"
echo "  2. Copy .env:  scp .env titan@<ip>:~/app/agentic_trader/"
echo "  3. Install deps:"
echo "       sudo su - titan"
echo "       cd ~/app && source .venv/bin/activate"
echo "       pip install -r requirements.txt"
echo "  4. Run preflight:"
echo "       cd agentic_trader && python _preflight.py"
echo "  5. Start paper trading:"
echo "       sudo systemctl start titan-bot"
echo "       journalctl -u titan-bot -f"
echo ""
