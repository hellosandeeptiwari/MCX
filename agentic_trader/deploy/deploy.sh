#!/bin/bash
# =====================================================
# TITAN TRADING BOT — AWS EC2 DEPLOYMENT GUIDE
# Instance: t3.medium (2 vCPU, 4GB RAM)
# OS: Ubuntu 24.04 LTS
# =====================================================
#
# STEP 1: Launch EC2 Instance
# ---------------------------
# 1. Go to AWS Console → EC2 → Launch Instance
# 2. Name: "titan-trader"
# 3. AMI: Ubuntu 24.04 LTS (free tier eligible)
# 4. Instance type: t3.medium
# 5. Key pair: Create new → "titan-key" → Download .pem
# 6. Security group: Allow SSH (port 22) from your IP
#    + Allow port 5000 (dashboard) from your IP (optional)
# 7. Storage: 20 GB gp3
# 8. Launch
#
# STEP 2: Connect
# ----------------
# From Windows PowerShell:
#   ssh -i titan-key.pem ubuntu@<EC2-PUBLIC-IP>
#
# STEP 3: Run this script on the VM
# -----------------------------------
#   chmod +x deploy.sh && ./deploy.sh
#

set -e

echo "╔═══════════════════════════════════════╗"
echo "║  TITAN TRADING BOT — EC2 SETUP        ║"
echo "╚═══════════════════════════════════════╝"

# ---- System deps ----
echo "📦 Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip git unzip curl

# ---- Chrome + Selenium (for auto-auth) ----
echo "🌐 Installing Chrome (headless) for Zerodha auto-login..."
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb || sudo apt-get -f install -y -qq
rm -f google-chrome-stable_current_amd64.deb

# ---- Project directory ----
echo "📂 Setting up project directory..."
sudo mkdir -p /opt/titan
sudo chown ubuntu:ubuntu /opt/titan

# ---- Python venv ----
echo "🐍 Creating Python virtual environment..."
python3.12 -m venv /opt/titan/venv
source /opt/titan/venv/bin/activate

# ---- Install deps ----
echo "📚 Installing Python packages..."
pip install --upgrade pip -q
pip install -r /opt/titan/agentic_trader/requirements.txt -q
pip install selenium pyotp -q

echo ""
echo "✅ System setup complete!"
echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  NEXT STEPS:                                   ║"
echo "║                                                 ║"
echo "║  1. Upload code:                                ║"
echo "║     scp -r -i titan-key.pem agentic_trader/     ║"
echo "║       ubuntu@<IP>:/opt/titan/agentic_trader/    ║"
echo "║                                                 ║"
echo "║  2. Upload .env with credentials:               ║"
echo "║     scp -i titan-key.pem .env                   ║"
echo "║       ubuntu@<IP>:/opt/titan/agentic_trader/    ║"
echo "║                                                 ║"
echo "║  3. Upload zerodha_token.json:                  ║"
echo "║     scp -i titan-key.pem zerodha_token.json     ║"
echo "║       ubuntu@<IP>:/opt/titan/                   ║"
echo "║                                                 ║"
echo "║  4. Upload ML models:                           ║"
echo "║     scp -r -i titan-key.pem ml_models/          ║"
echo "║       ubuntu@<IP>:/opt/titan/agentic_trader/    ║"
echo "║                                                 ║"
echo "║  5. Set timezone:                               ║"
echo "║     sudo timedatectl set-timezone Asia/Kolkata  ║"
echo "║                                                 ║"
echo "║  6. Install systemd service:                    ║"
echo "║     sudo cp titan.service                       ║"
echo "║       /etc/systemd/system/titan.service         ║"
echo "║     sudo systemctl daemon-reload                ║"
echo "║     sudo systemctl enable titan                 ║"
echo "║                                                 ║"
echo "║  7. Start:                                      ║"
echo "║     sudo systemctl start titan                  ║"
echo "║                                                 ║"
echo "║  8. View logs:                                  ║"
echo "║     journalctl -u titan -f                      ║"
echo "╚═══════════════════════════════════════════════╝"
