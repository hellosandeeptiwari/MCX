# =====================================================
# TITAN — Upload to EC2 from Windows
# =====================================================
# Usage: .\upload.ps1 -IP <EC2-IP> -PEM <path-to-key.pem>
# Example: .\upload.ps1 -IP 13.233.45.67 -PEM C:\Users\SandeepTiwari\titan-key.pem
# =====================================================

param(
    [Parameter(Mandatory=$true)][string]$IP,
    [Parameter(Mandatory=$true)][string]$PEM
)

$ErrorActionPreference = "Stop"
$BASE = "C:\Users\SandeepTiwari\MCX"
$REMOTE = "ubuntu@${IP}"

Write-Host "`n🚀 Uploading Titan to $IP..." -ForegroundColor Cyan

# Step 1: Upload agentic_trader code
Write-Host "📦 Uploading code + ML models..." -ForegroundColor Yellow
scp -i $PEM -r "$BASE\agentic_trader\*.py" "${REMOTE}:/opt/titan/agentic_trader/"
scp -i $PEM -r "$BASE\agentic_trader\ml_models" "${REMOTE}:/opt/titan/agentic_trader/"
scp -i $PEM -r "$BASE\agentic_trader\deploy" "${REMOTE}:/opt/titan/agentic_trader/"
scp -i $PEM -r "$BASE\agentic_trader\templates" "${REMOTE}:/opt/titan/agentic_trader/"

# Step 2: Upload .env
Write-Host "🔐 Uploading .env..." -ForegroundColor Yellow
scp -i $PEM "$BASE\agentic_trader\.env" "${REMOTE}:/opt/titan/agentic_trader/.env"

# Step 3: Upload requirements.txt
scp -i $PEM "$BASE\agentic_trader\requirements.txt" "${REMOTE}:/opt/titan/agentic_trader/requirements.txt"

# Step 4: Upload zerodha_token.json (if exists)
if (Test-Path "$BASE\zerodha_token.json") {
    Write-Host "🔑 Uploading token..." -ForegroundColor Yellow
    scp -i $PEM "$BASE\zerodha_token.json" "${REMOTE}:/opt/titan/zerodha_token.json"
}

# Step 5: Install deps + systemd on remote
Write-Host "⚙️ Installing on remote VM..." -ForegroundColor Yellow
ssh -i $PEM $REMOTE @"
    set -e
    cd /opt/titan
    source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
    pip install -r agentic_trader/requirements.txt -q
    pip install selenium pyotp -q

    sudo cp agentic_trader/deploy/titan.service /etc/systemd/system/
    sudo cp agentic_trader/deploy/titan-start.service /etc/systemd/system/
    sudo cp agentic_trader/deploy/titan-start.timer /etc/systemd/system/
    sudo cp agentic_trader/deploy/titan-stop.service /etc/systemd/system/
    sudo cp agentic_trader/deploy/titan-stop.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable titan-start.timer titan-stop.timer
    sudo systemctl start titan-start.timer titan-stop.timer
    sudo timedatectl set-timezone Asia/Kolkata
    echo '✅ Remote setup complete'
"@

Write-Host "`n✅ Upload & setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "COMMANDS:" -ForegroundColor Cyan
Write-Host "  Start NOW:     ssh -i $PEM $REMOTE 'sudo systemctl start titan'"
Write-Host "  View logs:     ssh -i $PEM $REMOTE 'journalctl -u titan -f'"
Write-Host "  Stop:          ssh -i $PEM $REMOTE 'sudo systemctl stop titan'"
Write-Host "  Status:        ssh -i $PEM $REMOTE 'sudo systemctl status titan'"
Write-Host "  SSH in:        ssh -i $PEM $REMOTE"
Write-Host ""
Write-Host "  Auto-schedule: 9:00 AM start, 3:45 PM stop (IST, weekdays)" -ForegroundColor DarkGray
