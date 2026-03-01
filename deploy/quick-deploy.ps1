# ═══════════════════════════════════════════════════════════
# Quick Deploy - Push code directly from local to EC2
# Usage: .\quick-deploy.ps1 [-NoRestart]
# ═══════════════════════════════════════════════════════════
param([switch]$NoRestart)

$Server = "13.232.89.92"
$User = "ubuntu"
$Pem = "$env:USERPROFILE\.ssh\titan-bot-key.pem"
$SshOpts = "-o StrictHostKeyChecking=no -o ServerAliveInterval=15"

Write-Host "`n=== Titan Quick Deploy ===" -ForegroundColor Cyan
Write-Host "Server: $Server"
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

# 1) Git commit check
$status = git status --porcelain 2>&1
if ($status) {
    Write-Host "[!] You have uncommitted changes:" -ForegroundColor Yellow
    Write-Host $status
    $confirm = Read-Host "Deploy anyway? (y/N)"
    if ($confirm -ne 'y') { exit 0 }
}

# 2) Sync .env to server
Write-Host "[1/4] Syncing .env..." -ForegroundColor Green
scp -i $Pem $SshOpts "agentic_trader\.env" "${User}@${Server}:/home/ubuntu/titan/agentic_trader/.env"

# 3) Create archive
Write-Host "[2/4] Creating archive..." -ForegroundColor Green
$tempTar = "$env:TEMP\titan-quick-deploy.tar.gz"
tar czf $tempTar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.db' --exclude='*.db-wal' --exclude='*.db-shm' --exclude='_gmm_cache_*' --exclude='.env' --exclude='*.log' agentic_trader/ requirements.txt deploy/

# 4) Upload & extract
Write-Host "[3/4] Uploading & extracting..." -ForegroundColor Green
scp -i $Pem $SshOpts $tempTar "${User}@${Server}:/tmp/titan-deploy.tar.gz"
ssh -i $Pem $SshOpts "${User}@${Server}" "cd /home/ubuntu/titan && tar xzf /tmp/titan-deploy.tar.gz && rm /tmp/titan-deploy.tar.gz && source venv/bin/activate && pip install -r requirements.txt -q 2>&1 | tail -2"

# 5) Restart
if (-not $NoRestart) {
    Write-Host "[4/4] Restarting bot..." -ForegroundColor Green
    ssh -i $Pem $SshOpts "${User}@${Server}" "sudo systemctl restart titan-bot && sleep 2 && sudo systemctl status titan-bot --no-pager | head -5"
} else {
    Write-Host "[4/4] Skipped restart (-NoRestart flag)" -ForegroundColor Yellow
}

Write-Host "`n=== Deploy Complete! ===" -ForegroundColor Cyan
Remove-Item $tempTar -ErrorAction SilentlyContinue
