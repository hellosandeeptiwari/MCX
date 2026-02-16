# Titan Cloud Deployment — Quick Reference

## Architecture
```
AWS EC2 (t3.medium, Ubuntu 24.04, Asia/Kolkata TZ)
├── /opt/titan/
│   ├── venv/                    # Python 3.12 virtualenv
│   ├── zerodha_token.json       # Daily auth token
│   └── agentic_trader/          # Bot code
│       ├── .env                 # Credentials + TRADING_MODE
│       ├── autonomous_trader.py # Main bot
│       ├── ml_models/           # XGBoost models
│       └── deploy/              # This folder
│           ├── auto_auth.py     # Selenium daily login
│           ├── titan.service    # systemd service
│           ├── titan-start.*    # 9:00 AM auto-start
│           └── titan-stop.*     # 3:45 PM auto-stop
```

## First-Time Setup

### 1. Launch EC2
- **AMI:** Ubuntu 24.04 LTS
- **Type:** t3.medium (2 vCPU, 4 GB)
- **Storage:** 20 GB gp3
- **Security Group:** SSH (22) + Dashboard (5000) from your IP
- **Key pair:** Download `.pem` file

### 2. SSH in and run deploy.sh
```bash
ssh -i titan-key.pem ubuntu@<EC2-IP>
# Upload deploy.sh first, then:
chmod +x deploy.sh && ./deploy.sh
```

### 3. Upload from Windows
```powershell
cd C:\Users\SandeepTiwari\MCX\agentic_trader\deploy
.\upload.ps1 -IP <EC2-IP> -PEM C:\path\to\titan-key.pem
```

### 4. Add auto-auth credentials to .env
SSH into VM and edit `/opt/titan/agentic_trader/.env`:
```
ZERODHA_USER_ID=AB1234
ZERODHA_PASSWORD=your_password
ZERODHA_TOTP_SECRET=your_totp_base32_secret
```
To get TOTP secret: Zerodha Console → My Profile → Password & Security → Re-setup TOTP → copy the secret key.

### 5. Test
```bash
# Test auth
/opt/titan/venv/bin/python /opt/titan/agentic_trader/deploy/auto_auth.py

# Manual start
sudo systemctl start titan

# Watch logs
journalctl -u titan -f
```

## Daily Operation (Automatic)
| Time (IST) | Action |
|---|---|
| 9:00 AM | `titan-start.timer` → runs `auto_auth.py` → starts bot |
| 9:15 AM | Bot begins scanning & trading |
| 3:30 PM | Bot's internal market-close logic kicks in |
| 3:45 PM | `titan-stop.timer` → stops bot process |

## Commands
```bash
# Start/stop
sudo systemctl start titan
sudo systemctl stop titan
sudo systemctl restart titan

# Logs
journalctl -u titan -f              # Live tail
journalctl -u titan --since today   # Today's logs
journalctl -u titan -n 200          # Last 200 lines

# Timer status
systemctl list-timers               # All timers
systemctl status titan-start.timer  # Start timer

# Check if running
systemctl status titan

# Switch between paper/live
# Edit /opt/titan/agentic_trader/.env → TRADING_MODE=PAPER or LIVE
# Then restart: sudo systemctl restart titan
```

## Cost
- **t3.medium:** ~₹2,800/month ($34) running 24/7
- **Tip:** Use AWS Instance Scheduler or stop instance on weekends to save ~30%
- **Even cheaper:** Stop instance after 3:45 PM, start at 8:45 AM via Lambda

## Troubleshooting
| Issue | Fix |
|---|---|
| Auth fails | Check TOTP secret, re-register if needed |
| Bot crashes | `journalctl -u titan -n 50` for error |
| Token expired mid-day | Rare (Kite tokens last full day). Restart: `sudo systemctl restart titan` |
| OOM | Unlikely on t3.medium (4GB). Check: `free -h` |
| Model files missing | Re-upload: `scp -r ml_models/ ubuntu@IP:/opt/titan/agentic_trader/` |
