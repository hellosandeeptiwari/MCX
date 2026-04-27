#!/bin/bash
set -e
TS=$(date +%s)
sudo cp /home/ubuntu/titan/agentic_trader/templates/titan_dashboard.html /home/ubuntu/titan/agentic_trader/templates/titan_dashboard.html.bak.$TS
sudo cp /home/ubuntu/titan/agentic_trader/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py.bak.$TS
sudo cp /tmp/titan_dashboard.html /home/ubuntu/titan/agentic_trader/templates/titan_dashboard.html
sudo cp /tmp/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py
sudo chown ubuntu:ubuntu /home/ubuntu/titan/agentic_trader/dashboard.py /home/ubuntu/titan/agentic_trader/templates/titan_dashboard.html
python3 -c "import py_compile; py_compile.compile('/home/ubuntu/titan/agentic_trader/dashboard.py', doraise=True); print('compile OK')"
sudo systemctl restart titan-bot
sleep 3
sudo systemctl is-active titan-bot
# Restart gunicorn dashboard too
sudo pkill -HUP -f 'gunicorn.*dashboard:app' || true
sleep 2
ps -ef | grep -E 'gunicorn.*dashboard' | grep -v grep | head -3
