#!/bin/bash
set -e
TS=$(date +%s)
sudo cp /home/ubuntu/titan/agentic_trader/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py.bak.$TS
sudo cp /home/ubuntu/titan/agentic_trader/zerodha_tools.py /home/ubuntu/titan/agentic_trader/zerodha_tools.py.bak.$TS
sudo cp /tmp/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py
sudo cp /tmp/zerodha_tools.py /home/ubuntu/titan/agentic_trader/zerodha_tools.py
sudo chown ubuntu:ubuntu /home/ubuntu/titan/agentic_trader/dashboard.py /home/ubuntu/titan/agentic_trader/zerodha_tools.py
python3 -c "import py_compile; py_compile.compile('/home/ubuntu/titan/agentic_trader/dashboard.py', doraise=True); py_compile.compile('/home/ubuntu/titan/agentic_trader/zerodha_tools.py', doraise=True); print('compile OK')"
sudo systemctl restart titan-bot
sudo pkill -HUP -f 'gunicorn.*dashboard:app' || true
sleep 3
sudo systemctl is-active titan-bot
ps -ef | grep -E 'gunicorn.*dashboard' | grep -v grep | head -2
