#!/bin/bash
set -e
TS=$(date +%s)
sudo cp /home/ubuntu/titan/agentic_trader/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py.bak.$TS
sudo cp /tmp/dashboard.py /home/ubuntu/titan/agentic_trader/dashboard.py
sudo chown ubuntu:ubuntu /home/ubuntu/titan/agentic_trader/dashboard.py
python3 -c "import py_compile; py_compile.compile('/home/ubuntu/titan/agentic_trader/dashboard.py', doraise=True); print('compile OK')"
sudo pkill -HUP -f 'gunicorn.*dashboard:app' || true
sleep 2
ps -ef | grep -E 'gunicorn.*dashboard' | grep -v grep | head -2
echo '---PAPER_MODE check---'
grep -n 'PAPER_MODE' /home/ubuntu/titan/agentic_trader/config.py | head -3
