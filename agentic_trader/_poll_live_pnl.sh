#!/bin/bash
# Poll until live_pnl has rows
for i in $(seq 1 12); do
    sleep 30
    rows=$(python3 -c 'import sqlite3; c=sqlite3.connect("/home/ubuntu/titan/agentic_trader/titan_state.db"); print(c.execute("SELECT count(*) FROM live_pnl").fetchone()[0])' 2>/dev/null)
    echo "Check $i: live_pnl rows=$rows"
    if [ "$rows" -gt 0 ] 2>/dev/null; then
        echo "DATA FOUND!"
        python3 /tmp/_check_api.py
        exit 0
    fi
done
echo "Timed out after 6 minutes"
tail -20 /home/ubuntu/titan/logs/titan.log
