#!/bin/bash
systemctl is-active titan-bot
echo "---"
curl -s http://localhost:5000/api/trade_summary > /tmp/r.json
python3 -c "
import json
d = json.load(open('/tmp/r.json'))
print('realized_pnl =', d.get('realized_pnl'))
print('unrealized_pnl =', d.get('unrealized_pnl'))
print('net_pnl =', d.get('net_pnl'))
print('open_count =', d.get('open_count'))
"
echo "---"
curl -s http://localhost:5000/api/trades/today > /tmp/h.json
python3 -c "
import json
d = json.load(open('/tmp/h.json'))
print('history total_pnl =', d.get('total_pnl'))
print('history total_trades =', d.get('total_trades'))
"
