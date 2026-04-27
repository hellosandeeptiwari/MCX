#!/bin/bash
echo "=== $(date +%H:%M:%S) ==="
curl -s http://localhost:5000/api/trade_summary > /tmp/ts.json
curl -s http://localhost:5000/api/active-trades > /tmp/at.json
python3 - << 'PYEOF'
import json
d = json.load(open('/tmp/ts.json'))
try:
    at_raw = json.load(open('/tmp/at.json'))
except Exception:
    at_raw = []
at = at_raw if isinstance(at_raw, list) else at_raw.get('trades', [])
unreal = sum(float(t.get('unrealized_pnl', 0) or 0) for t in at)
print(f"realized={d.get('realized_pnl',0):+,.0f}  unrealized={unreal:+,.0f}  active={len(at)}  capital={d.get('capital',0):,.0f}")
for t in at:
    sym = t.get('symbol','')
    qty = t.get('quantity',0)
    up = float(t.get('unrealized_pnl',0) or 0)
    pct = float(t.get('unrealized_pct',0) or 0)
    hold = t.get('holding_time','')
    print(f"   {sym:<45s} qty={qty:>5} uP={up:+8,.0f} ({pct:+.1f}%) hold={hold}")
PYEOF
echo "--- last 30 AP RULE_FIRED/EXECUTED/FAILED/VETO ---"
sudo tail -n 30000 /home/ubuntu/titan/logs/titan.log | grep -E 'AutoPilot (RULE_FIRED|EXECUTED|EXECUTION_FAILED|VETOED)|HANDS_OFF' | tail -n 30
echo "--- heartbeat last 4 ---"
sudo tail -n 500 /home/ubuntu/titan/logs/titan.log | grep 'heartbeat' | tail -n 4
echo "--- INFY trace ---"
sudo tail -n 20000 /home/ubuntu/titan/logs/titan.log | grep -E 'INFY26APR.*(RULE_FIRED|EXECUTED|EXECUTION_FAILED|VETOED)' | tail -n 10
