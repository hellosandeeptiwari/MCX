#!/bin/bash
# Granular 15-min recap — just take last ~1500 AP-related lines which covers ~15min
sudo tail -n 200000 /home/ubuntu/titan/logs/titan.log | grep -E 'AutoPilot (RULE_FIRED|EXECUTED|EXECUTION_FAILED|VETOED_BY_LLM|POST_REVERSE_LOCKOUT|HANDS_OFF)' | tail -n 300 > /tmp/w15.log
echo "Lines captured: $(wc -l < /tmp/w15.log)"
echo ""
echo "=== Counts by outcome × action ==="
grep -oE 'AutoPilot (EXECUTED|EXECUTION_FAILED|VETOED_BY_LLM) (REVERSE|ADD_LOT)' /tmp/w15.log | sort | uniq -c
echo ""
echo "=== All EXECUTED (chronological last 40) ==="
grep -E 'AutoPilot EXECUTED (REVERSE|ADD_LOT)' /tmp/w15.log | \
  sed -E 's/.*AutoPilot EXECUTED (REVERSE|ADD_LOT) (NFO:[^ ]+).*pnl%=([^ ]+).*$/\1  \2  pnl=\3/' | tail -n 40
echo ""
echo "=== EXECUTION_FAILED ==="
grep 'EXECUTION_FAILED' /tmp/w15.log | tail -n 20
echo ""
echo "=== VETOED (short) ==="
grep -E 'VETOED_BY_LLM' /tmp/w15.log | \
  sed -E 's/.*VETOED_BY_LLM (REVERSE|ADD_LOT) (NFO:[^ ]+).*pnl%=([^ |]+).*\| (.*)$/\1 \2 pnl=\3  → \4/' | tail -n 20
echo ""
echo "=== LOCKOUTS fired ==="
grep 'POST_REVERSE_LOCKOUT' /tmp/w15.log | tail -n 20
echo ""
echo "=== Per-symbol activity ==="
grep -oE 'AutoPilot (EXECUTED|EXECUTION_FAILED|VETOED_BY_LLM) (REVERSE|ADD_LOT) (NFO:[A-Z0-9.]+)' /tmp/w15.log | \
  awk '{print $2, $4}' | sort | uniq -c | sort -rn | head -n 20
echo ""
echo "=== Ledger recent (last 25 entries) + realized ==="
curl -s http://localhost:5000/api/trade_summary > /tmp/ts.json
python3 - << 'PYEOF'
import json
d = json.load(open('/tmp/ts.json'))
entries = d.get('ledger_entries', [])
for e in entries[-25:]:
    t = e.get('time','') or '?'
    sym = e.get('symbol','')
    dirn = e.get('direction','')
    px = float(e.get('entry_price',0) or 0)
    qty = e.get('quantity',0)
    src = e.get('source','')
    print(f"{t:>8s}  {dirn:>4s}  {sym:<45s}  qty={qty:>5}  px={px:>8.2f}  src={src}")
print(f"\nrealized={d.get('realized_pnl',0):+,.0f}  capital={d.get('capital',0):,.0f}")
PYEOF
