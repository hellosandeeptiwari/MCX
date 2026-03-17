import json
from collections import Counter
exits = []
with open("/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-17.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            if d.get("event") == "EXIT": exits.append(d)
        except: pass
long = [e for e in exits if e.get("hold_minutes",0) > 60]
print(f"Total exits: {len(exits)}, Long holds >60min: {len(long)}")
for e in sorted(long, key=lambda x: -x.get("hold_minutes",0))[:10]:
    sym = e.get("underlying","?").replace("NSE:","")
    print(f"  {sym:16s} hold={e.get('hold_minutes',0):4d}m pnl=Rs{e.get('pnl',0):8.0f} exit={e.get('exit_type','?')}")
print()
c = Counter(e.get("exit_type","?") for e in exits)
for t, n in c.most_common(10):
    avg_h = sum(e.get("hold_minutes",0) for e in exits if e.get("exit_type")==t) / max(1,n)
    avg_p = sum(e.get("pnl",0) for e in exits if e.get("exit_type")==t) / max(1,n)
    print(f"  {t:35s} n={n:3d} avg_hold={avg_h:5.0f}m avg_pnl=Rs{avg_p:8.0f}")
