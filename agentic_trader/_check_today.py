import json
for l in open("/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-04-03.jsonl"):
    d = json.loads(l)
    ev = d.get("event", "?")
    src = d.get("source", "?")
    sym = d.get("symbol", "?")
    pnl = d.get("pnl", "")
    ts = d.get("ts", "")[:19]
    print(f"{ev:6s} | {src:15s} | {sym:35s} | pnl={pnl} | {ts}")
