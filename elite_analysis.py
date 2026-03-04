import json, os

ledger_path = "/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-04.jsonl"
entries = []
exits = []
with open(ledger_path) as f:
    for l in f:
        l = l.strip()
        if not l: continue
        d = json.loads(l)
        ev = d.get("event","")
        if ev == "ENTRY": entries.append(d)
        elif ev == "EXIT": exits.append(d)

print("=== ELITE ENTRIES TODAY ===")
elite_entries = [e for e in entries if e.get("source") == "ELITE"]
for e in elite_entries:
    sym = e.get("symbol","")
    underlying = e.get("underlying","")
    direction = e.get("direction","")
    opt_type = e.get("option_type","")
    conf = e.get("ml_confidence",0)
    entry_price = e.get("entry_price",0)
    qty = e.get("quantity",0)
    xgb_dis = e.get("xgb_disagrees", False)
    ml_prob = e.get("ml_move_prob",0)
    setup = e.get("setup_type","")
    sl = e.get("stop_loss",0)
    target = e.get("target",0)
    ts = e.get("ts","")
    print(f"\n  {underlying} {opt_type} dir={direction}")
    print(f"    conf={conf:.3f} ml_prob={ml_prob:.3f} xgb_disagrees={xgb_dis}")
    print(f"    entry={entry_price} qty={qty} SL={sl} target={target}")
    print(f"    setup={setup} time={ts}")
    print(f"    symbol={sym}")
    # Find matching exit
    matching_exits = [x for x in exits if x.get("underlying") == underlying]
    for x in matching_exits:
        print(f"    EXIT: pnl={x.get('pnl',0):+.2f} exit_type={x.get('exit_type','')} exit_price={x.get('exit_price',0)} hold={x.get('hold_duration','')}")

# Check current open ELITE positions
print("\n=== ELITE POSITIONS STILL OPEN ===")
try:
    import subprocess
    result = subprocess.run(["curl", "-s", "http://localhost:5000/api/status"], capture_output=True, text=True)
    status = json.loads(result.stdout)
    for p in status.get("positions", []):
        src = p.get("source","")
        if src == "ELITE":
            sym = p.get("symbol","")[:55]
            upnl = p.get("unrealized_pnl",0)
            entry = p.get("avg_price",0)
            ltp = p.get("ltp",0)
            underlying = p.get("underlying","")
            print(f"  {underlying} {sym}")
            print(f"    entry={entry} ltp={ltp} upnl={upnl:.1f}")
except Exception as e:
    print(f"  Error checking open: {e}")

# Summary
print("\n=== ELITE P&L SUMMARY ===")
elite_underlyings = set(e.get("underlying","") for e in elite_entries)
all_elite_exits = [x for x in exits if x.get("underlying","") in elite_underlyings]
total_exit_pnl = sum(x.get("pnl",0) for x in all_elite_exits)
print(f"Total ELITE exit P&L: {total_exit_pnl:+.2f}")
print(f"ELITE entries: {len(elite_entries)}")
print(f"ELITE exits: {len(all_elite_exits)}")

# Also show ALL entries and their sources for context
print("\n=== ALL ENTRIES BY SOURCE ===")
from collections import Counter
sources = Counter(e.get("source","") for e in entries)
for src, cnt in sources.most_common():
    src_exits = [x for x in exits if x.get("source") == src]
    src_pnl = sum(x.get("pnl",0) for x in src_exits)
    print(f"  {src}: {cnt} entries, exits P&L={src_pnl:+.2f}")
