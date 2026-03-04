import json
from collections import defaultdict

entries = {}
exits = {}
with open("/home/ubuntu/titan/agentic_trader/trade_ledger/trade_ledger_2026-03-04.jsonl") as f:
    for line in f:
        t = json.loads(line.strip())
        ev = t.get("event", "")
        sym = t.get("symbol", "")
        if ev == "ENTRY":
            entries[sym] = t
        elif ev == "EXIT":
            exits[sym] = t

print("=== PNL BY SOURCE ===")
src_data = defaultdict(lambda: [0, 0, 0, 0, 0])
for sym, e in entries.items():
    src = e.get("source", "UNK")
    ex = exits.get(sym, {})
    pnl = ex.get("pnl", 0)
    src_data[src][0] += 1
    if ex:
        src_data[src][1] += pnl
        if pnl >= 0:
            src_data[src][2] += 1
        else:
            src_data[src][3] += 1
    else:
        src_data[src][4] += 1

tp = 0
for src in sorted(src_data):
    d = src_data[src]
    tp += d[1]
    print("  %-20s: %d trades  %dW/%dL/%d open  PnL=%+.0f" % (src, d[0], d[2], d[3], d[4], d[1]))
print("  %-20s: %d trades  PnL=%+.0f" % ("TOTAL", len(entries), tp))

print("\n=== ALL TRADES (by entry time) ===")
by_time = sorted(entries.items(), key=lambda x: x[1].get("timestamp", ""))
for sym, e in by_time:
    ex = exits.get(sym, {})
    src = e.get("source", "")
    d = e.get("direction", "")
    pnl = ex.get("pnl", 0)
    exit_r = (ex.get("exit_reason", "OPEN") or "OPEN")[:80]
    ts = e.get("timestamp", "")[:19]
    smart = e.get("smart_score", 0)
    mc = e.get("ml_confidence", 0)
    st = "OPEN" if not ex else ("WIN" if pnl >= 0 else "LOSS")
    print("  %s %-4s %-40s %-15s dir=%-4s smart=%.0f conf=%.2f pnl=%+.0f" % (ts, st, sym, src, d, smart, mc, pnl))
    if ex:
        print("           Exit: %s" % exit_r)

print("\n=== GPT / ALL_AGREE / LLM TRADE CHECK ===")
gpt = 0
for sym, e in entries.items():
    s = e.get("source", "").upper()
    if any(k in s for k in ["GPT", "ALL_AGREE", "COMBINED", "LLM"]):
        gpt += 1
        print("  Found: %s src=%s" % (sym, e.get("source", "")))
if gpt == 0:
    print("  NONE found in today's ledger")
