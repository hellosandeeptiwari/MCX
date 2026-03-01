"""Quick script to extract today's closed trades from the trade ledger."""
import json

exits = []
with open("trade_ledger/trade_ledger_2026-02-26.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("event") == "EXIT":
            exits.append(rec)

print(f"Total EXIT records today: {len(exits)}\n")
total_pnl = 0
for i, e in enumerate(exits, 1):
    pnl = e.get("pnl", 0)
    total_pnl += pnl
    sym = e["symbol"]
    etype = e["exit_type"]
    ep = e["entry_price"]
    xp = e["exit_price"]
    qty = e["quantity"]
    hold = e.get("hold_minutes", "?")
    ts = e["ts"][:19]
    strat = e.get("strategy_type", "?")
    src = e.get("source", "?")
    print(f"  {i}. {sym}")
    print(f"     Source={src}  Strategy={strat}  ExitType={etype}")
    print(f"     Entry={ep}  Exit={xp}  Qty={qty}  Hold={hold}min")
    print(f"     PnL = {pnl:+.2f}  ({e.get('pnl_pct', 0):+.2f}%)  @{ts}")
    print()

print(f"{'='*60}")
print(f"TOTAL REALIZED P&L TODAY: {total_pnl:+.2f}")
print(f"Wins:  {sum(1 for e in exits if e['pnl'] > 0)}")
print(f"Losses: {sum(1 for e in exits if e['pnl'] <= 0)}")
win_pnl = sum(e['pnl'] for e in exits if e['pnl'] > 0)
loss_pnl = sum(e['pnl'] for e in exits if e['pnl'] <= 0)
print(f"Win total:  {win_pnl:+.2f}")
print(f"Loss total: {loss_pnl:+.2f}")
