import json
d = json.load(open("/tmp/api_check2.json"))
positions = d.get("positions", [])
print("realized_pnl:", d.get("realized_pnl"))
print("capital:", d.get("capital"))
print("unrealized:", d.get("unrealized_pnl"))
print("open:", d.get("open_positions"))
print()
for p in positions:
    sym = p.get("symbol","")[:55]
    spread = p.get("is_debit_spread", False)
    hedged = p.get("hedged_from_tie", False)
    strat = p.get("strategy_type","")
    entry = p.get("avg_price",0)
    ltp = p.get("ltp",0)
    qty = p.get("quantity",0)
    upnl = p.get("unrealized_pnl",0)
    sell_p = p.get("sell_premium",0)
    nd = p.get("net_debit",0)
    buy_p = p.get("buy_premium",0)
    orig_entry = p.get("original_entry_price", p.get("buy_premium", entry))
    print(sym)
    print("  strat=%s spread=%s hedged=%s" % (strat, spread, hedged))
    print("  entry=%.2f ltp=%.2f qty=%d upnl=%.1f" % (entry, ltp, qty, upnl))
    if spread:
        print("  buy_prem=%.2f sell_prem=%.2f net_debit=%.2f" % (buy_p, sell_p, nd))
        # Hidden cost = original entry - net_debit (the premium given away)
        if buy_p > 0:
            hidden_cost_per_share = buy_p - nd
            hidden_cost_total = hidden_cost_per_share * qty
            print("  HIDDEN HEDGE COST: %.2f/share x %d = %.0f total" % (hidden_cost_per_share, qty, hidden_cost_total))
    print()

# Also check the raw realized_pnl vs exits
print("=== P&L GAP ANALYSIS ===")
print("Realized P&L (state_db):", d.get("realized_pnl"))
print("This is the gap unexplained by trade exits")
