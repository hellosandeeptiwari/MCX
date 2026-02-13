import json, os

print("=== ACTIVE TRADES ===")
if os.path.exists("active_trades.json"):
    d = json.load(open("active_trades.json"))
    trades = d.get("active_trades", [])
    print(f"Total: {len(trades)}")
    for t in trades:
        sym = t.get("symbol", "?")
        st = t.get("status", "?")
        side = t.get("side", "?")
        qty = t.get("quantity", 0)
        spread = " [SPREAD]" if t.get("is_credit_spread") else ""
        print(f"  {sym:35s} status={st:8s} side={side:5s} qty={qty}{spread}")
else:
    print("  No active_trades.json")

print("\n=== RISK STATE ===")
if os.path.exists("risk_state.json"):
    r = json.load(open("risk_state.json"))
    print(f"  State: {r.get('system_state')}")
    print(f"  Daily P&L: {r.get('daily_pnl', 0):+,.0f}")
    print(f"  Trades today: {r.get('trades_today', 0)}")
    print(f"  Date: {r.get('date')}")
else:
    print("  No risk_state.json")

print("\n=== EXIT MANAGER STATE ===")
if os.path.exists("exit_manager_state.json"):
    e = json.load(open("exit_manager_state.json"))
    states = e.get("trade_states", {})
    print(f"  Tracked symbols: {len(states)}")
    for sym, s in states.items():
        print(f"  {sym:35s} sl={s.get('current_sl','?')} spread={s.get('is_credit_spread', False)}")
else:
    print("  No exit_manager_state.json")
