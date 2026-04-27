import json, sys
trades = json.load(open("active_trades.json"))
grind_trades = [t for t in trades if "GRIND" in t.get("setup_type","") or "grind" in t.get("rationale","").lower()]
if not grind_trades:
    # Try trade_ledger
    try:
        from trade_ledger import TradeLedger
        ledger = TradeLedger()
        all_t = ledger.get_trades_with_pnl()
        grind_trades = [t for t in all_t if "GRIND" in str(t.get("setup_type","")) or "grind" in str(t.get("rationale","")).lower()]
    except:
        pass

print(f"=== GRIND TRADES TODAY: {len(grind_trades)} ===\n")
for t in grind_trades:
    sym = t.get("underlying", t.get("symbol", "?"))
    d = t.get("direction", "?")
    setup = t.get("setup_type", "?")
    pnl = t.get("pnl", t.get("unrealized_pnl", "?"))
    status = t.get("status", "?")
    entry = str(t.get("entry_time", "?"))[:19]
    rationale = t.get("rationale", "")[:120]
    ml = t.get("ml_data", {})
    score = ml.get("score", "?")
    pmove = ml.get("ml_move_prob", "?")
    dr = ml.get("ml_down_risk_score", "?")
    print(f"{sym:20s} {d:5s} pnl={pnl:>8} status={status:6s} entry={entry}")
    print(f"  setup={setup} score={score} P(move)={pmove} DR={dr}")
    print(f"  {rationale}")
    print()
