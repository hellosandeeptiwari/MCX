"""Find big losers from yesterday that should have been hedged."""
import json

lines = open('trade_ledger/trade_ledger_2026-02-24.jsonl').readlines()
exits = [json.loads(l) for l in lines if json.loads(l).get('event') == 'EXIT']

print("=== TRADES WITH >6% LOSS (should have triggered hedge) ===\n")
for e in exits:
    loss_pct = e.get('pnl_pct', 0)
    if loss_pct < -6:
        underlying = e.get('underlying', '?')
        pnl = e.get('pnl', 0)
        entry = e.get('entry_price', 0)
        exit_p = e.get('exit_price', 0)
        exit_type = e.get('exit_type', '?')
        strategy = e.get('strategy_type', '?')
        source = e.get('source', '?')
        hold = e.get('hold_minutes', '?')
        symbol = e.get('symbol', '?')
        print(f"  {underlying:20s} | PnL: {pnl:>+10,.2f} ({loss_pct:+.1f}%)")
        print(f"    Symbol: {symbol}")
        print(f"    Entry: {entry:.2f} -> Exit: {exit_p:.2f} | Hold: {hold}min")
        print(f"    Exit: {exit_type} | Strategy: {strategy} | Source: {source}")
        print()

# Also check entries for expiry field
print("\n=== ENTRY RECORDS (check expiry field) ===\n")
entries = [json.loads(l) for l in lines if json.loads(l).get('event') == 'ENTRY']
for e in entries:
    print(f"  {e.get('underlying','?'):20s} expiry='{e.get('expiry','')}' strike={e.get('strike',0)} option_type='{e.get('option_type','')}' symbol={e.get('symbol','')}")
