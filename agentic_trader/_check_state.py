"""Show current system state from SQLite DB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from state_db import get_state_db
from datetime import date

db = get_state_db()
today = str(date.today())

print("=== ACTIVE TRADES ===")
positions, pnl, cap = db.load_active_trades(today)
print(f"Total: {len(positions)} | Capital: ₹{cap:,.0f} | Realized PnL: ₹{pnl:+,.0f}")
for t in positions:
    sym = t.get("symbol", "?")
    st = t.get("status", "?")
    side = t.get("side", "?")
    qty = t.get("quantity", 0)
    spread = " [SPREAD]" if t.get("is_credit_spread") else ""
    ic = " [IC]" if t.get("is_iron_condor") else ""
    debit = " [DEBIT]" if t.get("is_debit_spread") else ""
    print(f"  {sym:35s} status={st:8s} side={side:5s} qty={qty}{spread}{ic}{debit}")
if not positions:
    print("  (no positions)")

print("\n=== RISK STATE ===")
r = db.load_risk_state(today)
if r:
    print(f"  State: {r.get('system_state')}")
    print(f"  Daily P&L: ₹{r.get('daily_pnl', 0):+,.0f}")
    print(f"  Trades today: {r.get('trades_today', 0)}")
else:
    print("  (no risk state for today — fresh day)")

print("\n=== EXIT MANAGER STATE ===")
states = db.load_exit_states(today)
print(f"  Tracked symbols: {len(states)}")
for sym, s in states.items():
    print(f"  {sym:35s} sl={s.get('current_sl','?')} trail={s.get('trailing_active', False)} be={s.get('breakeven_applied', False)}")
if not states:
    print("  (no exit states)")
