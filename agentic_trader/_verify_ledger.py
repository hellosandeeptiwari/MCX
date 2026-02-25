"""Quick verify that centralized ledger logging works."""
from trade_ledger import get_trade_ledger

ledger = get_trade_ledger()
print(f"Ledger OK: log_entry={hasattr(ledger, 'log_entry')}, log_exit={hasattr(ledger, 'log_exit')}")

import zerodha_tools
has_entry = hasattr(zerodha_tools.ZerodhaTools, '_log_entry_to_ledger')
has_exit = hasattr(zerodha_tools.ZerodhaTools, '_save_to_history')
print(f"zerodha_tools OK: _log_entry_to_ledger={has_entry}, _save_to_history={has_exit}")

if has_entry and has_exit:
    print("\n✅ CENTRALIZED: Both entry and exit logging are in zerodha_tools.py")
else:
    print("\n❌ MISSING methods!")
