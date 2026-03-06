"""Quick ORB_BREAKOUT trade analysis"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from trade_ledger import get_trade_ledger
from datetime import datetime

tl = get_trade_ledger()
today = datetime.now().strftime('%Y-%m-%d')
exits = tl.get_exits(today)
entries = tl.get_entries(today)

print("=== ORB_BREAKOUT EXITS ===")
orb_total = 0
for e in exits:
    src = e.get('source', '')
    if 'ORB' in src:
        sym = e.get('symbol', '').replace('NFO:', '')[:30]
        pnl = e.get('pnl', 0) or 0
        ext = e.get('exit_type', '')[:25]
        score = e.get('smart_score', 0) or 0
        entry = e.get('entry_price', 0) or 0
        exitp = e.get('exit_price', 0) or 0
        reason = str(e.get('exit_reason', ''))[:60]
        orb_total += pnl
        marker = '+' if pnl > 0 else '-'
        print(f"  {marker} {sym:30s} Score:{score:4.0f}  Entry:{entry:7.1f}  Exit:{exitp:7.1f}  PnL: Rs {pnl:+,.0f}")
        print(f"    Exit: {ext} | {reason}")

print(f"\nORB_BREAKOUT TOTAL: Rs {orb_total:+,.0f}")

print("\n=== ORB_BREAKOUT ENTRIES ===")
for e in entries:
    src = e.get('source', '')
    if 'ORB' in src:
        sym = e.get('symbol', '').replace('NFO:', '')[:30]
        score = e.get('smart_score', 0) or 0
        pre = e.get('pre_score', 0) or 0
        entry = e.get('entry_price', 0) or 0
        dr = e.get('dr_score', 0) or 0
        ml_dir = e.get('ml_direction', '')
        move = e.get('ml_move_prob', 0) or 0
        strat = e.get('strategy_type', '')
        print(f"  {sym:30s} SmartScore:{score:4.0f} PreScore:{pre:4.0f} DR:{dr:.3f} ML:{ml_dir} P(move):{move:.2f}")
