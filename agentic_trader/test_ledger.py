"""Quick test for TradeLedger"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from trade_ledger import get_trade_ledger

ledger = get_trade_ledger()

# Remove test file if exists
test_file = ledger._today_file()
if os.path.exists(test_file):
    os.remove(test_file)

# Test log_entry
ledger.log_entry(
    symbol='NFO:ABB26FEB5950CE', underlying='NSE:ABB', direction='BUY',
    source='GMM_BOOST', smart_score=65.6, pre_score=67.5, final_score=67.5,
    dr_score=0.000, gate_prob=0.65, gmm_action='BOOST', sector='INFRA',
    strategy_type='GMM_BOOST', entry_price=113.0, quantity=500, lots=4,
    stop_loss=81.36, target=180.8, total_premium=56500,
    order_id='PAPER_100001'
)

# Test log_exit (partial)
ledger.log_exit(
    symbol='NFO:ABB26FEB5950CE', underlying='NSE:ABB', direction='BUY',
    source='GMM_BOOST', exit_type='PARTIAL_PROFIT', entry_price=113.0,
    exit_price=137.4, quantity=250, pnl=6100, pnl_pct=21.6,
    smart_score=65.6, final_score=67.5, dr_score=0.000,
    candles_held=3, r_multiple=0.771, exit_reason='Partial 50%',
    entry_time='2026-02-21T09:18:07',
    order_id='PAPER_100001'
)

# Test log_exit (remainder)
ledger.log_exit(
    symbol='NFO:ABB26FEB5950CE', underlying='NSE:ABB', direction='BUY',
    source='GMM_BOOST', exit_type='TARGET_HIT', entry_price=113.0,
    exit_price=180.8, quantity=250, pnl=16950, pnl_pct=60.0,
    smart_score=65.6, final_score=67.5, dr_score=0.000,
    candles_held=8, r_multiple=2.14, exit_reason='Target hit',
    entry_time='2026-02-21T09:18:07',
    order_id='PAPER_100001'
)

# Test second trade entry + exit
ledger.log_entry(
    symbol='NFO:GAIL26FEB169CE', underlying='NSE:GAIL', direction='BUY',
    source='GMM_BOOST', smart_score=58.3, pre_score=54.8, final_score=54.8,
    dr_score=0.000, gate_prob=0.61, gmm_action='BOOST', sector='ENERGY',
    strategy_type='GMM_BOOST', entry_price=8.5, quantity=1000,
    order_id='PAPER_100002'
)

ledger.log_exit(
    symbol='NFO:GAIL26FEB169CE', underlying='NSE:GAIL', direction='BUY',
    source='GMM_BOOST', exit_type='STOPLOSS_HIT', entry_price=8.5,
    exit_price=3.1, quantity=1000, pnl=-5400,
    smart_score=58.3, final_score=54.8, dr_score=0.000,
    entry_time='2026-02-21T10:56:27',
    order_id='PAPER_100002'
)

# Read back
events = ledger.read_day()
print(f"Events logged: {len(events)}")
for e in events:
    ev = e['event']
    sym = e.get('underlying', '')
    d = e.get('direction', '')
    src = e.get('source', '')
    if ev == 'EXIT':
        print(f"  {ev}: {sym} {d} source={src} pnl=Rs {e['pnl']:+,.0f} {e['exit_type']}")
    else:
        print(f"  {ev}: {sym} {d} source={src} score={e.get('final_score',0)} dr={e.get('dr_score',0)}")

# Test get_trades_with_pnl - this should MERGE partial exits
trades = ledger.get_trades_with_pnl()
print(f"\nMerged trades: {len(trades)}")
for t in trades:
    sym = t.get('underlying', '').replace('NSE:', '')
    pnl = t.get('total_pnl', 0)
    exits = t.get('exit_count', 0)
    result = t.get('final_result', '?')
    print(f"  {sym}: PnL=Rs {pnl:+,.0f} ({exits} exit legs) -> {result}")

# Summary
summary = ledger.daily_summary()
print(f"\n=== SUMMARY ===")
print(f"Trades: {summary['total_trades']} | W:{summary['wins']} L:{summary['losses']} | WR: {summary['win_rate']}%")
print(f"PnL: Rs {summary['total_pnl']:+,.0f}")

# Cleanup test data
os.remove(test_file)
print("\nALL TESTS PASSED")
