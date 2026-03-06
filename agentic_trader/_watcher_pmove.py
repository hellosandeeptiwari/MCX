"""P(move) analysis for WATCHER and ORB trades"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from trade_ledger import get_trade_ledger
from datetime import datetime

tl = get_trade_ledger()
today = datetime.now().strftime('%Y-%m-%d')
entries = tl.get_entries(today)
exits = tl.get_exits(today)

# Build exit map — match by order_id first (handles debit spreads), fall back to symbol
exit_by_oid = {}
exit_by_sym = {}
for e in exits:
    oid = e.get('order_id', '')
    sym = e.get('symbol', '')
    if oid:
        exit_by_oid[oid] = e
    exit_by_sym[sym] = e

print("=" * 100)
print("WATCHER + ORB TRADES — P(move) Analysis")
print("=" * 100)
print(f"{'Symbol':30s} {'Source':20s} {'Score':>5s} {'P(move)':>8s} {'ML Dir':>7s} {'PnL':>10s} {'Exit Type':>25s}")
print("-" * 100)

watcher_wins = []
watcher_losses = []

for en in entries:
    src = en.get('source', '')
    if src not in ('WATCHER', 'ORB_BREAKOUT'):
        continue
    sym = en.get('symbol', '')
    score = en.get('smart_score', 0) or en.get('pre_score', 0) or 0
    move_prob = en.get('ml_move_prob', 0) or 0
    ml_dir = en.get('ml_direction', '') or ''
    
    oid = en.get('order_id', '')
    ex = exit_by_oid.get(oid) or exit_by_sym.get(sym, {})
    pnl = ex.get('pnl', 0) or 0
    exit_type = ex.get('exit_type', 'OPEN')[:25]
    
    marker = '✅' if pnl > 0 else '❌' if pnl < 0 else '⏳'
    short_sym = sym.replace('NFO:', '')[:28]
    print(f"{marker} {short_sym:28s} {src:20s} {score:5.0f} {move_prob:8.2f} {ml_dir:>7s} Rs{pnl:+9,.0f} {exit_type:>25s}")
    
    if pnl > 0:
        watcher_wins.append({'sym': sym, 'score': score, 'move': move_prob, 'pnl': pnl})
    elif pnl < 0:
        watcher_losses.append({'sym': sym, 'score': score, 'move': move_prob, 'pnl': pnl})

print("-" * 100)
if watcher_wins:
    avg_win_move = sum(w['move'] for w in watcher_wins) / len(watcher_wins)
    avg_win_score = sum(w['score'] for w in watcher_wins) / len(watcher_wins)
    total_win = sum(w['pnl'] for w in watcher_wins)
    print(f"WINNERS ({len(watcher_wins)}): Avg P(move)={avg_win_move:.2f}  Avg Score={avg_win_score:.0f}  Total=Rs{total_win:+,.0f}")
if watcher_losses:
    avg_loss_move = sum(w['move'] for w in watcher_losses) / len(watcher_losses)
    avg_loss_score = sum(w['score'] for w in watcher_losses) / len(watcher_losses)
    total_loss = sum(w['pnl'] for w in watcher_losses)
    print(f"LOSERS  ({len(watcher_losses)}): Avg P(move)={avg_loss_move:.2f}  Avg Score={avg_loss_score:.0f}  Total=Rs{total_loss:+,.0f}")
