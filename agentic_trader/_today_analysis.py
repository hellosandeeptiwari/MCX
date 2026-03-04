#!/usr/bin/env python3
"""Quick analysis of today's trades for dashboard review."""
import sys, json, os
sys.path.insert(0, os.path.dirname(__file__))

from state_db import get_state_db
from trade_ledger import get_trade_ledger
from datetime import date, datetime

db = get_state_db()
today = date.today().isoformat()
positions, realized_pnl, capital = db.load_active_trades(today)
live = db.load_live_pnl() or {}

print(f"{'='*60}")
print(f"  TITAN TRADE ANALYSIS — {today}")
print(f"{'='*60}")
print(f"  Capital: ₹{capital:,.2f}")
print(f"  Realized P&L: ₹{realized_pnl:+,.2f}")
print(f"  Open Positions: {len(positions)}")
print()

# ── Open Positions Detail ──
total_unreal = 0
winners = 0
losers = 0
print(f"{'─'*60}")
print("  OPEN POSITIONS")
print(f"{'─'*60}")
for i, p in enumerate(positions, 1):
    sym = p.get('symbol', '')
    entry = p.get('avg_price') or p.get('entry_price') or p.get('net_premium', 0)
    qty = abs(p.get('quantity', 0))
    d = p.get('direction', '')
    setup = p.get('setup_type', p.get('strategy_type', ''))
    score = p.get('smart_score', p.get('entry_score', 0))
    sl = p.get('stop_loss', 0)
    tgt = p.get('target', 0)
    ts = p.get('timestamp', '')
    spread = p.get('is_debit_spread') or p.get('is_credit_spread') or p.get('is_iron_condor')

    lp = live.get(sym) or live.get(sym.replace('NFO:', ''))
    ltp = 0
    unreal = 0
    if isinstance(lp, dict):
        ltp = lp.get('ltp', 0)
        unreal = lp.get('unrealized_pnl', 0)
    elif isinstance(lp, (int, float)):
        ltp = float(lp)

    if unreal == 0 and ltp > 0 and entry > 0 and not spread:
        if d in ('BUY', 'LONG'):
            unreal = (ltp - entry) * qty
        else:
            unreal = (entry - ltp) * qty

    total_unreal += unreal
    if unreal >= 0:
        winners += 1
    else:
        losers += 1

    pnl_pct = (unreal / (entry * qty) * 100) if entry > 0 and qty > 0 else 0
    status_icon = '🟢' if unreal >= 0 else '🔴'

    # Time in trade
    hold_str = ''
    try:
        if ts:
            entry_dt = datetime.fromisoformat(ts)
            hold_mins = int((datetime.now() - entry_dt).total_seconds() / 60)
            hold_str = f"{hold_mins}m" if hold_mins < 60 else f"{hold_mins//60}h{hold_mins%60}m"
    except:
        pass

    print(f"\n  {i}. {status_icon} {sym}")
    print(f"     {d} x {qty} @ ₹{entry:.2f} → LTP ₹{ltp:.2f}")
    print(f"     P&L: ₹{unreal:+,.2f} ({pnl_pct:+.1f}%)")
    print(f"     SL: {sl} | Target: {tgt} | Hold: {hold_str}")
    print(f"     Setup: {setup} | Score: {score}")
    if spread:
        net_prem = p.get('net_premium', 0)
        print(f"     [SPREAD] Net Premium: ₹{net_prem:.2f}")

# ── Trade Ledger Summary ──
print(f"\n{'─'*60}")
print("  TRADE LEDGER")
print(f"{'─'*60}")
try:
    ledger = get_trade_ledger()
    summary = ledger.daily_summary(today)
    print(f"  Total Trades: {summary.get('total_trades', 0)}")
    print(f"  Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}")
    print(f"  Win Rate: {summary.get('win_rate', 0):.0f}%")
    print(f"  Ledger Net P&L: ₹{summary.get('net_pnl', 0):+,.2f}")
    print(f"  Avg Win: ₹{summary.get('avg_win', 0):+,.2f} | Avg Loss: ₹{summary.get('avg_loss', 0):+,.2f}")
    
    # Count EXIT events
    events = ledger.get_today_events(today) if hasattr(ledger, 'get_today_events') else []
    entries = exits = 0
    for ev in events:
        if isinstance(ev, dict):
            if ev.get('event') == 'ENTRY':
                entries += 1
            elif ev.get('event') == 'EXIT':
                exits += 1
    if events:
        print(f"  Entry Events: {entries} | Exit Events: {exits}")
except Exception as e:
    print(f"  Ledger error: {e}")

# ── Risk State ──
print(f"\n{'─'*60}")
print("  RISK STATE")
print(f"{'─'*60}")
try:
    risk = db.load_risk_state(today) or {}
    print(f"  Daily Loss: {risk.get('daily_loss_pct', 0):.1f}%")
    print(f"  Circuit Breaker: {'ACTIVE' if risk.get('circuit_breaker') else 'OK'}")
    print(f"  Positions Used: {len(positions)} / {risk.get('max_positions', 80)}")
except Exception as e:
    print(f"  Risk error: {e}")

# ── Summary ──
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
net_pnl = realized_pnl + total_unreal
print(f"  Realized:    ₹{realized_pnl:+,.2f}")
print(f"  Unrealized:  ₹{total_unreal:+,.2f}  ({winners}W / {losers}L)")
print(f"  Net P&L:     ₹{net_pnl:+,.2f}")
pnl_pct_total = (net_pnl / capital * 100) if capital > 0 else 0
print(f"  Return:      {pnl_pct_total:+.2f}% on capital")
print(f"{'='*60}")
