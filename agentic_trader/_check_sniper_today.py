#!/usr/bin/env python3
"""Quick check: how many GMM_SNIPER trades placed today."""
import sys, json, datetime
sys.path.insert(0, "/home/ubuntu/titan/agentic_trader")

try:
    from state_db import get_state_db
    db = get_state_db()
    today = datetime.date.today().isoformat()
    trades, pnl, cap = db.load_active_trades(today)
    all_trades = trades
except Exception as e:
    print(f"DB load failed: {e}")
    # Fallback: trade_history.json
    try:
        with open("/home/ubuntu/titan/agentic_trader/trade_history.json") as f:
            all_trades = json.load(f)
    except:
        with open("/home/ubuntu/titan/agentic_trader/trade_log.json") as f:
            all_trades = json.load(f)

today_str = datetime.date.today().isoformat()
print(f"=== TODAY: {today_str} ===")
print(f"Total trades loaded: {len(all_trades)}")

snipers = [t for t in all_trades if t.get("setup_type") == "GMM_SNIPER"]
print(f"GMM_SNIPER trades: {len(snipers)}")
for t in snipers:
    et = str(t.get("entry_time", ""))[:19]
    print(f"  {t.get('underlying','?'):15s} {t.get('direction','?'):5s} {t.get('status','?'):8s} {et} P&L:{t.get('pnl',0)}")

# Also check trade_history for closed sniper trades
try:
    with open("/home/ubuntu/titan/agentic_trader/trade_history.json") as f:
        history = json.load(f)
    today_snipers = [t for t in history if t.get("setup_type") == "GMM_SNIPER" and today_str in str(t.get("entry_time", ""))]
    print(f"\n=== TRADE HISTORY: GMM_SNIPER from {today_str} ===")
    print(f"Count: {len(today_snipers)}")
    for t in today_snipers:
        et = str(t.get("entry_time", ""))[:19]
        xt = str(t.get("exit_time", ""))[:19]
        print(f"  {t.get('underlying','?'):15s} {t.get('direction','?'):5s} {t.get('status','?'):8s} entry={et} exit={xt} P&L:{t.get('pnl',0)}")
except Exception as e:
    print(f"History check: {e}")

# Check trade_log too
try:
    with open("/home/ubuntu/titan/agentic_trader/trade_log.json") as f:
        log = json.load(f)
    today_log_snipers = [t for t in log if t.get("setup_type") == "GMM_SNIPER" and today_str in str(t.get("entry_time", ""))]
    print(f"\n=== TRADE LOG: GMM_SNIPER from {today_str} ===")
    print(f"Count: {len(today_log_snipers)}")
    for t in today_log_snipers:
        et = str(t.get("entry_time", ""))[:19]
        print(f"  {t.get('underlying','?'):15s} {t.get('direction','?'):5s} {t.get('status','?'):8s} entry={et} P&L:{t.get('pnl',0)}")
except Exception as e:
    print(f"Log check: {e}")
