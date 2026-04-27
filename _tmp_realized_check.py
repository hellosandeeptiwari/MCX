import sqlite3, json
dbp = '/home/ubuntu/titan/agentic_trader/titan_state.db'
con = sqlite3.connect(dbp); cur = con.cursor()
# Current daily_state for today
r = cur.execute("SELECT date, realized_pnl, paper_capital, last_updated FROM daily_state WHERE date='2026-04-23'").fetchone()
print('daily_state today:', r)

# Sum all closed trades today from 'trades' table for comparison
rs = cur.execute("SELECT symbol, direction, entry_price, exit_price, quantity, pnl, exit_type, exit_reason FROM trades WHERE date='2026-04-23'").fetchall()
total = 0
reverses = 0
print('\nClosed trades today:')
for s, d, ep, xp, q, pnl, et, er in rs:
    print(f'  {s:<40} dir={d:<5} qty={q:<6} entry={ep:<7} exit={xp:<7} pnl={pnl:<10} exit={et} reason={er}')
    total += (pnl or 0)
    if et and 'REVERSE' in (et or ''):
        reverses += 1
print(f'\nSum of pnl in trades table: {total:+,.2f}')
print(f'Reverse exits count: {reverses}')
print(f'\nDelta (db.realized_pnl - sum(trades.pnl)): {(r[1] if r else 0) - total:+,.2f}')

# Also look at trade_log.json for MANUAL_REVERSE_EXIT events
import os
tlog = '/home/ubuntu/titan/agentic_trader/trade_log.json'
if os.path.exists(tlog):
    try:
        with open(tlog) as f:
            events = json.load(f)
        today_manual = [e for e in events if isinstance(e, dict) and e.get('event')=='EXIT' and e.get('exit_type','').startswith('MANUAL') and (e.get('ts') or '').startswith('2026-04-23')]
        print(f'\nMANUAL exits in trade_log today: {len(today_manual)}')
        for e in today_manual[-5:]:
            print(f'  {e.get("ts")[:19]} {e.get("symbol")} pnl={e.get("pnl")} type={e.get("exit_type")}')
    except Exception as e:
        print('trade_log err', e)
con.close()
