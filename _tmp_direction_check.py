import sqlite3, json
dbp = '/home/ubuntu/titan/agentic_trader/titan_state.db'
con = sqlite3.connect(dbp); cur = con.cursor()

# Get current live_pnl entries for today
rs = cur.execute("SELECT symbol, ltp, unrealized_pnl, last_updated FROM live_pnl WHERE date='2026-04-23'").fetchall()
print('live_pnl today (from bot):')
for r in rs:
    print(' ', r)

# Get today's OPEN active_trades positions with direction/side/entry/qty
rs2 = cur.execute("SELECT trade_json FROM active_trades WHERE date='2026-04-23'").fetchall()
print('\nOpen positions today:')
for (tj,) in rs2:
    t = json.loads(tj)
    if (t.get('status') or 'OPEN') != 'OPEN': continue
    s = t.get('symbol','')
    print(f"  {s:<45} side={t.get('side')} dir={t.get('direction')} entry={t.get('avg_price')} qty={t.get('quantity')}")
