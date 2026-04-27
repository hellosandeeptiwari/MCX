import sqlite3, json
dbp = '/home/ubuntu/titan/agentic_trader/titan_state.db'
con = sqlite3.connect(dbp); cur = con.cursor()
rs = cur.execute("SELECT id, trade_json FROM active_trades WHERE date='2026-04-23'").fetchall()
removed = 0
for rid, tj in rs:
    try:
        t = json.loads(tj)
        if (t.get('status') or 'OPEN') != 'OPEN':
            print('purging id=', rid, 'sym=', t.get('symbol'), 'status=', t.get('status'), 'reason=', t.get('exit_reason'))
            cur.execute("DELETE FROM active_trades WHERE id=?", (rid,))
            removed += 1
    except Exception as e:
        print('skip', rid, e)
con.commit(); con.close()
print('removed', removed, 'closed rows from today')
