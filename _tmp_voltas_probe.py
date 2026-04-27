import sqlite3, json
dbp = '/home/ubuntu/titan/agentic_trader/titan_state.db'
con = sqlite3.connect(dbp); cur = con.cursor()
rs = cur.execute("SELECT date, trade_json FROM active_trades WHERE date='2026-04-23'").fetchall()
print('rows today:', len(rs))
for date, tj in rs:
    try:
        t = json.loads(tj)
        s = (t.get('symbol') or t.get('option_symbol') or '')
        print('sym=', s, 'status=', t.get('status'), 'side=', t.get('side'), 'dir=', t.get('direction'))
        if 'VOLTAS' in s:
            print('==== VOLTAS FULL ====')
            print(json.dumps(t, indent=2, default=str))
    except Exception as e:
        print('parse err', e)



