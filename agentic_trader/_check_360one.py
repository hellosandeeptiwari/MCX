import sqlite3, json, sys, os

db = sqlite3.connect('/home/ubuntu/titan/agentic_trader/titan_state.db')

# Check active_trades for 360ONE 
rows = db.execute("SELECT trade_json FROM active_trades WHERE trade_json LIKE '%360ONE%' ORDER BY id DESC LIMIT 1").fetchall()
if rows:
    d = json.loads(rows[0][0])
    ml = d.get('ml_data', {})
    print('=== FROM active_trades ===')
    print('direction:', d.get('direction'))
    print('setup_type:', d.get('setup_type'))
    print('oi_signal:', ml.get('oi_signal', 'NOT_SET'))
    print('scored_direction:', ml.get('scored_direction', 'NOT_SET'))
    print('rationale:', d.get('rationale', '')[:200])
else:
    print('No 360ONE in active_trades')

# Check trades table
rows2 = db.execute("SELECT direction, source, strategy_type, rationale FROM trades WHERE symbol LIKE '%360ONE%' OR underlying LIKE '%360ONE%' ORDER BY id DESC LIMIT 1").fetchall()
if rows2:
    print('\n=== FROM trades ===')
    print('direction:', rows2[0][0])
    print('source:', rows2[0][1])
    print('strategy_type:', rows2[0][2])
    print('rationale:', (rows2[0][3] or '')[:200])
else:
    print('No 360ONE in trades')
