import json
from state_db import get_state_db
db = get_state_db()
trades, _, _ = db.load_active_trades()
for t in trades:
    if 'CGPOWER' in t.get('symbol', ''):
        print(json.dumps(t, indent=2, default=str))
