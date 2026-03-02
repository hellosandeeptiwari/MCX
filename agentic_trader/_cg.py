import sys, json
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from state_db import get_state_db
db = get_state_db()
positions = db.load_positions()
for p in positions:
    if 'CGPOWER' in p.get('symbol',''):
        keys = ['symbol','setup_type','strategy_type','side','entry_price','qty',
                'sl_price','target_price','smart_score','dr_score',
                'ml_scored_direction','gate_prob','rationale','entry_time',
                'gmm_model','xgb_model','is_sniper','score_tier','entry_score']
        print(json.dumps({k: p.get(k) for k in keys}, indent=2, default=str))
