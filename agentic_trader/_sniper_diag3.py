import json

# Check today's placed trades for their DR scores
decisions = json.load(open('scan_decisions.json'))
today = [d for d in decisions if '2026-02-20' in d.get('timestamp', '')]

# GMM_BOOST trades placed today — shows DR distribution of stocks that passed model-tracker
boost_placed = [d for d in today if d.get('outcome') == 'GMM_BOOST_PLACED']
print(f"GMM_BOOST placed today: {len(boost_placed)}")

# ML_DIRECTION_BLOCK — these are stocks where XGB disagrees
ml_blocks = [d for d in today if d.get('outcome') == 'ML_DIRECTION_BLOCK']
print(f"ML Direction blocks: {len(ml_blocks)}")
for d in ml_blocks:
    print(f"  {d.get('symbol','?')} reason={d.get('reason','?')}")

# Sniper placed
sniper_placed = [d for d in today if d.get('outcome') == 'GMM_SNIPER_PLACED']
print(f"\nGMM_SNIPER placed today: {len(sniper_placed)}")
for d in sniper_placed:
    print(f"  {d.get('symbol','?')} reason={d.get('reason','?')}")

# Check the active_trades for ML data on current positions
# These have actual dr_scores we can see
active = json.load(open('active_trades.json'))
trades = active.get('active_trades', [])
print(f"\nActive trades ML data:")
for t in trades:
    sym = t.get('underlying', '?').replace('NSE:', '')
    ml_data = t.get('ml_data', {})
    if isinstance(ml_data, dict):
        dr = ml_data.get('dr_score', '?')
        gmm = ml_data.get('gmm_model', {})
        dr2 = gmm.get('down_risk_score', '?') if isinstance(gmm, dict) else '?'
        gate = ml_data.get('ml_move_prob', '?')
        smart = ml_data.get('smart_score', '?')
        xgb = ml_data.get('xgb_model', {})
        signal = xgb.get('signal', '?') if isinstance(xgb, dict) else '?'
        print(f"  {sym:15} dr={dr} gate={gate} smart={smart} xgb={signal}")
    else:
        print(f"  {sym:15} no ml_data")

# Check trade_history for DR scores of sniper trades
history = json.load(open('trade_history.json'))
print(f"\nSniper trades DR scores:")
snipers = [t for t in history if t.get('setup_type') == 'GMM_SNIPER' or t.get('is_sniper')]
for t in snipers:
    sym = t.get('underlying', '?').replace('NSE:', '')
    ml_data = t.get('ml_data', {})
    if isinstance(ml_data, dict):
        dr = ml_data.get('dr_score', '?')
        gate = ml_data.get('ml_move_prob', '?')
        smart = ml_data.get('smart_score', '?')
        print(f"  {sym:15} dr={dr} gate={gate} smart={smart}")

# Also check how many unique symbols were evaluated in the latest full cycle
latest_cycle = [d for d in today if d.get('cycle') == today[-1].get('cycle', '')]
print(f"\nLatest cycle ({today[-1].get('cycle', '?')}): {len(latest_cycle)} stocks evaluated")
