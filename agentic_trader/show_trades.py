"""Show active trades with scores and ML predictions."""
import json

with open('active_trades.json') as f:
    data = json.load(f)

trades = data.get('active_trades', [])
rpnl = data.get('realized_pnl', 0)
cap = data.get('paper_capital', 0)
print(f'Active trades: {len(trades)} | Realized P&L: {rpnl:,.0f} | Capital: {cap:,.0f}')
print('='*80)

for t in trades:
    sym = t.get('symbol', '?')
    score = t.get('score', '?')
    ml = t.get('ml_prediction', {})
    if not isinstance(ml, dict):
        ml = {}
    ml_sig = ml.get('ml_signal', 'N/A')
    ml_boost = ml.get('ml_score_boost', 'N/A')
    ml_pm = ml.get('ml_p_move', 'N/A')
    ml_pu = ml.get('ml_p_up_given_move', 'N/A')
    ml_pd = ml.get('ml_p_down_given_move', 'N/A')
    entry = t.get('entry_price', '?')
    side = t.get('side', '?')
    ts = t.get('entry_time', '?')
    sl = t.get('stop_loss', '?')
    tgt = t.get('target', '?')
    qty = t.get('quantity', '?')
    
    print(f'{sym}')
    print(f'  Side: {side} | Qty: {qty} | Entry: {entry} | SL: {sl} | Tgt: {tgt}')
    print(f'  Score: {score} | Time: {ts}')
    print(f'  ML: signal={ml_sig} boost={ml_boost} p_move={ml_pm} p_up={ml_pu} p_down={ml_pd}')
    print()

# Also check scan_decisions.json for latest scored stocks
try:
    with open('scan_decisions.json') as f:
        decisions = json.load(f)
    if isinstance(decisions, list) and decisions:
        latest = decisions[-1] if len(decisions) <= 100 else decisions[-1]
        if isinstance(latest, dict):
            picks = latest.get('picks', latest.get('top_picks', []))
            if picks:
                print('='*80)
                print('LATEST SCAN TOP PICKS:')
                for p in picks[:10]:
                    if isinstance(p, dict):
                        s = p.get('symbol', '?')
                        sc = p.get('score', '?')
                        ml_s = p.get('ml_signal', p.get('ml_prediction', {}).get('ml_signal', '?') if isinstance(p.get('ml_prediction'), dict) else '?')
                        print(f'  {s}: score={sc} ml={ml_s}')
except Exception as e:
    print(f'Could not read scan_decisions: {e}')
