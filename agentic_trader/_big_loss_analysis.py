import json

with open('trade_history.json') as f:
    trades = json.load(f)

big_losses = [t for t in trades if '2026-02-16' in str(t.get('timestamp','')) and (t.get('pnl',0) or 0) < -5000]

for t in big_losses:
    sym = t.get('underlying','?')
    print(f"=== {sym} ===")
    print(f"  Direction: {t.get('direction')}, Option: {t.get('option_type')}")
    print(f"  Score: {t.get('entry_score')}")
    print(f"  Entry: {t.get('avg_price')}, Exit: {t.get('exit_price')}, PnL: {t.get('pnl')}")
    meta = t.get('entry_metadata', {}) or {}
    rationale = t.get('rationale', '')
    print(f"  Rationale: {str(rationale)[:200]}")
    print(f"  Score Audit: {meta.get('score_audit','')}")
    print(f"  Trend: {meta.get('trend_state','')}")
    print(f"  Reasons:")
    for r in (meta.get('reasons', []) or []):
        print(f"    {r}")
    print(f"  Warnings:")
    for w in (meta.get('warnings', []) or []):
        print(f"    {w}")
    print()

# Also show TIME_STOP losses > 2000
print("=== TIME_STOP LOSSES > 2000 ===")
time_stops = [t for t in trades if '2026-02-16' in str(t.get('timestamp','')) 
              and t.get('result') == 'TIME_STOP' and (t.get('pnl',0) or 0) < -2000]
for t in time_stops:
    sym = t.get('underlying','?')
    meta = t.get('entry_metadata', {}) or {}
    print(f"{sym} | Score:{t.get('entry_score')} | PnL:{t.get('pnl')} | Trend:{meta.get('trend_state','')}")
    print(f"  Audit: {meta.get('score_audit','')}")
    print()
