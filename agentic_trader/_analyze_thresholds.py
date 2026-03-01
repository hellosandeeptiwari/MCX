"""Threshold analysis script - one-time use"""
import json

r = json.load(open('ml_models/reports/down_risk_report_20260226_213910.json'))

up = r['UP_metrics']
print('=== UP REGIME MODEL ===')
print(f'  Threshold: {up["threshold"]:.4f}')
print(f'  Val AUROC: {up["val_auroc"]}')
print(f'  Val lift: {up["val_lift"]}')
print(f'  Val flag_rate: {up["val_flag_rate"]:.1%}')
print(f'  Val down_if_flagged: {up["val_down_if_flagged"]:.1%}')
print(f'  Val down_if_clean: {up["val_down_if_not_flagged"]:.1%}')
print(f'  Val decile: {[round(x,3) for x in up["val_decile_down_rates"]]}')
print(f'  Test AUROC: {r["test"]["UP"]["test_up_auroc"]}')
print(f'  Test lift: {r["test"]["UP"]["test_up_lift"]}')
print(f'  Test decile: {[round(x,3) for x in r["test"]["UP"]["test_up_decile_down_rates"]]}')

dn = r['DOWN_metrics']
print()
print('=== DOWN REGIME MODEL ===')
print(f'  Threshold: {dn["threshold"]:.4f}')
print(f'  Val AUROC: {dn["val_auroc"]}')
print(f'  Val lift: {dn["val_lift"]}')
print(f'  Val flag_rate: {dn["val_flag_rate"]:.1%}')
print(f'  Val down_if_flagged: {dn["val_down_if_flagged"]:.1%}')
print(f'  Val down_if_clean: {dn["val_down_if_not_flagged"]:.1%}')
print(f'  Val decile: {[round(x,3) for x in dn["val_decile_down_rates"]]}')
print(f'  Test AUROC: {r["test"]["DOWN"]["test_down_auroc"]}')
print(f'  Test lift: {r["test"]["DOWN"]["test_down_lift"]}')
print(f'  Test decile: {[round(x,3) for x in r["test"]["DOWN"]["test_down_decile_down_rates"]]}')

print()

# Analysis
up_test = r['test']['UP']['test_up_decile_down_rates']
dn_test = r['test']['DOWN']['test_down_decile_down_rates']

print('=== WHERE MODEL SIGNAL ACTUALLY EXISTS ===')
print('UP model (10 deciles, low→high score):')
for i, rate in enumerate(up_test):
    bar = '#' * int(rate * 100)
    useful = ' *** USEFUL' if i >= 8 and rate > 0.20 else ''
    print(f'  D{i+1:2d}: {rate:.3f} {bar}{useful}')

print()
print('DN model (10 deciles, low→high score):')
for i, rate in enumerate(dn_test):
    bar = '#' * int(rate * 100)
    useful = ' *** USEFUL' if i >= 6 and rate > 0.20 else ''
    print(f'  D{i+1:2d}: {rate:.3f} {bar}{useful}')

print()
print('=== SONACOMS TEST_GMM PUT TRADE ANALYSIS ===')
print('  Stock: SONACOMS, BUY PUT (bearish)')
print('  UP score: 0.0736 → Bottom 25% (NO anomaly)')
print('  DN score: 0.1431 → ~50th percentile (NO anomaly)')
print('  XGB: UP (67% move prob, BULLISH)')
print('  smart_score: 0 (bypassed)')
print()
print('  PROBLEM: Both GMM regimes show NO anomaly.')
print('  GMM gives ZERO directional signal for this stock.')
print('  XGB strongly says UP. We went PUT.')
print('  This trade has NO MODEL EDGE.')
print()

print('=== VERDICT ON EACH THRESHOLD CHANGE ===')
changes = [
    ('max_days_to_expiry 21→35', 'GOOD', 'Structural fix - monthly options need 35 DTE window'),
    ('Credit spread score 50→65', 'GOOD', 'Higher quality bar for theta trades'),
    ('Debit spread score 57→65', 'GOOD', 'Higher quality bar for momentum trades'),
    ('min_confirm_score 0.25→0.12', 'GOOD', 'Bug fix - was above flag thresholds, blocking all flags'),
    ('PCR overbought 0.65→0.55', 'RISKY', 'Very tight - PCR 0.55 filters aggressively, may kill PCR strategy'),
    ('OI_UNWINDING 7 params relaxed', 'RISKY', '7 filters relaxed simultaneously compounds error rate'),
    ('DOWN_RISK min_smart 60→55', 'OK', 'Moderate relaxation, acceptable'),
    ('GMM_CONTRARIAN dr 0.30→0.20', 'RISKY', f'UP AUROC={r["test"]["UP"]["test_up_auroc"]} (near random!), signal only in top 2 deciles'),
    ('GMM_CONTRARIAN gate 0.70→0.55', 'RISKY', 'At 0.55 gate, ~50% of stocks pass - too loose for contrarian'),
    ('ML_OVERRIDE move_prob 0.58→0.52', 'BAD', 'P(MOVE)=0.52 is coin flip, overriding direction on this is noise'),
    ('ML_OVERRIDE smart 58→55', 'OK', 'Moderate relaxation'),
    ('TEST_GMM asymmetric UP<0.20/DN<0.08 CALL, UP<0.08/DN<0.20 PUT', 'BAD',
     'Direction from clean regime is backwards. Low DR = no signal, not directional confirmation. SONACOMS proves this.'),
]

for name, verdict, reason in changes:
    icon = {'GOOD': '✅', 'OK': '⚠️', 'RISKY': '🔶', 'BAD': '❌'}[verdict]
    print(f'  {icon} {name}')
    print(f'     {reason}')
    print()
