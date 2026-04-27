"""Diagnose OI_WATCHER path 1 — why nse_oi_buildup_strength is always < 0.5"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options_flow_analyzer import OptionsFlowAnalyzer

oi = OptionsFlowAnalyzer()

# Stocks that had LONG_BUILDUP / SHORT_BUILDUP in Layer 1 today
test_syms = [
    'NSE:HINDUNILVR', 'NSE:BSE', 'NSE:SIEMENS', 'NSE:BRITANNIA',
    'NSE:RVNL', 'NSE:COLPAL', 'NSE:VBL', 'NSE:MARUTI',
    'NSE:PERSISTENT', 'NSE:DABUR', 'NSE:FORTIS',
]

print(f"{'Symbol':<20} {'Signal':<20} {'Strength':>10} {'FlowConf':>10} {'PCR':>8} {'Participant':<18}")
print("-" * 90)
for sym in test_syms:
    try:
        res = oi.analyze(sym)
        if not res:
            print(f"{sym:<20} {'NO DATA':<20}")
            continue
        sig = res.get('nse_oi_signal', res.get('oi_signal', 'UNKNOWN'))
        strength = res.get('nse_oi_buildup_strength', 0.0)
        flow_conf = res.get('flow_confidence', 0.0)
        pcr = res.get('pcr_oi', 0.0)
        part = res.get('oi_participant_id', 'UNKNOWN')
        
        # Also check all keys with 'strength' or 'buildup' in name
        extra_keys = {k: v for k, v in res.items() if 'strength' in k.lower() or 'buildup' in k.lower()}
        
        print(f"{sym:<20} {sig:<20} {strength:>10.4f} {flow_conf:>10.4f} {pcr:>8.3f} {part:<18} {extra_keys}")
    except Exception as e:
        print(f"{sym:<20} ERROR: {e}")
