"""Add temporary diagnostic logging to watcher_pipeline.py scoring loop"""
import sys

filepath = '/home/ubuntu/titan/agentic_trader/watcher_pipeline.py'

with open(filepath, 'r') as f:
    content = f.read()

# Find the line: _dec = _scorer.score_intraday_signal(_sig, market_data=_d, ...
target = "                    _dec = _scorer.score_intraday_signal(_sig, market_data=_d, caller_direction=None, source='watcher', trigger_type=_sym_trigger_type)"

diag_line = """                    # TEMP DIAG: per-stock scoring input/output
                    _diag_rsi = _d.get('rsi_14', 0)
                    _diag_adx = _d.get('adx', 0)
                    _diag_htf = _d.get('htf_alignment', '?')
                    _diag_vwap = _d.get('price_vs_vwap', '?')
                    _diag_vol = _d.get('volume_regime', '?')
                    _diag_orb = _d.get('orb_signal', '?')
                    _diag_ema = _d.get('ema_regime', '?')
                    _diag_ltp = _d.get('ltp', 0)
                    _diag_id = id(_d)
                    print(f"    DIAG [{_sym}] INPUT: rsi={_diag_rsi:.1f} adx={_diag_adx:.1f} htf={_diag_htf} vwap={_diag_vwap} vol={_diag_vol} orb={_diag_orb} ema={_diag_ema} ltp={_diag_ltp} dict_id={_diag_id}")
                    print(f"    DIAG [{_sym}] OUTPUT: score={_dec.confidence_score} dir={_dec.recommended_direction} audit={getattr(_scorer, '_last_score_audit', '?')[:80]}")"""

if target not in content:
    print("ERROR: target line not found!")
    sys.exit(1)

content = content.replace(target, target + "\n" + diag_line)

with open(filepath, 'w') as f:
    f.write(content)

print("OK: diagnostic added successfully")
