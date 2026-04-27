#!/usr/bin/env python3
"""Apply all 4 changes to autonomous_trader.py on remote server.
Run: python3 /tmp/apply_4_changes.py
"""
import re

FILE = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'

with open(FILE, 'r') as f:
    content = f.read()

original_len = len(content)
changes_made = 0

# ============================================================
# CHANGE 1: ARBTR Speed Gate — wrap in config toggle
# ============================================================
OLD_1 = """            try:
                from config import ARBTR_CONFIG as _arb_cfg
                _arb_speed_min = _arb_cfg.get('speed_gate_minutes', 15)
                _arb_speed_gain = _arb_cfg.get('speed_gate_min_gain_pct', 3.0)"""

# We need to find this block and:
# 1. Add the if/else after the import
# 2. Indent everything after the else by 4 extra spaces

# Find the full ARBTR speed gate try block
arbtr_pattern = r"(            try:\n                from config import ARBTR_CONFIG as _arb_cfg\n)(                _arb_speed_min = _arb_cfg\.get\('speed_gate_minutes', 15\)\n                _arb_speed_gain = _arb_cfg\.get\('speed_gate_min_gain_pct', 3\.0\).*?\n)(.*?)(            except Exception as _arb_speed_err:)"

match = re.search(arbtr_pattern, content, re.DOTALL)
if match:
    try_and_import = match.group(1)
    speed_vars = match.group(2)
    body_between = match.group(3)
    except_line = match.group(4)
    
    # The body between speed_vars and except is the rest of the speed gate logic
    # We need to indent speed_vars + body_between by 4 spaces (adding inside else block)
    lines_to_indent = (speed_vars + body_between).split('\n')
    indented_lines = []
    for line in lines_to_indent:
        if line.strip():  # non-empty lines get 4 extra spaces
            indented_lines.append('    ' + line)
        else:
            indented_lines.append(line)
    
    new_block = (
        try_and_import +
        "                if not _arb_cfg.get('speed_gate_enabled', True):\n"
        "                    pass  # Speed gate disabled — ARBTR runs to natural exit\n"
        "                else:\n" +
        '\n'.join(indented_lines) +
        except_line
    )
    
    content = content[:match.start()] + new_block + content[match.end():]
    changes_made += 1
    print(f"✅ CHANGE 1: ARBTR speed gate wrapped in config toggle")
else:
    print(f"❌ CHANGE 1 FAILED: Could not find ARBTR speed gate pattern")

# ============================================================
# CHANGE 2: B2 threshold relaxation for EARLYBIRD
# ============================================================
OLD_2 = "                if _b2_confirms < 2:"
NEW_2 = """                # Apr 2: EARLYBIRD uses relaxed threshold (1/4) — technicals unreliable at open
                _b2_min_needed = 1 if _is_earlybird else 2
                if _b2_confirms < _b2_min_needed:"""

if OLD_2 in content:
    content = content.replace(OLD_2, NEW_2, 1)
    changes_made += 1
    print(f"✅ CHANGE 2: B2 threshold relaxed for EARLYBIRD (1/4 vs 2/4)")
else:
    print(f"❌ CHANGE 2 FAILED: Could not find B2 threshold line")

# ============================================================
# CHANGE 3: Grind OI Anchor gate (B2-OI-ANCHOR) — insert after B2-CONVICTION
# ============================================================
OI_ANCHOR_MARKER = "                self._wlog(f\"  PASSED(B2-CONVICTION): {_stock_name} {_b2_confirms}/4 [{_conviction_tag}]: {_b2_detail}\")"

OI_ANCHOR_BLOCK = """                self._wlog(f"  PASSED(B2-CONVICTION): {_stock_name} {_b2_confirms}/4 [{_conviction_tag}]: {_b2_detail}")

                # --- GATE B2-OI-ANCHOR: Grind must pass 1/3 OI anchor factors (F, H, K) ---
                # Apr 2: SLOW_GRIND needs institutional OI validation via the OI_WATCHER
                # anchor pool. The OI_WATCHER engine computes _mf_F (Futures OI buildup),
                # _mf_H (OI concentration near ATM), _mf_K (Futures conviction) on Layer 1
                # data. Require at least 1 of 3 to confirm for grind trades.
                if _trigger_type in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):
                    _grind_oi_data = _oi_results.get(_sym, {})
                    _grind_mf_F = _grind_oi_data.get('_mf_F', False)
                    _grind_mf_H = _grind_oi_data.get('_mf_H', False)
                    _grind_mf_K = _grind_oi_data.get('_mf_K', False)
                    _grind_anchor_count = sum([_grind_mf_F, _grind_mf_H, _grind_mf_K])
                    _grind_anchor_detail = (f'F(FutOI)={"✓" if _grind_mf_F else "✗"} '
                                            f'H(ATM)={"✓" if _grind_mf_H else "✗"} '
                                            f'K(FutConv)={"✓" if _grind_mf_K else "✗"}')
                    if _grind_anchor_count < 1:
                        self._wlog(f"  BLOCKED(B2-OI-ANCHOR): {_stock_name} GRIND needs ≥1/3 anchor — {_grind_anchor_detail}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_GRIND_NO_OI_ANCHOR',
                                          reason=f'Grind OI anchor 0/3: {_grind_anchor_detail}',
                                          direction=direction)
                        continue
                    self._wlog(f"  PASSED(B2-OI-ANCHOR): {_stock_name} {_grind_anchor_count}/3 anchors — {_grind_anchor_detail}")"""

if OI_ANCHOR_MARKER in content and 'B2-OI-ANCHOR' not in content:
    content = content.replace(OI_ANCHOR_MARKER, OI_ANCHOR_BLOCK, 1)
    changes_made += 1
    print(f"✅ CHANGE 3: Grind OI Anchor gate (B2-OI-ANCHOR) inserted")
elif 'B2-OI-ANCHOR' in content:
    print(f"⏭️ CHANGE 3 SKIPPED: B2-OI-ANCHOR already exists")
else:
    print(f"❌ CHANGE 3 FAILED: Could not find B2-CONVICTION marker")

# ============================================================
# CHANGE 4: ADX gate — skip for EARLYBIRD
# ============================================================
OLD_4 = "                if adx_val < _adx_min and _trigger_type not in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN'):"
NEW_4 = """                # [Apr-2] Skip for EARLYBIRD — ADX unreliable at market open
                if adx_val < _adx_min and _trigger_type not in ('SLOW_GRIND_UP', 'SLOW_GRIND_DOWN') and not _is_earlybird:"""

if OLD_4 in content:
    content = content.replace(OLD_4, NEW_4, 1)
    changes_made += 1
    print(f"✅ CHANGE 4: ADX gate skipped for EARLYBIRD")
else:
    # Check if already applied
    if 'and not _is_earlybird:' in content:
        print(f"⏭️ CHANGE 4 SKIPPED: EARLYBIRD ADX skip already exists")
    else:
        print(f"❌ CHANGE 4 FAILED: Could not find ADX gate line")

# ============================================================
# CHANGE 5: EARLYBIRD max_trades cap before place_option_order
# ============================================================
# Insert before the first "with self._trade_lock:" + "result = self.tools.place_option_order"
# after the earlybird _lot_mult logic
EB_CAP_MARKER = """                    with self._trade_lock:
                        result = self.tools.place_option_order(
                            underlying=_sym,
                            direction=_direction,
                            strike_selection=_w_strike_sel,
                            rationale=(f"WATCHER\u2192FULL_PIPELINE: {_trigger_type}"""

EB_CAP_NEW = """                    # [Apr 2] EARLYBIRD max_trades cap — limit earlybird trades per morning
                    if 'EARLYBIRD' in _trigger_type:
                        _eb_max_trades = _EB_COM.get('max_trades', 3)
                        if self._earlybird_total_placed >= _eb_max_trades:
                            self._wlog(f"  ⛔ EARLYBIRD CAP: {_sym.replace('NSE:', '')} — already placed {self._earlybird_total_placed}/{_eb_max_trades} earlybird trades today")
                            continue

                    with self._trade_lock:
                        result = self.tools.place_option_order(
                            underlying=_sym,
                            direction=_direction,
                            strike_selection=_w_strike_sel,
                            rationale=(f"WATCHER\u2192FULL_PIPELINE: {_trigger_type}"""

if EB_CAP_MARKER in content and 'EARLYBIRD CAP' not in content:
    content = content.replace(EB_CAP_MARKER, EB_CAP_NEW, 1)
    changes_made += 1
    print(f"✅ CHANGE 5: EARLYBIRD max_trades cap added before place_option_order")
elif 'EARLYBIRD CAP' in content:
    print(f"⏭️ CHANGE 5 SKIPPED: EARLYBIRD cap already exists")
else:
    print(f"❌ CHANGE 5 FAILED: Could not find place_option_order marker")

# ============================================================
# Write result
# ============================================================
print(f"\nTotal changes applied: {changes_made}/5")
print(f"File size: {original_len} → {len(content)} chars")

if changes_made > 0:
    with open(FILE, 'w') as f:
        f.write(content)
    print(f"✅ File written successfully")
else:
    print(f"⚠️ No changes written")

# Verify
with open(FILE, 'r') as f:
    verify = f.read()

checks = [
    ("speed_gate_enabled", "ARBTR speed gate toggle"),
    ("_b2_min_needed", "B2 threshold variable"),
    ("B2-OI-ANCHOR", "Grind OI anchor gate"),
    ("and not _is_earlybird:", "ADX earlybird skip"),
    ("EARLYBIRD CAP", "EARLYBIRD max_trades cap"),
    ("commodities_trader", "MCX integration preserved"),
]

print("\n--- Verification ---")
for pattern, label in checks:
    count = verify.count(pattern)
    status = "✅" if count > 0 else "❌ MISSING"
    print(f"  {status} {label}: found {count}x")
