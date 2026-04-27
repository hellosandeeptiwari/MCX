#!/usr/bin/env python3
"""Final-final earlybird cleanup — fix all remaining _EB_* runtime crashes."""

AUTO = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'

with open(AUTO, 'r') as f:
    lines = f.readlines()

print(f"Total lines before: {len(lines)}")

# Work bottom-to-top to avoid line number shifts

# === FIX 5: Remove lot sizing elif EARLYBIRD block (lines 3461-3480) ===
# Line 3461: "                    elif 'EARLYBIRD' in _trigger_type:"
# Line 3480: "                                _surge_tag = f' [EB-{_eb_m}→{_lot_mult}x LOT]'"
# Verify anchors
if "'EARLYBIRD' in _trigger_type:" in lines[3460] and "EB-{_eb_m}" in lines[3479]:
    del lines[3460:3480]
    print("✅ Fix 5: Removed earlybird lot sizing block (3461-3480)")
else:
    print(f"❌ Fix 5: Anchor mismatch")
    print(f"  3461: {lines[3460].rstrip()}")
    print(f"  3480: {lines[3479].rstrip()}")

# === FIX 4: Remove earlybird P(move) floor — replace with direct news/default ===
# Lines 3336-3340 (after Fix 5 shift: 3336 is now the right line since Fix 5 was below)
# Wait — since I deleted from bottom-up, positions above are stable
# Line 3336: "                    # EARLYBIRD uses per-mode P(move) floor"
# Line 3337: "                    if _is_earlybird:"
# Line 3338: "                        _eb_mode_cfg = ..."  
# Line 3339: "                        _w_min_move = ..."
# Line 3340: "                    elif _is_news_trigger:"
# Line 3341: "                        _w_min_move = _EB_D.get('min_move_prob', 0.25)"
# Replace lines 3336-3341 with just the news and default branches
if "EARLYBIRD uses per-mode" in lines[3335]:
    lines[3335:3341] = [
        "                    if _is_news_trigger:\n",
        "                        _w_min_move = 0.25  # News P(move) floor\n",
    ]
    print("✅ Fix 4: Replaced P(move) floor with news/default only")
else:
    print(f"❌ Fix 4: Anchor mismatch at 3336: {lines[3335].rstrip()}")

# === FIX 3: Remove earlybird from C2 budget gate ===
# Line 2935: "                if not _is_earlybird and _c2_budget_used ..."
if "not _is_earlybird and _c2_budget" in lines[2934]:
    lines[2934] = lines[2934].replace("not _is_earlybird and ", "")
    print("✅ Fix 3: Simplified C2 budget gate")
else:
    print(f"❌ Fix 3: Anchor mismatch at 2935: {lines[2934].rstrip()}")

# === FIX 2: Remove earlybird score gate block ===
# Line 2769: "_eb_mode = ... if _is_earlybird else ''"  → simplify to _eb_mode = ''
# Line 2774-2776: "if _is_earlybird:" block → remove (dead code with undefined _EB_*)
# Line 2778: "_effective_min = _EB_D.get('min_score', 15)" → hardcode 15
if "_is_earlybird else ''" in lines[2768]:
    lines[2768] = "                _eb_mode = ''  # EARLYBIRD removed\n"
    print("✅ Fix 2a: Simplified _eb_mode assignment")
else:
    print(f"❌ Fix 2a: Anchor mismatch at 2769: {lines[2768].rstrip()}")

if "if _is_earlybird:" in lines[2773] and "_eb_mode_cfg" in lines[2774]:
    del lines[2773:2776]  # Remove 3 lines (if, cfg, effective_min)
    print("✅ Fix 2b: Removed earlybird score gate block")
    # Now line 2773 should be "elif _is_news_trigger:" 
    # and line 2774 should be "_effective_min = _EB_D.get('min_score', 15)"
    if "_EB_D.get('min_score'" in lines[2774]:
        lines[2774] = "                    _effective_min = 15  # Min score for news triggers\n"
        print("✅ Fix 2c: Hardcoded news min_score")
    else:
        print(f"❌ Fix 2c: Expected _EB_D at 2775, got: {lines[2774].rstrip()}")
else:
    print(f"❌ Fix 2b: Expected 'if _is_earlybird:' at 2774, got: {lines[2773].rstrip()}")

with open(AUTO, 'w') as f:
    f.writelines(lines)

print(f"Total lines after: {len(lines)}")

# === COMPILE CHECK ===
import py_compile
all_ok = True
for path in [AUTO,
             '/home/ubuntu/titan/agentic_trader/config.py',
             '/home/ubuntu/titan/agentic_trader/kite_ticker.py',
             '/home/ubuntu/titan/agentic_trader/zerodha_tools.py',
             '/home/ubuntu/titan/agentic_trader/options_trader.py',
             '/home/ubuntu/titan/agentic_trader/dashboard.py']:
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ {path.split('/')[-1]}: compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ {path.split('/')[-1]}: COMPILE ERROR: {e}")
        all_ok = False

# Final check: any remaining dangerous _EB_ refs (excluding safe local imports)?
print("\n=== REMAINING _EB_* REFERENCES: ===")
with open(AUTO, 'r') as f:
    for i, line in enumerate(f, 1):
        if ('_EB_A' in line or '_EB_B' in line or '_EB_C' in line or 
            '_EB_COM' in line) and 'from config' not in line:
            print(f"  ⚠️  Line {i}: {line.rstrip()}")

if all_ok:
    print("\n✅ ALL FILES COMPILE OK")
else:
    print("\n❌ COMPILE ERRORS — fix before restart")
