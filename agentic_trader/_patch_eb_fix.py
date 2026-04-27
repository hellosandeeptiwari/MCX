#!/usr/bin/env python3
"""Fix remaining earlybird patches that failed in first pass."""

FILES = {
    'kite': '/home/ubuntu/titan/agentic_trader/kite_ticker.py',
    'auto': '/home/ubuntu/titan/agentic_trader/autonomous_trader.py',
    'options': '/home/ubuntu/titan/agentic_trader/options_trader.py',
}

fixes = 0

# ═══ FIX 1: Empty import in autonomous_trader.py ═══
with open(FILES['auto'], 'r') as f:
    code = f.read()

old = """        from config import (

        )
"""
if old in code:
    code = code.replace(old, '', 1)
    fixes += 1
    print("✅ Fix 1: Removed empty config import block")
else:
    print("⏭️ Fix 1: Empty import not found (maybe already fixed)")

# ═══ FIX 2: kite_ticker state init (whitespace fix) ═══
with open(FILES['kite'], 'r') as f:
    kite = f.read()

# The anchors failed due to trailing whitespace. Let's use line-by-line matching
old_state = '        # === EARLYBIRD STATE (Opening Volatility 09:15-09:45) ==='
if old_state in kite:
    # Find this line and replace the block
    lines = kite.split('\n')
    start = None
    for i, line in enumerate(lines):
        if '=== EARLYBIRD STATE' in line:
            start = i
            break
    if start is not None:
        # The block runs from start to the _news_targets line
        end = start
        for i in range(start, min(start + 20, len(lines))):
            if '_news_targets' in lines[i]:
                end = i
                break
        # Replace this block with minimal stub
        new_lines = [
            '        # === EARLYBIRD — REMOVED Apr 20 ===',
            '        self._earlybird_enabled = False',
            '        self._earlybird_trades_fired = 0',
            '        self._prev_close: Dict[str, float] = {}',
            '        self._day_open: Dict[str, float] = {}',
            '        self._earlybird_fired: Dict[str, set] = {}',
            '        self._news_targets: Dict[str, dict] = {}',
        ]
        lines[start:end+1] = new_lines
        kite = '\n'.join(lines)
        fixes += 1
        print("✅ Fix 2: Replaced earlybird state init block")
else:
    print("⏭️ Fix 2: State init already removed or not found")

# ═══ FIX 3: kite_ticker sustain time branch ═══
old_sustain1 = "            if 'EARLYBIRD' in _ttype_s:"
if old_sustain1 in kite:
    # Find and remove the EARLYBIRD sustain branch (2 lines)
    lines = kite.split('\n')
    for i, line in enumerate(lines):
        if "if 'EARLYBIRD' in _ttype_s:" in line:
            # Remove this line and next line (_effective_sustain = ...)
            if i + 1 < len(lines) and '_earlybird_sustain' in lines[i + 1]:
                del lines[i:i+2]
                # Fix the elif to if
                if lines[i].strip().startswith("elif 'GRIND'"):
                    lines[i] = lines[i].replace('elif', 'if', 1)
                kite = '\n'.join(lines)
                fixes += 1
                print("✅ Fix 3: Removed earlybird sustain time branch")
                break
    else:
        print("⏭️ Fix 3: Sustain time branch not found in expected format")
else:
    print("⏭️ Fix 3: Sustain time already removed")

# ═══ FIX 4: kite_ticker sustain hold threshold ═══
old_hold = "                # Earlybird uses per-mode sustain hold threshold"
if old_hold in kite:
    lines = kite.split('\n')
    for i, line in enumerate(lines):
        if 'Earlybird uses per-mode sustain hold threshold' in line:
            # Find the block start (this comment) and end (elif 'SPIKE')
            end = i
            for j in range(i + 1, min(i + 12, len(lines))):
                if "elif 'SPIKE'" in lines[j] and 'EARLYBIRD' not in lines[j]:
                    end = j
                    break
            # Remove from comment to (but not including) the elif SPIKE line
            # Convert the elif to if
            del lines[i:end]
            if lines[i].strip().startswith("elif 'SPIKE'"):
                lines[i] = lines[i].replace('elif', 'if', 1)
            kite = '\n'.join(lines)
            fixes += 1
            print("✅ Fix 4: Removed earlybird sustain hold threshold block")
            break
else:
    print("⏭️ Fix 4: Sustain hold already removed")

# ═══ FIX 5: kite_ticker trigger metadata fields ═══
old_fields = "                          'earlybird_mode', 'earlybird_reason',"
if old_fields in kite:
    kite = kite.replace(old_fields + '\n', '', 1)
    fixes += 1
    print("✅ Fix 5: Removed earlybird fields from trigger metadata")
else:
    print("⏭️ Fix 5: Earlybird fields already removed")

with open(FILES['kite'], 'w') as f:
    f.write(kite)

# ═══ FIX 6: autonomous_trader range budget skip ═══
# The _is_earlybird was already set to False by earlier patch, so this line
# still references it. Let's fix it.
old_range = "                _is_earlybird = 'EARLYBIRD' in _trigger_type\n"
# Find this second occurrence (first was already changed)
if code.count("_is_earlybird = 'EARLYBIRD' in _trigger_type") > 0:
    # Replace all remaining occurrences
    code = code.replace("_is_earlybird = 'EARLYBIRD' in _trigger_type\n                if not _is_earlybird and ",
                        "if ")
    fixes += 1
    print("✅ Fix 6: Simplified range budget skip")
else:
    print("⏭️ Fix 6: Range budget already fixed")

# ═══ FIX 7: autonomous_trader ADX gate ═══
old_adx = " and not _is_earlybird:"
if old_adx in code:
    code = code.replace(old_adx, ':', 1)
    fixes += 1
    print("✅ Fix 7: Removed _is_earlybird from ADX gate")
else:
    print("⏭️ Fix 7: ADX gate already fixed")

# ═══ FIX 8: autonomous_trader earlybird metadata ═══
old_meta = "                # EARLYBIRD metadata for position tracking (per-mode)"
if old_meta in code:
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'EARLYBIRD metadata for position tracking' in line:
            # Remove from this line through earlybird_beta_driven
            end = i
            for j in range(i + 1, min(i + 15, len(lines))):
                if 'earlybird_beta_driven' in lines[j]:
                    end = j
                    break
            del lines[i:end+1]
            code = '\n'.join(lines)
            fixes += 1
            print("✅ Fix 8: Removed earlybird metadata storage")
            break
else:
    print("⏭️ Fix 8: Metadata already removed")

# ═══ FIX 9: autonomous_trader drain bypass ═══
old_drain = "        # EARLYBIRD triggers bypass this gate"
if old_drain in code:
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'EARLYBIRD triggers bypass this gate' in line:
            # Replace the comment + next 2 lines (import + assignment)
            code_line1 = lines[i]
            code_line2 = lines[i + 1] if i + 1 < len(lines) else ''
            code_line3 = lines[i + 2] if i + 2 < len(lines) else ''
            lines[i] = '        _earlybird_active = False  # EARLYBIRD REMOVED'
            if 'EARLYBIRD_COMMON' in code_line2:
                del lines[i + 1]
            if 'earlybird_active' in lines[i + 1]:
                del lines[i + 1]
            code = '\n'.join(lines)
            fixes += 1
            print("✅ Fix 9: Removed earlybird drain bypass")
            break
else:
    print("⏭️ Fix 9: Drain bypass already fixed")

# ═══ FIX 10: Daily summary earlybird stats ═══
old_summary = 'f"earlybird={self._earlybird_total_placed}'
if old_summary in code:
    # Find and remove the earlybird part of the f-string
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'earlybird={self._earlybird_total_placed}' in line:
            # Remove the earlybird portion from this f-string segment
            import re
            lines[i] = re.sub(
                r'f"earlybird=\{self\._earlybird_total_placed\}.*?"\s*', 
                '', lines[i]
            )
            # If line is now just whitespace, remove it
            if lines[i].strip() == '':
                del lines[i]
            code = '\n'.join(lines)
            fixes += 1
            print("✅ Fix 10: Removed earlybird from daily summary")
            break
else:
    print("⏭️ Fix 10: Daily summary already fixed")

with open(FILES['auto'], 'w') as f:
    f.write(code)

# ═══ FIX 11: options_trader micro bypass branch ═══
with open(FILES['options'], 'r') as f:
    opt = f.read()

old_micro = "        elif microstructure_block and _eb_micro_bypass:"
if old_micro in opt:
    # Find and remove this elif + next line
    lines = opt.split('\n')
    for i, line in enumerate(lines):
        if 'elif microstructure_block and _eb_micro_bypass:' in line:
            # Remove this line and the next (warning append)
            del lines[i:i+2]
            opt = '\n'.join(lines)
            fixes += 1
            print("✅ Fix 11: Removed earlybird micro bypass branch from options_trader")
            break
    with open(FILES['options'], 'w') as f:
        f.write(opt)
else:
    print("⏭️ Fix 11: Micro bypass branch already removed")


# ═══ VERIFY COMPILATION ═══
print(f"\n{'='*60}")
print("COMPILATION CHECK:")
import py_compile
all_ok = True
for name, path in FILES.items():
    try:
        py_compile.compile(path, doraise=True)
        print(f"  ✅ {name}: compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ {name}: COMPILE ERROR: {e}")
        all_ok = False

# Also check config and other files
for extra in ['/home/ubuntu/titan/agentic_trader/config.py',
              '/home/ubuntu/titan/agentic_trader/zerodha_tools.py',
              '/home/ubuntu/titan/agentic_trader/dashboard.py']:
    try:
        py_compile.compile(extra, doraise=True)
        print(f"  ✅ {extra.split('/')[-1]}: compile OK")
    except py_compile.PyCompileError as e:
        print(f"  ❌ {extra.split('/')[-1]}: COMPILE ERROR: {e}")
        all_ok = False

if all_ok:
    print(f"\n✅ ALL {fixes} fixes applied, ALL files compile OK")
else:
    print(f"\n❌ COMPILE ERRORS remain — check output above")
