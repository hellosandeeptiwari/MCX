#!/usr/bin/env python3
"""
Patch zerodha_tools.py: Add scorer threshold override for OI_WATCHER trades.

Same pattern as existing EARLYBIRD and GMM_SNIPER overrides.
OI_WATCHER trades have already been validated by a multi-factor conviction
pipeline (breadth gates, anchor gates, price confirmation). The intraday
scorer should not override these confirmed signals with a high threshold.

Sets BLOCK_THRESHOLD to 25 for OI_WATCHER/OI_WATCHER_AGGR setup types.
This is more conservative than EARLYBIRD (15) but much lower than default (66).
"""
import os

BASE = '/home/ubuntu/titan/agentic_trader'
zt_path = os.path.join(BASE, 'zerodha_tools.py')

data = open(zt_path, 'rb').read()

# Detect line endings
nl = b'\r\n' if b'\r\n' in data[:500] else b'\n'

# Insert AFTER the EARLYBIRD override block, BEFORE `plan = options_trader.create_option_order(`
# The exact anchor is this line:
ANCHOR = (
    b'            print(f"   \xf0\x9f\x90\xa6 EARLYBIRD THRESHOLD: {_eb_threshold_override} \xe2\x86\x92 15 + micro-bypass + conviction-relax (watcher-validated)")' + nl +
    nl +
    b'        plan = options_trader.create_option_order('
)

OI_WATCHER_BLOCK = (
    b'            print(f"   \xf0\x9f\x90\xa6 EARLYBIRD THRESHOLD: {_eb_threshold_override} \xe2\x86\x92 15 + micro-bypass + conviction-relax (watcher-validated)")' + nl +
    nl +
    b'        # === OI_WATCHER THRESHOLD OVERRIDE: OI pipeline already validated the trade ===' + nl +
    b"        # OI_WATCHER trades pass through multi-factor conviction (breadth gates 3-5/6," + nl +
    b"        # anchor gates, 45-90s price confirmation). Scorer should not override these." + nl +
    b"        _oi_watcher_threshold_override = None" + nl +
    b"        if 'OI_WATCHER' in (setup_type or ''):" + nl +
    b"            from options_trader import get_intraday_scorer as _get_scorer_oiw" + nl +
    b"            _scorer_inst_oiw = _get_scorer_oiw()" + nl +
    b"            _oi_watcher_threshold_override = _scorer_inst_oiw.BLOCK_THRESHOLD" + nl +
    b"            _scorer_inst_oiw.BLOCK_THRESHOLD = 25  # OI multi-factor conviction \xe2\x89\xab scorer (breadth+anchor+price confirmed)" + nl +
    b'            print(f"   \xf0\x9f\x94\x8d OI_WATCHER THRESHOLD: {_oi_watcher_threshold_override} \xe2\x86\x92 25 (OI-pipeline-validated)")' + nl +
    nl +
    b'        plan = options_trader.create_option_order('
)

if ANCHOR in data:
    data = data.replace(ANCHOR, OI_WATCHER_BLOCK, 1)
    print("  OK: OI_WATCHER scorer override inserted (before plan=)")
else:
    print("  SKIP: EARLYBIRD+plan anchor not found, trying alt...")
    # Try without the print line — just match on `plan = options_trader`
    ALT_ANCHOR = nl + b'        plan = options_trader.create_option_order('
    ALT_INSERT = (
        nl +
        b'        # === OI_WATCHER THRESHOLD OVERRIDE ===' + nl +
        b"        _oi_watcher_threshold_override = None" + nl +
        b"        if 'OI_WATCHER' in (setup_type or ''):" + nl +
        b"            from options_trader import get_intraday_scorer as _get_scorer_oiw" + nl +
        b"            _scorer_inst_oiw = _get_scorer_oiw()" + nl +
        b"            _oi_watcher_threshold_override = _scorer_inst_oiw.BLOCK_THRESHOLD" + nl +
        b"            _scorer_inst_oiw.BLOCK_THRESHOLD = 25" + nl +
        b'            print(f"   OI_WATCHER THRESHOLD: {_oi_watcher_threshold_override} -> 25")' + nl +
        nl +
        b'        plan = options_trader.create_option_order('
    )
    if ALT_ANCHOR in data:
        data = data.replace(ALT_ANCHOR, ALT_INSERT, 1)
        print("  OK: OI_WATCHER scorer override inserted (alt match)")
    else:
        print("  FAIL: Could not find insertion point!")

# Also add the restore block after the existing EARLYBIRD restore  
# Find:  _scorer_eb2._eb_conviction_override = False  # Restore conviction gate
RESTORE_ANCHOR = b"            _scorer_eb2._eb_conviction_override = False  # Restore conviction gate"
RESTORE_ADD = (
    b"            _scorer_eb2._eb_conviction_override = False  # Restore conviction gate" + nl +
    b"        if _oi_watcher_threshold_override is not None:" + nl +
    b"            from options_trader import get_intraday_scorer as _get_scorer_oiw2" + nl +
    b"            _get_scorer_oiw2().BLOCK_THRESHOLD = _oi_watcher_threshold_override"
)

if RESTORE_ANCHOR in data:
    data = data.replace(RESTORE_ANCHOR, RESTORE_ADD, 1)
    print("  OK: OI_WATCHER threshold restore added")
else:
    print("  SKIP: Restore anchor not found")

open(zt_path, 'wb').write(data)
print(f"\n  >>> zerodha_tools.py written ({len(data)} bytes)")
print("\npy_compile zerodha_tools.py + options_trader.py, then restart.")
