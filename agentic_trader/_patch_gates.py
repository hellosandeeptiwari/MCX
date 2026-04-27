#!/usr/bin/env python3
"""Patch: Fix OI-ANCHOR bypass + add breadth gate to watcher pipeline.

1) B2-OI-ANCHOR: Change "0/4 evaluable → bypass" to "0/4 evaluable → BLOCK"
2) Add BREADTH CONFLICT gate after direction is finalized in watcher pipeline
"""
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'

with open(TARGET, 'rb') as f:
    data = f.read()

original_len = len(data)

# ============================================================
# FIX 1: B2-OI-ANCHOR — change bypass to BLOCK
# Use byte-level search for robustness against encoding issues
# ============================================================
# Find the line "SKIPPED(B2-OI-ANCHOR)" and replace the 3-line block
marker = b'SKIPPED(B2-OI-ANCHOR)'
pos = data.find(marker)
if pos < 0:
    print("ERROR: Could not find SKIPPED(B2-OI-ANCHOR) marker")
    sys.exit(1)

# Find the start of the "if _grind_evaluable == 0:" line (go back from marker)
# Find the line containing "if _grind_evaluable == 0:" before the marker
search_start = data.rfind(b'if _grind_evaluable == 0:', 0, pos)
if search_start < 0:
    print("ERROR: Could not find 'if _grind_evaluable == 0:' before marker")
    sys.exit(1)

# Go to start of that line
line_start = data.rfind(b'\n', 0, search_start) + 1

# Find the end of the SKIPPED log line (next \n after marker)
line_end = data.find(b'\n', pos)

# Extract the old block
old_block = data[line_start:line_end]
print(f"OLD block ({len(old_block)} bytes): ...{old_block[-80:]}")

# Get indentation from the old block
indent = b''
for ch in old_block:
    if ch in (32, 9):  # space or tab
        indent += bytes([ch])
    else:
        break

# Build new block with same indentation
# Note: use \r\n if the file has it
nl = b'\r\n' if b'\r\n' in old_block else b'\n'
new_lines = [
    indent + b'if _grind_evaluable == 0:',
    indent + b'    # Apr 20 FIX: No institutional data = BLOCK (was bypass)',
    indent + b'    self._wlog(f"  BLOCKED(B2-OI-ANCHOR): {_stock_name} GRIND -- 0/4 factors evaluable (no data) -- {_grind_anchor_detail}")',
    indent + b'    self._watcher_total_gate_blocked += 1',
    indent + b'    self._log_decision(_ts, _sym, _final_score, \'WATCHER_GRIND_NO_OI_DATA\',',
    indent + b'                      reason=f\'Grind OI anchor: 0/4 factors evaluable (no institutional data): {_grind_anchor_detail}\',',
    indent + b'                      direction=direction)',
    indent + b'    continue',
]
new_block = nl.join(new_lines)

data = data[:line_start] + new_block + data[line_end:]
print(f"FIX 1 applied: replaced {len(old_block)} bytes with {len(new_block)} bytes")

# ============================================================
# FIX 2: Add BREADTH CONFLICT gate to watcher pipeline
# Insert after OI direction override, before VIX penalty
# ============================================================
vix_marker = b'# \xe2\x94\x80\xe2\x94\x80 Direction-aware VIX penalty'
vix_pos = data.find(vix_marker)
if vix_pos < 0:
    # Try simpler marker
    vix_marker = b'Direction-aware VIX penalty'
    vix_pos = data.find(vix_marker)
if vix_pos < 0:
    # Try without unicode
    vix_marker = b'VIX penalty'
    vix_pos = data.find(vix_marker)
    # Find the comment line (with "Direction-aware" or similar)
    # Back up to find "Direction" before it
    check = data[max(0, vix_pos-50):vix_pos]
    if b'Direction' not in check and b'direction' not in check:
        print("WARN: VIX penalty marker found but not Direction-aware variant")

if vix_pos < 0:
    print("ERROR: Could not find VIX penalty section")
    sys.exit(1)

# Find start of the comment line
vix_line_start = data.rfind(b'\n', 0, vix_pos) + 1
# Get indentation  
vix_indent = b''
for ch in data[vix_line_start:]:
    if ch in (32, 9):
        vix_indent += bytes([ch])
    else:
        break

# Use same nl as detected above
breadth_gate = nl.join([
    b'',
    vix_indent + b'# -- Gate: WATCHER BREADTH CONFLICT (Apr 20) --',
    vix_indent + b"# Don't buy CEs in BEARISH market, don't buy PEs in BULLISH market",
    vix_indent + b"if _market_breadth in ('BULLISH', 'BEARISH'):",
    vix_indent + b'    _watcher_breadth_conflict = (',
    vix_indent + b"        (direction == 'SELL' and _market_breadth == 'BULLISH') or",
    vix_indent + b"        (direction == 'BUY' and _market_breadth == 'BEARISH')",
    vix_indent + b'    )',
    vix_indent + b'    if _watcher_breadth_conflict:',
    vix_indent + b'        self._wlog(f"  BLOCKED(BREADTH): {_stock_name} {direction} conflicts with {_market_breadth} market")',
    vix_indent + b'        self._watcher_total_gate_blocked += 1',
    vix_indent + b"        self._log_decision(_ts, _sym, _final_score, 'WATCHER_BREADTH_CONFLICT',",
    vix_indent + b"                          reason=f'{direction} trade vs {_market_breadth} market breadth',",
    vix_indent + b'                          direction=direction)',
    vix_indent + b'        continue',
    b'',
])

data = data[:vix_line_start] + breadth_gate + data[vix_line_start:]
print(f"FIX 2 applied: inserted {len(breadth_gate)} bytes of breadth gate")

# Write back
with open(TARGET, 'wb') as f:
    f.write(data)

print(f"File patched: {original_len} -> {len(data)} bytes ({len(data) - original_len:+d})")
print("Run: python3 -m py_compile autonomous_trader.py && echo OK")
