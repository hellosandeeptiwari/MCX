#!/usr/bin/env python3
"""Patch: Speed up dashboard positions tab LTP updates.

1) dashboard.py _enrich_positions: Add direct Kite LTP bulk fetch 
   instead of only reading stale live_pnl from bot's write cycle.
2) titan_dashboard.html: Reduce positions polling from 15s to 5s.
"""
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else '/home/ubuntu/titan/agentic_trader'

# ============================================================
# FIX 1: dashboard.py — Add direct Kite LTP in _enrich_positions
# ============================================================
dash_path = f'{BASE}/dashboard.py'
with open(dash_path, 'rb') as f:
    data = f.read()

# Find the _enrich_positions function and replace it
old_marker = b'def _enrich_positions(positions: list, db) -> list:'
pos = data.find(old_marker)
if pos < 0:
    print("ERROR: Could not find _enrich_positions")
    sys.exit(1)

# Find end of function (next def or class at same indentation)
# The function ends with "    return positions\r\n"
end_marker = b'    return positions'
end_pos = data.find(end_marker, pos)
if end_pos < 0:
    print("ERROR: Could not find 'return positions' in _enrich_positions")
    sys.exit(1)

# Find the newline after "return positions"
end_line = data.find(b'\n', end_pos) + 1

old_func = data[pos:end_line]
print(f"Found _enrich_positions: {len(old_func)} bytes at offset {pos}")

# Detect line ending
nl = b'\r\n' if b'\r\n' in old_func else b'\n'

new_func = nl.join([
    b'def _enrich_positions(positions: list, db) -> list:',
    b'    """Merge live P&L data into each position dict so the dashboard',
    b'    can display current LTP, unrealized P&L, underlying symbol, etc."""',
    b'    import re',
    b'',
    b'    # Primary source: live_pnl table (updated every scan cycle by the bot)',
    b'    live = db.load_live_pnl() or {}',
    b'',
    b'    # Apr 20: Direct Kite LTP bulk fetch for fresh prices',
    b'    kite_ltp = {}',
    b'    try:',
    b'        kite = _get_dashboard_kite()',
    b'        if kite and positions:',
    b"            syms = [p.get('symbol') or p.get('option_symbol') or '' for p in positions]",
    b"            nfo_syms = [s if s.startswith('NFO:') else f'NFO:{s}' for s in syms if s]",
    b'            if nfo_syms:',
    b'                raw = kite.ltp(nfo_syms)',
    b'                for k, v in (raw or {}).items():',
    b"                    kite_ltp[k] = v.get('last_price', 0)",
    b"                    kite_ltp[k.replace('NFO:', '')] = v.get('last_price', 0)",
    b'    except Exception:',
    b'        pass  # Fall back to live_pnl',
    b'',
    b'    for pos in positions:',
    b"        sym = pos.get('symbol') or pos.get('option_symbol') or ''",
    b'        # Tag exchange',
    b"        if not pos.get('exchange'):",
    b"            pos['exchange'] = 'NSE'",
    b'        # Derive underlying from NFO symbol  e.g. NFO:DLF26MAR590PE -> DLF',
    b"        if not pos.get('underlying'):",
    b"            m = re.match(r'(?:NFO:)?([A-Z]+)\\d', sym.replace('NFO:', ''))",
    b'            pos[\'underlying\'] = f"NSE:{m.group(1)}" if m else \'\'',
    b'',
    b'        # 1) Try direct Kite LTP (freshest)',
    b"        _kite_price = kite_ltp.get(sym) or kite_ltp.get(sym.replace('NFO:', '')) or kite_ltp.get(f'NFO:{sym}')",
    b'        if _kite_price and _kite_price > 0:',
    b"            pos['ltp'] = _kite_price",
    b'            # Compute unrealized P&L from fresh LTP',
    b"            qty = pos.get('quantity', 0)",
    b"            avg = pos.get('avg_price', pos.get('entry_price', 0))",
    b"            side = pos.get('side', 'BUY')",
    b"            if side == 'BUY':",
    b"                pos['unrealized_pnl'] = round((_kite_price - avg) * qty, 2)",
    b'            else:',
    b"                pos['unrealized_pnl'] = round((avg - _kite_price) * qty, 2)",
    b'            continue',
    b'',
    b"        # 2) Fall back to live_pnl from bot's scan cycle",
    b"        lp = live.get(sym) or live.get(sym.replace('NFO:', ''))",
    b'        if isinstance(lp, dict):',
    b"            pos['ltp'] = lp.get('ltp', 0)",
    b"            pos['unrealized_pnl'] = lp.get('unrealized_pnl', 0)",
    b"            pos['ltp_updated'] = lp.get('last_updated', '')",
    b'        elif isinstance(lp, (int, float)):',
    b"            pos['ltp'] = float(lp)",
    b'    return positions',
    b'',
])

data = data[:pos] + new_func + data[end_line:]
with open(dash_path, 'wb') as f:
    f.write(data)
print(f"FIX 1: _enrich_positions now fetches live Kite LTP directly")

# ============================================================
# FIX 2: titan_dashboard.html — positions poll 15s → 5s
# ============================================================
html_path = f'{BASE}/templates/titan_dashboard.html'
with open(html_path, 'rb') as f:
    html = f.read()

old_interval = b'}, 15000);'
# Find the one near "positions" tab check
# There may be multiple }, 15000) — find the one after "tab-positions"
tab_pos = html.find(b'tab-positions')
if tab_pos < 0:
    print("WARN: Could not find tab-positions in HTML")
else:
    interval_pos = html.find(old_interval, tab_pos)
    if interval_pos > 0:
        html = html[:interval_pos] + b'}, 5000);' + html[interval_pos + len(old_interval):]
        with open(html_path, 'wb') as f:
            f.write(html)
        print("FIX 2: Positions tab polling 15s -> 5s")
    else:
        print("WARN: Could not find 15000 interval after tab-positions")

# Also speed up overview from 30s to 15s
with open(html_path, 'rb') as f:
    html = f.read()
old_overview = b'setInterval(loadOverview, 30000);'
new_overview = b'setInterval(loadOverview, 15000);'
if old_overview in html:
    html = html.replace(old_overview, new_overview, 1)
    with open(html_path, 'wb') as f:
        f.write(html)
    print("FIX 3: Overview polling 30s -> 15s")

print("\nDone. py_compile dashboard.py, then restart titan-dashboard.")
