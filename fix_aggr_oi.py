"""Fix: Reduce AGGR OI max symbols from 40 to 20 to fit DhanHQ rate limit.

DhanHQ serializes at 3.2s/call. With 40 symbols: 128s needed vs 80s timeout.
With 20 symbols: 64s needed, fits comfortably in 80s timeout.
Also increase per-symbol timeout multiplier from 2 to 4 as safety margin.
"""

fp = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'
content = open(fp, 'r').read()

# FIX: Increase timeout multiplier from 2 to 4 seconds per symbol
old_timeout = 'timeout=max(18, len(_scan_syms) * 2)'
new_timeout = 'timeout=max(30, len(_scan_syms) * 4)'
assert old_timeout in content, f'timeout pattern not found'
content = content.replace(old_timeout, new_timeout, 1)
print('FIX1 applied: timeout multiplier 2 -> 4 per symbol')

open(fp, 'w').write(content)

# Also fix autonomous_trader.py: reduce max symbols from 40 to 20
fp2 = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'
content2 = open(fp2, 'r').read()

old_max = 'self._oi_aggr_max_symbols = 40'
new_max = 'self._oi_aggr_max_symbols = 20'
assert old_max in content2, f'max_symbols pattern not found'
content2 = content2.replace(old_max, new_max, 1)
print('FIX2 applied: _oi_aggr_max_symbols 40 -> 20')

open(fp2, 'w').write(content2)
print('OK: both fixes applied')
