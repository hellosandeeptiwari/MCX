"""Smart AGGR OI: increase symbols to 40 with fixed 90s timeout.

Strategy: DhanHQ serializes at 3.2s/call ~ 25 stocks can complete in 90s.
But many will be cache hits (90s TTL), so effective throughput is higher.
Sort by change% ensures top movers are attempted first.
as_completed loop already handles partial results gracefully.

Changes:
1. autonomous_trader.py: _oi_aggr_max_symbols 20 -> 40 
2. oi_watcher_engine.py: timeout -> max(30, min(90, len * 4))
   Caps at 90s so the cycle doesn't stall, but still adaptive for small batches
"""

# Fix 1: autonomous_trader.py — increase max symbols back to 40
fp1 = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'
c1 = open(fp1, 'r').read()
old1 = 'self._oi_aggr_max_symbols = 20'
new1 = 'self._oi_aggr_max_symbols = 40       # DhanHQ serializes at 3.2s/call; 90s timeout gets ~25-28 results, top movers first'
assert old1 in c1, f'FIX1 not found'
c1 = c1.replace(old1, new1, 1)
open(fp1, 'w').write(c1)
print('FIX1: _oi_aggr_max_symbols 20 -> 40')

# Fix 2: oi_watcher_engine.py — smart timeout with 90s cap
fp2 = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'
c2 = open(fp2, 'r').read()
old2 = 'timeout=max(30, len(_scan_syms) * 4)'
new2 = 'timeout=max(30, min(90, len(_scan_syms) * 4))'
assert old2 in c2, f'FIX2 not found'
c2 = c2.replace(old2, new2, 1)
open(fp2, 'w').write(c2)
print('FIX2: timeout capped at 90s')

print('OK')
