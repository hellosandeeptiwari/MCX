"""Two-tier AGGR: unlimited cached stocks + capped uncached stocks.

Instead of sending all 40 to the executor and hoping for the best,
split the stock list into:
  - CACHED: stocks in analyzer cache (free, 0 API calls) — all included
  - UNCACHED: new stocks that need DhanHQ calls — capped at 25

This guarantees: 25 × 3.2s = 80s fits in 90s timeout,
while cached stocks return instantly and boost coverage to 40+.
"""
import re

filepath = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find the block boundaries
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# Parallel OI fetch' in line:
        start_idx = i
    if start_idx and '_fetch_dur = _oiag_t.time() - _fetch_start' in line:
        end_idx = i + 1
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: Could not find section (start={start_idx}, end={end_idx})")
    exit(1)

print(f"Replacing lines {start_idx+1}-{end_idx} ({end_idx - start_idx} lines)")

# Also need to find and patch the stock selection loop to allow 40 candidate stocks
# but split them into cached/uncached before the fetch
indent = '        '  # 8 spaces

new_block = f"""{indent}# ─── Two-tier AGGR fetch: unlimited cached + capped uncached ───
{indent}# Stocks already in analyzer cache (90s TTL) return instantly (0 API calls).
{indent}# Uncached stocks need ~3.2s each via DhanHQ throttle.
{indent}# Strategy: include ALL cached stocks, cap uncached at 25 for 80s budget.
{indent}from concurrent.futures import ThreadPoolExecutor as _OITP, as_completed as _oi_done
{indent}from datetime import datetime as _oiag_dt
{indent}_oi_raw = {{}}
{indent}_fetch_start = _oiag_t.time()

{indent}# Split scan list into cached (free) vs uncached (budget-limited)
{indent}_analyzer_cache = getattr(t._oi_analyzer, '_CACHE', {{}})
{indent}_cache_ttl = getattr(t._oi_analyzer, '_CACHE_TTL', 90)
{indent}_now_dt = _oiag_dt.now()
{indent}_cached_syms = []
{indent}_uncached_syms = []
{indent}for _s in _scan_syms:
{indent}    _ce = _analyzer_cache.get(_s)
{indent}    if _ce and (_now_dt - _ce[0]).total_seconds() < _cache_ttl:
{indent}        _cached_syms.append(_s)
{indent}    else:
{indent}        _uncached_syms.append(_s)

{indent}# Cap uncached at 25 (25 × 3.2s = 80s < 90s timeout budget)
{indent}_MAX_UNCACHED = 25
{indent}_uncached_syms = _uncached_syms[:_MAX_UNCACHED]
{indent}_final_syms = _cached_syms + _uncached_syms
{indent}_n_cached = len(_cached_syms)
{indent}_n_uncached = len(_uncached_syms)

{indent}# Timeout: uncached stocks need ~4s each, cached are instant
{indent}_smart_timeout = max(30, min(120, _n_uncached * 4 + 5))

{indent}with _OITP(max_workers=7, thread_name_prefix='oi-aggr') as _ex:
{indent}    _futs = {{_ex.submit(t._oi_analyzer.analyze, _s): _s for _s in _final_syms}}
{indent}    try:
{indent}        for _f in _oi_done(_futs, timeout=_smart_timeout):
{indent}            _sym = _futs[_f]
{indent}            try:
{indent}                _res = _f.result()
{indent}                if _res:
{indent}                    _oi_raw[_sym] = _res
{indent}            except Exception as e:
{indent}                t._wlog(f"\\u26a0\\ufe0f FALLBACK [oi_watcher/aggr_result]: {{e}}")
{indent}    except Exception as e:
{indent}        _unfinished = sum(1 for f in _futs if not f.done())
{indent}        t._wlog(f"\\u26a0\\ufe0f FALLBACK [oi_watcher/aggr_timeout]: {{_unfinished}} (of {{len(_final_syms)}}) futures unfinished after {{_oiag_t.time()-_fetch_start:.0f}}s \\u2014 got {{len(_oi_raw)}} results (cached={{_n_cached}} uncached={{_n_uncached}} timeout={{_smart_timeout}}s)")
{indent}_fetch_dur = _oiag_t.time() - _fetch_start
"""

new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print(f"OK: Two-tier AGGR fetch applied (lines {start_idx+1}-{end_idx} replaced)")
