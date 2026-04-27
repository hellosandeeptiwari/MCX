"""Replace AGGR fetch section with cache-aware version"""
filepath = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find start line: "Parallel OI fetch"
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# Parallel OI fetch' in line and 'oi-aggr' not in line:
        start_idx = i
    if start_idx and '_fetch_dur = _oiag_t.time() - _fetch_start' in line:
        end_idx = i + 1
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: Could not find section (start={start_idx}, end={end_idx})")
    exit(1)

print(f"Replacing lines {start_idx+1}-{end_idx} ({end_idx - start_idx} lines)")

# Get indentation from the start line
indent = '        '  # 8 spaces

new_block = f"""{indent}# Parallel OI fetch — cache-aware timeout
{indent}# Stocks already in analyzer cache (90s TTL) return instantly without
{indent}# DhanHQ API calls, so timeout only needs to cover uncached stocks.
{indent}from concurrent.futures import ThreadPoolExecutor as _OITP, as_completed as _oi_done
{indent}from datetime import datetime as _oiag_dt
{indent}_oi_raw = {{}}
{indent}_fetch_start = _oiag_t.time()

{indent}# Count how many stocks will actually hit DhanHQ (not cached)
{indent}_analyzer_cache = getattr(t._oi_analyzer, '_CACHE', {{}})
{indent}_cache_ttl = getattr(t._oi_analyzer, '_CACHE_TTL', 90)
{indent}_now_dt = _oiag_dt.now()
{indent}_n_cached = sum(1 for _s in _scan_syms
{indent}                if _s in _analyzer_cache
{indent}                and (_now_dt - _analyzer_cache[_s][0]).total_seconds() < _cache_ttl)
{indent}_n_uncached = len(_scan_syms) - _n_cached
{indent}# Budget: each uncached stock needs ~3.5s (throttle + API), cached = 0s
{indent}_smart_timeout = max(30, min(120, _n_uncached * 4 + 5))

{indent}with _OITP(max_workers=7, thread_name_prefix='oi-aggr') as _ex:
{indent}    _futs = {{_ex.submit(t._oi_analyzer.analyze, _s): _s for _s in _scan_syms}}
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
{indent}        t._wlog(f"\\u26a0\\ufe0f FALLBACK [oi_watcher/aggr_timeout]: {{_unfinished}} (of {{len(_scan_syms)}}) futures unfinished after {{_oiag_t.time()-_fetch_start:.0f}}s \\u2014 got {{len(_oi_raw)}} results (cached={{_n_cached}} uncached={{_n_uncached}} timeout={{_smart_timeout}}s)")
{indent}_fetch_dur = _oiag_t.time() - _fetch_start
"""

new_lines = lines[:start_idx] + [new_block] + lines[end_idx:]

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("OK: Smart cache-aware AGGR timeout applied")
