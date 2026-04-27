"""Smart AGGR scan: cache-aware batching.

Instead of blindly capping at 25 stocks, we:
1. Allow up to 40 stocks into the scan
2. Pre-check the options_flow_analyzer cache — cached stocks are FREE (no API call)
3. Set timeout based only on UNCACHED stock count
4. This gets us 40 stocks scanned while staying within DhanHQ rate limits
"""
filepath = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'

with open(filepath, 'r') as f:
    content = f.read()

old = """        # Parallel OI fetch (7 workers — Kite calls benefit from parallelism)
        from concurrent.futures import ThreadPoolExecutor as _OITP, as_completed as _oi_done
        _oi_raw = {}
        _fetch_start = _oiag_t.time()
        with _OITP(max_workers=7, thread_name_prefix='oi-aggr') as _ex:
            _futs = {_ex.submit(t._oi_analyzer.analyze, _s): _s for _s in _scan_syms}
            try:
                for _f in _oi_done(_futs, timeout=max(30, min(90, len(_scan_syms) * 4))):
                    _sym = _futs[_f]
                    try:
                        _res = _f.result()
                        if _res:
                            _oi_raw[_sym] = _res
                    except Exception as e:
                        t._wlog(f"\\u26a0\\ufe0f FALLBACK [oi_watcher/aggr_result]: {e}")   
            except Exception as e:
                _unfinished = sum(1 for f in _futs if not f.done())
                t._wlog(f"\\u26a0\\ufe0f FALLBACK [oi_watcher/aggr_timeout]: {_unfinished} ((of {len(_scan_syms)}) futures unfinished after {_oiag_t.time()-_fetch_start:.0f}s \\u2014 got {len(_oi_raw)} results")
        _fetch_dur = _oiag_t.time() - _fetch_start"""

count = content.count(old)
print(f"Match count: {count}")
if count != 1:
    # Print actual lines around the fetch section for debugging
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Parallel OI fetch' in line or 'oi-aggr' in line or 'aggr_timeout' in line:
            print(f"L{i+1}: {line}")
