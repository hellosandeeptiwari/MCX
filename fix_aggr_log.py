"""Improve AGGR OI timeout logging to show completed/total and duration."""

fp = '/home/ubuntu/titan/agentic_trader/oi_watcher_engine.py'
c = open(fp, 'r').read()

old = """            except Exception as e:
                t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_timeout]: {e}")  # Timeout — proceed with partial data
        _fetch_dur = _oiag_t.time() - _fetch_start"""

new = """            except Exception as e:
                _unfinished = sum(1 for f in _futs if not f.done())
                t._wlog(f"⚠️ FALLBACK [oi_watcher/aggr_timeout]: {_unfinished} (of {len(_scan_syms)}) futures unfinished after {_oiag_t.time()-_fetch_start:.0f}s — got {len(_oi_raw)} results")
        _fetch_dur = _oiag_t.time() - _fetch_start"""

assert old in c, 'pattern not found'
c = c.replace(old, new, 1)
open(fp, 'w').write(c)
print('OK: improved timeout logging')
