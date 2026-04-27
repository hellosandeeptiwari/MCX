"""Patch EC2 autonomous_trader.py: add startup regression guard"""
p = 'autonomous_trader.py'
with open(p, encoding='utf-8') as f:
    content = f.read()

old = '        # OI Watcher Engine \u2014 extracted module with 13-factor + anchor gate\n        self._oi_engine = OIWatcherEngine(self)'

new = """        # OI Watcher Engine \u2014 extracted module with 13-factor + anchor gate
        # GUARD: If this import or init is missing, inline OI code has regressed
        from oi_watcher_engine import _ENGINE_LOADED
        assert _ENGINE_LOADED, "oi_watcher_engine not loaded - OI_WATCHER regression detected!"
        self._oi_engine = OIWatcherEngine(self)
        assert hasattr(self._oi_engine, 'run_watcher_scan'), "OIWatcherEngine missing run_watcher_scan!"
        assert hasattr(self._oi_engine, 'aggressive_buildup_scan'), "OIWatcherEngine missing aggressive_buildup_scan!\""""

if old in content:
    content = content.replace(old, new)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    print('Patched startup guard successfully')
else:
    print('Guard already present or old text not found')
    # Debug
    for i, line in enumerate(content.splitlines()):
        if 'OI Watcher Engine' in line:
            print(f'  L{i+1}: {line.rstrip()[:80]}')
