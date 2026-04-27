"""Fix: Store audit per-stock instead of reading from singleton scorer."""
import sys

fp = '/home/ubuntu/titan/agentic_trader/watcher_pipeline.py'
content = open(fp, 'r').read()

# FIX 1: Store audit per-stock in scoring loop (around line 151)
old1 = "'raw_score': _dec.confidence_score,\r\n                    }"
new1 = "'raw_score': _dec.confidence_score,\r\n                        'audit': getattr(_scorer, '_last_score_audit', ''),\r\n                    }"
if old1 not in content:
    print('FIX1 pattern not found (trying without \\r)')
    old1 = "'raw_score': _dec.confidence_score,\n                    }"
    new1 = "'raw_score': _dec.confidence_score,\n                        'audit': getattr(_scorer, '_last_score_audit', ''),\n                    }"
assert old1 in content, 'FIX1 pattern still not found'
content = content.replace(old1, new1, 1)
print('FIX1 applied: audit stored per-stock in _cycle_decisions')

# FIX 2: Read audit from per-stock dict instead of singleton (around line 711)
old2 = "_score_audit = getattr(_scorer, '_last_score_audit', '')"
new2 = "_score_audit = _cycle_decisions.get(_sym, {}).get('audit', '')"
assert old2 in content, 'FIX2 pattern not found'
content = content.replace(old2, new2, 1)
print('FIX2 applied: audit read from per-stock _cycle_decisions')

open(fp, 'w').write(content)
print('OK: both fixes applied successfully')
