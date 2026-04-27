"""Fix audit display bug in autonomous_trader.py:
1. Store per-stock audit immediately after scoring (line ~1922)
2. Read from per-stock storage instead of singleton (line ~2540)
"""
filepath = '/home/ubuntu/titan/agentic_trader/autonomous_trader.py'

with open(filepath, 'r') as f:
    content = f.read()

# Fix 1: Add 'audit' to _cycle_decisions storage
old1 = """                    _cycle_decisions[_sym] = {
                        'decision': _dec,
                        'direction': None,
                        'score': _dec.confidence_score,
                        'raw_score': _dec.confidence_score,
                    }"""

new1 = """                    _cycle_decisions[_sym] = {
                        'decision': _dec,
                        'direction': None,
                        'score': _dec.confidence_score,
                        'raw_score': _dec.confidence_score,
                        'audit': getattr(_scorer, '_last_score_audit', ''),
                    }"""

count1 = content.count(old1)
if count1 != 1:
    print(f"ERROR: Fix 1 matched {count1} times (expected 1)")
    exit(1)
content = content.replace(old1, new1)
print("Fix 1 applied: audit stored per-stock in _cycle_decisions")

# Fix 2: Read audit from _cycle_decisions instead of singleton
old2 = "                _score_audit = getattr(_scorer, '_last_score_audit', '')"
new2 = "                _score_audit = _cycle_decisions.get(_sym, {}).get('audit', '')"

count2 = content.count(old2)
if count2 != 1:
    print(f"ERROR: Fix 2 matched {count2} times (expected 1)")
    exit(1)
content = content.replace(old2, new2)
print("Fix 2 applied: audit read from _cycle_decisions per-stock")

with open(filepath, 'w') as f:
    f.write(content)

print("OK: Both fixes applied successfully")
