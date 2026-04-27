import json
with open(r'C:\Users\SandeepTiwari\MCX\pyright_json.json', 'r', encoding='utf-16') as f:
    data = json.load(f)
diags = data.get('generalDiagnostics', [])
print(f"Total diagnostics: {len(diags)}")
print()
# Group by rule
from collections import Counter
rules = Counter(d.get('rule', 'unknown') for d in diags)
print("By rule:")
for rule, count in rules.most_common():
    print(f"  {rule}: {count}")
print()
# Show first 100 with line numbers
for d in diags[:100]:
    line = d['range']['start']['line'] + 1
    sev = d.get('severity', '?')
    rule = d.get('rule', '')
    msg = d.get('message', '')[:150]
    print(f"  L{line} [{sev}] ({rule}) {msg}")
