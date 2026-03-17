import json, urllib.request
data = json.load(urllib.request.urlopen('http://localhost:5000/api/exits'))
for k, v in data.items():
    trail = v.get('trailing_active', 'MISSING')
    be = v.get('breakeven_applied', 'MISSING')
    sl = v.get('current_sl', '?')
    init_sl = v.get('initial_sl', '?')
    print(f"{k:45s} trail={trail!s:8s} be={be!s:8s} sl={sl} init={init_sl}")
