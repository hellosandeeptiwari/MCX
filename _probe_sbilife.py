import json, urllib.request
d = json.loads(urllib.request.urlopen('http://localhost:5000/api/status').read())
for p in d.get('positions', []):
    if 'SBILIFE' in (p.get('symbol') or ''):
        print(json.dumps(p, indent=2, default=str))
