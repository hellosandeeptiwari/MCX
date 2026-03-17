import json, urllib.request
data = json.load(urllib.request.urlopen('http://localhost:5000/api/status'))
positions = data.get('positions', [])
for p in positions:
    sym = p.get('symbol', '') or p.get('option_symbol', '')
    underlying = p.get('underlying', '')
    if 'ANGEL' in sym.upper() or 'ANGEL' in underlying.upper():
        print(f"symbol={sym}")
        print(f"option_symbol={p.get('option_symbol','')}")
        print(f"underlying={underlying}")
        print(f"all keys: {list(p.keys())}")
