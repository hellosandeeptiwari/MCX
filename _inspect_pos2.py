import json
d = json.load(open("/tmp/_s.json"))
keys = ("symbol","underlying","exchange","is_option","option_type","strike","quantity","lots","avg_price","entry_price")
for p in d.get("positions", []):
    print(json.dumps({k: p.get(k) for k in keys}))
