import urllib.request, json
d = json.loads(urllib.request.urlopen('http://localhost:8090/api/pnl_live', timeout=10).read())
print(f"total_unreal={d.get('total_unrealized')}  realized={d.get('realized_pnl')}  net={d.get('net_pnl')}")
for k, v in (d.get('positions') or {}).items():
    print(f"  {k:<45} ltp={v.get('ltp'):<8} upnl={v.get('upnl'):>+10}  pct={v.get('pct'):>+6.2f}%")
