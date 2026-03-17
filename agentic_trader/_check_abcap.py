import sys, os
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
os.chdir('/home/ubuntu/titan/agentic_trader')
from _auth_kite import get_kite
kite = get_kite()
nfo = kite.instruments('NFO')
abcap = [i for i in nfo if 'ABCAP' in i.get('name','').upper() or 'BIRLA' in i.get('name','').upper()]
print(f"ABCAP matches: {len(abcap)}")
for a in abcap[:5]:
    print(f"  name={a.get('name')} tsym={a.get('tradingsymbol')} type={a.get('instrument_type')} seg={a.get('segment')}")

print("---")
fut_names = sorted(set(i['name'] for i in nfo if i.get('instrument_type')=='FUT' and i.get('segment')=='NFO-FUT'))
print(f"Total FUT names: {len(fut_names)}")
# Check a few names around ABC
for n in fut_names:
    if n.startswith('A'):
        print(f"  {n}")
