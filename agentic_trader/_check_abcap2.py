import json, os
os.chdir('/home/ubuntu/titan/agentic_trader')

# Read the token file
token_data = json.load(open('zerodha_token.json'))
access_token = token_data.get('access_token', '')
api_key = token_data.get('api_key', '')

from kiteconnect import KiteConnect
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

nfo = kite.instruments('NFO')
print(f"Total NFO instruments: {len(nfo)}")

# Check for ABCAPITAL
abcap = [i for i in nfo if 'ABCAP' in i.get('name','').upper() or 'BIRLA' in i.get('tradingsymbol','').upper()]
print(f"\nABCAPITAL matches: {len(abcap)}")
for a in abcap[:5]:
    print(f"  name={a.get('name')} tsym={a.get('tradingsymbol')} type={a.get('instrument_type')} seg={a.get('segment')}")

# All FUT names starting with A
fut_names = sorted(set(i['name'] for i in nfo if i.get('instrument_type')=='FUT' and i.get('segment')=='NFO-FUT'))
print(f"\nTotal F&O stocks (FUT): {len(fut_names)}")
print("\nAll names starting with A:")
for n in fut_names:
    if n.startswith('A'):
        print(f"  {n}")
