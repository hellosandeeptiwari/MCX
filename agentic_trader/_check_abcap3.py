import os
os.chdir('/home/ubuntu/titan/agentic_trader')

# Load env
from dotenv import load_dotenv
load_dotenv('.env')

from kiteconnect import KiteConnect
kite = KiteConnect(api_key=os.environ['ZERODHA_API_KEY'])
kite.set_access_token(os.environ['ZERODHA_ACCESS_TOKEN'])

nfo = kite.instruments('NFO')
print(f"Total NFO instruments: {len(nfo)}")

# Check for ABCAPITAL
abcap = [i for i in nfo if 'ABCAP' in (i.get('name','') + i.get('tradingsymbol','')).upper()]
print(f"\nABCAPITAL matches: {len(abcap)}")
for a in abcap[:5]:
    print(f"  name={a['name']} tsym={a['tradingsymbol']} type={a['instrument_type']} seg={a['segment']}")

# All FUT names starting with A
fut_names = sorted(set(i['name'] for i in nfo if i.get('instrument_type')=='FUT' and i.get('segment')=='NFO-FUT'))
print(f"\nTotal F&O stocks (FUT): {len(fut_names)}")
print("\nAll FUT names starting with A:")
for n in fut_names:
    if n.startswith('A'):
        print(f"  {n}")

# Also check if in TIER_1 or TIER_2
from config import TIER_1_OPTIONS, TIER_2_OPTIONS
all_tiers = TIER_1_OPTIONS + TIER_2_OPTIONS
abcap_in_tiers = [s for s in all_tiers if 'ABCAP' in s.upper()]
print(f"\nABCAPITAL in TIER_1/TIER_2: {abcap_in_tiers}")
