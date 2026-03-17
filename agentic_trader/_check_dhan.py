import os, sys, json
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from dotenv import load_dotenv
load_dotenv('/home/ubuntu/titan/agentic_trader/.env', override=True)

from kiteconnect import KiteConnect
k = KiteConnect(api_key=os.environ['ZERODHA_API_KEY'])
k.set_access_token(os.environ['ZERODHA_ACCESS_TOKEN'])

insts = k.instruments('NSE')
# Search for LTIM variants
for pattern in ['LTIM', 'MINDTREE', 'LTIMIND']:
    hits = [i for i in insts if pattern in i['tradingsymbol'].upper()]
    if hits:
        for h in hits[:5]:
            print(f"  {h['tradingsymbol']} ({h['name']}) token={h['instrument_token']}")

# Also search NFO for LTIM
nfo = k.instruments('NFO')
ltim_nfo = [i for i in nfo if 'LTIM' in i['tradingsymbol'].upper()][:5]
if ltim_nfo:
    print(f'NFO LTIM: {[(i["tradingsymbol"], i["name"]) for i in ltim_nfo]}')
else:
    print('No LTIM in NFO either')

# Search for any symbol with 'LT' in the name field that mentions mindtree  
name_hits = [i for i in insts if 'MINDTREE' in i.get('name', '').upper() or 'LTIMINDTREE' in i.get('name', '').upper()]
if name_hits:
    print(f'Name search MINDTREE: {[(i["tradingsymbol"], i["name"]) for i in name_hits[:5]]}')

# Also search by company name containing 'LTI'
lti_name = [i for i in insts if i.get('name', '').upper().startswith('LTI')]
if lti_name:
    print(f'Name starts with LTI: {[(i["tradingsymbol"], i["name"]) for i in lti_name[:10]]}')

# Check NFO for any LTI-related futures
lti_nfo = [i for i in nfo if i['tradingsymbol'].startswith('LTI') and i['instrument_type'] == 'FUT']
if lti_nfo:
    print(f'NFO LTI FUT: {[(i["tradingsymbol"], i["name"]) for i in lti_nfo[:5]]}')
else:
    print('No LTI futures in NFO')

# Check what IT stocks ARE in F&O
it_stocks = ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'COFORGE', 'MPHASIS', 'PERSISTENT', 'LTIMINDTREE']
for s in it_stocks:
    nfo_hits = [i for i in nfo if i['tradingsymbol'].startswith(s) and i['instrument_type'] == 'FUT']
    nse_hits = [i for i in insts if i['tradingsymbol'] == s]
    status = 'OK' if nse_hits else 'MISSING'
    fno = 'F&O' if nfo_hits else 'NO-FNO'
    print(f'  {s}: {status} {fno}')
