import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
from kiteconnect import KiteConnect
from datetime import datetime

api_key = os.environ.get('ZERODHA_API_KEY')
api_secret = os.environ.get('ZERODHA_API_SECRET')
request_token = os.environ.get('ZERODHA_REQUEST_TOKEN')

if not request_token:
    print("ERROR: ZERODHA_REQUEST_TOKEN not found in .env")
    print("1. Open: https://kite.zerodha.com/connect/login?v=3&api_key=" + (api_key or ''))
    print("2. Login and copy the request_token from the redirect URL")
    print("3. Add to .env: ZERODHA_REQUEST_TOKEN=<your_token>")
    sys.exit(1)

kite = KiteConnect(api_key=api_key, timeout=15)

data = kite.generate_session(request_token, api_secret=api_secret)
token = data['access_token']
kite.set_access_token(token)

# Save to .env only
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
lines = open(env_path).readlines()
new_lines = []
found = False
for line in lines:
    if line.startswith('ZERODHA_ACCESS_TOKEN='):
        new_lines.append('ZERODHA_ACCESS_TOKEN=' + token + '\n')
        found = True
    else:
        new_lines.append(line)
if not found:
    new_lines.append('ZERODHA_ACCESS_TOKEN=' + token + '\n')
with open(env_path, 'w') as f:
    f.writelines(new_lines)

# Verify connection
profile = kite.profile()
uname = profile.get('user_name', 'N/A')
email = profile.get('email', 'N/A')
broker = profile.get('broker', 'N/A')
print('=' * 50)
print('KITE CONNECTED SUCCESSFULLY')
print('=' * 50)
print('User:   ' + uname)
print('Email:  ' + email)
print('Broker: ' + broker)
print('Token saved to .env')
