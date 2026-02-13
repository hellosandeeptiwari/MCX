"""Quick Kite connection check"""
import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# 1. Check credentials
api_key = os.environ.get('ZERODHA_API_KEY', '')
api_secret = os.environ.get('ZERODHA_API_SECRET', '')
env_token = os.environ.get('ZERODHA_ACCESS_TOKEN', '')

print("=" * 50)
print("KITE CONNECTION CHECK")
print("=" * 50)
print(f"API Key:    {'SET (' + api_key[:8] + '...)' if api_key else 'NOT SET'}")
print(f"API Secret: {'SET' if api_secret else 'NOT SET'}")
print(f"Env Token:  {'SET (' + env_token[:10] + '...)' if env_token else 'NOT SET'}")

# 2. Check saved token file
token_path = os.path.join(os.path.dirname(__file__), '..', 'zerodha_token.json')
try:
    with open(token_path, 'r') as f:
        data = json.load(f)
    print(f"Token File: date={data.get('date')}, token={data.get('access_token','')[:10]}...")
except Exception as e:
    print(f"Token File: {e}")

# 3. Try connecting
from kiteconnect import KiteConnect
kite = KiteConnect(api_key=api_key, timeout=15)

# Try env token first
if env_token:
    kite.set_access_token(env_token)
    try:
        profile = kite.profile()
        print(f"\n✅ CONNECTED via .env token!")
        print(f"   User: {profile.get('user_name', 'N/A')}")
        print(f"   Email: {profile.get('email', 'N/A')}")
        print(f"   Broker: {profile.get('broker', 'N/A')}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ .env token FAILED: {e}")

# Try saved token file
try:
    with open(token_path, 'r') as f:
        data = json.load(f)
    file_token = data.get('access_token', '')
    if file_token:
        kite.set_access_token(file_token)
        try:
            profile = kite.profile()
            print(f"\n✅ CONNECTED via saved token file!")
            print(f"   User: {profile.get('user_name', 'N/A')}")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Saved token FAILED: {e}")
except:
    pass

# Neither worked
print(f"\n⚠️ No valid token. You need to authenticate.")
print(f"Login URL: {kite.login_url()}")
print(f"\nOpen the URL above, login, copy the request_token from redirect URL,")
print(f"then run: python autonomous_trader.py --capital 200000")
print(f"It will prompt you to paste the request_token.")
