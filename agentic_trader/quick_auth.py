"""Quick Kite authentication - run this, login in browser, paste the redirect URL"""
import json, os, re, webbrowser
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()
api_key = os.getenv("ZERODHA_API_KEY")
api_secret = os.getenv("ZERODHA_API_SECRET")

kite = KiteConnect(api_key=api_key)
login_url = kite.login_url()

print(f"\n🔐 Opening Zerodha login in your browser...")
webbrowser.open(login_url)

print(f"\nAfter login, paste the FULL redirect URL here:")
url = input("> ").strip()

# Extract request_token from URL
match = re.search(r'request_token=([^&]+)', url)
if not match:
    # Maybe they pasted just the token
    request_token = url
else:
    request_token = match.group(1)

print(f"\n⏳ Generating access token...")
try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    
    # Save token
    token_path = os.path.join(os.path.dirname(__file__), '..', 'zerodha_token.json')
    with open(token_path, 'w') as f:
        json.dump({'access_token': access_token, 'date': str(__import__('datetime').datetime.now().date())}, f)
    
    # Also update .env
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
        with open(env_path, 'w') as f:
            for line in lines:
                if line.startswith('ZERODHA_ACCESS_TOKEN='):
                    f.write(f'ZERODHA_ACCESS_TOKEN={access_token}\n')
                else:
                    f.write(line)
    
    kite.set_access_token(access_token)
    profile = kite.profile()
    print(f"\n✅ Authentication successful!")
    print(f"   Logged in as: {profile.get('user_name', '?')}")
    print(f"   Token saved for today.")
    print(f"\n   Now run: python autonomous_trader.py --capital 200000")
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    print("   Token may have expired. Run this script again and paste URL immediately after login.")
