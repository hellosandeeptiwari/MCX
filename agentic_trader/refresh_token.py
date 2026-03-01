"""
Kite Token Refresher — run this every morning before launching the bot.

Usage:
  python refresh_token.py <request_token>

Example:
  python refresh_token.py fiXOzOd6LVx2Z4C1oamoLLW4qPmVugwv
"""
import sys
import json
import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

def refresh_token(request_token: str):
    from kiteconnect import KiteConnect
    
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    
    if not api_key or not api_secret:
        print("ERROR: ZERODHA_API_KEY or ZERODHA_API_SECRET not found in .env")
        sys.exit(1)
    
    # Generate session
    kite = KiteConnect(api_key=api_key)
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as e:
        print(f"ERROR: Failed to generate session: {e}")
        sys.exit(1)
    
    access_token = data["access_token"]
    today = date.today().isoformat()
    
    # 1. Update .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        lines = open(env_path, "r").readlines()
        new_lines = []
        token_updated = False
        for line in lines:
            if line.startswith("ZERODHA_ACCESS_TOKEN="):
                new_lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")
                token_updated = True
            else:
                new_lines.append(line)
        if not token_updated:
            new_lines.append(f"ZERODHA_ACCESS_TOKEN={access_token}\n")
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        print(f"[OK] Saved token to .env (date: {today})")
    
    # 2. Verify login
    kite.set_access_token(access_token)
    try:
        profile = kite.profile()
        user_name = profile.get("user_name", "Unknown")
        user_id = profile.get("user_id", "Unknown")
        print(f"[OK] Logged in as: {user_name} ({user_id})")
        print(f"\nToken: {access_token}")
        print("\nReady to launch: python autonomous_trader.py --capital 500000")
    except Exception as e:
        print(f"WARNING: Token saved but verification failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check if token URL was pasted instead
        print("Usage: python refresh_token.py <request_token>")
        print("       python refresh_token.py <full_callback_url>")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    # Extract request_token from URL if full URL was pasted
    if "request_token=" in arg:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(arg)
        params = parse_qs(parsed.query)
        request_token = params.get("request_token", [None])[0]
        if not request_token:
            print("ERROR: Could not extract request_token from URL")
            sys.exit(1)
    else:
        request_token = arg
    
    refresh_token(request_token)
