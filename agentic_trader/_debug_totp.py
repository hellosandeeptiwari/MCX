#!/usr/bin/env python3
"""Debug TOTP generation for DhanHQ."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)

import pyotp
import requests

secret = os.environ.get('DHAN_TOTP_SECRET', '')
pin = os.environ.get('DHAN_PIN', '')
client_id = os.environ.get('DHAN_CLIENT_ID', '')

print(f"Secret len: {len(secret)}, chars: {secret[:4]}...{secret[-4:]}")
print(f"PIN len: {len(pin)}, value: {pin}")
print(f"Client ID: {client_id}")

totp = pyotp.TOTP(secret)
code = totp.now()
print(f"Generated TOTP code: {code}")

# Try the API
url = f"https://auth.dhan.co/app/generateAccessToken?dhanClientId={client_id}&pin={pin}&totp={code}"
print(f"\nCalling: {url[:80]}...")
resp = requests.post(url, timeout=15)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:500]}")
