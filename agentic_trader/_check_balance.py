"""Quick check: Zerodha live balance"""
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv
load_dotenv()

kite = KiteConnect(api_key=os.environ.get('ZERODHA_API_KEY'))
kite.set_access_token(os.environ.get('ZERODHA_ACCESS_TOKEN'))

m = kite.margins()
eq = m.get('equity', {})
avail = eq.get('available', {})
util = eq.get('utilised', {})

print('=== EQUITY MARGINS ===')
print(f"Live Balance:    Rs {avail.get('live_balance', 0):,.2f}")
print(f"Cash:            Rs {avail.get('cash', 0):,.2f}")
print(f"Opening Balance: Rs {avail.get('opening_balance', 0):,.2f}")
print(f"Collateral:      Rs {avail.get('collateral', 0):,.2f}")
print()
print('=== UTILISED ===')
print(f"Exposure:     Rs {util.get('exposure', 0):,.2f}")
print(f"Span:         Rs {util.get('span', 0):,.2f}")
print(f"Option Prem:  Rs {util.get('option_premium', 0):,.2f}")
print(f"Debits:       Rs {util.get('debits', 0):,.2f}")
print()
print(f"NET AVAILABLE: Rs {eq.get('net', 0):,.2f}")
