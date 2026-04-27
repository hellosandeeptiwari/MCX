#!/usr/bin/env python3
"""Quick check: what expiry dates does Zerodha list for stock options today?"""
import json
from kiteconnect import KiteConnect

tok = json.load(open("zerodha_token.json"))
k = KiteConnect(api_key=tok["api_key"])
k.set_access_token(tok["access_token"])
insts = k.instruments("NFO")

for stock in ["MOTHERSON", "INOXWIND", "RELIANCE", "NIFTY"]:
    opts = [i for i in insts if i["name"] == stock]
    expiries = sorted(set(str(i["expiry"]) for i in opts))
    print(f"\n{stock} expiries: {expiries}")
    # Show a 26MAR sample
    for i in opts:
        ts = i.get("tradingsymbol", "")
        if "26MAR" in ts:
            print(f"  {ts}  expiry_field={i['expiry']}  type={i['instrument_type']}  strike={i['strike']}")
            break
    # Show nearest non-26MAR sample
    for i in opts:
        ts = i.get("tradingsymbol", "")
        if "26MAR" not in ts and i["instrument_type"] in ("CE", "PE"):
            print(f"  {ts}  expiry_field={i['expiry']}  type={i['instrument_type']}  strike={i['strike']}")
            break
