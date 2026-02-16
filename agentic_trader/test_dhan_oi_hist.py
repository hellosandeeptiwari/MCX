"""Test DhanHQ intraday futures OI and daily for backfill feasibility."""
import requests, pandas as pd
from dhan_risk_tools import DhanRiskTools

drt = DhanRiskTools()

# Test 1: SBIN Feb futures INTRADAY 5-min with OI
print("=== SBIN Feb FUT (59466) INTRADAY 5min with OI ===")
body = {
    "securityId": "59466", "exchangeSegment": "NSE_FNO", "instrument": "FUTSTK",
    "interval": "5", "oi": True,
    "fromDate": "2026-02-10 09:15:00", "toDate": "2026-02-12 15:30:00"
}
resp = requests.post("https://api.dhan.co/v2/charts/intraday",
                     headers=drt._headers, json=body, timeout=10)
if resp.status_code == 200:
    d = resp.json()
    if d.get("timestamp"):
        df = pd.DataFrame({
            "date": pd.to_datetime(d["timestamp"], unit="s"),
            "close": d["close"],
            "volume": d["volume"],
            "oi": d.get("open_interest", [0]*len(d["timestamp"]))
        })
        print(f"  rows={len(df)}")
        print(df.head(5).to_string())
        print("  ...")
        print(df.tail(5).to_string())
        print(f"\n  OI range: {df['oi'].min():,.0f} — {df['oi'].max():,.0f}")
        print(f"  OI non-zero: {(df['oi'] > 0).sum()}/{len(df)}")
    else:
        print(f"  No data: {str(d)[:200]}")
else:
    print(f"  {resp.status_code}: {resp.text[:200]}")

# Test 2: SBIN DAILY with OI for longer lookback
print("\n=== SBIN Feb FUT (59466) DAILY 6 months with OI ===")
body2 = {
    "securityId": "59466", "exchangeSegment": "NSE_FNO", "instrument": "FUTSTK",
    "oi": True, "fromDate": "2025-08-01", "toDate": "2026-02-15"
}
resp2 = requests.post("https://api.dhan.co/v2/charts/historical",
                      headers=drt._headers, json=body2, timeout=10)
if resp2.status_code == 200:
    d = resp2.json()
    if d.get("timestamp"):
        df2 = pd.DataFrame({
            "date": pd.to_datetime(d["timestamp"], unit="s"),
            "close": d["close"],
            "volume": d["volume"],
            "oi": d.get("open_interest", [0]*len(d["timestamp"]))
        })
        print(f"  rows={len(df2)}")
        print(df2.head(3).to_string())
        print("  ...")
        print(df2.tail(3).to_string())
        print(f"\n  OI range: {df2['oi'].min():,.0f} — {df2['oi'].max():,.0f}")
    else:
        print(f"  No data: {str(d)[:200]}")
else:
    print(f"  {resp2.status_code}: {resp2.text[:200]}")

# Test 3: Equity spot price for same period (to compute basis)
print("\n=== SBIN EQUITY (3045) DAILY for basis calc ===")
body3 = {
    "securityId": "3045", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
    "fromDate": "2025-08-01", "toDate": "2026-02-15"
}
resp3 = requests.post("https://api.dhan.co/v2/charts/historical",
                      headers=drt._headers, json=body3, timeout=10)
if resp3.status_code == 200:
    d = resp3.json()
    if d.get("timestamp"):
        df3 = pd.DataFrame({
            "date": pd.to_datetime(d["timestamp"], unit="s"),
            "close": d["close"]
        })
        print(f"  rows={len(df3)}")
        print(df3.tail(3).to_string())
