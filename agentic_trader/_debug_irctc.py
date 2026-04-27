#!/usr/bin/env python3
"""Debug IRCTC and SYNGENE."""
import sys, requests, time
sys.path.insert(0, '/home/ubuntu/titan/agentic_trader')
from dhan_oi_fetcher import DhanOIFetcher, DHAN_SCRIP_MAP
f = DhanOIFetcher()
for sym in ['IRCTC', 'SYNGENE']:
    r = DHAN_SCRIP_MAP[sym]
    print(f"{sym}: scrip_id={r['scrip_id']}  segment={r['segment']}")
    time.sleep(3.5)
    payload = {'UnderlyingScrip': r['scrip_id'], 'UnderlyingSeg': r['segment']}
    resp = requests.post(f.EXPIRY_URL, headers=f._headers(), json=payload, timeout=15)
    print(f"  Expiry: status={resp.status_code}  data={resp.text[:300]}")
