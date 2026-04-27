#!/usr/bin/env python3
"""Quick test of intraday news scan."""
import logging
logging.basicConfig(level=logging.INFO)

from news_scanner import NewsScanner
from datetime import datetime

ns = NewsScanner()
cutoff = datetime.now().replace(hour=9, minute=15, second=0)
headlines = ns._fetch_intraday_headlines(cutoff)
print(f"Headlines since 09:15: {len(headlines)}")
for h in (headlines or [])[:10]:
    title = h.get('title', '')[:90]
    print(f"  {title}")

print("---")
if headlines:
    relevant = ns._count_relevant_headlines(headlines)
    print(f"Relevant: {relevant}")

print("---")
targets = ns.intraday_scan()
print(f"Targets: {len(targets) if targets else 0}")
for t in (targets or []):
    print(f"  {t.symbol} {t.sentiment} conf={t.confidence} {t.reason[:80]}")
