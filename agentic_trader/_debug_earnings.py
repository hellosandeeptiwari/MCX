"""Debug: Show ALL headlines from RSS feeds, highlight results/earnings related ones."""
import os, sys, logging, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
logging.basicConfig(level=logging.WARNING, format='%(message)s')

from news_scanner import NewsScanner, _RSS_FEEDS
import requests
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta

EARNINGS_KEYWORDS = re.compile(
    r'result|earning|quarter|Q4|Q3|Q1|Q2|FY2[0-9]|profit|revenue|net income|PAT|EBITDA|dividend|board meet|annual',
    re.IGNORECASE
)

print("=" * 80)
print("RAW RSS HEADLINES — checking for earnings/results keywords")
print("=" * 80)

total_items = 0
earnings_items = 0

for feed in _RSS_FEEDS:
    try:
        resp = requests.get(
            feed['url'], timeout=12,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"\n--- {feed['name']} --- FAILED: {e}")
        continue

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', resp.text)
        try:
            root = ET.fromstring(text)
        except:
            print(f"\n--- {feed['name']} --- XML PARSE FAILED")
            continue

    items = list(root.iter('item'))
    feed_earnings = 0
    print(f"\n--- {feed['name']} ({len(items)} items) ---")

    for item in items:
        title_el = item.find('title')
        pub_el = item.find('pubDate')
        title = title_el.text.strip() if title_el is not None and title_el.text else ''
        title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title).strip()
        pub_str = pub_el.text.strip() if pub_el is not None and pub_el.text else ''

        pub_dt_str = ''
        if pub_str:
            try:
                pub_dt = parsedate_to_datetime(pub_str)
                pub_dt_str = pub_dt.strftime('%m/%d %H:%M')
            except:
                pass

        total_items += 1
        if EARNINGS_KEYWORDS.search(title):
            earnings_items += 1
            feed_earnings += 1
            print(f"  🎯 [{pub_dt_str}] {title[:120]}")

    if feed_earnings == 0:
        print(f"  (no earnings/results headlines found)")

print(f"\n{'=' * 80}")
print(f"TOTAL: {total_items} headlines, {earnings_items} earnings-related ({earnings_items*100//max(total_items,1)}%)")
print(f"{'=' * 80}")

# Now check the scanner's cutoff — are we dropping earnings news as "too old"?
scanner = NewsScanner({
    'llm_model': 'gpt-4o-mini', 'max_targets': 10,
    'trade_targets': 5, 'min_confidence': 45
})
cutoff = scanner._get_news_cutoff()
print(f"\nScanner news cutoff: {cutoff}")
print(f"Current time:        {datetime.now()}")
print(f"Lookback window:     {datetime.now() - cutoff}")

# Also check: does _is_during_market_hours drop them?
print(f"\nMarket-hours filter test:")
test_times = [
    datetime(2026, 4, 8, 10, 0),  # 10 AM today
    datetime(2026, 4, 8, 14, 0),  # 2 PM today
    datetime(2026, 4, 8, 16, 0),  # 4 PM today (after market)
    datetime(2026, 4, 8, 20, 0),  # 8 PM today
    datetime(2026, 4, 7, 22, 0),  # 10 PM yesterday
]
for dt in test_times:
    ts = dt.timestamp()
    is_mkt = scanner._is_during_market_hours(ts)
    print(f"  {dt.strftime('%b %d %H:%M')} → {'DROPPED (market hours)' if is_mkt else 'KEPT'}")
