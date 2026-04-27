"""Quick script to show raw headlines with timestamps and sources."""
from news_scanner import NewsScanner
from datetime import datetime, timedelta

s = NewsScanner({'llm_model': 'gpt-4o-mini', 'max_targets': 10, 'trade_targets': 5})
cutoff = s._get_news_cutoff()
headlines = s._fetch_all_feeds()

print()
print('=' * 100)
print(f'HEADLINES FETCHED: {len(headlines)} (after all filters)')
print(f'Cutoff: {cutoff.strftime("%Y-%m-%d %H:%M")} (post-market only, ~{s._get_lookback_hours()}h)')
print(f'Rule: ONLY news after prev market close (15:30). Market-hours news is discarded.')
print('=' * 100)
for i, h in enumerate(headlines[:50], 1):
    ts = ''
    if h.get('pub_ts') and h['pub_ts'] > 0:
        ts = datetime.fromtimestamp(h['pub_ts']).strftime('%Y-%m-%d %H:%M')
    src = h.get('source', '?')
    title = h.get('title', '')[:90]
    print(f'{i:3d}. [{ts:16s}] [{src:22s}] {title}')

print()
sources = set(h['source'] for h in headlines)
for src in sorted(sources):
    count = sum(1 for h in headlines if h['source'] == src)
    print(f'  {src:25s} → {count} headlines')
print()

# Also show what was FILTERED as reactive
print('--- REACTIVE HEADLINES (filtered out) ---')
from news_scanner import _RSS_FEEDS
import requests, re
from email.utils import parsedate_to_datetime
cutoff = datetime.now() - timedelta(hours=s._get_lookback_hours())
all_raw = []
for feed in _RSS_FEEDS:
    try:
        import xml.etree.ElementTree as ET
        resp = requests.get(feed['url'], timeout=10, headers={'User-Agent': 'TitanBot/1.0'})
        items = s._parse_rss(resp.text, feed['name'], cutoff)
        all_raw.extend(items)
    except:
        pass

reactive_count = 0
for h in all_raw:
    if s._is_reactive_headline(h['title']):
        ts = ''
        if h.get('pub_ts') and h['pub_ts'] > 0:
            ts = datetime.fromtimestamp(h['pub_ts']).strftime('%Y-%m-%d %H:%M')
        print(f'  ❌ [{ts:16s}] [{h["source"]:22s}] {h["title"][:90]}')
        reactive_count += 1
print(f'\nTotal reactive filtered: {reactive_count}')
