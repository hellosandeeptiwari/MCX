#!/usr/bin/env python3
"""Manual trigger for pre-market news scan — DIAGNOSTIC mode showing full pipeline."""
import sys, os, re, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable verbose logging to see what's happening inside the scanner
logging.basicConfig(level=logging.INFO, format='%(message)s')

from news_scanner import NewsScanner, _RSS_FEEDS, _FNO_UNIVERSE
from config import EARLYBIRD_D

cfg = {
    'openai_api_key': EARLYBIRD_D.get('openai_api_key', ''),
    'model': EARLYBIRD_D.get('model', 'gpt-4o-mini'),
    'min_confidence': EARLYBIRD_D.get('min_confidence', 60),
}
scanner = NewsScanner(cfg)

# Step 1: Fetch raw headlines (before any filtering)
print('\n' + '='*80)
print('STEP 1: RAW HEADLINE FETCH')
print('='*80)
cutoff = scanner._get_news_cutoff()
print(f'Cutoff time: {cutoff}')
print(f'F&O Universe size: {len(_FNO_UNIVERSE)} stocks')
print(f'RSS Feeds configured: {len(_RSS_FEEDS)}')

raw_headlines = []
seen_titles = set()
feed_stats = {}
for feed in _RSS_FEEDS:
    try:
        resp = scanner._fetch_feed(feed)
        items = scanner._parse_rss(resp.text, feed['name'], cutoff)
        feed_count = 0
        for item in items:
            title_key = re.sub(r'[^a-z0-9]', '', item['title'].lower())[:100]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                item['priority'] = feed['priority']
                raw_headlines.append(item)
                feed_count += 1
        feed_stats[feed['name']] = feed_count
        if feed_count > 0:
            print(f'  {feed["name"]:25s} → {feed_count:3d} headlines (priority={feed["priority"]})')
    except Exception as e:
        feed_stats[feed['name']] = f'FAIL: {str(e)[:50]}'
        print(f'  {feed["name"]:25s} → FAILED: {str(e)[:60]}')

print(f'\nTotal raw headlines (after dedup, after cutoff): {len(raw_headlines)}')

# Step 2: Show what gets filtered by reactive filter
print('\n' + '='*80)
print('STEP 2: REACTIVE FILTER')
print('='*80)
reactive_dropped = []
reactive_kept = []
for h in raw_headlines:
    if scanner._is_reactive_headline(h['title']):
        reactive_dropped.append(h)
    else:
        reactive_kept.append(h)

print(f'Kept: {len(reactive_kept)}, Dropped as reactive: {len(reactive_dropped)}')
if reactive_dropped:
    print('\nDROPPED (reactive/stale):')
    for h in reactive_dropped[:15]:
        print(f'  [{h["source"]:20s}] {h["title"][:100]}')
    if len(reactive_dropped) > 15:
        print(f'  ... and {len(reactive_dropped)-15} more')

# Step 3: Show what gets filtered by market-hours filter
print('\n' + '='*80)
print('STEP 3: MARKET-HOURS FILTER')
print('='*80)
mkt_dropped = [h for h in reactive_kept if scanner._is_during_market_hours(h.get('pub_ts', 0))]
mkt_kept = [h for h in reactive_kept if not scanner._is_during_market_hours(h.get('pub_ts', 0))]
print(f'Kept: {len(mkt_kept)}, Dropped (during market hours): {len(mkt_dropped)}')
if mkt_dropped:
    print('\nDROPPED (market hours):')
    for h in mkt_dropped[:10]:
        from datetime import datetime as dt
        ts_str = dt.fromtimestamp(h['pub_ts']).strftime('%H:%M') if h.get('pub_ts') else '?'
        print(f'  [{ts_str}] [{h["source"]:20s}] {h["title"][:90]}')

# Step 4: Show headlines going to GPT
print('\n' + '='*80)
print(f'STEP 4: HEADLINES GOING TO GPT ({len(mkt_kept)} items)')
print('='*80)
for i, h in enumerate(mkt_kept[:40], 1):
    from datetime import datetime as dt
    ts_str = dt.fromtimestamp(h['pub_ts']).strftime('%Y-%m-%d %H:%M') if h.get('pub_ts') else 'no-date'
    print(f'  {i:2d}. [{h["source"]:20s}] [{ts_str}] {h["title"][:100]}')

# Step 5: Run full scan
print('\n' + '='*80)
print('STEP 5: GPT ANALYSIS + SCORING')
print('='*80)
targets = scanner.scan(force=True)

print(f'\n=== FINAL TARGETS (conf >= {cfg["min_confidence"]}) ===')
if not targets:
    print('No actionable targets found')
else:
    for t in targets:
        print(f'  {t.symbol:15s} {t.sentiment:8s} conf={t.confidence:3d} catalyst={t.catalyst_type:16s} direct={t.directness:8s}')
        print(f'    Headline: {t.headline[:100]}')
        print(f'    Score: {t.score_breakdown}')
        print(f'    Reason: {t.reason}')

all_scanned = scanner.get_all_scanned()
below = [x for x in (all_scanned or []) if x.get('confidence', 0) < cfg['min_confidence']]
if below:
    print(f'\n=== BELOW THRESHOLD ({len(below)} items) ===')
    for item in below:
        print(f'  {item.get("symbol","?"):15s} {item.get("sentiment","?"):8s} conf={item.get("confidence",0):3d} catalyst={item.get("catalyst_type","?"):16s} | {item.get("headline","")[:80]}')
        if item.get('score_breakdown'):
            print(f'    Score: {item["score_breakdown"]}')
