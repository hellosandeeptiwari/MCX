"""Quick test of new RSS feeds + scanner pipeline."""
import os, sys, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv('.env')
logging.basicConfig(level=logging.INFO, format='%(message)s')

from news_scanner import NewsScanner, _RSS_FEEDS
import requests, time

# Step 1: Test each RSS feed individually
print("=" * 70)
print("STEP 1: Testing individual RSS feeds")
print("=" * 70)
for feed in _RSS_FEEDS:
    t0 = time.time()
    try:
        resp = requests.get(feed['url'], timeout=10,
                           headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        elapsed = time.time() - t0
        items = resp.text.count('<item>')
        status = f"✅ {resp.status_code} | {items} items | {elapsed:.1f}s | {len(resp.text)//1024}KB"
    except Exception as e:
        status = f"❌ FAILED: {e}"
    print(f"  {feed['name']:25s} P{feed['priority']}  {status}")

# Step 2: Run full scan pipeline
print(f"\n{'=' * 70}")
print("STEP 2: Full scan pipeline (RSS → GPT → scoring)")
print("=" * 70)
scanner = NewsScanner({
    'llm_model': 'gpt-4o-mini',
    'max_targets': 10,
    'trade_targets': 5,
    'min_confidence': 45,
})
results = scanner.scan(force=True)
all_scanned = scanner.get_all_scanned()

print(f"\n{'=' * 70}")
print(f"RESULTS: {len(all_scanned)} scanned, {len(results)} tradeable (DIRECT-only filter active)")
print(f"{'=' * 70}")
for i, r in enumerate(all_scanned):
    arrow = '🟢' if r['sentiment'] == 'BULLISH' else '🔴'
    tag = '🎯 TRADE' if r.get('tradeable') else '📋 watch'
    print(f"  {i+1}. {arrow} {r['symbol']:15s} {r['sentiment']:8s} conf={r['confidence']:3d}  [{tag}]")
    print(f"     type={r.get('catalyst_type','?')}/{r.get('directness','?')}  pub={r.get('pub_time','?')}  src={r.get('source','')}")
    print(f"     scoring: {r.get('score_breakdown','')}")
    print(f"     headline: {r['headline'][:120]}")
    print(f"     reason: {r['reason'][:120]}")
    print()

if not all_scanned:
    print("  (No actionable targets found — may be normal if no overnight news)")
