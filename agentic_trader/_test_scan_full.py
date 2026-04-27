"""Show ALL GPT picks including SECTOR/INDIRECT ones that get filtered."""
import os, sys, logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
logging.basicConfig(level=logging.INFO, format='%(message)s')
from news_scanner import NewsScanner

scanner = NewsScanner({
    'llm_model': 'gpt-4o-mini',
    'max_targets': 10,
    'trade_targets': 5,
    'min_confidence': 45
})

results = scanner.scan(force=True)
all_targets = scanner.get_all_scanned()

print(f"\n{'='*70}")
print(f"FINAL OUTPUT: {len(all_targets)} targets passed ALL filters")
print(f"{'='*70}")
for r in all_targets:
    sym = r['symbol']
    conf = r['confidence']
    cat = r.get('catalyst_type', '?')
    dr = r.get('directness', '?')
    sent = r['sentiment']
    hl = r['headline'][:100]
    print(f"  {sym:15s} conf={conf:3d}  {cat}/{dr}  [{sent}]")
    print(f"    {hl}")
    print()

if not all_targets:
    print("  (Zero targets — all dropped by DIRECT-only filter or GPT returned none)")
    print("  This is normal at 9 PM. Morning 8:50 AM scan with fresh overnight news will get more.")
