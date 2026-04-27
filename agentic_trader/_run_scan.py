"""Run live news scan and show results."""
from news_scanner import NewsScanner

s = NewsScanner({'llm_model': 'gpt-4o-mini', 'max_targets': 10, 'trade_targets': 5})
results = s.scan(force=True)
allsc = s.get_all_scanned()
cutoff = s._get_news_cutoff()

print(f'\nCutoff: {cutoff.strftime("%Y-%m-%d %H:%M")} (~{s._get_lookback_hours()}h)')
print(f'Scanned: {len(allsc)} | Tradeable: {len(results)}')
print()
print('ALL SCANNED:')
for i, t in enumerate(allsc, 1):
    flag = 'TRADE' if t.get('tradeable') else 'WATCH'
    print(f'  {i}. [{flag}] {t["symbol"]:15s} {t["sentiment"]:8s} conf={t["confidence"]}  {t["headline"][:70]}')
print()
print('TOP 5 TRADEABLE:')
for i, t in enumerate(results, 1):
    print(f'  {i}. {t.symbol:15s} {t.sentiment:8s} conf={t.confidence}  {t.headline[:70]}')
