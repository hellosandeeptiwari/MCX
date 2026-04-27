from agentic_trader.news_scanner import NewsScanner
import json
import traceback

scanner = NewsScanner({
    'llm_model': 'gpt-4o-mini',
    'max_targets': 10,
    'trade_targets': 5,
    'min_confidence': 45,
})

print('DEBUG: OPENAI_API_KEY env =', repr(__import__('os').getenv('OPENAI_API_KEY')))
print('DEBUG: client =', type(scanner._client).__name__ if scanner._client else None)

print('DEBUG: running feed fetch')
try:
    headlines = scanner._fetch_all_feeds()
    print('HEADLINES_COUNT:', len(headlines))
    for i, h in enumerate(headlines[:20], start=1):
        print(f'{i}. {h.get("source")} | {h.get("pub_ts")} | {h.get("title")[:120]}')
except Exception as e:
    print('FETCH_ERROR', e)
    traceback.print_exc()

print('\nDEBUG: running analysis on fetched headlines')
try:
    analysis_targets = scanner._analyze_with_llm(headlines)
    print('ANALYSIS_COUNT:', len(analysis_targets))
    for i, t in enumerate(analysis_targets[:20], start=1):
        print(f'{i}. {t.symbol} | {t.catalyst_type} | {t.directness} | {t.confidence} | {t.headline[:120]}')
except Exception as e:
    print('ANALYSIS_ERROR', e)
    traceback.print_exc()

print('\nDEBUG: running scan()')
try:
    targets = scanner.scan(force=True)
    print('SCAN_COUNT:', len(targets))
    print(json.dumps([
        {
            'symbol': t.symbol,
            'sentiment': t.sentiment,
            'confidence': t.confidence,
            'catalyst_type': t.catalyst_type,
            'directness': t.directness,
            'headline': t.headline,
            'source': t.source,
            'pub_time': t.pub_time,
        }
        for t in targets
    ], indent=2))
except Exception as e:
    print('SCAN_ERROR', e)
    traceback.print_exc()
