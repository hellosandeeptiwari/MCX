import news_scanner
feed = {
    'name': 'BusinessStandard_Markets',
    'url': 'https://www.business-standard.com/rss/markets-106.rss',
    'alt_url': 'https://www.business-standard.com/rss/markets.rss',
}

orig = news_scanner.requests.get

def debug_get(*args, **kwargs):
    print('DEBUG CALL', args, kwargs)
    return orig(*args, **kwargs)

news_scanner.requests.get = debug_get
s = news_scanner.NewsScanner({'llm_model': 'gpt-4o-mini', 'max_targets': 10, 'trade_targets': 5, 'min_confidence': 45})
try:
    resp = s._fetch_feed(feed)
    print('RESP', resp.status_code, resp.url)
except Exception as e:
    print('EXC', type(e).__name__, e)
