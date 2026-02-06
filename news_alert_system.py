"""
MCX NEWS SENTIMENT ALERT SYSTEM
Tracks news and recommends CE/PE based on sentiment

Sources:
- Google News RSS (free, real-time)
- NewsAPI (free tier: 100/day)
- BSE Announcements
"""

import feedparser
import requests
import json
import re
from datetime import datetime, timedelta
from collections import Counter

# ========== CONFIGURATION ==========
STOCK_NAME = "MCX"
STOCK_FULL = "Multi Commodity Exchange"

# NewsAPI Key (free tier - 100 requests/day)
# Get yours at: https://newsapi.org/register
NEWSAPI_KEY = ""  # Add your key if you have one

# Google News RSS for MCX
GOOGLE_NEWS_URL = f"https://news.google.com/rss/search?q={STOCK_NAME}+India+stock&hl=en-IN&gl=IN&ceid=IN:en"

# Bullish keywords with weights
BULLISH_WORDS = {
    'profit': 3, 'revenue': 2, 'growth': 3, 'up': 1, 'surge': 4, 'jump': 3,
    'rise': 2, 'gain': 2, 'bullish': 4, 'buy': 3, 'upgrade': 4, 'outperform': 4,
    'record': 3, 'beat': 3, 'exceed': 3, 'positive': 2, 'strong': 2, 'rally': 4,
    'momentum': 2, 'breakout': 4, 'expansion': 3, 'acquisition': 2, 'partnership': 2,
    'dividend': 3, 'bonus': 3, 'approval': 3, 'launch': 2, 'success': 3,
    'robust': 3, 'soar': 4, 'high': 1, 'boost': 3, 'optimistic': 3,
    'highest': 3, 'best': 2, 'improve': 2, 'improvement': 2, 'increase': 2
}

# Bearish keywords with weights
BEARISH_WORDS = {
    'loss': 3, 'decline': 3, 'down': 1, 'fall': 2, 'drop': 2, 'crash': 5,
    'bearish': 4, 'sell': 3, 'downgrade': 4, 'underperform': 4, 'miss': 3,
    'negative': 2, 'weak': 2, 'concern': 2, 'risk': 2, 'warning': 3,
    'investigation': 4, 'fraud': 5, 'scam': 5, 'penalty': 4, 'fine': 3,
    'lawsuit': 4, 'regulation': 2, 'ban': 4, 'resign': 3, 'exit': 2,
    'layoff': 3, 'cut': 2, 'reduce': 2, 'reduction': 2, 'slowdown': 3,
    'worst': 3, 'low': 1, 'lowest': 3, 'trouble': 3, 'crisis': 5,
    'debt': 2, 'default': 5, 'slump': 4, 'plunge': 4, 'tumble': 4
}

def fetch_google_news():
    """Fetch news from Google News RSS"""
    try:
        feed = feedparser.parse(GOOGLE_NEWS_URL)
        news = []
        for entry in feed.entries[:20]:
            # Parse date
            try:
                pub_date = datetime(*entry.published_parsed[:6])
            except:
                pub_date = datetime.now()
            
            news.append({
                'title': entry.title,
                'link': entry.link,
                'source': entry.source.title if hasattr(entry, 'source') else 'Google News',
                'published': pub_date,
                'summary': entry.get('summary', '')[:200]
            })
        return news
    except Exception as e:
        print(f"Error fetching Google News: {e}")
        return []

def fetch_newsapi():
    """Fetch from NewsAPI (if key available)"""
    if not NEWSAPI_KEY:
        return []
    
    try:
        url = f"https://newsapi.org/v2/everything"
        params = {
            'q': f'"{STOCK_NAME}" OR "{STOCK_FULL}"',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 20,
            'apiKey': NEWSAPI_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') != 'ok':
            return []
        
        news = []
        for article in data.get('articles', []):
            news.append({
                'title': article['title'],
                'link': article['url'],
                'source': article['source']['name'],
                'published': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                'summary': article.get('description', '')[:200]
            })
        return news
    except Exception as e:
        print(f"Error fetching NewsAPI: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment of text and return score"""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    bullish_score = 0
    bearish_score = 0
    bullish_matches = []
    bearish_matches = []
    
    for word in words:
        if word in BULLISH_WORDS:
            bullish_score += BULLISH_WORDS[word]
            bullish_matches.append(word)
        if word in BEARISH_WORDS:
            bearish_score += BEARISH_WORDS[word]
            bearish_matches.append(word)
    
    total = bullish_score + bearish_score
    if total == 0:
        return 0, 'NEUTRAL', [], []
    
    # Net sentiment (-1 to +1)
    net_score = (bullish_score - bearish_score) / total
    
    if net_score > 0.2:
        sentiment = 'BULLISH'
    elif net_score < -0.2:
        sentiment = 'BEARISH'
    else:
        sentiment = 'NEUTRAL'
    
    return net_score, sentiment, bullish_matches, bearish_matches

def get_all_news():
    """Fetch news from all sources"""
    all_news = []
    
    print("📡 Fetching news from sources...")
    
    # Google News
    google_news = fetch_google_news()
    all_news.extend(google_news)
    print(f"   Google News: {len(google_news)} articles")
    
    # NewsAPI
    newsapi_news = fetch_newsapi()
    all_news.extend(newsapi_news)
    if NEWSAPI_KEY:
        print(f"   NewsAPI: {len(newsapi_news)} articles")
    
    # Remove duplicates by title
    seen = set()
    unique_news = []
    for item in all_news:
        title_key = item['title'][:50].lower()
        if title_key not in seen:
            seen.add(title_key)
            unique_news.append(item)
    
    # Sort by date (newest first)
    unique_news.sort(key=lambda x: x['published'], reverse=True)
    
    return unique_news

def analyze_news():
    """Main function to analyze news and give trading signal"""
    
    print()
    print("=" * 60)
    print(f"   📰 MCX NEWS SENTIMENT ANALYZER")
    print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    news = get_all_news()
    
    if not news:
        print("❌ No news found! Check your internet connection.")
        return
    
    print(f"\n📰 Found {len(news)} news articles for MCX\n")
    print("-" * 60)
    
    total_bullish = 0
    total_bearish = 0
    recent_news = []  # Last 24 hours
    
    for i, item in enumerate(news[:15], 1):
        title = item['title']
        score, sentiment, bull_words, bear_words = analyze_sentiment(title + " " + item.get('summary', ''))
        
        # Time ago
        time_diff = datetime.now() - item['published'].replace(tzinfo=None)
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}d ago"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}h ago"
        else:
            time_ago = f"{time_diff.seconds // 60}m ago"
        
        # Emoji for sentiment
        if sentiment == 'BULLISH':
            emoji = '🟢'
            total_bullish += abs(score)
        elif sentiment == 'BEARISH':
            emoji = '🔴'
            total_bearish += abs(score)
        else:
            emoji = '⚪'
        
        # Track recent news
        if time_diff.days == 0:
            recent_news.append({'sentiment': sentiment, 'score': score})
        
        print(f"{emoji} [{time_ago}] {sentiment}")
        print(f"   {title[:70]}{'...' if len(title) > 70 else ''}")
        if bull_words:
            print(f"   📈 Keywords: {', '.join(set(bull_words)[:3])}")
        if bear_words:
            print(f"   📉 Keywords: {', '.join(set(bear_words)[:3])}")
        print()
    
    # Calculate overall sentiment
    print("-" * 60)
    print()
    print("📊 SENTIMENT SUMMARY")
    print()
    
    if total_bullish + total_bearish == 0:
        overall = "NEUTRAL"
        signal = "NO CLEAR SIGNAL"
    else:
        net = (total_bullish - total_bearish) / (total_bullish + total_bearish)
        if net > 0.3:
            overall = "BULLISH"
            signal = "BUY CE"
        elif net < -0.3:
            overall = "BEARISH"
            signal = "BUY PE"
        else:
            overall = "MIXED"
            signal = "WAIT FOR CLEAR DIRECTION"
    
    # Recent sentiment (last 24h)
    if recent_news:
        recent_bullish = sum(1 for n in recent_news if n['sentiment'] == 'BULLISH')
        recent_bearish = sum(1 for n in recent_news if n['sentiment'] == 'BEARISH')
        print(f"   Last 24 Hours: {len(recent_news)} articles")
        print(f"   🟢 Bullish: {recent_bullish}  |  🔴 Bearish: {recent_bearish}")
    
    print()
    print("=" * 60)
    print(f"   OVERALL: {overall}")
    print(f"   SIGNAL: {signal}")
    print("=" * 60)
    print()
    
    # Load live data for actionable advice
    try:
        with open('mcx_options_live.json', 'r') as f:
            data = json.load(f)
        
        spot = data['spot']
        chain = {float(k): v for k, v in data['chain'].items()}
        strikes = sorted(chain.keys())
        atm = min(strikes, key=lambda x: abs(x - spot))
        
        ce_prem = chain[atm]['CE']['ltp']
        pe_prem = chain[atm]['PE']['ltp']
        
        print("💡 ACTIONABLE TRADE:")
        print()
        if signal == "BUY CE":
            print(f"   ✅ BUY {int(atm)} CE @ ₹{ce_prem:.2f}")
            print(f"   📍 Entry: ₹{ce_prem:.2f}")
            print(f"   🎯 Target: ₹{ce_prem * 1.15:.2f} (+15%)")
            print(f"   🛑 Stop Loss: ₹{ce_prem * 0.85:.2f} (-15%)")
            print(f"   📊 Lot Size: 625 | Cost: ₹{ce_prem * 625:,.0f}/lot")
        elif signal == "BUY PE":
            print(f"   ✅ BUY {int(atm)} PE @ ₹{pe_prem:.2f}")
            print(f"   📍 Entry: ₹{pe_prem:.2f}")
            print(f"   🎯 Target: ₹{pe_prem * 1.15:.2f} (+15%)")
            print(f"   🛑 Stop Loss: ₹{pe_prem * 0.85:.2f} (-15%)")
            print(f"   📊 Lot Size: 625 | Cost: ₹{pe_prem * 625:,.0f}/lot")
        else:
            print("   ⏸️  No clear direction - wait for news confirmation")
        print()
    except FileNotFoundError:
        print("   Run options dashboard first to get live prices!")
        print()
    
    return {
        'overall_sentiment': overall,
        'signal': signal,
        'bullish_score': total_bullish,
        'bearish_score': total_bearish,
        'news_count': len(news)
    }


if __name__ == "__main__":
    # Install required package if needed
    try:
        import feedparser
    except ImportError:
        print("Installing feedparser...")
        import subprocess
        subprocess.run(['pip', 'install', 'feedparser'], capture_output=True)
        import feedparser
    
    result = analyze_news()
