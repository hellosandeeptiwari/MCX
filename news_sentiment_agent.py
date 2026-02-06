"""
MCX News Sentiment Agent
=========================
Scrapes financial news from multiple sources and uses ChatGPT to analyze sentiment.
Adds weighted sentiment score to trading decisions.

Sources:
- CNBC TV18
- Moneycontrol
- Economic Times
- Google News
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta
import re
import time

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# User agent for web scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
}

# Shorter timeout for unreliable sites
TIMEOUT_SHORT = 8
TIMEOUT_LONG = 15


class NewsSentimentAgent:
    def __init__(self, openai_api_key=None):
        self.api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            print("✅ ChatGPT API initialized for news analysis")
        else:
            print("⚠️ ChatGPT API not available")
        
        self.news_cache = []
        self.sentiment_cache = None
        self.cache_time = None
    
    def scrape_google_news(self, query="MCX India stock"):
        """Scrape Google News for MCX related articles"""
        print(f"  📰 Fetching Google News for '{query}'...")
        
        try:
            url = f"https://news.google.com/search?q={query.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            # Find article elements
            for article in soup.find_all('article')[:10]:
                try:
                    # Get title
                    title_elem = article.find('a', class_='JtKRv')
                    if not title_elem:
                        title_elem = article.find('a')
                    
                    title = title_elem.get_text(strip=True) if title_elem else None
                    
                    # Get source
                    source_elem = article.find('div', class_='vr1PYe')
                    source = source_elem.get_text(strip=True) if source_elem else 'Unknown'
                    
                    # Get time
                    time_elem = article.find('time')
                    pub_time = time_elem.get('datetime', '') if time_elem else ''
                    
                    if title and len(title) > 20:
                        articles.append({
                            'title': title,
                            'source': source,
                            'time': pub_time,
                            'origin': 'Google News'
                        })
                except:
                    continue
            
            print(f"    Found {len(articles)} articles from Google News")
            return articles
            
        except Exception as e:
            print(f"    ❌ Google News error: {e}")
            return []
    
    def scrape_moneycontrol(self):
        """Scrape Moneycontrol for MCX news"""
        print("  📰 Fetching Moneycontrol news...")
        
        articles = []
        
        try:
            # Try multiple endpoints
            urls = [
                "https://www.moneycontrol.com/news/business/stocks/",
                "https://www.moneycontrol.com/news/business/markets/",
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SHORT)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find all headlines
                        for item in soup.find_all(['h2', 'h3', 'a'], limit=50):
                            text = item.get_text(strip=True)
                            if text and len(text) > 25:
                                # Look for MCX or commodity related
                                keywords = ['mcx', 'commodity', 'gold', 'silver', 'crude', 'exchange']
                                if any(kw in text.lower() for kw in keywords):
                                    articles.append({
                                        'title': text[:150],
                                        'source': 'Moneycontrol',
                                        'time': '',
                                        'origin': 'Moneycontrol'
                                    })
                except:
                    continue
            
            print(f"    Found {len(articles)} articles from Moneycontrol")
            return articles
            
        except Exception as e:
            print(f"    ❌ Moneycontrol error: {e}")
            return []
    
    def scrape_economic_times(self):
        """Scrape Economic Times for MCX news"""
        print("  📰 Fetching Economic Times news...")
        
        try:
            url = "https://economictimes.indiatimes.com/multi-commodity-exchange-of-india-ltd/stocks/companyid-42aborr.cms"
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                # Try search
                url = "https://economictimes.indiatimes.com/topic/mcx"
                response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            # Find news items
            for item in soup.find_all(['div', 'li'], class_=re.compile('news|story|article'))[:10]:
                try:
                    link = item.find('a')
                    if link:
                        title = link.get_text(strip=True)
                        if title and len(title) > 20:
                            articles.append({
                                'title': title,
                                'source': 'Economic Times',
                                'time': '',
                                'origin': 'Economic Times'
                            })
                except:
                    continue
            
            print(f"    Found {len(articles)} articles from Economic Times")
            return articles
            
        except Exception as e:
            print(f"    ❌ Economic Times error: {e}")
            return []
    
    def scrape_cnbc_tv18(self):
        """Scrape CNBC TV18 for MCX news"""
        print("  📰 Fetching CNBC TV18 news...")
        
        try:
            # CNBC TV18 search
            url = "https://www.cnbctv18.com/search/?q=MCX"
            response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                # Try market news
                url = "https://www.cnbctv18.com/market/"
                response = requests.get(url, headers=HEADERS, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            # Find news items
            for item in soup.find_all(['div', 'article', 'li'], class_=re.compile('news|story|article|item'))[:15]:
                try:
                    link = item.find('a')
                    title_elem = item.find(['h2', 'h3', 'h4', 'span'])
                    
                    title = None
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    elif link:
                        title = link.get_text(strip=True)
                    
                    if title and len(title) > 20:
                        # Check if related to MCX or commodities
                        keywords = ['mcx', 'commodity', 'gold', 'silver', 'crude', 'exchange', 'trading', 'market']
                        if any(kw in title.lower() for kw in keywords):
                            articles.append({
                                'title': title,
                                'source': 'CNBC TV18',
                                'time': '',
                                'origin': 'CNBC TV18'
                            })
                except:
                    continue
            
            print(f"    Found {len(articles)} articles from CNBC TV18")
            return articles
            
        except Exception as e:
            print(f"    ❌ CNBC TV18 error: {e}")
            return []
    
    def scrape_livemint(self):
        """Scrape LiveMint for MCX news"""
        print("  📰 Fetching LiveMint news...")
        
        try:
            url = "https://www.livemint.com/market/stock-market-news"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SHORT)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = []
            
            for item in soup.find_all(['h2', 'h3', 'a'], limit=50):
                try:
                    title = item.get_text(strip=True)
                    if title and len(title) > 25:
                        keywords = ['mcx', 'commodity', 'gold', 'silver', 'crude', 'exchange', 'trading']
                        if any(kw in title.lower() for kw in keywords):
                            articles.append({
                                'title': title[:150],
                                'source': 'LiveMint',
                                'time': '',
                                'origin': 'LiveMint'
                            })
                except:
                    continue
            
            print(f"    Found {len(articles)} articles from LiveMint")
            return articles
            
        except Exception as e:
            print(f"    ❌ LiveMint error: {e}")
            return []
    
    def scrape_yahoo_finance(self):
        """Scrape Yahoo Finance for MCX news - more reliable"""
        print("  📰 Fetching Yahoo Finance news...")
        
        try:
            url = "https://finance.yahoo.com/quote/MCX.NS/news/"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_LONG)
            
            articles = []
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.find_all(['h3', 'a'], limit=30):
                    title = item.get_text(strip=True)
                    if title and len(title) > 25:
                        articles.append({
                            'title': title[:150],
                            'source': 'Yahoo Finance',
                            'time': '',
                            'origin': 'Yahoo Finance'
                        })
            
            print(f"    Found {len(articles)} articles from Yahoo Finance")
            return articles
            
        except Exception as e:
            print(f"    ❌ Yahoo Finance error: {e}")
            return []
    
    def scrape_tradingview(self):
        """Get TradingView ideas/news for MCX"""
        print("  📰 Fetching TradingView insights...")
        
        try:
            url = "https://www.tradingview.com/symbols/NSE-MCX/"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SHORT)
            
            articles = []
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.find_all(['div', 'span', 'a'], class_=re.compile('title|idea|content'))[:20]:
                    title = item.get_text(strip=True)
                    if title and len(title) > 20 and len(title) < 200:
                        articles.append({
                            'title': title,
                            'source': 'TradingView',
                            'time': '',
                            'origin': 'TradingView'
                        })
            
            print(f"    Found {len(articles)} insights from TradingView")
            return articles
            
        except Exception as e:
            print(f"    ❌ TradingView error: {e}")
            return []
    
    def scrape_investing_com(self):
        """Scrape Investing.com for MCX news"""
        print("  📰 Fetching Investing.com news...")
        
        try:
            url = "https://www.investing.com/equities/multi-comm-exchange-india-news"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_LONG)
            
            articles = []
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.find_all(['a', 'article'], class_=re.compile('title|article'))[:15]:
                    title = item.get_text(strip=True)
                    if title and len(title) > 25:
                        articles.append({
                            'title': title[:150],
                            'source': 'Investing.com',
                            'time': '',
                            'origin': 'Investing.com'
                        })
            
            print(f"    Found {len(articles)} articles from Investing.com")
            return articles
            
        except Exception as e:
            print(f"    ❌ Investing.com error: {e}")
            return []
    
    def scrape_bse_announcements(self):
        """Get BSE corporate announcements for MCX"""
        print("  📰 Fetching BSE announcements...")
        
        try:
            # BSE scrip code for MCX is 534091
            url = "https://www.bseindia.com/stock-share-price/multi-commodity-exchange-of-india-ltd/mcx/534091/corp-announcements/"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SHORT)
            
            articles = []
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.find_all(['td', 'div'], limit=30):
                    text = item.get_text(strip=True)
                    if text and len(text) > 20 and any(kw in text.lower() for kw in ['mcx', 'board', 'result', 'dividend', 'agm']):
                        articles.append({
                            'title': text[:150],
                            'source': 'BSE',
                            'time': '',
                            'origin': 'BSE Announcements'
                        })
            
            print(f"    Found {len(articles)} from BSE")
            return articles
            
        except Exception as e:
            print(f"    ❌ BSE error: {e}")
            return []
    
    def fetch_all_news(self):
        """Fetch news from all sources"""
        print("\n🔍 FETCHING NEWS FROM MULTIPLE SOURCES...")
        
        all_news = []
        
        # More reliable sources first
        all_news.extend(self.scrape_yahoo_finance())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_google_news("MCX stock Multi Commodity Exchange"))
        time.sleep(0.3)
        
        all_news.extend(self.scrape_google_news("MCX India commodity exchange news"))
        time.sleep(0.3)
        
        all_news.extend(self.scrape_cnbc_tv18())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_economic_times())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_moneycontrol())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_livemint())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_investing_com())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_tradingview())
        time.sleep(0.3)
        
        all_news.extend(self.scrape_bse_announcements())
        
        # Remove duplicates based on title similarity
        unique_news = []
        seen_titles = set()
        
        for article in all_news:
            # Normalize title for dedup
            title_key = re.sub(r'[^a-z0-9]', '', article['title'][:40].lower())
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_news.append(article)
        
        print(f"\n✅ Total unique articles collected: {len(unique_news)}")
        
        self.news_cache = unique_news
        return unique_news
    
    def analyze_sentiment_with_gpt(self, news_articles):
        """Use ChatGPT to analyze sentiment of news articles"""
        if not self.client:
            return {
                'available': False,
                'sentiment': 'NEUTRAL',
                'score': 50,
                'confidence': 50,
                'reasoning': 'ChatGPT API not configured'
            }
        
        if not news_articles:
            return {
                'available': True,
                'sentiment': 'NEUTRAL',
                'score': 50,
                'confidence': 30,
                'reasoning': 'No news articles found'
            }
        
        # Prepare news summary for GPT
        news_text = "\n".join([
            f"- [{a['source']}] {a['title']}"
            for a in news_articles[:20]  # Limit to 20 articles
        ])
        
        prompt = f"""You are a financial news sentiment analyst. Analyze these news headlines about MCX Ltd (Multi Commodity Exchange of India) stock.

NEWS HEADLINES:
{news_text}

IMPORTANT CONTEXT:
- MCX is India's largest commodity exchange
- Stock is traded on NSE (MCX.NS)
- We need sentiment for TRADING decisions
- Be conservative - news doesn't always predict price

Analyze and respond with ONLY this JSON:
{{
    "sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "score": 0-100 (50=neutral, >50=bullish, <50=bearish),
    "confidence": 30-90 (how confident based on news quality/quantity),
    "key_themes": ["theme1", "theme2", "theme3"],
    "bullish_factors": ["factor1", "factor2"],
    "bearish_factors": ["factor1", "factor2"],
    "reasoning": "2-3 sentence analysis of overall news sentiment"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial news sentiment analyst. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON if wrapped in code blocks
            if '```' in content:
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            result['available'] = True
            result['article_count'] = len(news_articles)
            
            return result
            
        except Exception as e:
            return {
                'available': False,
                'sentiment': 'NEUTRAL',
                'score': 50,
                'confidence': 30,
                'reasoning': f'Analysis error: {str(e)}'
            }
    
    def get_sentiment_analysis(self, use_cache=True):
        """Get complete sentiment analysis"""
        # Check cache (5 minute expiry)
        if use_cache and self.sentiment_cache and self.cache_time:
            if (datetime.now() - self.cache_time).seconds < 300:
                print("📦 Using cached sentiment analysis")
                return self.sentiment_cache
        
        # Fetch fresh news
        news = self.fetch_all_news()
        
        # Analyze with GPT
        print("\n🤖 ANALYZING SENTIMENT WITH CHATGPT...")
        sentiment = self.analyze_sentiment_with_gpt(news)
        
        # Cache results
        self.sentiment_cache = {
            'news': news,
            'sentiment': sentiment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.cache_time = datetime.now()
        
        return self.sentiment_cache
    
    def display_analysis(self, analysis):
        """Display sentiment analysis results"""
        news = analysis['news']
        sent = analysis['sentiment']
        
        print("\n" + "="*70)
        print("📰 NEWS SENTIMENT ANALYSIS")
        print("="*70)
        
        print(f"\n📊 ARTICLES ANALYZED: {len(news)}")
        print("\nRecent Headlines:")
        for i, article in enumerate(news[:10], 1):
            print(f"  {i}. [{article['source']}] {article['title'][:70]}...")
        
        print(f"\n" + "="*70)
        print("🎯 SENTIMENT ANALYSIS RESULTS")
        print("="*70)
        
        if sent.get('available'):
            # Color indicator
            color = '🟢' if sent['sentiment'] == 'BULLISH' else '🔴' if sent['sentiment'] == 'BEARISH' else '🟡'
            
            print(f"\n   {color} SENTIMENT: {sent['sentiment']}")
            print(f"   📊 SCORE: {sent['score']}/100 (50=neutral)")
            print(f"   🎯 CONFIDENCE: {sent['confidence']}%")
            
            if sent.get('key_themes'):
                print(f"\n   📌 KEY THEMES:")
                for theme in sent['key_themes']:
                    print(f"      • {theme}")
            
            if sent.get('bullish_factors'):
                print(f"\n   🟢 BULLISH FACTORS:")
                for factor in sent['bullish_factors']:
                    print(f"      + {factor}")
            
            if sent.get('bearish_factors'):
                print(f"\n   🔴 BEARISH FACTORS:")
                for factor in sent['bearish_factors']:
                    print(f"      - {factor}")
            
            print(f"\n   💡 REASONING:")
            print(f"      {sent.get('reasoning', 'N/A')}")
        else:
            print(f"\n   ❌ {sent.get('reasoning', 'Analysis not available')}")
        
        print(f"\n   ⏰ Timestamp: {analysis['timestamp']}")
        print("="*70)
        
        return sent


def main():
    """Main function to run news sentiment analysis"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   📰 MCX NEWS SENTIMENT AGENT                                               ║
║                                                                              ║
║   Sources: CNBC TV18, Moneycontrol, Economic Times, LiveMint, Google News   ║
║   Analysis: ChatGPT-4 Sentiment Analysis                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set!")
        api_key = input("Enter OpenAI API key (or Enter to skip): ").strip()
    
    # Create agent and run analysis
    agent = NewsSentimentAgent(openai_api_key=api_key)
    analysis = agent.get_sentiment_analysis(use_cache=False)
    agent.display_analysis(analysis)
    
    return analysis


if __name__ == '__main__':
    main()
