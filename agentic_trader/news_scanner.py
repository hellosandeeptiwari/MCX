"""
News Scanner — Pre-market news-based stock identification for Early Bird Mode D.

Runs at ~8:50 AM before market open. Fetches headlines from multiple Indian
financial news RSS feeds, uses GPT to identify F&O stocks directly impacted
by overnight/morning news, and returns actionable targets with sentiment.

Flow:
  1. Fetch RSS headlines from ET Markets + MoneyControl + Livemint
  2. Filter to last 14 hours (overnight + pre-market)
  3. Send headlines batch to GPT with F&O universe context
  4. GPT returns: [{symbol, sentiment, confidence, headline}]
  5. Filter to top 10 by confidence, matching our F&O universe
  6. Pass to kite_ticker as Early Bird Mode D news targets

Usage:
  from news_scanner import NewsScanner
  scanner = NewsScanner()
  targets = scanner.scan()  # Returns list of NewsTarget dicts
"""

import os
import re
import json
import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from email.utils import parsedate_to_datetime
import calendar

import requests
from openai import OpenAI

logger = logging.getLogger('news_scanner')

# ── RSS Feed Sources (curated for trading signal quality + speed) ────────────
_RSS_FEEDS = [
    # ── Tier 0 — Earnings & company-specific feeds (highest priority) ──────────
    {
        'name': 'MC_Results',
        'url': 'https://www.moneycontrol.com/rss/results.xml',
        'priority': 1,   # [FIX Apr 9] Dedicated quarterly results feed
    },
    {
        'name': 'MC_MarketReports',
        'url': 'https://www.moneycontrol.com/rss/marketreports.xml',
        'priority': 1,   # [FIX Apr 9] Market reports — result summaries, guidance
    },
    {
        'name': 'ET_CompanyNews',
        'url': 'https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms',
        'priority': 1,   # [FIX Apr 9] Official ET Company News — earnings, mgmt changes, deals
    },
    {
        'name': 'ET_ExpertView',
        'url': 'https://economictimes.indiatimes.com/markets/expert-view/rssfeeds/50649960.cms',
        'priority': 1,   # [FIX Apr 9] Broker upgrades/downgrades, target price changes
    },
    {
        'name': 'FE_Companies',
        'url': 'https://www.financialexpress.com/about/rss-feeds/market/rss',
        'priority': 1,   # [FIX Apr 10] Financial Express Market — buybacks, broker calls, results
    },
    # ── Tier 1 — Core feeds (fastest, cleanest, best for intraday triggers) ──
    {
        'name': 'Moneycontrol',
        'url': 'https://www.moneycontrol.com/rss/latestnews.xml',
        'priority': 1,   # Fastest India-specific breaking news, clean headlines
    },
    {
        'name': 'ET_Markets',
        'url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'priority': 1,   # Structured macro + sector coverage
    },
    {
        'name': 'ET_Markets_Stocks',
        'url': 'https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms',
        'priority': 1,   # Stock-specific news (earnings, deals, recos)
    },
    {
        'name': 'Moneycontrol_Business',
        'url': 'https://www.moneycontrol.com/rss/business.xml',
        'priority': 1,   # Business and corporate news
    },
    # ── Tier 2 — Intelligence layer (policy, macro, institutional signals) ──
    {
        'name': 'CNBCTV18_Stocks',
        'url': 'https://www.cnbctv18.com/commonfeeds/v1/cne/rss/stocks.xml',
        'priority': 2,   # High volume stock-specific news (200 items)
    },
    {
        'name': 'Livemint_Markets',
        'url': 'https://www.livemint.com/rss/markets',
        'priority': 2,   # Policy + macro (RBI, inflation, global cues)
    },
    {
        'name': 'Livemint_Companies',
        'url': 'https://www.livemint.com/rss/companies',
        'priority': 2,   # [FIX Apr 9] Company-specific results & announcements
    },
    {
        'name': 'NDTV_Profit',
        'url': 'https://feeds.feedburner.com/ndtvprofit-latest',
        'priority': 2,   # Stock-specific breaking news, M&A, results
    },
    # ── Tier 3 — High-diversity feeds ──────────────────────────────────────────
    {
        'name': 'TheHindu_Business',
        'url': 'https://www.thehindu.com/business/feeder/default.rss',
        'priority': 1,   # Excellent stock-specific: earnings, RBI, SEBI, rupee, oil
    },
    {
        'name': 'FE_Market',
        'url': 'https://www.financialexpress.com/section/market/feed/',
        'priority': 1,   # [FIX Apr 10] Financial Express — buybacks, broker calls, results, corporate actions
    },
    {
        'name': 'MC_StocksNews',
        'url': 'https://www.moneycontrol.com/rss/stocksnews.xml',
        'priority': 1,   # [FIX Apr 10] Moneycontrol stock-specific: earnings, upgrades, corp actions
    },
    {
        'name': 'MC_Economy',
        'url': 'https://www.moneycontrol.com/rss/economy.xml',
        'priority': 2,   # [FIX Apr 10] Macro: RBI, GDP, inflation, policy
    },
]

# ── Expanded F&O Universe for News Matching ──────────────────────────────────
# Full NSE F&O symbols the system can trade (superset of APPROVED_UNIVERSE)
# GPT will match news to these symbols
_FNO_UNIVERSE = [
    # Tier-1 (always-scan)
    'SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK',
    'BAJFINANCE', 'RELIANCE', 'BHARTIARTL', 'INFY', 'TCS',
    # Tier-2
    'TATASTEEL', 'JSWSTEEL', 'JINDALSTEL', 'HINDALCO', 'LT',
    'MARUTI', 'TITAN', 'SUNPHARMA', 'ONGC', 'NTPC', 'ITC',
    'TATAMOTORS', 'CIPLA',
    # Extended F&O universe (commonly traded)
    'WIPRO', 'HCLTECH', 'TECHM', 'DRREDDY', 'APOLLOHOSP',
    'ADANIENT', 'ADANIPORTS', 'BAJAJFINSV', 'NESTLEIND', 'DIVISLAB',
    'COALINDIA', 'POWERGRID', 'BPCL', 'IOC', 'HINDPETRO',
    'VEDL', 'TATAPOWER', 'INDUSINDBK', 'BANDHANBNK', 'PNB',
    'BANKBARODA', 'HEROMOTOCO', 'M&M', 'EICHERMOT', 'ASHOKLEY',
    'DABUR', 'GODREJCP', 'HINDUNILVR', 'BRITANNIA', 'MARICO',
    'GRASIM', 'ULTRACEMCO', 'SHREECEM', 'AMBUJACEM', 'ACC',
    'SAIL', 'NMDC', 'TATACOMM', 'JUBLFOOD', 'ZOMATO',
    'DMART', 'NYKAA', 'PAYTM', 'POLICYBZR', 'IRCTC',
    'HAL', 'BEL', 'BHEL', 'SIEMENS', 'ABB',
    'DLF', 'OBEROIRLTY', 'GODREJPROP', 'PRESTIGE',
    'SBILIFE', 'HDFCLIFE', 'ICICIPRULI',
    'CHOLAFIN', 'MUTHOOTFIN', 'MANAPPURAM', 'PEL',
    'TRENT', 'PAGEIND', 'VOLTAS', 'BLUESTARCO',
    'SRF', 'PIIND', 'ATUL', 'DEEPAKNTR',
    'MPHASIS', 'LTIM', 'PERSISTENT', 'COFORGE',
    # Additional F&O stocks (earnings season coverage)
    'NAUKRI', 'PHOENIXLTD', 'ANGELONE', 'INDIGO', 'CANBK',
    'IDFCFIRSTB', 'FEDERALBNK', 'AUBANK', 'MCX', 'RECLTD',
    'PFC', 'NHPC', 'INDUSTOWER', 'IDEA', 'LICI',
    'JSWENERGY', 'TORNTPHARM', 'LUPIN', 'BIOCON', 'AUROPHARMA',
    'LAURUSLABS', 'MRF', 'BATAINDIA', 'PVRINOX', 'INDIAMART',
]

# Company name → NSE symbol mapping for GPT output normalization
_COMPANY_ALIASES = {
    'reliance industries': 'RELIANCE', 'reliance': 'RELIANCE', 'ril': 'RELIANCE',
    'tata steel': 'TATASTEEL', 'tata motors': 'TATAMOTORS', 'tata power': 'TATAPOWER',
    'tata consultancy': 'TCS', 'tata communications': 'TATACOMM',
    'infosys': 'INFY', 'wipro': 'WIPRO', 'hcl tech': 'HCLTECH', 'hcl technologies': 'HCLTECH',
    'tech mahindra': 'TECHM', 'hdfc bank': 'HDFCBANK', 'icici bank': 'ICICIBANK',
    'axis bank': 'AXISBANK', 'kotak bank': 'KOTAKBANK', 'kotak mahindra': 'KOTAKBANK',
    'sbi': 'SBIN', 'state bank': 'SBIN', 'bajaj finance': 'BAJFINANCE',
    'bharti airtel': 'BHARTIARTL', 'airtel': 'BHARTIARTL',
    'sun pharma': 'SUNPHARMA', 'sun pharmaceutical': 'SUNPHARMA',
    'cipla': 'CIPLA', 'dr reddy': 'DRREDDY', "dr reddy's": 'DRREDDY',
    'jsw steel': 'JSWSTEEL', 'jindal steel': 'JINDALSTEL',
    'hindalco': 'HINDALCO', 'vedanta': 'VEDL',
    'larsen': 'LT', 'l&t': 'LT', 'maruti': 'MARUTI', 'maruti suzuki': 'MARUTI',
    'titan': 'TITAN', 'itc': 'ITC', 'ongc': 'ONGC', 'ntpc': 'NTPC',
    'apollo hospitals': 'APOLLOHOSP', 'adani enterprises': 'ADANIENT',
    'adani ports': 'ADANIPORTS', 'jubilant foodworks': 'JUBLFOOD',
    'zomato': 'ZOMATO', 'dmart': 'DMART', 'avenue supermarts': 'DMART',
    'nykaa': 'NYKAA', 'paytm': 'PAYTM', 'one97': 'PAYTM',
    'hal': 'HAL', 'hindustan aeronautics': 'HAL',
    'bhel': 'BHEL', 'bharat heavy': 'BHEL',
    'bel': 'BEL', 'bharat electronics': 'BEL',
    'dlf': 'DLF', 'coal india': 'COALINDIA',
    'hero motocorp': 'HEROMOTOCO', 'mahindra': 'M&M', 'm&m': 'M&M',
    'eicher motors': 'EICHERMOT', 'ashok leyland': 'ASHOKLEY',
    'dabur': 'DABUR', 'godrej consumer': 'GODREJCP',
    'hindustan unilever': 'HINDUNILVR', 'hul': 'HINDUNILVR',
    'britannia': 'BRITANNIA', 'marico': 'MARICO',
    'ultratech': 'ULTRACEMCO', 'ambuja cements': 'AMBUJACEM',
    'sail': 'SAIL', 'nmdc': 'NMDC', 'irctc': 'IRCTC',
    'siemens': 'SIEMENS', 'abb': 'ABB',
    'mphasis': 'MPHASIS', 'ltimindtree': 'LTIM', 'persistent': 'PERSISTENT',
    'trent': 'TRENT', 'voltas': 'VOLTAS', 'kalyan jewellers': 'KALYANKJIL',
    'sbi life': 'SBILIFE', 'hdfc life': 'HDFCLIFE',
    'bandhan bank': 'BANDHANBNK', 'pnb': 'PNB', 'bank of baroda': 'BANKBARODA',
    'indusind bank': 'INDUSINDBK', 'bpcl': 'BPCL', 'ioc': 'IOC',
    'power grid': 'POWERGRID', 'srf': 'SRF',
    # Additional earnings-season stocks
    'info edge': 'NAUKRI', 'naukri': 'NAUKRI', 'infoedge': 'NAUKRI',
    'phoenix mills': 'PHOENIXLTD', 'angel one': 'ANGELONE', 'angel broking': 'ANGELONE',
    'indigo': 'INDIGO', 'interglobe': 'INDIGO', 'interglobe aviation': 'INDIGO',
    'canara bank': 'CANBK', 'idfc first': 'IDFCFIRSTB', 'federal bank': 'FEDERALBNK',
    'au bank': 'AUBANK', 'au small finance': 'AUBANK',
    'rec': 'RECLTD', 'pfc': 'PFC', 'nhpc': 'NHPC',
    'indus towers': 'INDUSTOWER', 'vodafone idea': 'IDEA', 'vi': 'IDEA',
    'lic': 'LICI', 'life insurance': 'LICI',
    'jsw energy': 'JSWENERGY', 'torrent pharma': 'TORNTPHARM',
    'lupin': 'LUPIN', 'biocon': 'BIOCON', 'aurobindo': 'AUROPHARMA',
    'laurus labs': 'LAURUSLABS', 'mrf': 'MRF', 'bata': 'BATAINDIA',
    'pvr inox': 'PVRINOX', 'pvr': 'PVRINOX', 'indiamart': 'INDIAMART',
}


# ── Sector Keywords → Symbols (for corroboration boosting) ───────────────────
# When these keywords appear in headlines, credit the mapped symbols
_SECTOR_KEYWORDS = {
    # Oil & Gas
    'crude oil': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO', 'RELIANCE'],
    'oil prices': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO', 'RELIANCE'],
    'oil price': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO', 'RELIANCE'],
    'brent': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    'wti': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    'kharg island': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    'strait of hormuz': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    'petroleum': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    'opec': ['BPCL', 'ONGC', 'IOC', 'HINDPETRO'],
    # Metals
    'metal prices': ['TATASTEEL', 'JSWSTEEL', 'JINDALSTEL', 'HINDALCO', 'VEDL', 'SAIL', 'NMDC'],
    'steel prices': ['TATASTEEL', 'JSWSTEEL', 'JINDALSTEL', 'SAIL'],
    'aluminium': ['HINDALCO', 'VEDL'],
    'copper prices': ['HINDALCO', 'VEDL'],
    'iron ore': ['TATASTEEL', 'JSWSTEEL', 'NMDC', 'SAIL'],
    # IT sector
    'it sector': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
    'tech layoffs': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM'],
    'h-1b': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM'],
    # Banking
    'rbi policy': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK'],
    'rbi rate': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK'],
    'interest rate': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'BAJFINANCE'],
    'npa': ['SBIN', 'PNB', 'BANKBARODA', 'INDUSINDBK'],
    # Pharma
    'pharma sector': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB'],
    'fda approval': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB'],
    'usfda': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB'],
    # Auto
    'auto sales': ['MARUTI', 'TATAMOTORS', 'M&M', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY'],
    'vehicle sales': ['MARUTI', 'TATAMOTORS', 'M&M', 'HEROMOTOCO'],
    # Defence
    'defence order': ['HAL', 'BEL', 'BHEL'],
    'defense spending': ['HAL', 'BEL', 'BHEL'],
    # Real estate
    'real estate': ['DLF', 'OBEROIRLTY', 'GODREJPROP', 'PRESTIGE'],
    'housing': ['DLF', 'OBEROIRLTY', 'GODREJPROP'],
}

# ── Catalyst Type Weights (base score contribution) ──────────────────────────
_CATALYST_WEIGHTS = {
    'EARNINGS':        35,   # [FIX Apr 9] Raised 30→35 — Q results are highest-alpha catalyst
    'BUYBACK':         30,   # [FIX Apr 10] Share buyback/tender offer — strong bullish capital return signal
    'BROKER_ACTION':   25,   # Upgrade/downgrade/target price change
    'DIVIDEND':        24,   # [FIX Apr 10] Special/interim dividend, high yield payout
    'ORDER_WIN':       22,   # Contract win, deal, partnership
    'MANAGEMENT':      22,   # CEO/CFO change, restructuring, layoffs
    'CAPEX_EXPANSION': 22,   # [FIX Apr 10] New plant, capacity expansion, investment announcement
    'STAKE_CHANGE':    20,   # [FIX Apr 10] Promoter stake increase/decrease, block/bulk deal, FII/DII
    'REGULATORY':      20,   # SEBI action, govt policy, approval
    'SPLIT_BONUS':     20,   # [FIX Apr 10] Stock split, bonus issue
    'RATING_OUTLOOK':  18,   # [FIX Apr 10] Credit rating upgrade/downgrade (Moody's, S&P, CRISIL, ICRA)
    'MACRO_SECTOR':    15,   # Crude oil, metal prices, rate changes, sector rotation
    'GLOBAL_CUE':      12,   # US Fed, Dow Jones, global risk-on/off
    'OTHER':           10,
}

# Directness bonus — how directly the stock is mentioned
_DIRECTNESS_BONUS = {
    'DIRECT':   20,   # Company named explicitly in headline
    'SECTOR':    8,   # Sector-level news mapped to liquid names
    'INDIRECT':  0,   # Loosely related / thematic
}


class NewsTarget:
    """A stock identified by news analysis as likely to move at open."""
    __slots__ = ('symbol', 'sentiment', 'confidence', 'headline', 'source', 'reason',
                 'catalyst_type', 'directness', 'score_breakdown', 'pub_time')

    def __init__(self, symbol: str, sentiment: str, confidence: int,
                 headline: str, source: str, reason: str,
                 catalyst_type: str = '', directness: str = '',
                 score_breakdown: str = '', pub_time: str = ''):
        self.symbol = symbol            # NSE F&O symbol e.g. 'TATASTEEL'
        self.sentiment = sentiment      # 'BULLISH' or 'BEARISH'
        self.confidence = confidence    # 1-100 (deterministic hybrid score)
        self.headline = headline        # Source headline
        self.source = source            # Feed name
        self.reason = reason            # Brief LLM rationale
        self.catalyst_type = catalyst_type
        self.directness = directness
        self.score_breakdown = score_breakdown
        self.pub_time = pub_time        # ISO timestamp of headline publication

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol, 'sentiment': self.sentiment,
            'confidence': self.confidence, 'headline': self.headline,
            'source': self.source, 'reason': self.reason,
            'catalyst_type': self.catalyst_type,
            'directness': self.directness,
            'score_breakdown': self.score_breakdown,
            'pub_time': self.pub_time,
        }


class NewsScanner:
    """Pre-market news scanner using RSS + GPT sentiment analysis."""

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}
        self._max_targets = self._cfg.get('max_targets', 10)
        self._trade_targets = self._cfg.get('trade_targets', 5)  # Top N to actually trade
        self._min_confidence = self._cfg.get('min_confidence', 60)
        self._lookback_hours = self._cfg.get('lookback_hours', 18)
        self._model = self._cfg.get('llm_model', 'gpt-4o-mini')
        self._timeout = self._cfg.get('feed_timeout_sec', 15)

        # OpenAI client
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
            api_key = os.getenv('OPENAI_API_KEY', '')
        self._client = OpenAI(api_key=api_key) if api_key else None

        # Cache to avoid re-scanning within same session
        self._last_scan_result: List[NewsTarget] = []
        self._all_scan_result: List[NewsTarget] = []  # All identified (incl non-traded)
        self._last_scan_ts: float = 0.0

        # Intraday delta tracking — avoid re-processing seen headlines
        # [FIX Apr 9] Changed from set → dict{key: timestamp} with TTL to avoid stale blocking
        # [FIX Apr 10] Reduced TTL from 2h → 45min so developing stories re-trigger after cooloff
        self._seen_title_keys: dict = {}             # {dedup_key: epoch_ts}
        self._seen_ttl_sec: float = 2700             # 45-min TTL for dedup keys
        self._last_intraday_ts: float = 0.0          # Timestamp of last intraday check
        self._intraday_targets: List[NewsTarget] = []  # Breaking news targets found during market hours
        self._intraday_gpt_calls: int = 0            # Track GPT token usage

        # Acted-symbol tracking — don't trade same news twice
        self._acted_symbols: set = set()             # Symbols already traded on from news

    def _purge_stale_dedup_keys(self, now_ts: float):
        """[FIX Apr 9] Remove dedup keys older than TTL so follow-up updates are not blocked."""
        if not self._seen_title_keys:
            return
        stale = [k for k, ts in self._seen_title_keys.items() if (now_ts - ts) > self._seen_ttl_sec]
        for k in stale:
            del self._seen_title_keys[k]
        if stale:
            logger.info(f"NEWS_SCAN: Purged {len(stale)} stale dedup keys (>45min old), {len(self._seen_title_keys)} remain")

    def mark_acted(self, symbol: str):
        """Mark a symbol as acted-on (trade placed from news). Prevents re-triggering."""
        sym = symbol.replace('NSE:', '')
        self._acted_symbols.add(sym)
        logger.info(f"NEWS_SCAN: Marked {sym} as acted — will not re-trigger from news")

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(self, force: bool = False) -> List[NewsTarget]:
        """
        Run full news scan pipeline. Returns up to max_targets NewsTarget objects.
        Results are cached for 30 minutes unless force=True.
        """
        if not force and self._last_scan_ts and (time.time() - self._last_scan_ts) < 1800:
            logger.info(f"NEWS_SCAN: Using cached results ({len(self._last_scan_result)} targets)")
            return self._last_scan_result

        if not self._client:
            logger.warning("NEWS_SCAN: No OpenAI API key — skipping news scan")
            return []

        logger.info("NEWS_SCAN: Starting pre-market news scan...")
        t0 = time.time()

        # Step 1: Fetch headlines from all RSS feeds
        headlines = self._fetch_all_feeds()
        if not headlines:
            logger.warning("NEWS_SCAN: No headlines fetched from any feed")
            return []

        logger.info(f"NEWS_SCAN: Fetched {len(headlines)} headlines in {time.time()-t0:.1f}s")

        # Step 2: Send to GPT for analysis
        targets = self._analyze_with_llm(headlines)

        # Step 3: Filter and rank
        targets = [t for t in targets if t.confidence >= self._min_confidence]

        # Pre-market gate: stock-specific news preferred,
        # but allow high-confidence sector-level macro picks as well.
        _before_direct = len(targets)
        targets = [t for t in targets if (
            t.directness == 'DIRECT'
            or (t.directness == 'SECTOR' and t.confidence >= 50)
        )]
        if _before_direct > len(targets):
            logger.info(f"NEWS_SCAN: Pre-market filter — dropped {_before_direct - len(targets)} SECTOR/INDIRECT picks (stock-specific or high-confidence only)")

        # Filter out already-acted symbols (don't trade same news twice)
        targets = [t for t in targets if t.symbol not in self._acted_symbols]

        targets.sort(key=lambda x: x.confidence, reverse=True)
        all_scanned = targets[:self._max_targets]  # All identified (up to 10)
        trade_targets = targets[:self._trade_targets]  # Top N to actually trade (default 5)

        self._last_scan_result = trade_targets  # Only tradeable targets used by Mode D
        self._all_scan_result = all_scanned     # All scanned for dashboard display
        self._last_scan_ts = time.time()

        elapsed = time.time() - t0
        logger.info(f"NEWS_SCAN: Complete — {len(all_scanned)} scanned, {len(trade_targets)} tradeable in {elapsed:.1f}s")
        for i, t in enumerate(all_scanned):
            tag = '🎯' if i < self._trade_targets else '📋'
            logger.info(f"  {tag} {t.symbol} {t.sentiment} conf={t.confidence} — {t.reason}")

        # Persist to disk for debugging
        self._save_scan_results(all_scanned, trade_targets)

        return trade_targets

    def get_news_direction(self, symbol: str) -> Optional[str]:
        """Return 'BUY' or 'SELL' if symbol has a fresh news target, else None."""
        for t in self._last_scan_result:
            if t.symbol == symbol:
                return 'BUY' if t.sentiment == 'BULLISH' else 'SELL'
        return None

    def get_news_target(self, symbol: str) -> Optional[NewsTarget]:
        """Return the NewsTarget for a symbol if it exists."""
        for t in self._last_scan_result:
            if t.symbol == symbol:
                return t
        return None

    def get_all_news_symbols(self) -> set:
        """Return set of all symbols identified in latest scan."""
        return {t.symbol for t in self._last_scan_result}

    def get_all_scanned(self) -> List[dict]:
        """Return all scanned targets (up to 10) as dicts, with 'tradeable' flag."""
        trade_syms = {t.symbol for t in self._last_scan_result}
        result = []
        for t in self._all_scan_result:
            d = t.to_dict()
            d['tradeable'] = t.symbol in trade_syms
            result.append(d)
        return result

    # ── Intraday Breaking News Scan ─────────────────────────────────────────────

    def intraday_scan(self) -> List[NewsTarget]:
        """
        Lightweight intraday scan for breaking news during market hours.

        Design: RSS fetch is free (no GPT tokens). We poll every 15 min but only
        call GPT when genuinely NEW stock-relevant headlines appear since last check.

        Flow:
          1. Fetch RSS headlines (free) — only keep those published AFTER last check
          2. Dedup against all previously seen headlines (from pre-market + prior intraday)
          3. Pre-filter: count how many new headlines mention F&O stocks/sectors
          4. If >= 2 relevant new headlines → call GPT (costs tokens)
          5. If < 2 relevant → skip GPT, save tokens
          6. Return any new targets found

        Returns empty list if no breaking news worth GPT analysis.
        """
        if not self._client:
            print(f"[NEWS_DBG] intraday_scan: NO OpenAI client — returning []")
            return []

        now = time.time()
        # [FIX Apr 9] Reduced cooldown from 10 → 5 min for faster reaction to breaking results
        if self._last_intraday_ts and (now - self._last_intraday_ts) < 300:
            print(f"[NEWS_DBG] intraday_scan: COOLDOWN — {now - self._last_intraday_ts:.0f}s < 300s")
            return []

        t0 = time.time()

        # Step 1: Fetch headlines from RSS — only DURING market hours
        # [FIX Apr 9] Widened freshness gate from 30 → 60 min to catch results announced 35+ min ago
        cutoff_dt = datetime.now() - timedelta(minutes=60)
        # Ensure we don't go before market open
        market_open_dt = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        if cutoff_dt < market_open_dt:
            cutoff_dt = market_open_dt

        new_headlines = self._fetch_intraday_headlines(cutoff_dt)
        self._last_intraday_ts = now
        print(f"[NEWS_DBG] intraday_scan: fetched {len(new_headlines)} new headlines (cutoff={cutoff_dt.strftime('%H:%M')}, seen_keys={len(self._seen_title_keys)})")

        if not new_headlines:
            print(f"[NEWS_DBG] intraday_scan: 0 headlines — returning []")
            logger.debug("INTRADAY_NEWS: No new headlines since last check")
            return []

        logger.info(f"INTRADAY_NEWS: {len(new_headlines)} new headlines since {cutoff_dt.strftime('%H:%M')}")

        # Step 2: Pre-filter — how many mention F&O stocks or sector keywords?
        relevant = self._count_relevant_headlines(new_headlines)
        print(f"[NEWS_DBG] intraday_scan: {relevant} relevant out of {len(new_headlines)} headlines")
        if relevant < 1:
            logger.info(f"INTRADAY_NEWS: No relevant headlines — skipping GPT (saving tokens)")
            return []

        logger.info(f"INTRADAY_NEWS: {relevant} relevant headlines — calling GPT for analysis")
        self._intraday_gpt_calls += 1

        # Step 3: Analyze with GPT (same hybrid scoring pipeline)
        targets = self._analyze_with_llm(new_headlines)
        targets = [t for t in targets if t.confidence >= self._min_confidence]
        # Filter out already-acted symbols (don't trade same news twice)
        targets = [t for t in targets if t.symbol not in self._acted_symbols]
        targets.sort(key=lambda x: x.confidence, reverse=True)
        targets = targets[:5]  # Max 5 intraday targets per scan

        if targets:
            self._intraday_targets = targets
            logger.info(f"INTRADAY_NEWS: Found {len(targets)} breaking news targets in {time.time()-t0:.1f}s (GPT call #{self._intraday_gpt_calls} today)")
            for t in targets:
                logger.info(f"  ⚡ BREAKING: {t.symbol} {t.sentiment} conf={t.confidence} — {t.reason}")
            self._save_intraday_results(targets)
        else:
            logger.info(f"INTRADAY_NEWS: GPT found no actionable targets (call #{self._intraday_gpt_calls})")

        return targets

    def get_intraday_targets(self) -> List[dict]:
        """Return latest intraday breaking news targets as dicts."""
        return [t.to_dict() for t in self._intraday_targets]

    def _fetch_intraday_headlines(self, cutoff_dt: datetime) -> List[Dict]:
        """Fetch only NEW headlines published after cutoff, excluding already-seen ones."""
        all_headlines = []
        seen_titles = set()
        now_ts = time.time()

        # [FIX Apr 9] Purge stale dedup keys (>2h old) before fetching
        self._purge_stale_dedup_keys(now_ts)

        _feed_ok = 0
        _feed_fail = 0
        _total_items = 0
        _dedup_skip = 0
        for feed in _RSS_FEEDS:
            try:
                resp = requests.get(
                    feed['url'],
                    timeout=self._timeout,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'},
                )
                resp.raise_for_status()
                _feed_ok += 1
                items = self._parse_rss(resp.text, feed['name'], cutoff_dt)
                _total_items += len(items)
                for item in items:
                    # [FIX Apr 9] Increased dedup key from 60→100 chars
                    title_key = re.sub(r'[^a-z0-9]', '', item['title'].lower())[:100]
                    # Skip if we've already seen this headline (from pre-market or prior intraday)
                    if title_key in self._seen_title_keys:
                        _dedup_skip += 1
                        continue
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)
                    self._seen_title_keys[title_key] = now_ts
                    item['priority'] = feed['priority']
                    all_headlines.append(item)
            except Exception as e:
                _feed_fail += 1
                logger.warning(f"INTRADAY_NEWS: Failed to fetch {feed['name']}: {e}")

        print(f"[NEWS_DBG] _fetch_intraday: feeds={_feed_ok}ok/{_feed_fail}fail, rss_items={_total_items}, dedup_skipped={_dedup_skip}, new_rss={len(all_headlines)}")

        # ── [FIX Apr 10] Pulse by Zerodha — real-time all-source aggregator ──
        pulse_headlines = self._fetch_pulse_headlines(cutoff_dt)
        _pulse_new = 0
        for item in pulse_headlines:
            title_key = re.sub(r'[^a-z0-9]', '', item['title'].lower())[:100]
            if title_key in self._seen_title_keys:
                continue
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            self._seen_title_keys[title_key] = now_ts
            all_headlines.append(item)
            _pulse_new += 1
        print(f"[NEWS_DBG] _fetch_intraday: pulse_fetched={len(pulse_headlines)}, pulse_new={_pulse_new}, total_new={len(all_headlines)}")

        # Sort by priority then recency
        all_headlines.sort(key=lambda x: (x['priority'], -x.get('pub_ts', 0)))

        # Filter out reactive headlines
        all_headlines = self._filter_reactive_headlines(all_headlines)

        return all_headlines[:50]  # Cap at 50 for intraday (increased from 40 with Pulse)

    # ── [FIX Apr 10] Pulse by Zerodha — real-time aggregator of all Indian financial news ──
    def _fetch_pulse_headlines(self, cutoff_dt: datetime) -> List[Dict]:
        """Fetch headlines from Pulse by Zerodha (aggregates ET, NDTV, TheHindu, Livemint, etc)."""
        try:
            resp = requests.get(
                'https://pulse.zerodha.com/',
                timeout=self._timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/125.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml',
                },
            )
            resp.raise_for_status()
            html = resp.text

            headlines = []
            now = datetime.now()
            cutoff_ts = calendar.timegm(cutoff_dt.timetuple())

            # Extract headline blocks from Pulse HTML
            # Pattern: <a ... href="URL" ...>Title</a> ... Xmin/hr ago ... Source
            # Pulse renders news cards with title links + relative timestamps
            title_pattern = re.compile(
                r'<a[^>]+href=["\']'           # opening <a> with href
                r'(https?://[^"\']+)["\']'     # capture URL
                r'[^>]*>\s*'                    # closing >
                r'([^<]{15,200})'               # capture title (15-200 chars)
                r'\s*</a>',
                re.IGNORECASE
            )
            time_pattern = re.compile(
                r'(\d+(?:\.\d+)?)\s+(minute|hour|day)s?\s+ago',
                re.IGNORECASE
            )
            source_pattern = re.compile(
                r'[—–-]\s*([A-Za-z][A-Za-z &\.]+?)(?:\s*<|$)',
            )

            # Split HTML into card-like chunks by common delimiters
            # Each news item typically lives in its own div/li block
            chunks = re.split(r'<(?:div|li|article)[^>]*class=["\'][^"\']*(?:feed|story|item|card|news)[^"\']*["\']', html, flags=re.IGNORECASE)

            for chunk in chunks:
                title_match = title_pattern.search(chunk)
                if not title_match:
                    continue

                url = title_match.group(1)
                title = title_match.group(2).strip()

                # Skip non-news links (navigation, ads, etc.)
                if len(title) < 20:
                    continue
                news_domains = ('economictimes.', 'ndtvprofit.', 'thehindu.', 'livemint.',
                                'moneycontrol.', 'business-standard.', 'cnbctv18.', 'finshots.')
                if not any(d in url for d in news_domains):
                    continue

                # Parse relative timestamp
                time_match = time_pattern.search(chunk)
                if not time_match:
                    continue

                time_val = float(time_match.group(1))
                time_unit = time_match.group(2).lower()
                if 'minute' in time_unit:
                    delta = timedelta(minutes=time_val)
                elif 'hour' in time_unit:
                    delta = timedelta(hours=time_val)
                elif 'day' in time_unit:
                    delta = timedelta(days=time_val)
                else:
                    continue

                pub_dt = now - delta
                pub_ts = calendar.timegm(pub_dt.timetuple())
                if pub_ts < cutoff_ts:
                    continue

                # Extract source
                source_name = 'Pulse'
                source_match = source_pattern.search(chunk[time_match.end():time_match.end()+100])
                if source_match:
                    source_name = f"Pulse_{source_match.group(1).strip()}"

                headlines.append({
                    'title': title,
                    'source': source_name,
                    'published': pub_dt.isoformat(),
                    'pub_ts': pub_ts,
                    'description': '',
                    'priority': 1,  # Pulse is curated, high-quality aggregation
                })

            logger.info(f"PULSE: Fetched {len(headlines)} headlines from pulse.zerodha.com")
            return headlines
        except Exception as e:
            logger.warning(f"PULSE: Failed to fetch pulse.zerodha.com: {e}")
            return []

    def _count_relevant_headlines(self, headlines: List[Dict]) -> int:
        """Count how many headlines mention F&O stocks or sector keywords. No GPT cost."""
        relevant = 0
        for h in headlines:
            text = (h['title'] + ' ' + (h.get('description') or '')).lower()
            # Check company aliases
            for kw in _COMPANY_ALIASES:
                if len(kw) >= 3 and kw in text:
                    relevant += 1
                    break
            else:
                # Check sector keywords
                for sk in _SECTOR_KEYWORDS:
                    if sk in text:
                        relevant += 1
                        break
        return relevant

    def _save_intraday_results(self, targets: List[NewsTarget]):
        """Save intraday scan results to disk."""
        try:
            out_dir = os.path.dirname(os.path.abspath(__file__))
            out_file = os.path.join(out_dir, 'news_intraday_results.json')
            data = {
                'scan_time': datetime.now().isoformat(),
                'gpt_calls_today': self._intraday_gpt_calls,
                'targets': [t.to_dict() for t in targets],
            }
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"INTRADAY_NEWS: Failed to save results: {e}")

    def seed_seen_headlines(self, headlines_keys: set):
        """Seed the seen-headline tracker from pre-market scan to avoid re-processing."""
        self._seen_title_keys.update(headlines_keys)

    # ── Junk/Filler Headline Filter ────────────────────────────────────────────
    # Auto-generated filler headlines waste GPT tokens and crowd out real news.
    # e.g. "HDFC Bank Share Price Live Updates: closing price analysis"
    # These carry zero trading signal — strip them before LLM analysis.

    _JUNK_PATTERNS = re.compile(
        r'(?:'
        r'share\s+price\s+(?:live\s+)?updates?'
        r'|(?:closing|opening)\s+(?:price|figure|value)\s+(?:analysis|update|summary|overview|details)'
        r'|(?:previous|last)\s+(?:day|trading\s+day)(?:[\'\u2019]s)?\s+(?:close|closing|performance)'
        r'|(?:stock|share)\s+(?:price|closing)\s+(?:movement|snapshot|overview)'
        r'|price\s+(?:update|movement)\s*$'
        r'|end[- ]of[- ]day\s+price'
        r'|market\s+(?:closing|opening)\s+(?:details|summary|update)'
        r'|(?:latest|today[\'\u2019]s)\s+closing\s+figure'
        r'|settled\s+at\s+rs\s+\d+'
        r'|(?:closes?|ends?)\s+(?:trading\s+)?at\s+rs\s+\d+'
        r'|stock\s+closes\s+at\s+rs'
        r')',
        re.IGNORECASE
    )

    def _is_junk_headline(self, title: str) -> bool:
        """Return True if headline is auto-generated filler with no trading signal."""
        return bool(self._JUNK_PATTERNS.search(title))

    # ── Stale/Reactive Headline Filter ─────────────────────────────────────────
    # Headlines that describe moves ALREADY HAPPENED are useless — they're post-hoc.
    # e.g. "JUBLFOOD shares drop 10% after Q4 miss" → the drop already occurred.
    # We want FORWARD-LOOKING catalysts: upgrades, upcoming results, policy changes.

    _REACTIVE_PATTERNS = re.compile(
        r'\b(?:'
        r'shares?\s+(?:fall|drop|crash|tumble|sink|slide|slip|tank|plunge|dive|decline|lose|shed|dip|slump|skid|plummet|nosedive)'
        r'|shares?\s+(?:rise|rally|surge|soar|jump|gain|climb|spike|zoom|up)'
        r'|stock\s+(?:falls?|drops?|crashes?|tumbles?|sinks?|slides?|slips?|tanks?|plunges?|dives?|declines?|loses?|sheds?|dips?|slumps?)'
        r'|stock\s+(?:rises?|rallies|surges?|soars?|jumps?|gains?|climbs?|spikes?|zooms?)'
        r'|(?:hits?|touch(?:es)?|near)\s+(?:52[- ]?week|all[- ]?time|record|multi[- ]?year|multi[- ]?month)\s+(?:low|high)'
        r'|(?:plunges?|tumbles?|crashes?|tanks?|nosedives?|rallies|surges?|soars?|zooms?|jumps?)\s+\d+\s*%'
        r'|down\s+\d+\s*%|up\s+\d+\s*%'
        r'|(?:biggest|sharpest|worst|steepest)\s+(?:fall|drop|decline|loss|gain|rise)'
        r'|market\s*cap\s+(?:eroded|wiped|lost)'
        r'|investors\s+(?:lose|lost|poorer|richer)'
        r')\b',
        re.IGNORECASE
    )

    # Corporate action keywords that should NEVER be filtered as reactive,
    # even if the headline also contains price-move words ("shares jump on buyback")
    _CORPORATE_ACTION_WHITELIST = re.compile(
        r'\b(?:'
        r'buyback|buy[- ]?back|share\s+repurchase|tender\s+offer'
        r'|bonus\s+(?:issue|shares?|ratio)'
        r'|stock\s+split|(?:sub[- ]?)?split(?:ting)?\s+(?:shares?|stock)'
        r'|(?:special|interim|final|record)\s+dividend'
        r'|dividend\s+(?:of|at|\₹|rs)'
        r'|promoter\s+(?:stake|holding|buying|selling|pledge)'
        r'|block\s+deal|bulk\s+deal'
        r'|(?:FII|DII|FPI)\s+(?:bought|sold|buying|selling|stake)'
        r'|(?:credit\s+)?rating\s+(?:upgrade|downgrade|revised|affirmed)'
        r'|(?:capacity|capex|plant|expansion|greenfield|brownfield)\s+(?:expansion|investment|announcement|plan)'
        r'|new\s+(?:plant|factory|facility|capacity)'
        r')\b',
        re.IGNORECASE
    )

    def _is_reactive_headline(self, title: str) -> bool:
        """Return True if headline describes a move that already happened (post-hoc).
        Exception: corporate action headlines (buyback, split, dividend, etc.) are whitelisted."""
        if self._CORPORATE_ACTION_WHITELIST.search(title):
            return False
        return bool(self._REACTIVE_PATTERNS.search(title))

    def _filter_reactive_headlines(self, headlines: List[Dict]) -> List[Dict]:
        """Remove post-hoc/reactive headlines. Keep only forward-looking catalysts."""
        kept = []
        dropped = 0
        for h in headlines:
            if self._is_reactive_headline(h['title']):
                dropped += 1
                logger.debug(f"NEWS_SCAN: DROPPED reactive headline: {h['title'][:80]}")
            else:
                kept.append(h)
        if dropped:
            logger.info(f"NEWS_SCAN: Filtered out {dropped} reactive/stale headlines, kept {len(kept)}")
        return kept

    # ── Pre-GPT Corroboration Map ──────────────────────────────────────────────

    def _build_corroboration_map(self, headlines: List[Dict]) -> Dict[str, Dict]:
        """
        Before GPT call, count how many DISTINCT feeds mention each company.
        Also track the freshest headline timestamp per symbol.
        Returns: {symbol: {'feed_count': N, 'feeds': set(), 'freshest_ts': float, 'best_priority': int}}
        """
        corr: Dict[str, Dict] = {}

        # Build reverse lookup: lowercase keyword → symbol
        kw_to_sym: Dict[str, str] = {}
        for alias, sym in _COMPANY_ALIASES.items():
            kw_to_sym[alias] = sym
        # Also add symbol names themselves as keywords
        for sym in _FNO_UNIVERSE:
            kw_to_sym[sym.lower()] = sym

        for h in headlines:
            title_lower = h['title'].lower()
            desc_lower = (h.get('description') or '').lower()
            text = title_lower + ' ' + desc_lower
            feed_name = h.get('source', '')
            pub_ts = h.get('pub_ts', 0)
            priority = h.get('priority', 2)

            matched_syms = set()
            for kw, sym in kw_to_sym.items():
                if len(kw) >= 3 and kw in text:
                    matched_syms.add(sym)

            # Sector keyword matching — oil/metal/IT/banking sector headlines
            for sector_kw, sector_syms in _SECTOR_KEYWORDS.items():
                if sector_kw in text:
                    matched_syms.update(sector_syms)

            for sym in matched_syms:
                if sym not in corr:
                    corr[sym] = {'feed_count': 0, 'feeds': set(), 'freshest_ts': 0.0, 'best_priority': 9}
                if feed_name not in corr[sym]['feeds']:
                    corr[sym]['feeds'].add(feed_name)
                    corr[sym]['feed_count'] = len(corr[sym]['feeds'])
                if pub_ts > corr[sym]['freshest_ts']:
                    corr[sym]['freshest_ts'] = pub_ts
                if priority < corr[sym]['best_priority']:
                    corr[sym]['best_priority'] = priority

        if corr:
            multi = {s: d['feed_count'] for s, d in corr.items() if d['feed_count'] >= 2}
            if multi:
                logger.info(f"NEWS_SCAN: Multi-source corroboration: {multi}")

        return corr

    def _compute_confidence(self, catalyst_type: str, directness: str,
                            symbol: str, headline_nums: List[int],
                            headlines: List[Dict],
                            corr_map: Dict[str, Dict]) -> tuple:
        """
        Compute deterministic confidence score from structured factors.
        Returns (score, breakdown_string).
        """
        parts = []

        # 1. Catalyst type base score
        ct = catalyst_type.upper() if catalyst_type else 'OTHER'
        base = _CATALYST_WEIGHTS.get(ct, _CATALYST_WEIGHTS['OTHER'])
        parts.append(f"catalyst({ct})={base}")

        # 2. Directness bonus
        dr = directness.upper() if directness else 'INDIRECT'
        dir_bonus = _DIRECTNESS_BONUS.get(dr, 0)
        parts.append(f"direct({dr})={dir_bonus}")

        # 3. Multi-source corroboration: +10 per extra feed, cap +20
        corr_bonus = 0
        if symbol in corr_map:
            extra_feeds = max(0, corr_map[symbol]['feed_count'] - 1)
            corr_bonus = min(extra_feeds * 10, 20)
        if corr_bonus:
            parts.append(f"corroboration={corr_bonus}")

        # 4. Recency bonus — based on freshest headline timestamp
        recency_bonus = 0
        freshest_ts = 0.0
        for hnum in headline_nums:
            if 0 < hnum <= len(headlines) and headlines[hnum - 1].get('pub_ts', 0) > freshest_ts:
                freshest_ts = headlines[hnum - 1]['pub_ts']
        if freshest_ts <= 0 and symbol in corr_map:
            freshest_ts = corr_map[symbol].get('freshest_ts', 0)
        if freshest_ts > 0:
            age_hours = (time.time() - freshest_ts) / 3600
            if age_hours < 2:
                recency_bonus = 10
            elif age_hours < 6:
                recency_bonus = 5
        if recency_bonus:
            parts.append(f"recency={recency_bonus}")

        # 5. Source priority bonus: best priority=1 → +5
        priority_bonus = 0
        if symbol in corr_map and corr_map[symbol]['best_priority'] == 1:
            priority_bonus = 5
        else:
            for hnum in headline_nums:
                if 0 < hnum <= len(headlines) and headlines[hnum - 1].get('priority', 2) == 1:
                    priority_bonus = 5
                    break
        if priority_bonus:
            parts.append(f"src_pri={priority_bonus}")

        # 6. Description richness: has numbers/financials → +5
        richness_bonus = 0
        for hnum in headline_nums:
            if 0 < hnum <= len(headlines):
                desc = headlines[hnum - 1].get('description', '')
                if re.search(r'(?:₹|\$|%|crore|lakh|billion|million|revenue|profit|EPS|EBITDA|PAT|\d{2,})', desc, re.IGNORECASE):
                    richness_bonus = 5
                    break
        if richness_bonus:
            parts.append(f"richness={richness_bonus}")

        # 7. [FIX Apr 9] Earnings-season boost: EARNINGS + DIRECT gets extra +5
        earnings_boost = 0
        if ct == 'EARNINGS' and dr == 'DIRECT':
            earnings_boost = 5
            parts.append(f"earnings_boost={earnings_boost}")

        total = min(base + dir_bonus + corr_bonus + recency_bonus + priority_bonus + richness_bonus + earnings_boost, 100)
        breakdown = ' + '.join(parts) + f" = {total}"
        return total, breakdown

    # ── RSS Fetching ──────────────────────────────────────────────────────────

    def _get_news_cutoff(self) -> datetime:
        """
        Return the cutoff datetime — only news AFTER this time is relevant.

        Logic: News during market hours (9:15-15:30) is already priced in.
        Only post-market news is actionable for Early Bird next day.

        Two modes based on CURRENT time:
        1. Post-market (after 15:30 today) → cutoff = TODAY 15:30
           We're preparing for tomorrow; only fresh post-close news matters.
        2. Pre-market (before 9:15 today) → cutoff = LAST trading day 15:30
           We're about to trade; post-close news from yesterday matters.

        Weekend/Monday handling:
        - Mon pre-market → Friday 15:30 (covers weekend news)
        - Sat/Sun → Friday 15:30
        """
        now = datetime.now()
        today_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        weekday = now.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

        if now >= today_close and weekday < 5:
            # Post-market: cutoff = today's close (only fresh post-close news)
            return today_close

        # Pre-market or during market hours: cutoff = last trading day's close
        if weekday == 0:      # Monday pre-market → Friday 15:30
            return today_close - timedelta(days=3)
        elif weekday == 5:    # Saturday → Friday 15:30
            return today_close - timedelta(days=1)
        elif weekday == 6:    # Sunday → Friday 15:30
            return today_close - timedelta(days=2)
        else:                 # Tue-Fri pre-market → yesterday 15:30
            return today_close - timedelta(days=1)

    def _is_during_market_hours(self, pub_ts: float) -> bool:
        """Return True if headline was published during market hours (9:15-15:30) on a weekday.
        Such news is already priced in and should be discarded."""
        if pub_ts <= 0:
            return False
        dt = datetime.fromtimestamp(pub_ts)
        if dt.weekday() >= 5:  # Sat/Sun — not market hours
            return False
        market_open = dt.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = dt.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= dt <= market_close

    def _get_lookback_hours(self) -> int:
        """Return effective lookback hours for display/logging purposes."""
        cutoff = self._get_news_cutoff()
        return int((datetime.now() - cutoff).total_seconds() / 3600)

    def _fetch_feed(self, feed: Dict) -> requests.Response:
        """Fetch an RSS feed with retries and an optional alternate URL."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        }
        urls = [feed['url']]
        if feed.get('alt_url'):
            urls.append(feed['alt_url'])

        # If HTTPS is blocked on the same host, try HTTP as a fallback.
        def _add_http_fallback(url):
            if url.startswith('https://'):
                return url.replace('https://', 'http://', 1)
            return None

        http_fallback = _add_http_fallback(feed['url'])
        if http_fallback:
            urls.append(http_fallback)
        if feed.get('alt_url'):
            alt_http = _add_http_fallback(feed['alt_url'])
            if alt_http:
                urls.append(alt_http)

        # Deduplicate while preserving order
        seen_urls = set()
        urls = [u for u in urls if u and not (u in seen_urls or seen_urls.add(u))]

        last_error: Optional[Exception] = None
        for url in urls:
            for attempt in range(1, 4):
                try:
                    resp = requests.get(url, timeout=self._timeout, headers=headers)
                    resp.raise_for_status()
                    return resp
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep(1)
                        continue
                    logger.warning(f"NEWS_SCAN: Failed attempt {attempt} for {feed['name']} ({url}): {exc}")
            if feed.get('alt_url') and url == feed['url']:
                logger.info(f"NEWS_SCAN: Falling back to alternate URL for {feed['name']}")

        if last_error is None:
            raise RuntimeError(f"NEWS_SCAN: Unable to fetch feed {feed['name']} and no error was captured")
        raise last_error

    def _fetch_all_feeds(self) -> List[Dict]:
        """Fetch headlines from all configured RSS feeds."""
        cutoff = self._get_news_cutoff()
        all_headlines = []
        seen_titles = set()
        now_ts = time.time()

        # [FIX Apr 9] Purge stale dedup keys (>2h old) before fetching
        self._purge_stale_dedup_keys(now_ts)

        for feed in _RSS_FEEDS:
            try:
                resp = self._fetch_feed(feed)
                items = self._parse_rss(resp.text, feed['name'], cutoff)
                for item in items:
                    # [FIX Apr 9] Increased dedup key from 60→100 chars to avoid false collisions
                    title_key = re.sub(r'[^a-z0-9]', '', item['title'].lower())[:100]
                    if title_key not in seen_titles:
                        seen_titles.add(title_key)
                        self._seen_title_keys[title_key] = now_ts  # Seed for intraday delta
                        item['priority'] = feed['priority']
                        all_headlines.append(item)
            except Exception as e:
                logger.warning(f"NEWS_SCAN: Failed to fetch {feed['name']}: {e}")

        # Sort by priority (1=highest) then recency
        all_headlines.sort(key=lambda x: (x['priority'], -x.get('pub_ts', 0)))

        # [FIX Apr 10] Drop zero-value filler headlines before GPT (saves token budget)
        pre_junk = len(all_headlines)
        all_headlines = [h for h in all_headlines if not self._is_junk_headline(h['title'])]
        junk_dropped = pre_junk - len(all_headlines)
        if junk_dropped:
            logger.info(f"NEWS_SCAN: Dropped {junk_dropped} junk/filler headlines, kept {len(all_headlines)}")

        # Filter out reactive/stale headlines BEFORE sending to GPT
        all_headlines = self._filter_reactive_headlines(all_headlines)

        # Drop headlines published DURING market hours (9:15-15:30) — already priced in
        pre_count = len(all_headlines)
        all_headlines = [h for h in all_headlines if not self._is_during_market_hours(h.get('pub_ts', 0))]
        mkt_dropped = pre_count - len(all_headlines)
        if mkt_dropped:
            logger.info(f"NEWS_SCAN: Dropped {mkt_dropped} market-hours headlines (already priced in), kept {len(all_headlines)}")

        return all_headlines[:120]  # [FIX Apr 10] Bumped 100→120 for 20+ feeds after junk pre-filter removes filler

    def _parse_rss(self, xml_text: str, source: str, cutoff: datetime) -> List[Dict]:
        """Parse RSS XML and extract headline items newer than cutoff."""
        items = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            # Try fixing common XML issues
            xml_text = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', xml_text)
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError:
                logger.warning(f"NEWS_SCAN: XML parse failed for {source}")
                return []

        for item in root.iter('item'):
            title_el = item.find('title')
            desc_el = item.find('description')
            pub_el = item.find('pubDate')

            title = title_el.text.strip() if title_el is not None and title_el.text else ''
            desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ''
            pub_str = pub_el.text.strip() if pub_el is not None and pub_el.text else ''

            # Clean CDATA wrappers
            title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title).strip()
            desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc, flags=re.DOTALL).strip()
            # Strip HTML tags from description
            desc = re.sub(r'<[^>]+>', '', desc).strip()

            if not title:
                continue

            # Parse publish date
            pub_ts = 0
            if pub_str:
                try:
                    pub_dt = parsedate_to_datetime(pub_str)
                    pub_ts = pub_dt.timestamp()
                    if pub_dt.replace(tzinfo=None) < cutoff:
                        continue  # Too old
                except Exception:
                    pass  # Keep item even if date parse fails

            items.append({
                'title': title,
                'description': desc[:300],  # Truncate long descriptions
                'source': source,
                'pub_ts': pub_ts,
            })

        return items

    # ── LLM Analysis ──────────────────────────────────────────────────────────

    def _analyze_with_llm(self, headlines: List[Dict]) -> List[NewsTarget]:
        """Send headlines to GPT for structured classification, then compute deterministic scores."""

        # Pre-GPT: build corroboration map
        corr_map = self._build_corroboration_map(headlines)

        # Build headline text for prompt
        headline_text = ""
        for i, h in enumerate(headlines, 1):
            headline_text += f"{i}. [{h['source']}] {h['title']}"
            if h.get('description'):
                headline_text += f"\n   {h['description'][:200]}"
            headline_text += "\n"

        universe_str = ', '.join(sorted(set(_FNO_UNIVERSE)))

        # [FIX Apr 9] Auto-compute current earnings quarter instead of hardcoding
        _now = datetime.now()
        _fy_year = _now.year if _now.month >= 4 else _now.year - 1
        _q_map = {1: ('Q3', f'Oct-Dec {_fy_year}'), 2: ('Q3', f'Oct-Dec {_fy_year}'),
                  3: ('Q3', f'Oct-Dec {_fy_year}'), 4: ('Q4', f'Jan-Mar {_fy_year + 1}'),
                  5: ('Q4', f'Jan-Mar {_fy_year + 1}'), 6: ('Q4', f'Jan-Mar {_fy_year + 1}'),
                  7: ('Q1', f'Apr-Jun {_fy_year}'), 8: ('Q1', f'Apr-Jun {_fy_year}'),
                  9: ('Q1', f'Apr-Jun {_fy_year}'), 10: ('Q2', f'Jul-Sep {_fy_year}'),
                  11: ('Q2', f'Jul-Sep {_fy_year}'), 12: ('Q2', f'Jul-Sep {_fy_year}')}
        _q_label, _q_period = _q_map[_now.month]
        _earnings_line = f"We are in {_q_label}/FY{(_fy_year + 1) % 100} results season ({_q_period} quarter)."

        prompt = f"""You are a pre-market news analyst for an Indian stock options trading system.

TASK: Analyze these overnight/morning news headlines and identify which NSE F&O stocks
will be DIRECTLY and SIGNIFICANTLY impacted when the market opens at 9:15 AM TODAY.
You are looking for FORWARD-LOOKING catalysts that will CAUSE a move, NOT news about
moves that ALREADY happened.

TRADEABLE F&O UNIVERSE (only pick from these):
{universe_str}

CRITICAL — REJECT STALE/REACTIVE NEWS:
- REJECT headlines about price moves that already happened ("shares fell 10%", "stock crashed", "tumbled", "rallied X%")
- REJECT post-hoc analysis of yesterday's move ("why X stock fell", "after losing X%")
- REJECT headlines where the stock has ALREADY reacted to the news (the move is done)
- ONLY ACCEPT forward-looking catalysts: NEW earnings/results not yet priced in, broker upgrades/downgrades
  issued after hours, policy changes, regulatory approvals, contract wins, management changes,
  share buybacks, special dividends, stock splits, bonus issues, credit rating changes,
  capacity expansion/capex announcements, promoter stake changes, block/bulk deals,
  sector-wide macro shifts (crude oil, metal prices, rate changes), global cues from overnight US/Asia markets
- EXCEPTION: Headlines about corporate actions (buyback, dividend, split, bonus) are ALWAYS valid even if they mention price moves

IMPORTANT — EARNINGS SEASON:
{_earnings_line} Any headline about quarterly results,
business updates, revenue numbers, profit/loss, guidance, or board meetings to approve
accounts for a SPECIFIC company → catalyst_type=EARNINGS, directness=DIRECT.
Do NOT classify company-specific results as SECTOR.

CATALYST TYPES (you MUST classify each pick):
- EARNINGS:        Quarterly results, revenue/profit numbers, guidance, business updates — just announced
- BUYBACK:         Share buyback announced, tender offer, share repurchase program
- BROKER_ACTION:   Upgrade, downgrade, target price change, initiation by brokerage
- DIVIDEND:        Special dividend, interim dividend, large dividend payout, record date
- ORDER_WIN:       Contract win, deal, partnership, JV, acquisition
- MANAGEMENT:      CEO/CFO change, restructuring, layoffs, board reshuffle
- CAPEX_EXPANSION: New plant, factory, capacity expansion, large capex investment announced
- STAKE_CHANGE:    Promoter stake increase/decrease, block deal, bulk deal, FII/DII buying/selling
- REGULATORY:      SEBI action, govt policy, approval, ban, tax change
- SPLIT_BONUS:     Stock split, bonus issue, bonus shares
- RATING_OUTLOOK:  Credit rating upgrade/downgrade by Moody's, S&P, Fitch, CRISIL, ICRA, CARE
- MACRO_SECTOR:    Crude oil move, metal price shift, rate change, sector rotation
- GLOBAL_CUE:      US Fed, Dow Jones crash/rally, global risk-on/off, war impact on markets

DIRECTNESS (you MUST classify each pick):
- DIRECT:   Company is explicitly named in the headline
- SECTOR:   Sector-level news, you are mapping to the most liquid F&O name
- INDIRECT: Loosely related / thematic connection

RULES:
1. Only pick stocks DIRECTLY mentioned in or clearly impacted by the news
2. Sector-wide news → pick THE SINGLE MOST LIQUID F&O name, mark directness=SECTOR
3. Macro news (crude oil, US markets) → pick THE SINGLE MOST SENSITIVE stock, mark directness=SECTOR or INDIRECT
4. ONE STOCK PER HEADLINE — if a headline mentions multiple companies, pick only the PRIMARY subject
5. NEVER pick multiple correlated stocks from the same headline (e.g., don't pick TATASTEEL + JSWSTEEL + SAIL from one steel headline)
6. Earnings JUST ANNOUNCED (after market close) → catalyst_type=EARNINGS
7. Broker upgrades/downgrades → catalyst_type=BROKER_ACTION
8. Ignore generic market commentary, opinion pieces, educational articles, and technical analysis
9. Maximum 10 stocks — only high-conviction forward-looking directional calls
10. If a headline says a stock "fell X%" or "rose X%" — that move is DONE. Do NOT include it.
11. headline_nums = list ALL headline numbers that support this call (enables multi-source scoring)

HEADLINES:
{headline_text}

Respond with ONLY a JSON array (no markdown, no explanation):
[
  {{
    "symbol": "TATASTEEL",
    "sentiment": "BULLISH",
    "catalyst_type": "EARNINGS",
    "directness": "DIRECT",
    "headline_nums": [16, 23],
    "reason": "Record India output in FY26, strong Q4 volumes — catalyst not yet priced in"
  }},
  ...
]

If no actionable stock-specific news found, return empty array: []"""

        try:
            if not self._client:
                return []
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a financial news analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = (response.choices[0].message.content or '').strip()

            # Extract JSON from potential markdown wrapper
            if raw.startswith('```'):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                logger.warning(f"NEWS_SCAN: LLM returned non-list: {type(parsed)}")
                return []

            targets = []
            _seen_syms = set()  # One stock per unique catalyst — reject correlated duplicates
            for item in parsed:
                sym = str(item.get('symbol', '')).upper().strip()
                sentiment = str(item.get('sentiment', '')).upper().strip()
                catalyst_type = str(item.get('catalyst_type', 'OTHER')).upper().strip()
                directness = str(item.get('directness', 'INDIRECT')).upper().strip()
                reason = str(item.get('reason', ''))

                # Support both single headline_num and list headline_nums
                h_nums = item.get('headline_nums', [])
                if not h_nums:
                    single = item.get('headline_num', 0)
                    h_nums = [int(single)] if single else []
                else:
                    h_nums = [int(x) for x in h_nums]

                # Normalize symbol
                sym = self._normalize_symbol(sym)

                if not sym or sym not in _FNO_UNIVERSE:
                    continue
                if sentiment not in ('BULLISH', 'BEARISH'):
                    continue
                # Skip if we already have this symbol (GPT may still output duplicates)
                if sym in _seen_syms:
                    logger.info(f"NEWS_SCAN: Skipping duplicate {sym} (already selected)")
                    continue
                _seen_syms.add(sym)

                # Compute deterministic confidence from structured factors
                confidence, breakdown = self._compute_confidence(
                    catalyst_type, directness, sym, h_nums, headlines, corr_map,
                )

                # Get source headline + pub_time (use first headline_num)
                headline = ''
                source = 'LLM'
                pub_time = ''
                for hnum in h_nums:
                    if 0 < hnum <= len(headlines):
                        headline = headlines[hnum - 1]['title']
                        source = headlines[hnum - 1]['source']
                        _pts = headlines[hnum - 1].get('pub_ts', 0)
                        if _pts > 0:
                            pub_time = datetime.fromtimestamp(_pts).strftime('%Y-%m-%dT%H:%M')
                        break

                logger.info(f"NEWS_SCAN: SCORE {sym} [{catalyst_type}/{directness}] → {confidence} ({breakdown}) pub={pub_time}")

                targets.append(NewsTarget(
                    symbol=sym, sentiment=sentiment, confidence=confidence,
                    headline=headline[:200], source=source, reason=reason[:150],
                    catalyst_type=catalyst_type, directness=directness,
                    score_breakdown=breakdown, pub_time=pub_time,
                ))

            return targets

        except json.JSONDecodeError as e:
            logger.error(f"NEWS_SCAN: JSON parse error from LLM: {e}")
            return []
        except Exception as e:
            logger.error(f"NEWS_SCAN: LLM analysis failed: {e}")
            return []

    def _normalize_symbol(self, sym: str) -> str:
        """Try to map company names or variations to NSE symbols."""
        if sym in _FNO_UNIVERSE:
            return sym
        # Strip NSE: prefix
        if sym.startswith('NSE:'):
            sym = sym[4:]
            if sym in _FNO_UNIVERSE:
                return sym
        # Check aliases
        sym_lower = sym.lower()
        if sym_lower in _COMPANY_ALIASES:
            return _COMPANY_ALIASES[sym_lower]
        # Partial match
        for alias, nse_sym in _COMPANY_ALIASES.items():
            if alias in sym_lower or sym_lower in alias:
                return nse_sym
        return sym

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_scan_results(self, all_targets: List[NewsTarget], trade_targets: List[NewsTarget]):
        """Save scan results to disk for debugging and audit."""
        try:
            out_dir = os.path.dirname(os.path.abspath(__file__))
            out_file = os.path.join(out_dir, 'news_scan_results.json')

            trade_syms = {t.symbol for t in trade_targets}
            data = {
                'scan_time': datetime.now().isoformat(),
                'lookback_hours': self._get_lookback_hours(),
                'total_scanned': len(all_targets),
                'tradeable_count': len(trade_targets),
                'targets': [{**t.to_dict(), 'tradeable': t.symbol in trade_syms} for t in all_targets],
            }
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"NEWS_SCAN: Failed to save results: {e}")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    scanner = NewsScanner({'llm_model': 'gpt-4o-mini', 'max_targets': 10, 'trade_targets': 5, 'min_confidence': 45})
    results = scanner.scan(force=True)
    all_scanned = scanner.get_all_scanned()
    cutoff = scanner._get_news_cutoff()
    print(f"\n{'='*70}")
    print(f"NEWS SCAN RESULTS: {len(all_scanned)} scanned, {len(results)} tradeable (top 5)")
    print(f"Cutoff: {cutoff.strftime('%Y-%m-%d %H:%M')} (post-market only, ~{scanner._get_lookback_hours()}h)")
    print(f"{'='*70}")
    for r in all_scanned:
        arrow = '🟢' if r['sentiment'] == 'BULLISH' else '🔴'
        tag = '🎯 TRADE' if r.get('tradeable') else '📋 watch'
        ct = r.get('catalyst_type', '')
        dr = r.get('directness', '')
        bd = r.get('score_breakdown', '')
        print(f"  {arrow} {r['symbol']:15s} {r['sentiment']:8s} conf={r['confidence']:3d}  [{tag}]  {ct}/{dr}")
        print(f"     Scoring: {bd}")
        print(f"     Reason: {r['reason']}")
    print()
