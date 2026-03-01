"""
TITAN QUANTITATIVE PERFORMANCE ANALYTICS ENGINE
================================================
Institutional-grade performance analysis for the Titan trading system.

Metrics computed:
  - Portfolio-level: Sharpe, Sortino, Calmar, Profit Factor, Max Drawdown
  - Trade-level: Win Rate, Avg R, Payoff Ratio, Expectancy
  - Attribution: P&L by strategy, setup, score tier, time-of-day, sector
  - Execution: Slippage, hold time, speed gate efficiency
  - Edge Decay: Rolling win rate, R-multiple trend, score→outcome correlation
  - Sector Flow: Money-weighted sector rotation detector

Usage:
    python quant_analytics.py                # Full report for all dates
    python quant_analytics.py --date 2026-02-12  # Single day
    python quant_analytics.py --live         # Live dashboard (refreshes)
"""

import json
import os
import sys
import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import statistics

# ============================================================
# SECTOR MAP — Every F&O stock → sector
# ============================================================
SECTOR_MAP = {
    # Banks
    "SBIN": "BANKS", "HDFCBANK": "BANKS", "ICICIBANK": "BANKS", "AXISBANK": "BANKS",
    "KOTAKBANK": "BANKS", "BANKBARODA": "BANKS", "PNB": "BANKS", "INDUSINDBK": "BANKS",
    "FEDERALBNK": "BANKS", "IDFCFIRSTB": "BANKS", "BANDHANBNK": "BANKS", "CANBK": "BANKS",
    "UNIONBANK": "BANKS", "AUBANK": "BANKS",
    # Financial Services
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC", "MUTHOOTFIN": "NBFC",
    "M&MFIN": "NBFC", "SHRIRAMFIN": "NBFC", "LICHSGFIN": "NBFC", "MANAPPURAM": "NBFC",
    "PFC": "NBFC", "RECLTD": "NBFC",
    # IT
    "INFY": "IT", "TCS": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    # Metals
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "JINDALSTEL": "METALS", "HINDALCO": "METALS",
    "VEDL": "METALS", "NATIONALUM": "METALS", "NMDC": "METALS", "SAIL": "METALS",
    "HINDCOPPER": "METALS", "COALINDIA": "METALS",
    # Oil & Energy
    "RELIANCE": "OIL_ENERGY", "ONGC": "OIL_ENERGY", "BPCL": "OIL_ENERGY", "IOC": "OIL_ENERGY",
    "GAIL": "OIL_ENERGY", "NTPC": "OIL_ENERGY", "POWERGRID": "OIL_ENERGY", "TATAPOWER": "OIL_ENERGY",
    "ADANIPOWER": "OIL_ENERGY", "ADANIENT": "OIL_ENERGY", "ADANIGREEN": "OIL_ENERGY",
    "PETRONET": "OIL_ENERGY", "HINDPETRO": "OIL_ENERGY",
    # Auto
    "TATAMOTORS": "AUTO", "MARUTI": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO", "ASHOKLEY": "AUTO", "TVSMOTOR": "AUTO",
    "MOTHERSON": "AUTO", "BALKRISIND": "AUTO", "MRF": "AUTO", "APOLLOTYRE": "AUTO",
    "EXIDEIND": "AUTO",
    # Pharma
    "SUNPHARMA": "PHARMA", "CIPLA": "PHARMA", "DRREDDY": "PHARMA", "DIVISLAB": "PHARMA",
    "AUROPHARMA": "PHARMA", "BIOCON": "PHARMA", "LUPIN": "PHARMA", "LAURUSLABS": "PHARMA",
    "GLENMARK": "PHARMA", "IPCALAB": "PHARMA", "ALKEM": "PHARMA",
    # FMCG
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG", "DABUR": "FMCG",
    "BRITANNIA": "FMCG", "GODREJCP": "FMCG", "MARICO": "FMCG", "COLPAL": "FMCG",
    "TATACONSUM": "FMCG", "UBL": "FMCG", "MCDOWELL-N": "FMCG",
    # Infra / Capital Goods
    "LT": "INFRA", "LTTS": "INFRA", "ABB": "INFRA", "SIEMENS": "INFRA",
    "BHEL": "INFRA", "BEL": "INFRA", "HAL": "INFRA", "GRASIM": "INFRA",
    "ULTRACEMCO": "INFRA", "SHREECEM": "INFRA", "AMBUJACEM": "INFRA",
    "ACC": "INFRA", "DALBHARAT": "INFRA", "RAMCOCEM": "INFRA",
    # Telecom
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM",
    # Consumer / Retail
    "TITAN": "CONSUMER", "TRENT": "CONSUMER", "PAGEIND": "CONSUMER",
    "ZOMATO": "CONSUMER", "NYKAA": "CONSUMER", "DMART": "CONSUMER",
    # Insurance
    "SBILIFE": "INSURANCE", "HDFCLIFE": "INSURANCE", "ICICIGI": "INSURANCE",
    "ICICIPRULI": "INSURANCE",
    # Chemicals
    "PIDILITIND": "CHEMICALS", "UPL": "CHEMICALS", "ATUL": "CHEMICALS",
    "DEEPAKNTR": "CHEMICALS", "SRF": "CHEMICALS", "NAVINFLUOR": "CHEMICALS",
    # Index
    "NIFTY 50": "INDEX", "NIFTY BANK": "INDEX", "NIFTY FIN SERVICE": "INDEX",
}

def _get_sector(symbol: str) -> str:
    """Extract sector from trade symbol"""
    # Handle NSE:SYMBOL, NFO:SYMBOLyyMMMddSTRIKECE/PE, etc.
    name = symbol.split(":")[-1] if ":" in symbol else symbol
    # Strip NFO suffixes (e.g., SBIN26FEB180CE → SBIN)
    # F&O symbols: NAME + yy + MONTH + STRIKE + CE/PE
    import re
    match = re.match(r'^([A-Z&-]+?)(?:\d{2}[A-Z]{3})', name)
    if match:
        name = match.group(1)
    return SECTOR_MAP.get(name, "OTHER")


def _get_underlying_name(symbol: str) -> str:
    """Extract clean underlying name from any symbol format"""
    name = symbol.split(":")[-1] if ":" in symbol else symbol
    import re
    match = re.match(r'^([A-Z&-]+?)(?:\d{2}[A-Z]{3})', name)
    if match:
        return match.group(1)
    return name


# ============================================================
# DATA LOADER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_trade_history() -> List[Dict]:
    """Load all historical trades"""
    path = os.path.join(BASE_DIR, 'trade_history.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

def load_daily_summary(dt: str) -> Optional[Dict]:
    """Load a daily summary file"""
    path = os.path.join(BASE_DIR, 'daily_summaries', f'daily_summary_{dt}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def load_slippage_log() -> List[Dict]:
    """Load slippage tracking data"""
    path = os.path.join(BASE_DIR, 'slippage_log.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# CORE METRICS
# ============================================================

@dataclass
class PortfolioMetrics:
    """Institutional-grade portfolio metrics"""
    # Basic
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    
    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk-adjusted
    profit_factor: float = 0.0        # gross_profit / |gross_loss|
    payoff_ratio: float = 0.0         # avg_winner / |avg_loser|
    expectancy: float = 0.0           # (win_rate × avg_win) - (loss_rate × avg_loss)
    expectancy_per_rupee: float = 0.0 # expectancy / avg_risk
    
    # R-Multiple Analysis
    avg_r_multiple: float = 0.0
    median_r_multiple: float = 0.0
    r_above_1: int = 0               # Trades that made > 1R
    r_above_2: int = 0               # Trades that made > 2R
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    
    # Time
    avg_hold_minutes: float = 0.0
    median_hold_minutes: float = 0.0
    
    # Sharpe-like metrics (using trade returns)
    sharpe_ratio: float = 0.0        # mean(returns) / std(returns) * sqrt(252)
    sortino_ratio: float = 0.0       # mean(returns) / downside_std * sqrt(252)
    
    # Streaks
    current_streak: int = 0           # +N = win streak, -N = loss streak
    best_streak: int = 0
    worst_streak: int = 0


def compute_portfolio_metrics(trades: List[Dict], capital: float = 500000) -> PortfolioMetrics:
    """Compute all portfolio metrics from trade list"""
    m = PortfolioMetrics()
    
    if not trades:
        return m
    
    m.total_trades = len(trades)
    
    pnls = []
    r_multiples = []
    hold_times = []
    returns = []  # % returns for Sharpe calculation
    
    for t in trades:
        pnl = t.get('pnl', 0) or 0
        pnls.append(pnl)
        
        # Classify
        if pnl > 0:
            m.wins += 1
            m.gross_profit += pnl
        elif pnl < 0:
            m.losses += 1
            m.gross_loss += abs(pnl)
        else:
            m.breakevens += 1
        
        # R-multiple from exit_detail
        ed = t.get('exit_detail', {}) or {}
        r = ed.get('r_multiple_achieved', 0) or 0
        r_multiples.append(r)
        if r >= 1.0:
            m.r_above_1 += 1
        if r >= 2.0:
            m.r_above_2 += 1
        
        # Hold time
        ts_entry = t.get('timestamp', '')
        ts_exit = t.get('exit_time', '') or t.get('closed_at', '')
        if ts_entry and ts_exit:
            try:
                dt_entry = datetime.fromisoformat(ts_entry)
                dt_exit = datetime.fromisoformat(ts_exit)
                hold_min = (dt_exit - dt_entry).total_seconds() / 60
                hold_times.append(hold_min)
            except Exception:
                pass
        
        # % return for Sharpe
        cost = abs(t.get('avg_price', 1) * t.get('quantity', 1))
        if cost > 0:
            returns.append(pnl / cost)
    
    # Win rate
    m.win_rate = m.wins / m.total_trades * 100 if m.total_trades > 0 else 0
    
    # P&L stats
    m.total_pnl = sum(pnls)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    m.avg_winner = statistics.mean(winners) if winners else 0
    m.avg_loser = statistics.mean(losers) if losers else 0
    m.largest_win = max(pnls) if pnls else 0
    m.largest_loss = min(pnls) if pnls else 0
    
    # Risk-adjusted
    m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss > 0 else float('inf')
    m.payoff_ratio = m.avg_winner / abs(m.avg_loser) if m.avg_loser != 0 else float('inf')
    m.expectancy = (m.win_rate/100 * m.avg_winner) + ((1 - m.win_rate/100) * m.avg_loser)
    
    # R-multiples
    if r_multiples:
        m.avg_r_multiple = statistics.mean(r_multiples)
        m.median_r_multiple = statistics.median(r_multiples)
    
    # Drawdown
    equity_curve = []
    running = capital
    for pnl in pnls:
        running += pnl
        equity_curve.append(running)
    
    if equity_curve:
        peak = capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        m.max_drawdown = max_dd
        m.max_drawdown_pct = (max_dd / capital * 100) if capital > 0 else 0
    
    # Consecutive losses
    streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    current = 0
    for pnl in pnls:
        if pnl >= 0:
            if current < 0:
                current = 0
            current += 1
            max_win_streak = max(max_win_streak, current)
        else:
            if current > 0:
                current = 0
            current -= 1
            max_loss_streak = max(max_loss_streak, abs(current))
    
    m.max_consecutive_losses = max_loss_streak
    m.best_streak = max_win_streak
    m.worst_streak = max_loss_streak
    m.current_streak = current
    
    # Hold time
    if hold_times:
        m.avg_hold_minutes = statistics.mean(hold_times)
        m.median_hold_minutes = statistics.median(hold_times)
    
    # Sharpe & Sortino (annualized using ~252 trading days, ~40 trades/day potential)
    if len(returns) > 1:
        mean_ret = statistics.mean(returns)
        std_ret = statistics.stdev(returns)
        m.sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0
        
        downside = [r for r in returns if r < 0]
        if downside:
            downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
            m.sortino_ratio = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 0 else 0
    
    return m


# ============================================================
# ATTRIBUTION ANALYSIS (P&L breakdown by dimension)
# ============================================================

@dataclass
class AttributionBucket:
    """P&L attribution for one category"""
    name: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0
    avg_score: float = 0.0
    avg_hold_min: float = 0.0


def compute_attribution(trades: List[Dict], key_fn, label: str = "Category") -> List[AttributionBucket]:
    """Group trades by key_fn and compute stats per bucket"""
    buckets: Dict[str, AttributionBucket] = {}
    
    for t in trades:
        key = key_fn(t)
        if key not in buckets:
            buckets[key] = AttributionBucket(name=key)
        b = buckets[key]
        b.trades += 1
        pnl = t.get('pnl', 0) or 0
        b.pnl += pnl
        if pnl > 0:
            b.wins += 1
        elif pnl < 0:
            b.losses += 1
        
        ed = t.get('exit_detail', {}) or {}
        r = ed.get('r_multiple_achieved', 0) or 0
        b.avg_r += r
        b.avg_score += (t.get('entry_score', 0) or 0)
        
        ts_entry = t.get('timestamp', '')
        ts_exit = t.get('exit_time', '') or t.get('closed_at', '')
        if ts_entry and ts_exit:
            try:
                dt_entry = datetime.fromisoformat(ts_entry)
                dt_exit = datetime.fromisoformat(ts_exit)
                b.avg_hold_min += (dt_exit - dt_entry).total_seconds() / 60
            except Exception:
                pass
    
    result = []
    for b in sorted(buckets.values(), key=lambda x: x.pnl, reverse=True):
        if b.trades > 0:
            b.win_rate = b.wins / b.trades * 100
            b.avg_r = b.avg_r / b.trades
            b.avg_score = b.avg_score / b.trades
            b.avg_hold_min = b.avg_hold_min / b.trades
        result.append(b)
    
    return result


# ============================================================
# SECTOR MONEY FLOW ANALYSIS
# ============================================================

def compute_sector_flow(trades: List[Dict]) -> List[Dict]:
    """
    Compute sector-level money flow from trades.
    Positive P&L in a sector = Titan correctly rode that sector's flow.
    Negative P&L = Titan was contra-flow.
    
    Returns sorted list of sector performance dicts.
    """
    sector_data = defaultdict(lambda: {
        'sector': '', 'trades': 0, 'wins': 0, 'losses': 0,
        'pnl': 0.0, 'win_rate': 0.0, 'stocks': set(),
        'avg_score': 0.0, 'avg_r': 0.0
    })
    
    for t in trades:
        symbol = t.get('underlying', t.get('symbol', ''))
        sector = _get_sector(symbol)
        underlying = _get_underlying_name(symbol)
        
        s = sector_data[sector]
        s['sector'] = sector
        s['trades'] += 1
        pnl = t.get('pnl', 0) or 0
        s['pnl'] += pnl
        if pnl > 0:
            s['wins'] += 1
        else:
            s['losses'] += 1
        s['stocks'].add(underlying)
        s['avg_score'] += (t.get('entry_score', 0) or 0)
        ed = t.get('exit_detail', {}) or {}
        s['avg_r'] += (ed.get('r_multiple_achieved', 0) or 0)
    
    result = []
    for s in sector_data.values():
        if s['trades'] > 0:
            s['win_rate'] = s['wins'] / s['trades'] * 100
            s['avg_score'] = s['avg_score'] / s['trades']
            s['avg_r'] = s['avg_r'] / s['trades']
            s['stocks'] = sorted(s['stocks'])
            s['stock_count'] = len(s['stocks'])
        result.append(s)
    
    return sorted(result, key=lambda x: x['pnl'], reverse=True)


# ============================================================
# SCORE → OUTCOME CORRELATION (Edge Quality)
# ============================================================

def compute_score_edge_analysis(trades: List[Dict]) -> Dict:
    """
    Analyze whether higher entry scores produce better outcomes.
    This is the MOST IMPORTANT quant metric — it validates the entire scoring system.
    
    Returns correlation data + score band performance.
    """
    scored_trades = [(t.get('entry_score', 0) or 0, t.get('pnl', 0) or 0) for t in trades if t.get('entry_score')]
    
    if len(scored_trades) < 5:
        return {'correlation': 0, 'bands': [], 'edge_valid': False, 'sample_size': len(scored_trades), 'verdict': 'INSUFFICIENT DATA'}
    
    scores, pnls = zip(*scored_trades)
    
    # Pearson correlation (score vs P&L)
    n = len(scores)
    mean_s = sum(scores) / n
    mean_p = sum(pnls) / n
    
    cov = sum((s - mean_s) * (p - mean_p) for s, p in zip(scores, pnls)) / n
    std_s = math.sqrt(sum((s - mean_s)**2 for s in scores) / n)
    std_p = math.sqrt(sum((p - mean_p)**2 for p in pnls) / n)
    
    correlation = cov / (std_s * std_p) if (std_s > 0 and std_p > 0) else 0
    
    # Score bands
    bands = [
        ("45-54 (Base)", 45, 54),
        ("55-64 (Standard)", 55, 64),
        ("65-74 (Premium)", 65, 74),
        ("75+ (Elite)", 75, 100),
    ]
    
    band_stats = []
    for label, lo, hi in bands:
        band_trades = [t for t in trades if lo <= (t.get('entry_score', 0) or 0) <= hi]
        if band_trades:
            band_pnl = sum(t.get('pnl', 0) or 0 for t in band_trades)
            band_wins = sum(1 for t in band_trades if (t.get('pnl', 0) or 0) > 0)
            band_wr = band_wins / len(band_trades) * 100
            band_stats.append({
                'band': label, 'trades': len(band_trades),
                'pnl': band_pnl, 'win_rate': band_wr,
                'avg_pnl': band_pnl / len(band_trades)
            })
    
    return {
        'correlation': round(correlation, 3),
        'bands': band_stats,
        'edge_valid': correlation > 0.1,  # Positive correlation = scoring works
        'sample_size': len(scored_trades),
        'verdict': 'SCORING SYSTEM HAS EDGE' if correlation > 0.1 else 
                   'SCORING SYSTEM NEEDS CALIBRATION' if correlation > -0.05 else
                   'SCORING SYSTEM IS INVERTED — FIX URGENTLY'
    }


# ============================================================
# TIME-OF-DAY ANALYSIS
# ============================================================

def compute_time_analysis(trades: List[Dict]) -> List[Dict]:
    """Break down performance by entry hour"""
    hourly = defaultdict(lambda: {'hour': '', 'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    for t in trades:
        ts = t.get('timestamp', '')
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            hour_label = f"{dt.hour:02d}:00-{dt.hour:02d}:59"
            h = hourly[hour_label]
            h['hour'] = hour_label
            h['trades'] += 1
            pnl = t.get('pnl', 0) or 0
            h['pnl'] += pnl
            if pnl > 0:
                h['wins'] += 1
        except Exception:
            pass
    
    result = []
    for h in sorted(hourly.values(), key=lambda x: x['hour']):
        if h['trades'] > 0:
            h['win_rate'] = h['wins'] / h['trades'] * 100
            h['avg_pnl'] = h['pnl'] / h['trades']
        result.append(h)
    return result


# ============================================================
# EXIT QUALITY ANALYSIS
# ============================================================

def compute_exit_analysis(trades: List[Dict]) -> Dict:
    """
    Analyze exit quality — are we capturing enough of the favorable move?
    Key metric: MFE (Max Favorable Excursion) capture ratio
    """
    exit_types = defaultdict(lambda: {'type': '', 'count': 0, 'pnl': 0.0, 'avg_r': 0.0})
    
    mfe_captured = []  # How much of MFE did we capture?
    r_at_exit = []
    candles = []
    
    for t in trades:
        ed = t.get('exit_detail', {}) or {}
        
        # Exit type breakdown
        etype = ed.get('exit_type', t.get('result', 'UNKNOWN'))
        e = exit_types[etype]
        e['type'] = etype
        e['count'] += 1
        pnl = t.get('pnl', 0) or 0
        e['pnl'] += pnl
        r = ed.get('r_multiple_achieved', 0) or 0
        e['avg_r'] += r
        
        # MFE capture
        mfe = ed.get('max_favorable_excursion', 0) or 0
        entry = t.get('avg_price', 0) or 0
        exit_p = t.get('exit_price', 0) or 0
        if mfe > 0 and entry > 0:
            actual_move = abs(exit_p - entry)
            max_move = abs(mfe - entry)
            if max_move > 0:
                capture = actual_move / max_move
                mfe_captured.append(capture)
        
        if r:
            r_at_exit.append(r)
        
        c = ed.get('candles_held', 0) or 0
        if c > 0:
            candles.append(c)
    
    # Finalize exit type averages
    exit_list = []
    for e in sorted(exit_types.values(), key=lambda x: x['count'], reverse=True):
        if e['count'] > 0:
            e['avg_r'] = e['avg_r'] / e['count']
            e['avg_pnl'] = e['pnl'] / e['count']
        exit_list.append(e)
    
    return {
        'exit_types': exit_list,
        'mfe_capture_avg': statistics.mean(mfe_captured) * 100 if mfe_captured else 0,
        'mfe_capture_median': statistics.median(mfe_captured) * 100 if mfe_captured else 0,
        'avg_candles_held': statistics.mean(candles) if candles else 0,
        'avg_r_at_exit': statistics.mean(r_at_exit) if r_at_exit else 0,
    }


# ============================================================
# DAILY EQUITY CURVE
# ============================================================

def compute_daily_equity_curve(trades: List[Dict], capital: float = 500000) -> List[Dict]:
    """Build day-by-day equity curve"""
    daily = defaultdict(lambda: {'date': '', 'trades': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0})
    
    for t in trades:
        ts = t.get('timestamp', '')
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            d = str(dt.date())
            day = daily[d]
            day['date'] = d
            day['trades'] += 1
            pnl = t.get('pnl', 0) or 0
            day['pnl'] += pnl
            if pnl > 0:
                day['wins'] += 1
            elif pnl < 0:
                day['losses'] += 1
        except Exception:
            pass
    
    curve = []
    running = capital
    for d in sorted(daily.keys()):
        day = daily[d]
        running += day['pnl']
        day['equity'] = running
        day['return_pct'] = (running - capital) / capital * 100
        day['win_rate'] = day['wins'] / day['trades'] * 100 if day['trades'] > 0 else 0
        curve.append(day)
    
    return curve


# ============================================================
# ROLLING METRICS (Edge Decay Detection)
# ============================================================

def compute_rolling_metrics(trades: List[Dict], window: int = 10) -> List[Dict]:
    """Rolling win rate and R-multiple over last N trades"""
    if len(trades) < window:
        return []
    
    results = []
    for i in range(window, len(trades) + 1):
        batch = trades[i-window:i]
        wins = sum(1 for t in batch if (t.get('pnl', 0) or 0) > 0)
        wr = wins / window * 100
        r_vals = [((t.get('exit_detail', {}) or {}).get('r_multiple_achieved', 0) or 0) for t in batch]
        avg_r = statistics.mean(r_vals) if r_vals else 0
        pnl = sum(t.get('pnl', 0) or 0 for t in batch)
        
        results.append({
            'trade_num': i,
            'rolling_win_rate': round(wr, 1),
            'rolling_avg_r': round(avg_r, 3),
            'rolling_pnl': round(pnl, 0),
            'symbol': trades[i-1].get('symbol', ''),
            'timestamp': trades[i-1].get('timestamp', ''),
        })
    
    return results


# ============================================================
# REPORT GENERATOR
# ============================================================

def generate_full_report(trades: List[Dict], capital: float = 500000, 
                         filter_date: str = None) -> str:
    """Generate full quantitative performance report"""
    
    # Filter by date if specified
    if filter_date:
        trades = [t for t in trades if t.get('timestamp', '').startswith(filter_date)]
    
    if not trades:
        return "No trades found for the specified period."
    
    # Sort by timestamp
    trades.sort(key=lambda t: t.get('timestamp', ''))
    
    # Compute all metrics
    metrics = compute_portfolio_metrics(trades, capital)
    
    # Attribution
    by_strategy = compute_attribution(trades, 
        lambda t: t.get('strategy_type', 'EQUITY'))
    by_tier = compute_attribution(trades,
        lambda t: t.get('score_tier', t.get('entry_metadata', {}).get('score_tier', 'unknown')))
    by_exit = compute_attribution(trades,
        lambda t: (t.get('exit_detail', {}) or {}).get('exit_type', t.get('result', 'UNKNOWN')))
    by_sector = compute_sector_flow(trades)
    
    # Edge analysis
    score_edge = compute_score_edge_analysis(trades)
    time_analysis = compute_time_analysis(trades)
    exit_analysis = compute_exit_analysis(trades)
    equity_curve = compute_daily_equity_curve(trades, capital)
    rolling = compute_rolling_metrics(trades, window=min(10, max(3, len(trades)//3)))
    
    # Build report
    lines = []
    w = lines.append
    
    date_range = f"{trades[0].get('timestamp', '?')[:10]} → {trades[-1].get('timestamp', '?')[:10]}"
    w("=" * 72)
    w("  TITAN QUANTITATIVE PERFORMANCE REPORT")
    w(f"  Period: {date_range}  |  {metrics.total_trades} trades")
    w("=" * 72)
    
    # 1. PORTFOLIO OVERVIEW
    w("\n┌─── PORTFOLIO OVERVIEW ──────────────────────────────────────┐")
    w(f"│  Total P&L:        ₹{metrics.total_pnl:>+12,.0f}                          │")
    w(f"│  Win Rate:         {metrics.win_rate:>6.1f}%  ({metrics.wins}W / {metrics.losses}L / {metrics.breakevens}BE)       │")
    w(f"│  Profit Factor:    {metrics.profit_factor:>6.2f}    (>1.5 = good)                │")
    w(f"│  Payoff Ratio:     {metrics.payoff_ratio:>6.2f}    (avg win / avg loss)          │")
    w(f"│  Expectancy:       ₹{metrics.expectancy:>+10,.0f}/trade                      │")
    w(f"│  Sharpe Ratio:     {metrics.sharpe_ratio:>6.2f}    (>1.0 = good, >2.0 = great) │")
    w(f"│  Sortino Ratio:    {metrics.sortino_ratio:>6.2f}    (>1.5 = good)                │")
    w(f"│  Max Drawdown:     ₹{metrics.max_drawdown:>10,.0f}  ({metrics.max_drawdown_pct:.1f}%)             │")
    w(f"│  Max Loss Streak:  {metrics.max_consecutive_losses:>3d} trades                              │")
    w(f"│  Avg Hold Time:    {metrics.avg_hold_minutes:>5.0f} min  (median: {metrics.median_hold_minutes:.0f} min)       │")
    w(f"│  Avg R-Multiple:   {metrics.avg_r_multiple:>5.2f}   (median: {metrics.median_r_multiple:.2f})           │")
    w(f"│  Trades > 1R:      {metrics.r_above_1:>3d}    Trades > 2R: {metrics.r_above_2:>3d}                │")
    w(f"│  Largest Win:      ₹{metrics.largest_win:>+10,.0f}                              │")
    w(f"│  Largest Loss:     ₹{metrics.largest_loss:>+10,.0f}                              │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 2. SCORE → OUTCOME EDGE VALIDATION
    w("\n┌─── SCORE → OUTCOME EDGE VALIDATION ────────────────────────┐")
    w(f"│  Correlation (score ↔ P&L): {score_edge['correlation']:>+6.3f}                        │")
    w(f"│  Verdict: {score_edge['verdict']:<49s}│")
    w(f"│  Sample: {score_edge['sample_size']} scored trades                                │")
    if score_edge['bands']:
        w("│                                                            │")
        w("│  Band           Trades  Win%   Avg P&L    Total P&L       │")
        w("│  ──────────────────────────────────────────────────────    │")
        for b in score_edge['bands']:
            w(f"│  {b['band']:<16s} {b['trades']:>4d}   {b['win_rate']:>5.1f}%  ₹{b['avg_pnl']:>+8,.0f}  ₹{b['pnl']:>+10,.0f}   │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 3. STRATEGY ATTRIBUTION
    w("\n┌─── P&L BY STRATEGY ────────────────────────────────────────┐")
    w("│  Strategy         Trades  Win%    P&L        Avg R          │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for s in by_strategy:
        w(f"│  {s.name:<18s} {s.trades:>4d}   {s.win_rate:>5.1f}%  ₹{s.pnl:>+10,.0f}  {s.avg_r:>+5.2f}R       │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 4. SCORE TIER ATTRIBUTION
    w("\n┌─── P&L BY SCORE TIER ──────────────────────────────────────┐")
    w("│  Tier            Trades  Win%    P&L        Avg Score       │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for s in by_tier:
        w(f"│  {s.name:<17s} {s.trades:>4d}   {s.win_rate:>5.1f}%  ₹{s.pnl:>+10,.0f}  {s.avg_score:>6.1f}         │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 5. SECTOR FLOW
    w("\n┌─── SECTOR MONEY FLOW ──────────────────────────────────────┐")
    w("│  Sector         Trades  Win%    P&L        Stocks           │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for s in by_sector:
        stocks_str = ','.join(s['stocks'][:3])
        if len(s['stocks']) > 3:
            stocks_str += f"+{len(s['stocks'])-3}"
        w(f"│  {s['sector']:<15s} {s['trades']:>4d}   {s['win_rate']:>5.1f}%  ₹{s['pnl']:>+10,.0f}  {stocks_str:<15s}│")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 6. TIME OF DAY
    w("\n┌─── TIME OF DAY ANALYSIS ───────────────────────────────────┐")
    w("│  Hour         Trades  Win%    P&L        Avg P&L            │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for h in time_analysis:
        bar = "█" * max(0, min(20, int(h['pnl'] / 500))) if h['pnl'] > 0 else "▓" * max(0, min(20, int(abs(h['pnl']) / 500)))
        prefix = "+" if h['pnl'] > 0 else "-"
        w(f"│  {h['hour']:<13s} {h['trades']:>4d}   {h['win_rate']:>5.1f}%  ₹{h['pnl']:>+10,.0f}  ₹{h.get('avg_pnl',0):>+7,.0f}    │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 7. EXIT ANALYSIS
    w("\n┌─── EXIT QUALITY ───────────────────────────────────────────┐")
    w(f"│  MFE Capture (avg):  {exit_analysis['mfe_capture_avg']:>5.1f}%  (how much of the best move we keep) │")
    w(f"│  MFE Capture (med):  {exit_analysis['mfe_capture_median']:>5.1f}%                                    │")
    w(f"│  Avg Candles Held:   {exit_analysis['avg_candles_held']:>5.1f}                                      │")
    w(f"│  Avg R at Exit:      {exit_analysis['avg_r_at_exit']:>+5.2f}                                      │")
    w("│                                                              │")
    w("│  Exit Type           Count   P&L        Avg R               │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for e in exit_analysis['exit_types'][:8]:
        w(f"│  {e['type']:<21s} {e['count']:>4d}   ₹{e['pnl']:>+10,.0f}  {e['avg_r']:>+5.2f}R            │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 8. EQUITY CURVE
    w("\n┌─── DAILY EQUITY CURVE ─────────────────────────────────────┐")
    w("│  Date        Trades  W/L   Day P&L      Cum Return         │")
    w("│  ──────────────────────────────────────────────────────────  │")
    for d in equity_curve:
        emoji = "🟢" if d['pnl'] > 0 else "🔴" if d['pnl'] < 0 else "⚪"
        w(f"│  {emoji} {d['date']}   {d['trades']:>4d}   {d['wins']}/{d['losses']:>d}   ₹{d['pnl']:>+10,.0f}  {d['return_pct']:>+6.2f}%       │")
    w("└────────────────────────────────────────────────────────────────┘")
    
    # 9. ROLLING EDGE DECAY
    if rolling:
        w("\n┌─── ROLLING EDGE (last 10-trade window) ────────────────────┐")
        # Show first, middle, last entries
        samples = [rolling[0], rolling[len(rolling)//2], rolling[-1]]
        for r in samples:
            w(f"│  Trade #{r['trade_num']:>3d}: WR {r['rolling_win_rate']:>5.1f}%  R {r['rolling_avg_r']:>+.2f}  P&L ₹{r['rolling_pnl']:>+8,.0f}  │")
        
        # Trend detection
        if len(rolling) >= 3:
            first_wr = rolling[0]['rolling_win_rate']
            last_wr = rolling[-1]['rolling_win_rate']
            wr_trend = "IMPROVING ↑" if last_wr > first_wr + 5 else "DECLINING ↓" if last_wr < first_wr - 5 else "STABLE →"
            w(f"│  Win Rate Trend: {wr_trend:<42s}│")
        w("└────────────────────────────────────────────────────────────────┘")
    
    # 10. ACTIONABLE INSIGHTS
    w("\n┌─── QUANT INSIGHTS & RECOMMENDATIONS ───────────────────────┐")
    insights = []
    
    # Score edge
    if score_edge['correlation'] < 0:
        insights.append("⚠️  CRITICAL: Score correlation is NEGATIVE — higher scores produce WORSE outcomes. Scoring system needs recalibration.")
    elif score_edge['correlation'] < 0.1:
        insights.append("⚡ Score→P&L correlation is weak. Consider tightening entry thresholds to concentrate on higher-conviction setups.")
    else:
        insights.append(f"✅ Scoring system has positive edge (r={score_edge['correlation']:.3f}). Higher scores → better outcomes.")
    
    # Win rate
    if metrics.win_rate < 40:
        insights.append(f"⚠️  Win rate {metrics.win_rate:.1f}% is below 40%. Either tighten entries or widen targets.")
    elif metrics.win_rate > 60:
        insights.append(f"✅ Win rate {metrics.win_rate:.1f}% is strong. May be leaving money on table — check if targets are too tight.")
    
    # Profit factor
    if metrics.profit_factor < 1.0:
        insights.append(f"🔴 Profit Factor {metrics.profit_factor:.2f} < 1.0 — system is LOSING money. Fix before adding capital.")
    elif metrics.profit_factor < 1.5:
        insights.append(f"⚡ Profit Factor {metrics.profit_factor:.2f} — marginal edge. Focus on cutting losers faster (trailing SL).")
    
    # Best/worst sectors
    if by_sector:
        best = by_sector[0]
        worst = by_sector[-1]
        if best['pnl'] > 0:
            insights.append(f"✅ Best sector: {best['sector']} (+₹{best['pnl']:,.0f} from {','.join(best['stocks'][:3])})")
        if worst['pnl'] < 0:
            insights.append(f"🔴 Worst sector: {worst['sector']} (₹{worst['pnl']:,.0f}) — consider sector filters or reduced sizing")
    
    # Best/worst time
    if time_analysis:
        best_time = max(time_analysis, key=lambda x: x['pnl'])
        worst_time = min(time_analysis, key=lambda x: x['pnl'])
        if best_time['pnl'] > 0:
            insights.append(f"✅ Best hour: {best_time['hour']} → ₹{best_time['pnl']:+,.0f} ({best_time['trades']} trades)")
        if worst_time['pnl'] < 0:
            insights.append(f"🔴 Worst hour: {worst_time['hour']} → ₹{worst_time['pnl']:+,.0f} — consider blocking this window")
    
    # MFE capture
    if exit_analysis['mfe_capture_avg'] < 30:
        insights.append(f"⚡ MFE capture only {exit_analysis['mfe_capture_avg']:.0f}% — exits are too early. Consider wider trailing SL.")
    elif exit_analysis['mfe_capture_avg'] > 70:
        insights.append(f"✅ MFE capture {exit_analysis['mfe_capture_avg']:.0f}% — excellent exit quality.")
    
    # Best strategy
    if by_strategy:
        best_strat = max(by_strategy, key=lambda x: x.pnl)
        worst_strat = min(by_strategy, key=lambda x: x.pnl)
        if best_strat.pnl > 0:
            insights.append(f"✅ Best strategy: {best_strat.name} (₹{best_strat.pnl:+,.0f}, {best_strat.win_rate:.0f}% WR)")
        if worst_strat.pnl < 0 and worst_strat.name != best_strat.name:
            insights.append(f"🔴 Worst strategy: {worst_strat.name} (₹{worst_strat.pnl:+,.0f}) — review or disable")
    
    for i, insight in enumerate(insights):
        w(f"│  {insight:<60s}│")
    w("└────────────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Titan Quantitative Analytics')
    parser.add_argument('--date', '-d', type=str, help='Filter to specific date (YYYY-MM-DD)')
    parser.add_argument('--json', action='store_true', help='Output raw JSON instead of report')
    parser.add_argument('--capital', '-c', type=float, default=500000, help='Starting capital')
    args = parser.parse_args()
    
    trades = load_trade_history()
    if not trades:
        print("No trade history found.")
        return
    
    if args.json:
        # JSON output for programmatic consumption
        if args.date:
            trades = [t for t in trades if t.get('timestamp', '').startswith(args.date)]
        metrics = compute_portfolio_metrics(trades, args.capital)
        score_edge = compute_score_edge_analysis(trades)
        sector = compute_sector_flow(trades)
        time_data = compute_time_analysis(trades)
        
        output = {
            'metrics': {k: v for k, v in metrics.__dict__.items()},
            'score_edge': score_edge,
            'sector_flow': [{k: v for k, v in s.items() if k != 'stocks'} for s in sector],
            'time_analysis': time_data,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        report = generate_full_report(trades, args.capital, args.date)
        print(report)


if __name__ == '__main__':
    main()
