"""
Check exact monthly expiry dates for stock F&O from Zerodha API.
Shows all upcoming expiry dates for each F&O stock.
Usage: python _check_stock_expiry.py [SYMBOL]
    No args → shows summary of all stocks
    With SYMBOL → shows details for that stock
"""
import json, os, sys
from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

def get_kite():
    """Initialize KiteConnect with token from .env"""
    from kiteconnect import KiteConnect
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    
    if not access_token:
        print("❌ ZERODHA_ACCESS_TOKEN not set in .env")
        sys.exit(1)
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def check_expiry(symbol=None):
    kite = get_kite()
    print("📡 Fetching NFO instruments from Zerodha...")
    instruments = kite.instruments("NFO")
    print(f"   Total NFO instruments: {len(instruments)}")
    
    # Group by underlying → expiry dates
    stock_expiries = defaultdict(set)
    stock_types = defaultdict(set)  # Track instrument types per stock
    
    for inst in instruments:
        name = inst.get('name', '')
        inst_type = inst.get('instrument_type', '')
        expiry = inst.get('expiry')
        
        if not expiry or not name:
            continue
        
        # We want stock options (CE/PE), not index options
        if inst_type in ('CE', 'PE'):
            exp_date = expiry if isinstance(expiry, date) else expiry.date() if hasattr(expiry, 'date') else None
            if exp_date and exp_date >= date.today():
                stock_expiries[name].add(exp_date)
                stock_types[name].add(inst_type)
    
    if symbol:
        # Detailed view for one stock
        symbol = symbol.upper()
        if symbol not in stock_expiries:
            # Try partial match
            matches = [s for s in stock_expiries if symbol in s]
            if matches:
                print(f"\n   Symbol '{symbol}' not found exactly. Did you mean: {', '.join(sorted(matches)[:10])}?")
            else:
                print(f"\n   ❌ No options found for '{symbol}'")
            return
        
        expiries = sorted(stock_expiries[symbol])
        print(f"\n{'='*60}")
        print(f"  📊 {symbol} — {len(expiries)} upcoming expiry dates")
        print(f"{'='*60}")
        
        today = date.today()
        current_month = today.month
        current_year = today.year
        
        # Group by month
        by_month = defaultdict(list)
        for exp in expiries:
            key = f"{exp.year}-{exp.month:02d}"
            by_month[key].append(exp)
        
        for month_key in sorted(by_month.keys()):
            month_expiries = sorted(by_month[month_key])
            month_label = datetime.strptime(month_key, "%Y-%m").strftime("%B %Y")
            
            # Find monthly expiry (last Thursday of month)
            last_exp = month_expiries[-1]
            is_current = (last_exp.month == current_month and last_exp.year == current_year)
            
            weekly_count = len(month_expiries) - 1  # Last one is monthly
            
            print(f"\n  📅 {month_label}:")
            for exp in month_expiries:
                day_name = exp.strftime("%A")
                days_away = (exp - today).days
                is_monthly = (exp == last_exp)
                marker = " ◀ MONTHLY" if is_monthly else ""
                if days_away == 0:
                    timing = "TODAY"
                elif days_away == 1:
                    timing = "TOMORROW"
                elif days_away < 0:
                    timing = f"{abs(days_away)}d ago"
                else:
                    timing = f"in {days_away}d"
                
                print(f"     {exp.strftime('%Y-%m-%d')} ({day_name:<9}) — {timing}{marker}")
            
            if weekly_count > 0:
                print(f"     ({weekly_count} weekly + 1 monthly expiry)")
            else:
                print(f"     (monthly expiry only)")
    
    else:
        # Summary view — all stocks
        print(f"\n{'='*70}")
        print(f"  STOCK F&O EXPIRY SUMMARY — {len(stock_expiries)} underlyings found")
        print(f"{'='*70}")
        
        today = date.today()
        
        # Separate indices from stocks
        INDICES = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX', 'NIFTYNXT50'}
        
        # Show indices first
        index_names = sorted([s for s in stock_expiries if s in INDICES])
        stock_names = sorted([s for s in stock_expiries if s not in INDICES])
        
        print(f"\n  📈 INDICES ({len(index_names)}):")
        for name in index_names:
            expiries = sorted(stock_expiries[name])
            nearest = expiries[0]
            monthly_candidates = [e for e in expiries if e.day >= 23]
            monthly = monthly_candidates[0] if monthly_candidates else expiries[-1]
            days_to_nearest = (nearest - today).days
            days_to_monthly = (monthly - today).days
            print(f"     {name:<15} next: {nearest} ({days_to_nearest}d) | monthly: {monthly} ({days_to_monthly}d) | {len(expiries)} expiries")
        
        # Show stocks — just nearest and monthly
        print(f"\n  📊 STOCKS ({len(stock_names)}):")
        print(f"     {'SYMBOL':<16} {'NEAREST EXPIRY':<16} {'DAYS':<6} {'MONTHLY EXPIRY':<16} {'DAYS':<6} {'TOTAL'}")
        print(f"     {'─'*16} {'─'*16} {'─'*6} {'─'*16} {'─'*6} {'─'*6}")
        
        for name in stock_names:
            expiries = sorted(stock_expiries[name])
            nearest = expiries[0]
            
            # Find monthly (last expiry of nearest month, typically last Thursday)
            nearest_month = nearest.month
            same_month = [e for e in expiries if e.month == nearest_month]
            monthly = same_month[-1]  # Last expiry in the nearest month = monthly
            
            days_to_nearest = (nearest - today).days
            days_to_monthly = (monthly - today).days
            
            print(f"     {name:<16} {nearest.isoformat():<16} {days_to_nearest:<6} {monthly.isoformat():<16} {days_to_monthly:<6} {len(expiries)}")
        
        # Summary stats
        all_monthly_dates = set()
        for name in stock_names:
            expiries = sorted(stock_expiries[name])
            nearest_month = expiries[0].month
            same_month = [e for e in expiries if e.month == nearest_month]
            all_monthly_dates.add(same_month[-1])
        
        if all_monthly_dates:
            print(f"\n  📌 UNIQUE MONTHLY EXPIRY DATES FOR STOCKS:")
            for d in sorted(all_monthly_dates):
                day_name = d.strftime("%A")
                days_away = (d - today).days
                print(f"     {d.isoformat()} ({day_name}) — in {days_away} days")


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else None
    check_expiry(target)
