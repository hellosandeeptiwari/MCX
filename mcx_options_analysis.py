"""
MCX Options Chain Fetcher - Robust NSE API Access
==================================================
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

class NSEOptionsChain:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }
        self.base_url = 'https://www.nseindia.com'
        self.cookies_set = False
    
    def _get_cookies(self):
        """Initialize session with NSE cookies"""
        try:
            # Visit main page first
            response = self.session.get(
                self.base_url,
                headers=self.headers,
                timeout=15
            )
            time.sleep(1)
            
            # Visit option chain page
            self.session.get(
                f'{self.base_url}/option-chain',
                headers=self.headers,
                timeout=15
            )
            time.sleep(1)
            
            self.cookies_set = True
            return True
        except Exception as e:
            print(f"Cookie fetch error: {e}")
            return False
    
    def get_option_chain(self, symbol='MCX'):
        """Fetch option chain for a symbol"""
        if not self.cookies_set:
            if not self._get_cookies():
                return None
        
        api_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.nseindia.com/option-chain',
            'Connection': 'keep-alive',
        }
        
        url = f'{self.base_url}/api/option-chain-equities?symbol={symbol}'
        
        try:
            response = self.session.get(url, headers=api_headers, timeout=20)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Status: {response.status_code}")
                return None
        except Exception as e:
            print(f"API Error: {e}")
            return None


def analyze_strategies(spot, options_df, lot_size=300):
    """Analyze various options strategies"""
    
    strategies = []
    
    # Find ATM strike
    atm_strike = options_df.iloc[(options_df['Strike'] - spot).abs().argsort()[:1]]['Strike'].values[0]
    
    # Get ATM row
    atm_row = options_df[options_df['Strike'] == atm_strike].iloc[0]
    
    # Nearby strikes
    otm_ce_strikes = options_df[options_df['Strike'] > atm_strike].head(3)['Strike'].tolist()
    otm_pe_strikes = options_df[options_df['Strike'] < atm_strike].tail(3)['Strike'].tolist()
    
    # 1. ATM Straddle
    straddle_premium = atm_row['CE_LTP'] + atm_row['PE_LTP']
    straddle_cost = straddle_premium * lot_size
    upper_be = atm_strike + straddle_premium
    lower_be = atm_strike - straddle_premium
    be_move = (straddle_premium / spot) * 100
    
    strategies.append({
        'name': 'Short ATM Straddle',
        'setup': f'Sell {atm_strike} CE @ Rs.{atm_row["CE_LTP"]:.2f} + Sell {atm_strike} PE @ Rs.{atm_row["PE_LTP"]:.2f}',
        'premium': straddle_cost,
        'max_profit': straddle_cost,
        'max_loss': 'Unlimited',
        'breakeven': f'Rs.{lower_be:.0f} to Rs.{upper_be:.0f}',
        'move_required': f'+/-{be_move:.1f}%',
        'view': 'Neutral - expect low volatility',
        'risk_level': 'HIGH',
        'score': 65 - be_move
    })
    
    strategies.append({
        'name': 'Long ATM Straddle',
        'setup': f'Buy {atm_strike} CE @ Rs.{atm_row["CE_LTP"]:.2f} + Buy {atm_strike} PE @ Rs.{atm_row["PE_LTP"]:.2f}',
        'premium': -straddle_cost,
        'max_profit': 'Unlimited',
        'max_loss': straddle_cost,
        'breakeven': f'Rs.{lower_be:.0f} to Rs.{upper_be:.0f}',
        'move_required': f'+/-{be_move:.1f}%',
        'view': 'High volatility expected - big move expected',
        'risk_level': 'MEDIUM',
        'score': 50 + (20 if be_move < 5 else 0)
    })
    
    # 2. Strangle
    if len(otm_ce_strikes) >= 1 and len(otm_pe_strikes) >= 1:
        ce_strike = otm_ce_strikes[0]
        pe_strike = otm_pe_strikes[-1]
        
        ce_prem = options_df[options_df['Strike'] == ce_strike].iloc[0]['CE_LTP']
        pe_prem = options_df[options_df['Strike'] == pe_strike].iloc[0]['PE_LTP']
        
        strangle_premium = ce_prem + pe_prem
        strangle_cost = strangle_premium * lot_size
        
        strategies.append({
            'name': 'Short Strangle',
            'setup': f'Sell {ce_strike} CE @ Rs.{ce_prem:.2f} + Sell {pe_strike} PE @ Rs.{pe_prem:.2f}',
            'premium': strangle_cost,
            'max_profit': strangle_cost,
            'max_loss': 'Unlimited',
            'breakeven': f'Rs.{pe_strike - strangle_premium:.0f} to Rs.{ce_strike + strangle_premium:.0f}',
            'move_required': f'{((ce_strike-spot)/spot)*100:.1f}% up / {((spot-pe_strike)/spot)*100:.1f}% down',
            'view': 'Neutral - wider range than straddle',
            'risk_level': 'HIGH',
            'score': 70
        })
    
    # 3. Iron Condor
    if len(otm_ce_strikes) >= 2 and len(otm_pe_strikes) >= 2:
        sell_ce = otm_ce_strikes[0]
        buy_ce = otm_ce_strikes[1]
        sell_pe = otm_pe_strikes[-1]
        buy_pe = otm_pe_strikes[-2]
        
        sell_ce_prem = options_df[options_df['Strike'] == sell_ce].iloc[0]['CE_LTP']
        buy_ce_prem = options_df[options_df['Strike'] == buy_ce].iloc[0]['CE_LTP']
        sell_pe_prem = options_df[options_df['Strike'] == sell_pe].iloc[0]['PE_LTP']
        buy_pe_prem = options_df[options_df['Strike'] == buy_pe].iloc[0]['PE_LTP']
        
        ic_credit = (sell_ce_prem - buy_ce_prem + sell_pe_prem - buy_pe_prem) * lot_size
        wing_width = (buy_ce - sell_ce) * lot_size
        ic_max_loss = wing_width - ic_credit
        
        strategies.append({
            'name': 'Iron Condor',
            'setup': f'Sell {sell_pe} PE, Buy {buy_pe} PE, Sell {sell_ce} CE, Buy {buy_ce} CE',
            'premium': ic_credit,
            'max_profit': ic_credit,
            'max_loss': ic_max_loss,
            'breakeven': f'Rs.{sell_pe - ic_credit/lot_size:.0f} to Rs.{sell_ce + ic_credit/lot_size:.0f}',
            'move_required': f'Stay within Rs.{sell_pe:.0f} - Rs.{sell_ce:.0f}',
            'view': 'Neutral - defined risk',
            'risk_level': 'MEDIUM',
            'score': 80 if ic_credit > 0 else 50
        })
    
    # 4. Bull Call Spread
    if len(otm_ce_strikes) >= 1:
        buy_strike = atm_strike
        sell_strike = otm_ce_strikes[0]
        
        buy_prem = atm_row['CE_LTP']
        sell_prem = options_df[options_df['Strike'] == sell_strike].iloc[0]['CE_LTP']
        
        spread_debit = (buy_prem - sell_prem) * lot_size
        spread_max_profit = (sell_strike - buy_strike) * lot_size - spread_debit
        
        strategies.append({
            'name': 'Bull Call Spread',
            'setup': f'Buy {buy_strike} CE @ Rs.{buy_prem:.2f}, Sell {sell_strike} CE @ Rs.{sell_prem:.2f}',
            'premium': -spread_debit,
            'max_profit': spread_max_profit,
            'max_loss': spread_debit,
            'breakeven': f'Rs.{buy_strike + spread_debit/lot_size:.0f}',
            'move_required': f'+{((buy_strike + spread_debit/lot_size - spot)/spot)*100:.1f}%',
            'view': 'Bullish - moderate upside expected',
            'risk_level': 'LOW',
            'score': 75 if spread_max_profit > spread_debit else 55
        })
    
    # 5. Bear Put Spread
    if len(otm_pe_strikes) >= 1:
        buy_strike = atm_strike
        sell_strike = otm_pe_strikes[-1]
        
        buy_prem = atm_row['PE_LTP']
        sell_prem = options_df[options_df['Strike'] == sell_strike].iloc[0]['PE_LTP']
        
        spread_debit = (buy_prem - sell_prem) * lot_size
        spread_max_profit = (buy_strike - sell_strike) * lot_size - spread_debit
        
        strategies.append({
            'name': 'Bear Put Spread',
            'setup': f'Buy {buy_strike} PE @ Rs.{buy_prem:.2f}, Sell {sell_strike} PE @ Rs.{sell_prem:.2f}',
            'premium': -spread_debit,
            'max_profit': spread_max_profit,
            'max_loss': spread_debit,
            'breakeven': f'Rs.{buy_strike - spread_debit/lot_size:.0f}',
            'move_required': f'-{((spot - (buy_strike - spread_debit/lot_size))/spot)*100:.1f}%',
            'view': 'Bearish - moderate downside expected',
            'risk_level': 'LOW',
            'score': 75 if spread_max_profit > spread_debit else 55
        })
    
    return strategies, atm_strike


def main():
    print("\n" + "=" * 80)
    print("MCX LTD OPTIONS CHAIN ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Try fetching from NSE
    print("\n[1] Attempting to fetch options chain from NSE...")
    
    nse = NSEOptionsChain()
    data = nse.get_option_chain('MCX')
    
    if data:
        records = data.get('records', {})
        spot = records.get('underlyingValue', 0)
        expiries = records.get('expiryDates', [])
        chain = records.get('data', [])
        timestamp = records.get('timestamp', 'N/A')
        
        print(f"\n    SUCCESS - Data fetched!")
        print(f"    Spot Price: Rs.{spot:,.2f}")
        print(f"    Timestamp: {timestamp}")
        
        if not expiries:
            print("    No expiry dates found!")
            return
        
        nearest_expiry = expiries[0]
        print(f"    Nearest Expiry: {nearest_expiry}")
        
        # Build options dataframe
        options_data = []
        for item in chain:
            if item.get('expiryDate') != nearest_expiry:
                continue
            
            strike = item.get('strikePrice', 0)
            ce = item.get('CE', {})
            pe = item.get('PE', {})
            
            options_data.append({
                'Strike': strike,
                'CE_LTP': ce.get('lastPrice', 0) or 0,
                'CE_OI': ce.get('openInterest', 0) or 0,
                'CE_IV': ce.get('impliedVolatility', 0) or 0,
                'PE_LTP': pe.get('lastPrice', 0) or 0,
                'PE_OI': pe.get('openInterest', 0) or 0,
                'PE_IV': pe.get('impliedVolatility', 0) or 0,
            })
        
        df = pd.DataFrame(options_data).sort_values('Strike')
        
    else:
        print("\n    NSE API unavailable. Using simulated data based on current market...")
        
        # Get current price from yfinance
        import yfinance as yf
        mcx = yf.Ticker('MCX.NS')
        spot = mcx.info.get('currentPrice', 2436)
        
        print(f"\n    Current MCX Price: Rs.{spot:,.2f}")
        print(f"    (Using estimated options data based on typical IV)")
        
        # Generate realistic options chain
        nearest_expiry = "27-Feb-2026"
        days_to_exp = 22
        
        # Typical IV for MCX around 25-35%
        base_iv = 0.30
        
        # Strike range
        strike_step = 50
        strikes = list(range(int(spot - 500), int(spot + 500), strike_step))
        
        options_data = []
        for strike in strikes:
            # Calculate theoretical prices using Black-Scholes approximation
            from scipy.stats import norm
            import math
            
            T = days_to_exp / 365
            r = 0.07  # Risk-free rate
            sigma = base_iv
            
            d1 = (math.log(spot/strike) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            ce_price = spot * norm.cdf(d1) - strike * math.exp(-r*T) * norm.cdf(d2)
            pe_price = strike * math.exp(-r*T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            
            # Add some realistic noise
            ce_price = max(ce_price, 0.05) * (1 + np.random.uniform(-0.05, 0.05))
            pe_price = max(pe_price, 0.05) * (1 + np.random.uniform(-0.05, 0.05))
            
            # Estimate OI (higher at ATM)
            moneyness = abs(strike - spot) / spot
            ce_oi = int(50000 * math.exp(-moneyness * 10) + np.random.randint(1000, 10000))
            pe_oi = int(50000 * math.exp(-moneyness * 10) + np.random.randint(1000, 10000))
            
            options_data.append({
                'Strike': strike,
                'CE_LTP': round(ce_price, 2),
                'CE_OI': ce_oi,
                'CE_IV': base_iv * 100 * (1 + moneyness * 0.5),
                'PE_LTP': round(pe_price, 2),
                'PE_OI': pe_oi,
                'PE_IV': base_iv * 100 * (1 + moneyness * 0.5),
            })
        
        df = pd.DataFrame(options_data)
    
    lot_size = 300  # MCX lot size
    
    # Display options chain
    print(f"\n{'=' * 80}")
    print(f"OPTIONS CHAIN - Expiry: {nearest_expiry}")
    print(f"{'=' * 80}")
    
    # Filter around ATM
    atm_strike = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]['Strike'].values[0]
    df_display = df[(df['Strike'] >= spot - 300) & (df['Strike'] <= spot + 300)]
    
    print(f"\n{'Strike':>8} | {'CE LTP':>10} {'CE OI':>10} {'CE IV':>8} | {'PE LTP':>10} {'PE OI':>10} {'PE IV':>8}")
    print("-" * 85)
    
    for _, row in df_display.iterrows():
        marker = " <-- ATM" if row['Strike'] == atm_strike else ""
        print(f"{row['Strike']:>8.0f} | Rs.{row['CE_LTP']:>8.2f} {row['CE_OI']:>10,.0f} {row['CE_IV']:>7.1f}% | Rs.{row['PE_LTP']:>8.2f} {row['PE_OI']:>10,.0f} {row['PE_IV']:>7.1f}%{marker}")
    
    # Analyze strategies
    strategies, atm = analyze_strategies(spot, df, lot_size)
    
    print(f"\n{'=' * 80}")
    print(f"STRATEGY ANALYSIS (Lot Size: {lot_size} shares)")
    print(f"{'=' * 80}")
    
    for i, strat in enumerate(strategies, 1):
        print(f"\n{i}. {strat['name'].upper()}")
        print(f"   Setup: {strat['setup']}")
        if strat['premium'] >= 0:
            print(f"   Credit Received: Rs.{strat['premium']:,.0f}")
        else:
            print(f"   Debit Paid: Rs.{abs(strat['premium']):,.0f}")
        
        if isinstance(strat['max_profit'], str):
            print(f"   Max Profit: {strat['max_profit']}")
        else:
            print(f"   Max Profit: Rs.{strat['max_profit']:,.0f}")
        
        if isinstance(strat['max_loss'], str):
            print(f"   Max Loss: {strat['max_loss']}")
        else:
            print(f"   Max Loss: Rs.{strat['max_loss']:,.0f}")
        
        print(f"   Breakeven: {strat['breakeven']}")
        print(f"   Move Required: {strat['move_required']}")
        print(f"   Market View: {strat['view']}")
        print(f"   Risk Level: {strat['risk_level']}")
    
    # Rank strategies
    strategies_sorted = sorted(strategies, key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'=' * 80}")
    print("TOP RECOMMENDATIONS")
    print(f"{'=' * 80}")
    
    for i, strat in enumerate(strategies_sorted[:3], 1):
        print(f"\n  #{i} {strat['name']}")
        print(f"      Score: {strat['score']:.0f}/100")
        print(f"      {strat['setup']}")
        
        if strat['premium'] >= 0:
            print(f"      Collect: Rs.{strat['premium']:,.0f}")
        else:
            print(f"      Cost: Rs.{abs(strat['premium']):,.0f}")
        
        if isinstance(strat['max_profit'], (int, float)) and isinstance(strat['max_loss'], (int, float)):
            rr = strat['max_profit'] / strat['max_loss'] if strat['max_loss'] > 0 else 0
            print(f"      Risk/Reward: 1:{rr:.2f}")
    
    # PCR Analysis
    total_ce_oi = df['CE_OI'].sum()
    total_pe_oi = df['PE_OI'].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    print(f"\n{'=' * 80}")
    print("MARKET SENTIMENT")
    print(f"{'=' * 80}")
    print(f"\nPut-Call Ratio (PCR): {pcr:.2f}")
    
    if pcr > 1.2:
        print("Signal: BULLISH (High put writing indicates support)")
    elif pcr < 0.8:
        print("Signal: BEARISH (High call writing indicates resistance)")
    else:
        print("Signal: NEUTRAL (Balanced market)")
    
    # Max Pain
    max_pain_losses = []
    for strike in df['Strike'].unique():
        total_loss = 0
        for _, row in df.iterrows():
            if row['Strike'] <= strike:
                total_loss += row['CE_OI'] * (strike - row['Strike'])
            if row['Strike'] >= strike:
                total_loss += row['PE_OI'] * (row['Strike'] - strike)
        max_pain_losses.append({'Strike': strike, 'Loss': total_loss})
    
    mp_df = pd.DataFrame(max_pain_losses)
    max_pain = mp_df.loc[mp_df['Loss'].idxmin()]['Strike']
    
    print(f"\nMax Pain Strike: Rs.{max_pain:,.0f}")
    print(f"Current Spot: Rs.{spot:,.2f}")
    print(f"Distance: {((spot - max_pain)/spot)*100:+.2f}%")
    
    if spot > max_pain:
        print("Expectation: Stock may drift DOWN towards max pain")
    else:
        print("Expectation: Stock may drift UP towards max pain")
    
    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
