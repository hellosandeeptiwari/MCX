"""
MCX Ltd Options Chain Analysis - Find Most Profitable Combinations
===================================================================
Fetches live options data and analyzes various strategies
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm

# NSE Headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/option-chain',
    'Connection': 'keep-alive',
}

def fetch_nse_option_chain():
    """Fetch MCX option chain from NSE"""
    print("=" * 70)
    print("FETCHING MCX LTD OPTIONS CHAIN FROM NSE")
    print("=" * 70)
    
    session = requests.Session()
    
    try:
        # Get cookies first
        print("\n[1] Getting NSE session cookies...")
        session.get('https://www.nseindia.com', headers=headers, timeout=10)
        
        # Fetch option chain
        print("[2] Fetching MCX option chain...")
        url = 'https://www.nseindia.com/api/option-chain-equities?symbol=MCX'
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"    NSE API returned status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"    Error fetching from NSE: {e}")
        return None

def calculate_iv(option_price, S, K, T, r, option_type):
    """Calculate implied volatility using Newton-Raphson"""
    if T <= 0 or option_price <= 0:
        return 0
    
    sigma = 0.3  # Initial guess
    for _ in range(100):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'CE':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        vega = S*np.sqrt(T)*norm.pdf(d1)
        
        if vega < 1e-10:
            break
            
        sigma = sigma - (price - option_price) / vega
        
        if sigma <= 0:
            sigma = 0.01
            
    return sigma * 100  # Return as percentage

def analyze_options(data):
    """Analyze options chain and find best strategies"""
    if not data:
        print("No data to analyze!")
        return
    
    records = data.get('records', {})
    spot = records.get('underlyingValue', 0)
    expiries = records.get('expiryDates', [])
    chain = records.get('data', [])
    timestamp = records.get('timestamp', 'N/A')
    
    print(f"\n{'='*70}")
    print(f"MCX LTD - LIVE OPTIONS DATA")
    print(f"{'='*70}")
    print(f"Timestamp: {timestamp}")
    print(f"Spot Price: ₹{spot:,.2f}")
    print(f"Available Expiries: {len(expiries)}")
    
    if not expiries:
        print("No expiry dates found!")
        return
    
    nearest_expiry = expiries[0]
    print(f"Nearest Expiry: {nearest_expiry}")
    
    # Calculate days to expiry
    try:
        exp_date = datetime.strptime(nearest_expiry, '%d-%b-%Y')
        days_to_exp = (exp_date - datetime.now()).days + 1
    except:
        days_to_exp = 7
    
    print(f"Days to Expiry: {days_to_exp}")
    
    # Build options table
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
            'CE_Volume': ce.get('totalTradedVolume', 0) or 0,
            'CE_IV': ce.get('impliedVolatility', 0) or 0,
            'CE_Bid': ce.get('bidprice', 0) or 0,
            'CE_Ask': ce.get('askPrice', 0) or 0,
            'PE_LTP': pe.get('lastPrice', 0) or 0,
            'PE_OI': pe.get('openInterest', 0) or 0,
            'PE_Volume': pe.get('totalTradedVolume', 0) or 0,
            'PE_IV': pe.get('impliedVolatility', 0) or 0,
            'PE_Bid': pe.get('bidprice', 0) or 0,
            'PE_Ask': pe.get('askPrice', 0) or 0,
        })
    
    df = pd.DataFrame(options_data)
    df = df.sort_values('Strike')
    
    # Find ATM strike
    atm_strike = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]['Strike'].values[0]
    
    print(f"ATM Strike: ₹{atm_strike}")
    print(f"Total Strikes Available: {len(df)}")
    
    # Display options near ATM
    print(f"\n{'='*70}")
    print(f"OPTIONS CHAIN (Near ATM) - Expiry: {nearest_expiry}")
    print(f"{'='*70}")
    
    # Filter strikes around ATM
    strike_range = spot * 0.10  # 10% around spot
    df_display = df[(df['Strike'] >= spot - strike_range) & (df['Strike'] <= spot + strike_range)]
    
    print(f"\n{'Strike':>8} | {'CE LTP':>10} {'CE OI':>10} {'CE IV':>8} | {'PE LTP':>10} {'PE OI':>10} {'PE IV':>8}")
    print("-" * 80)
    
    for _, row in df_display.iterrows():
        marker = " <-- ATM" if row['Strike'] == atm_strike else ""
        print(f"{row['Strike']:>8.0f} | ₹{row['CE_LTP']:>9.2f} {row['CE_OI']:>10,.0f} {row['CE_IV']:>7.1f}% | ₹{row['PE_LTP']:>9.2f} {row['PE_OI']:>10,.0f} {row['PE_IV']:>7.1f}%{marker}")
    
    # =========================================================================
    # STRATEGY ANALYSIS
    # =========================================================================
    
    lot_size = 300  # MCX lot size
    
    print(f"\n{'='*70}")
    print(f"STRATEGY ANALYSIS (Lot Size: {lot_size} shares)")
    print(f"{'='*70}")
    
    strategies = []
    
    # 1. ATM STRADDLE (Sell)
    atm_row = df[df['Strike'] == atm_strike].iloc[0] if len(df[df['Strike'] == atm_strike]) > 0 else None
    if atm_row is not None:
        straddle_premium = atm_row['CE_LTP'] + atm_row['PE_LTP']
        straddle_revenue = straddle_premium * lot_size
        upper_be = atm_strike + straddle_premium
        lower_be = atm_strike - straddle_premium
        be_pct = (straddle_premium / spot) * 100
        
        strategies.append({
            'Strategy': 'Short ATM Straddle',
            'Description': f'Sell {atm_strike} CE + PE',
            'Premium_Received': straddle_revenue,
            'Max_Profit': straddle_revenue,
            'Max_Loss': 'Unlimited',
            'Breakeven': f'₹{lower_be:.0f} - ₹{upper_be:.0f}',
            'BE_Move': f'±{be_pct:.1f}%',
            'Win_Prob': f'{100 - be_pct*2:.0f}%' if be_pct < 50 else 'N/A',
            'Risk_Reward': 'High Risk/Limited Reward',
            'Score': 70 - be_pct
        })
        
        strategies.append({
            'Strategy': 'Long ATM Straddle',
            'Description': f'Buy {atm_strike} CE + PE',
            'Premium_Received': -straddle_revenue,
            'Max_Profit': 'Unlimited',
            'Max_Loss': straddle_revenue,
            'Breakeven': f'₹{lower_be:.0f} - ₹{upper_be:.0f}',
            'BE_Move': f'±{be_pct:.1f}%',
            'Win_Prob': f'{be_pct*2:.0f}%' if be_pct < 50 else 'N/A',
            'Risk_Reward': 'Limited Risk/Unlimited Reward',
            'Score': be_pct * 2 + 10
        })
    
    # 2. STRANGLE strategies
    otm_strikes = df_display[df_display['Strike'] > atm_strike].head(2)['Strike'].tolist()
    itm_strikes = df_display[df_display['Strike'] < atm_strike].tail(2)['Strike'].tolist()
    
    if len(otm_strikes) >= 1 and len(itm_strikes) >= 1:
        ce_strike = otm_strikes[0]
        pe_strike = itm_strikes[-1]
        
        ce_row = df[df['Strike'] == ce_strike].iloc[0]
        pe_row = df[df['Strike'] == pe_strike].iloc[0]
        
        strangle_premium = ce_row['CE_LTP'] + pe_row['PE_LTP']
        strangle_revenue = strangle_premium * lot_size
        upper_be = ce_strike + strangle_premium
        lower_be = pe_strike - strangle_premium
        
        strategies.append({
            'Strategy': 'Short Strangle',
            'Description': f'Sell {ce_strike} CE + {pe_strike} PE',
            'Premium_Received': strangle_revenue,
            'Max_Profit': strangle_revenue,
            'Max_Loss': 'Unlimited',
            'Breakeven': f'₹{lower_be:.0f} - ₹{upper_be:.0f}',
            'BE_Move': f'{((ce_strike-spot)/spot)*100:.1f}% / {((spot-pe_strike)/spot)*100:.1f}%',
            'Win_Prob': 'Higher than Straddle',
            'Risk_Reward': 'High Risk/Limited Reward',
            'Score': 65
        })
    
    # 3. IRON CONDOR
    if len(df_display) >= 4:
        strikes_sorted = sorted(df_display['Strike'].tolist())
        n = len(strikes_sorted)
        
        # Find strikes for iron condor
        atm_idx = strikes_sorted.index(atm_strike) if atm_strike in strikes_sorted else n//2
        
        if atm_idx >= 2 and atm_idx < n - 2:
            sell_put = strikes_sorted[atm_idx - 1]
            buy_put = strikes_sorted[atm_idx - 2]
            sell_call = strikes_sorted[atm_idx + 1]
            buy_call = strikes_sorted[atm_idx + 2]
            
            sell_put_prem = df[df['Strike'] == sell_put].iloc[0]['PE_LTP']
            buy_put_prem = df[df['Strike'] == buy_put].iloc[0]['PE_LTP']
            sell_call_prem = df[df['Strike'] == sell_call].iloc[0]['CE_LTP']
            buy_call_prem = df[df['Strike'] == buy_call].iloc[0]['CE_LTP']
            
            ic_credit = (sell_put_prem - buy_put_prem + sell_call_prem - buy_call_prem) * lot_size
            ic_max_loss = ((sell_put - buy_put) * lot_size) - ic_credit
            
            strategies.append({
                'Strategy': 'Iron Condor',
                'Description': f'Sell {sell_put}P/{sell_call}C, Buy {buy_put}P/{buy_call}C',
                'Premium_Received': ic_credit,
                'Max_Profit': ic_credit,
                'Max_Loss': ic_max_loss,
                'Breakeven': f'₹{sell_put - (ic_credit/lot_size):.0f} - ₹{sell_call + (ic_credit/lot_size):.0f}',
                'BE_Move': 'Defined Range',
                'Win_Prob': '~60-70%',
                'Risk_Reward': f'1:{ic_max_loss/ic_credit:.1f}' if ic_credit > 0 else 'N/A',
                'Score': 80 if ic_credit > 0 else 0
            })
    
    # 4. BULL CALL SPREAD
    if len(otm_strikes) >= 1:
        buy_strike = atm_strike
        sell_strike = otm_strikes[0]
        
        buy_prem = df[df['Strike'] == buy_strike].iloc[0]['CE_LTP']
        sell_prem = df[df['Strike'] == sell_strike].iloc[0]['CE_LTP']
        
        spread_cost = (buy_prem - sell_prem) * lot_size
        max_profit = ((sell_strike - buy_strike) * lot_size) - spread_cost
        
        strategies.append({
            'Strategy': 'Bull Call Spread',
            'Description': f'Buy {buy_strike} CE, Sell {sell_strike} CE',
            'Premium_Received': -spread_cost,
            'Max_Profit': max_profit,
            'Max_Loss': spread_cost,
            'Breakeven': f'₹{buy_strike + (spread_cost/lot_size):.0f}',
            'BE_Move': f'+{((buy_strike + (spread_cost/lot_size) - spot)/spot)*100:.1f}%',
            'Win_Prob': '~45-50%',
            'Risk_Reward': f'{max_profit/spread_cost:.2f}:1' if spread_cost > 0 else 'N/A',
            'Score': 75 if max_profit > spread_cost else 50
        })
    
    # 5. BEAR PUT SPREAD
    if len(itm_strikes) >= 1:
        buy_strike = atm_strike
        sell_strike = itm_strikes[-1]
        
        buy_prem = df[df['Strike'] == buy_strike].iloc[0]['PE_LTP']
        sell_prem = df[df['Strike'] == sell_strike].iloc[0]['PE_LTP']
        
        spread_cost = (buy_prem - sell_prem) * lot_size
        max_profit = ((buy_strike - sell_strike) * lot_size) - spread_cost
        
        strategies.append({
            'Strategy': 'Bear Put Spread',
            'Description': f'Buy {buy_strike} PE, Sell {sell_strike} PE',
            'Premium_Received': -spread_cost,
            'Max_Profit': max_profit,
            'Max_Loss': spread_cost,
            'Breakeven': f'₹{buy_strike - (spread_cost/lot_size):.0f}',
            'BE_Move': f'-{((spot - (buy_strike - (spread_cost/lot_size)))/spot)*100:.1f}%',
            'Win_Prob': '~45-50%',
            'Risk_Reward': f'{max_profit/spread_cost:.2f}:1' if spread_cost > 0 else 'N/A',
            'Score': 75 if max_profit > spread_cost else 50
        })
    
    # 6. Covered Call (if holding stock)
    if atm_row is not None and len(otm_strikes) >= 1:
        sell_strike = otm_strikes[0]
        sell_prem = df[df['Strike'] == sell_strike].iloc[0]['CE_LTP']
        premium_yield = (sell_prem / spot) * 100
        
        strategies.append({
            'Strategy': 'Covered Call',
            'Description': f'Own stock + Sell {sell_strike} CE',
            'Premium_Received': sell_prem * lot_size,
            'Max_Profit': f'₹{(sell_strike - spot + sell_prem) * lot_size:,.0f}',
            'Max_Loss': f'Stock drops to 0 minus premium',
            'Breakeven': f'₹{spot - sell_prem:.0f}',
            'BE_Move': f'-{premium_yield:.1f}% cushion',
            'Win_Prob': '~65-70%',
            'Risk_Reward': 'Income Strategy',
            'Score': 72
        })
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    
    print("\n" + "=" * 90)
    print("STRATEGY COMPARISON")
    print("=" * 90)
    
    for i, strat in enumerate(strategies, 1):
        print(f"\n{i}. {strat['Strategy'].upper()}")
        print(f"   Setup: {strat['Description']}")
        if isinstance(strat['Premium_Received'], (int, float)):
            if strat['Premium_Received'] >= 0:
                print(f"   Premium Received: Rs.{strat['Premium_Received']:,.0f}")
            else:
                print(f"   Cost: Rs.{abs(strat['Premium_Received']):,.0f}")
        max_profit = strat['Max_Profit']
        max_loss = strat['Max_Loss']
        if isinstance(max_profit, str):
            print(f"   Max Profit: {max_profit}")
        else:
            print(f"   Max Profit: Rs.{max_profit:,.0f}")
        if isinstance(max_loss, str):
            print(f"   Max Loss: {max_loss}")
        else:
            print(f"   Max Loss: Rs.{max_loss:,.0f}")
        print(f"   Breakeven: {strat['Breakeven']}")
        print(f"   Required Move: {strat['BE_Move']}")
        print(f"   Est. Win Probability: {strat['Win_Prob']}")
    
    # =========================================================================
    # RECOMMENDATION
    # =========================================================================
    
    # Sort by score
    strategies_scored = [s for s in strategies if isinstance(s.get('Score'), (int, float))]
    strategies_scored.sort(key=lambda x: x['Score'], reverse=True)
    
    print("\n" + "=" * 90)
    print("🏆 TOP RECOMMENDATIONS (Based on Risk-Reward Analysis)")
    print("=" * 90)
    
    for i, strat in enumerate(strategies_scored[:3], 1):
        print(f"\n#{i} - {strat['Strategy']}")
        print(f"    {strat['Description']}")
        if isinstance(strat['Max_Profit'], (int, float)) and isinstance(strat['Max_Loss'], (int, float)):
            print(f"    Potential P/L: ₹{strat['Max_Profit']:,.0f} / -₹{strat['Max_Loss']:,.0f}")
        print(f"    Score: {strat['Score']:.0f}/100")
    
    # =========================================================================
    # OI ANALYSIS - Max Pain
    # =========================================================================
    
    print("\n" + "=" * 90)
    print("OPEN INTEREST ANALYSIS - MAX PAIN")
    print("=" * 90)
    
    # Calculate max pain
    max_pain_losses = []
    for strike in df['Strike'].unique():
        total_loss = 0
        for _, row in df.iterrows():
            if row['Strike'] <= strike:
                # CE is ITM
                total_loss += row['CE_OI'] * (strike - row['Strike'])
            if row['Strike'] >= strike:
                # PE is ITM
                total_loss += row['PE_OI'] * (row['Strike'] - strike)
        max_pain_losses.append({'Strike': strike, 'Loss': total_loss})
    
    if max_pain_losses:
        mp_df = pd.DataFrame(max_pain_losses)
        max_pain_strike = mp_df.loc[mp_df['Loss'].idxmin()]['Strike']
        
        print(f"\nMax Pain Strike: ₹{max_pain_strike:,.0f}")
        print(f"Current Spot: ₹{spot:,.2f}")
        print(f"Distance from Max Pain: {((spot - max_pain_strike)/spot)*100:+.2f}%")
        
        if spot > max_pain_strike:
            print(f"📉 Stock may drift DOWN towards ₹{max_pain_strike:,.0f}")
        else:
            print(f"📈 Stock may drift UP towards ₹{max_pain_strike:,.0f}")
    
    # PCR Analysis
    total_ce_oi = df['CE_OI'].sum()
    total_pe_oi = df['PE_OI'].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    print(f"\nPut-Call Ratio (PCR): {pcr:.2f}")
    if pcr > 1.2:
        print("📈 HIGH PCR - Bullish sentiment (more put writing)")
    elif pcr < 0.8:
        print("📉 LOW PCR - Bearish sentiment (more call writing)")
    else:
        print("➡️ NEUTRAL PCR - Market balanced")
    
    return df, strategies

# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == "__main__":
    print("\n" + "🔄" * 35)
    print("MCX LTD OPTIONS CHAIN ANALYZER")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄" * 35)
    
    # Fetch data
    data = fetch_nse_option_chain()
    
    if data:
        df, strategies = analyze_options(data)
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
    else:
        print("\n❌ Failed to fetch data from NSE. Trying alternate methods...")
