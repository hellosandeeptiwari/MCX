"""
MCX Options Dashboard - Live Zerodha Data
==========================================
Beautiful web dashboard for MCX options chain and strategy analysis
"""

from flask import Flask, render_template, jsonify
from kiteconnect import KiteConnect
import json
from datetime import datetime
import os

app = Flask(__name__)

# Zerodha credentials
API_KEY = os.environ.get("ZERODHA_API_KEY", "")
API_SECRET = os.environ.get("ZERODHA_API_SECRET", "")
LOT_SIZE = 625

def get_kite():
    """Get authenticated Kite instance"""
    kite = KiteConnect(api_key=API_KEY)
    
    # Load saved token
    try:
        with open("zerodha_token.json", "r") as f:
            data = json.load(f)
            if data.get("date") == str(datetime.now().date()):
                kite.set_access_token(data["access_token"])
                return kite
    except:
        pass
    
    return None

def fetch_options_data():
    """Fetch live MCX options data"""
    kite = get_kite()
    if not kite:
        # Return cached data if available
        try:
            with open("mcx_options_live.json", "r") as f:
                return json.load(f)
        except:
            return None
    
    try:
        # Get instruments
        instruments = kite.instruments("NFO")
        mcx_options = [i for i in instruments 
                       if i['name'] == 'MCX' 
                       and i['instrument_type'] in ['CE', 'PE']]
        
        # Get expiries
        expiries = sorted(set([i['expiry'] for i in mcx_options]))
        nearest_expiry = expiries[0]
        
        # Filter options
        options = [i for i in mcx_options if i['expiry'] == nearest_expiry]
        
        # Get spot
        spot_quote = kite.quote(["NSE:MCX"])
        spot = spot_quote["NSE:MCX"]["last_price"]
        
        # Get quotes
        relevant_options = [o for o in options if abs(o['strike'] - spot) <= 400]
        tokens = [o['instrument_token'] for o in relevant_options]
        quotes = kite.quote(tokens)
        
        # Build chain with market depth
        chain = {}
        for opt in relevant_options:
            strike = opt['strike']
            if strike not in chain:
                chain[strike] = {'CE': {}, 'PE': {}}
            
            token = opt['instrument_token']
            q = quotes.get(str(token), {})
            
            # Extract market depth
            depth = q.get('depth', {'buy': [], 'sell': []})
            best_bid = depth['buy'][0]['price'] if depth['buy'] else 0
            best_ask = depth['sell'][0]['price'] if depth['sell'] else 0
            bid_qty = sum(level['quantity'] for level in depth['buy'])
            ask_qty = sum(level['quantity'] for level in depth['sell'])
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
            
            chain[strike][opt['instrument_type']] = {
                'ltp': q.get('last_price', 0) or 0,
                'oi': q.get('oi', 0) or 0,
                'volume': q.get('volume', 0) or 0,
                'buy_qty': q.get('buy_quantity', 0) or 0,
                'sell_qty': q.get('sell_quantity', 0) or 0,
                'bid': best_bid,
                'ask': best_ask,
                'bid_depth': bid_qty,
                'ask_depth': ask_qty,
                'spread': spread,
                'spread_pct': round(spread_pct, 2),
            }
        
        data = {
            'spot': spot,
            'expiry': str(nearest_expiry),
            'timestamp': str(datetime.now()),
            'chain': {str(k): v for k, v in chain.items()}
        }
        
        # Save
        with open("mcx_options_live.json", "w") as f:
            json.dump(data, f, indent=2)
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            with open("mcx_options_live.json", "r") as f:
                return json.load(f)
        except:
            return None


def get_liquidity_score(option_data):
    """Calculate liquidity score based on market depth"""
    score = 100  # Start with perfect score
    
    # Penalize for wide spread
    spread_pct = option_data.get('spread_pct', 0)
    if spread_pct > 2:
        score -= 30  # Very wide spread
    elif spread_pct > 1:
        score -= 15  # Moderate spread
    elif spread_pct > 0.5:
        score -= 5   # Acceptable spread
    
    # Penalize for low volume
    volume = option_data.get('volume', 0)
    if volume < 10000:
        score -= 25
    elif volume < 50000:
        score -= 10
    
    # Penalize for low OI
    oi = option_data.get('oi', 0)
    if oi < 100000:
        score -= 20
    elif oi < 500000:
        score -= 10
    
    # Penalize for low bid depth
    bid_depth = option_data.get('bid_depth', 0)
    if bid_depth < 5000:
        score -= 15
    
    return max(0, score)


def analyze_strategies(data):
    """Analyze BUY-ONLY options strategies with market depth consideration"""
    if not data:
        return []
    
    spot = data['spot']
    chain = {float(k): v for k, v in data['chain'].items()}
    strikes = sorted(chain.keys())
    
    atm_strike = min(strikes, key=lambda x: abs(x - spot))
    atm_idx = strikes.index(atm_strike)
    
    strategies = []
    BUDGET = 300000  # Rs 3 lakh budget
    
    # Calculate liquidity for ATM options
    atm_ce_data = chain[atm_strike]['CE']
    atm_pe_data = chain[atm_strike]['PE']
    atm_ce_liq = get_liquidity_score(atm_ce_data)
    atm_pe_liq = get_liquidity_score(atm_pe_data)
    
    # 1. Long Straddle (Buy ATM CE + Buy ATM PE) - Volatility Play
    atm_ce = atm_ce_data.get('ltp', 0)
    atm_pe = atm_pe_data.get('ltp', 0)
    straddle_cost = atm_ce + atm_pe
    straddle_total = straddle_cost * LOT_SIZE
    lots_straddle = int(BUDGET / straddle_total)
    
    # Calculate slippage impact from spread
    ce_spread = atm_ce_data.get('spread', 0)
    pe_spread = atm_pe_data.get('spread', 0)
    slippage_cost = (ce_spread + pe_spread) * LOT_SIZE * lots_straddle
    
    strategies.append({
        'name': 'Long Straddle',
        'description': f'Buy {int(atm_strike)} CE @ ₹{atm_ce:.2f} + Buy {int(atm_strike)} PE @ ₹{atm_pe:.2f}',
        'premium': -straddle_total,
        'max_profit': 'Unlimited',
        'max_loss': straddle_total,
        'breakeven_low': atm_strike - straddle_cost,
        'breakeven_high': atm_strike + straddle_cost,
        'risk_level': 'Medium',
        'view': 'High Volatility Expected',
        'color': 'primary',
        'lots_possible': lots_straddle,
        'total_cost': straddle_total * lots_straddle,
        'profit_scenario': f'If MCX moves ±₹{straddle_cost:.0f} from {int(atm_strike)}, you profit',
        'liquidity_score': int((atm_ce_liq + atm_pe_liq) / 2),
        'slippage_cost': slippage_cost,
        'spread_info': f'CE spread: ₹{ce_spread:.2f} | PE spread: ₹{pe_spread:.2f}'
    })
    
    # 2. Long Strangle (Buy OTM CE + Buy OTM PE) - Cheaper Volatility Play
    if atm_idx >= 1 and atm_idx < len(strikes) - 1:
        buy_pe = strikes[atm_idx - 1]
        buy_ce = strikes[atm_idx + 1]
        
        pe_data = chain[buy_pe]['PE']
        ce_data = chain[buy_ce]['CE']
        pe_prem = pe_data.get('ltp', 0)
        ce_prem = ce_data.get('ltp', 0)
        
        strangle_cost = pe_prem + ce_prem
        strangle_total = strangle_cost * LOT_SIZE
        lots_strangle = int(BUDGET / strangle_total)
        
        pe_liq = get_liquidity_score(pe_data)
        ce_liq = get_liquidity_score(ce_data)
        
        strategies.append({
            'name': 'Long Strangle',
            'description': f'Buy {int(buy_pe)} PE @ ₹{pe_prem:.2f} + Buy {int(buy_ce)} CE @ ₹{ce_prem:.2f}',
            'premium': -strangle_total,
            'max_profit': 'Unlimited',
            'max_loss': strangle_total,
            'breakeven_low': buy_pe - strangle_cost,
            'breakeven_high': buy_ce + strangle_cost,
            'risk_level': 'Medium',
            'view': 'High Volatility Expected',
            'color': 'info',
            'lots_possible': lots_strangle,
            'total_cost': strangle_total * lots_strangle,
            'profit_scenario': f'Needs bigger move than straddle but cheaper entry',
            'liquidity_score': int((pe_liq + ce_liq) / 2),
            'spread_info': f'PE spread: ₹{pe_data.get("spread", 0):.2f} | CE spread: ₹{ce_data.get("spread", 0):.2f}'
        })
    
    # 3. Buy OTM Calls (Bullish) - Multiple strike options
    otm_ce_strikes = [s for s in strikes if s > spot][:3]  # Next 3 OTM strikes
    for strike in otm_ce_strikes:
        ce_data = chain[strike]['CE']
        ce_prem = ce_data.get('ltp', 0)
        if ce_prem > 0:
            cost_per_lot = ce_prem * LOT_SIZE
            lots_possible = int(BUDGET / cost_per_lot)
            ce_liq = get_liquidity_score(ce_data)
            
            # Calculate profit at different targets
            target_10pct = spot * 1.10
            profit_10pct = max(0, target_10pct - strike - ce_prem) * LOT_SIZE * lots_possible
            
            rr = profit_10pct / (cost_per_lot * lots_possible) if lots_possible > 0 else 0
            
            strategies.append({
                'name': f'Buy {int(strike)} CE',
                'description': f'Buy {int(strike)} CE @ ₹{ce_prem:.2f} ({lots_possible} lots with ₹3L)',
                'premium': -cost_per_lot,
                'max_profit': 'Unlimited',
                'max_loss': cost_per_lot * lots_possible,
                'breakeven_low': strike,
                'breakeven_high': strike + ce_prem,
                'risk_reward': f'{rr:.1f}:1 at 10% up',
                'risk_level': 'High' if strike > spot * 1.05 else 'Medium',
                'view': 'Bullish',
                'color': 'success',
                'lots_possible': lots_possible,
                'total_cost': cost_per_lot * lots_possible,
                'profit_scenario': f'If MCX hits ₹{target_10pct:.0f} (+10%): Profit ₹{profit_10pct:,.0f}',
                'liquidity_score': ce_liq,
                'spread_info': f'Bid-Ask spread: ₹{ce_data.get("spread", 0):.2f} ({ce_data.get("spread_pct", 0):.1f}%)'
            })
    
    # 4. Buy OTM Puts (Bearish) - Multiple strike options
    otm_pe_strikes = [s for s in strikes if s < spot][-3:][::-1]  # 3 OTM strikes below
    for strike in otm_pe_strikes:
        pe_data = chain[strike]['PE']
        pe_prem = pe_data.get('ltp', 0)
        if pe_prem > 0:
            cost_per_lot = pe_prem * LOT_SIZE
            lots_possible = int(BUDGET / cost_per_lot)
            pe_liq = get_liquidity_score(pe_data)
            
            # Calculate profit at different targets
            target_10pct = spot * 0.90
            profit_10pct = max(0, strike - target_10pct - pe_prem) * LOT_SIZE * lots_possible
            
            rr = profit_10pct / (cost_per_lot * lots_possible) if lots_possible > 0 else 0
            
            strategies.append({
                'name': f'Buy {int(strike)} PE',
                'description': f'Buy {int(strike)} PE @ ₹{pe_prem:.2f} ({lots_possible} lots with ₹3L)',
                'premium': -cost_per_lot,
                'max_profit': f'₹{strike * LOT_SIZE * lots_possible:,.0f}',
                'max_loss': cost_per_lot * lots_possible,
                'breakeven_low': strike - pe_prem,
                'breakeven_high': strike,
                'risk_reward': f'{rr:.1f}:1 at 10% down',
                'risk_level': 'High' if strike < spot * 0.95 else 'Medium',
                'view': 'Bearish',
                'color': 'danger',
                'lots_possible': lots_possible,
                'total_cost': cost_per_lot * lots_possible,
                'profit_scenario': f'If MCX drops to ₹{target_10pct:.0f} (-10%): Profit ₹{profit_10pct:,.0f}',
                'liquidity_score': pe_liq,
                'spread_info': f'Bid-Ask spread: ₹{pe_data.get("spread", 0):.2f} ({pe_data.get("spread_pct", 0):.1f}%)'
            })
    
    # 5. Buy ATM Call (Slightly Bullish)
    atm_ce_cost = atm_ce * LOT_SIZE
    lots_atm_ce = int(BUDGET / atm_ce_cost)
    target_5pct = spot * 1.05
    profit_atm_ce = max(0, target_5pct - atm_strike - atm_ce) * LOT_SIZE * lots_atm_ce
    
    strategies.append({
        'name': f'Buy ATM {int(atm_strike)} CE',
        'description': f'Buy {int(atm_strike)} CE @ ₹{atm_ce:.2f} ({lots_atm_ce} lots with ₹3L)',
        'premium': -atm_ce_cost,
        'max_profit': 'Unlimited',
        'max_loss': atm_ce_cost * lots_atm_ce,
        'breakeven_low': atm_strike,
        'breakeven_high': atm_strike + atm_ce,
        'risk_level': 'Medium',
        'view': 'Bullish',
        'color': 'success',
        'lots_possible': lots_atm_ce,
        'total_cost': atm_ce_cost * lots_atm_ce,
        'profit_scenario': f'If MCX hits ₹{target_5pct:.0f} (+5%): Profit ₹{profit_atm_ce:,.0f}',
        'liquidity_score': atm_ce_liq,
        'spread_info': f'Bid-Ask spread: ₹{atm_ce_data.get("spread", 0):.2f} ({atm_ce_data.get("spread_pct", 0):.1f}%)'
    })
    
    # 6. Buy ATM Put (Slightly Bearish)
    atm_pe_cost = atm_pe * LOT_SIZE
    lots_atm_pe = int(BUDGET / atm_pe_cost)
    target_5pct_down = spot * 0.95
    profit_atm_pe = max(0, atm_strike - target_5pct_down - atm_pe) * LOT_SIZE * lots_atm_pe
    
    strategies.append({
        'name': f'Buy ATM {int(atm_strike)} PE',
        'description': f'Buy {int(atm_strike)} PE @ ₹{atm_pe:.2f} ({lots_atm_pe} lots with ₹3L)',
        'premium': -atm_pe_cost,
        'max_profit': f'₹{atm_strike * LOT_SIZE * lots_atm_pe:,.0f}',
        'max_loss': atm_pe_cost * lots_atm_pe,
        'breakeven_low': atm_strike - atm_pe,
        'breakeven_high': atm_strike,
        'risk_level': 'Medium',
        'view': 'Bearish',
        'color': 'danger',
        'lots_possible': lots_atm_pe,
        'total_cost': atm_pe_cost * lots_atm_pe,
        'profit_scenario': f'If MCX drops to ₹{target_5pct_down:.0f} (-5%): Profit ₹{profit_atm_pe:,.0f}',
        'liquidity_score': atm_pe_liq,
        'spread_info': f'Bid-Ask spread: ₹{atm_pe_data.get("spread", 0):.2f} ({atm_pe_data.get("spread_pct", 0):.1f}%)'
    })
    
    # Calculate scores for each strategy (optimized for BUY strategies with liquidity)
    for strat in strategies:
        score = 0
        
        # LIQUIDITY SCORE (Most Important for execution)
        liquidity = strat.get('liquidity_score', 50)
        score += liquidity * 0.4  # Up to 40 points for liquidity
        
        # For buy strategies, look at risk/reward from profit_scenario
        if 'risk_reward' in strat:
            try:
                rr_str = strat['risk_reward'].split(':')[0]
                rr_ratio = float(rr_str)
                score += min(rr_ratio * 10, 25)  # Higher R/R = better
            except:
                pass
        
        # Unlimited profit potential is good for buy strategies
        if strat.get('max_profit') == 'Unlimited':
            score += 15
        
        # Lower cost strategies get bonus (more lots possible)
        if strat.get('lots_possible', 0) > 0:
            score += min(strat['lots_possible'] * 2, 15)
        
        # Risk level scoring
        if strat.get('risk_level') == 'Low':
            score += 10
        elif strat.get('risk_level') == 'Medium':
            score += 7
        elif strat.get('risk_level') == 'High':
            score += 3
        
        strat['score'] = round(score, 2)
    
    # Sort by score and mark best
    if strategies:
        strategies.sort(key=lambda x: x['score'], reverse=True)
        strategies[0]['is_best'] = True
    
    return strategies


def get_best_strategy(strategies):
    """Get the best strategy with explanation"""
    if not strategies:
        return None
    
    best = strategies[0] if strategies else None
    if best:
        reasons = []
        if best.get('max_profit') == 'Unlimited':
            reasons.append("Unlimited profit potential")
        if best.get('lots_possible', 0) > 0:
            reasons.append(f"{best['lots_possible']} lots possible with ₹3L")
        if best.get('risk_level') in ['Low', 'Medium']:
            reasons.append(f"{best['risk_level']} risk")
        if best.get('profit_scenario'):
            reasons.append(best['profit_scenario'])
        
        best['reasons'] = reasons if reasons else ["Best risk/reward ratio"]
    
    return best


@app.route('/')
def dashboard():
    """Main dashboard"""
    data = fetch_options_data()
    strategies = analyze_strategies(data)
    best_strategy = get_best_strategy(strategies)
    
    if data:
        chain = data['chain']
        spot = data['spot']
        
        # Calculate PCR
        total_ce_oi = sum(v['CE'].get('oi', 0) for v in chain.values())
        total_pe_oi = sum(v['PE'].get('oi', 0) for v in chain.values())
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Max Pain
        max_pain = None
        min_loss = float('inf')
        for strike in chain.keys():
            strike_f = float(strike)
            total_loss = 0
            for s, v in chain.items():
                s_f = float(s)
                if s_f <= strike_f:
                    total_loss += v['CE'].get('oi', 0) * (strike_f - s_f)
                if s_f >= strike_f:
                    total_loss += v['PE'].get('oi', 0) * (s_f - strike_f)
            if total_loss < min_loss:
                min_loss = total_loss
                max_pain = strike_f
    else:
        spot = 0
        pcr = 0
        max_pain = 0
        chain = {}
    
    return render_template('options_dashboard.html',
                          data=data,
                          strategies=strategies,
                          best_strategy=best_strategy,
                          spot=spot,
                          pcr=pcr,
                          max_pain=max_pain,
                          lot_size=LOT_SIZE,
                          timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


@app.route('/api/refresh')
def refresh_data():
    """API to refresh data"""
    data = fetch_options_data()
    strategies = analyze_strategies(data)
    return jsonify({
        'data': data,
        'strategies': strategies,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


if __name__ == '__main__':
    print("=" * 60)
    print("MCX OPTIONS DASHBOARD")
    print("=" * 60)
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)
