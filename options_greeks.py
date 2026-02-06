"""
Options Greeks Calculator & Integration for MCX Trading
========================================================
Calculates Options Greeks (Delta, Gamma, Theta, Vega, Rho)
for MCX Ltd stock options and integrates with sync analysis.

Author: Options Trader Tool
Date: February 2026
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptionContract:
    """Option contract details"""
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'CALL' or 'PUT'
    premium: float
    lot_size: int = 1
    

@dataclass
class Greeks:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float  # Implied Volatility


class BlackScholesCalculator:
    """
    Black-Scholes Option Pricing and Greeks Calculator
    """
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2"""
        if T <= 0 or sigma <= 0:
            return 0
        return BlackScholesCalculator.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate call option price"""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate put option price"""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='CALL'):
        """Calculate Delta"""
        if T <= 0:
            if option_type == 'CALL':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        if option_type == 'CALL':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate Gamma (same for call and put)"""
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='CALL'):
        """Calculate Theta (per day)"""
        if T <= 0:
            return 0
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'CALL':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            return (term1 + term2) / 365  # Daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate Vega (for 1% change in IV)"""
        if T <= 0:
            return 0
        d1 = BlackScholesCalculator.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def rho(S, K, T, r, sigma, option_type='CALL'):
        """Calculate Rho (for 1% change in rate)"""
        if T <= 0:
            return 0
        d2 = BlackScholesCalculator.d2(S, K, T, r, sigma)
        if option_type == 'CALL':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='CALL', max_iter=100):
        """Calculate Implied Volatility using Brent's method"""
        if T <= 0:
            return 0.0
            
        def objective(sigma):
            if option_type == 'CALL':
                return BlackScholesCalculator.call_price(S, K, T, r, sigma) - market_price
            else:
                return BlackScholesCalculator.put_price(S, K, T, r, sigma) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iter)
            return iv
        except:
            return 0.3  # Default to 30% if calculation fails


class MCXOptionsAnalyzer:
    """
    Options analyzer for MCX Ltd stock with sync-based strategy integration.
    """
    
    def __init__(self, risk_free_rate=0.065):  # RBI repo rate ~6.5%
        self.risk_free_rate = risk_free_rate
        self.mcx_ticker = "MCX.NS"
        self.bs = BlackScholesCalculator()
        self.current_price = None
        self.historical_vol = None
        
    def fetch_underlying_data(self):
        """Fetch current MCX stock data"""
        print("📈 Fetching MCX Ltd stock data...")
        
        mcx = yf.Ticker(self.mcx_ticker)
        hist = mcx.history(period='1y')
        
        if hist.empty:
            print("   ❌ Failed to fetch data")
            return None
            
        self.current_price = hist['Close'].iloc[-1]
        
        # Calculate historical volatility (annualized)
        returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        self.historical_vol = returns.std() * np.sqrt(252)
        
        print(f"   ✅ Current Price: ₹{self.current_price:.2f}")
        print(f"   ✅ Historical Volatility: {self.historical_vol*100:.1f}%")
        
        return {
            'price': self.current_price,
            'historical_vol': self.historical_vol,
            'history': hist
        }
        
    def calculate_greeks(self, strike: float, days_to_expiry: int, 
                         option_type: str = 'CALL', iv: float = None) -> Greeks:
        """Calculate all Greeks for an option"""
        if self.current_price is None:
            self.fetch_underlying_data()
            
        S = self.current_price
        K = strike
        T = days_to_expiry / 365
        r = self.risk_free_rate
        sigma = iv if iv else self.historical_vol
        
        return Greeks(
            delta=self.bs.delta(S, K, T, r, sigma, option_type),
            gamma=self.bs.gamma(S, K, T, r, sigma),
            theta=self.bs.theta(S, K, T, r, sigma, option_type),
            vega=self.bs.vega(S, K, T, r, sigma),
            rho=self.bs.rho(S, K, T, r, sigma, option_type),
            iv=sigma
        )
        
    def price_option(self, strike: float, days_to_expiry: int,
                     option_type: str = 'CALL', iv: float = None) -> float:
        """Price an option using Black-Scholes"""
        if self.current_price is None:
            self.fetch_underlying_data()
            
        S = self.current_price
        K = strike
        T = days_to_expiry / 365
        r = self.risk_free_rate
        sigma = iv if iv else self.historical_vol
        
        if option_type == 'CALL':
            return self.bs.call_price(S, K, T, r, sigma)
        else:
            return self.bs.put_price(S, K, T, r, sigma)
            
    def generate_option_chain(self, days_to_expiry: int = 30, 
                              num_strikes: int = 11) -> pd.DataFrame:
        """Generate option chain around current price"""
        if self.current_price is None:
            self.fetch_underlying_data()
            
        S = self.current_price
        
        # Generate strikes (5% intervals around spot)
        strike_interval = S * 0.025  # 2.5% intervals
        strikes = [round(S + (i - num_strikes//2) * strike_interval, -1) 
                   for i in range(num_strikes)]
        
        chain_data = []
        
        for K in strikes:
            # Call
            call_greeks = self.calculate_greeks(K, days_to_expiry, 'CALL')
            call_price = self.price_option(K, days_to_expiry, 'CALL')
            
            # Put
            put_greeks = self.calculate_greeks(K, days_to_expiry, 'PUT')
            put_price = self.price_option(K, days_to_expiry, 'PUT')
            
            chain_data.append({
                'Strike': K,
                'Call_Price': call_price,
                'Call_Delta': call_greeks.delta,
                'Call_Gamma': call_greeks.gamma,
                'Call_Theta': call_greeks.theta,
                'Call_Vega': call_greeks.vega,
                'Put_Price': put_price,
                'Put_Delta': put_greeks.delta,
                'Put_Gamma': put_greeks.gamma,
                'Put_Theta': put_greeks.theta,
                'Put_Vega': put_greeks.vega,
                'IV': call_greeks.iv * 100
            })
            
        chain = pd.DataFrame(chain_data)
        return chain
        
    def sync_based_option_strategy(self, divergence_z: float, 
                                   days_to_expiry: int = 30) -> Dict:
        """
        Recommend option strategy based on MCX-Metal sync divergence.
        
        Args:
            divergence_z: Current divergence Z-score from sync analysis
            days_to_expiry: Days until option expiry
        """
        if self.current_price is None:
            self.fetch_underlying_data()
            
        S = self.current_price
        
        recommendations = {
            'signal': '',
            'strategy': '',
            'details': {},
            'risk_reward': {},
            'greeks_exposure': {}
        }
        
        if divergence_z < -1.5:
            # MCX underperforming - expect reversion UP
            recommendations['signal'] = 'BULLISH'
            
            if divergence_z < -2.0:
                # Strong signal - buy call
                recommendations['strategy'] = 'BUY ATM CALL'
                strike = round(S, -1)
                greeks = self.calculate_greeks(strike, days_to_expiry, 'CALL')
                price = self.price_option(strike, days_to_expiry, 'CALL')
                
                recommendations['details'] = {
                    'action': 'BUY',
                    'type': 'CALL',
                    'strike': strike,
                    'expiry_days': days_to_expiry,
                    'premium': price,
                    'max_loss': price,
                    'breakeven': strike + price
                }
            else:
                # Moderate signal - bull call spread
                recommendations['strategy'] = 'BULL CALL SPREAD'
                lower_strike = round(S, -1)
                upper_strike = round(S * 1.05, -1)
                
                buy_price = self.price_option(lower_strike, days_to_expiry, 'CALL')
                sell_price = self.price_option(upper_strike, days_to_expiry, 'CALL')
                
                recommendations['details'] = {
                    'action': 'BUY CALL + SELL CALL',
                    'buy_strike': lower_strike,
                    'sell_strike': upper_strike,
                    'expiry_days': days_to_expiry,
                    'net_premium': buy_price - sell_price,
                    'max_loss': buy_price - sell_price,
                    'max_profit': upper_strike - lower_strike - (buy_price - sell_price),
                    'breakeven': lower_strike + (buy_price - sell_price)
                }
                
        elif divergence_z > 1.5:
            # MCX outperforming - expect reversion DOWN
            recommendations['signal'] = 'BEARISH'
            
            if divergence_z > 2.0:
                # Strong signal - buy put
                recommendations['strategy'] = 'BUY ATM PUT'
                strike = round(S, -1)
                greeks = self.calculate_greeks(strike, days_to_expiry, 'PUT')
                price = self.price_option(strike, days_to_expiry, 'PUT')
                
                recommendations['details'] = {
                    'action': 'BUY',
                    'type': 'PUT',
                    'strike': strike,
                    'expiry_days': days_to_expiry,
                    'premium': price,
                    'max_loss': price,
                    'breakeven': strike - price
                }
            else:
                # Moderate signal - bear put spread
                recommendations['strategy'] = 'BEAR PUT SPREAD'
                upper_strike = round(S, -1)
                lower_strike = round(S * 0.95, -1)
                
                buy_price = self.price_option(upper_strike, days_to_expiry, 'PUT')
                sell_price = self.price_option(lower_strike, days_to_expiry, 'PUT')
                
                recommendations['details'] = {
                    'action': 'BUY PUT + SELL PUT',
                    'buy_strike': upper_strike,
                    'sell_strike': lower_strike,
                    'expiry_days': days_to_expiry,
                    'net_premium': buy_price - sell_price,
                    'max_loss': buy_price - sell_price,
                    'max_profit': upper_strike - lower_strike - (buy_price - sell_price),
                    'breakeven': upper_strike - (buy_price - sell_price)
                }
                
        else:
            # Neutral - sell premium (theta strategy)
            recommendations['signal'] = 'NEUTRAL'
            recommendations['strategy'] = 'IRON CONDOR / SELL STRADDLE'
            
            atm_strike = round(S, -1)
            call_strike = round(S * 1.05, -1)
            put_strike = round(S * 0.95, -1)
            
            call_premium = self.price_option(call_strike, days_to_expiry, 'CALL')
            put_premium = self.price_option(put_strike, days_to_expiry, 'PUT')
            
            recommendations['details'] = {
                'action': 'SELL OTM CALL + SELL OTM PUT',
                'call_strike': call_strike,
                'put_strike': put_strike,
                'expiry_days': days_to_expiry,
                'total_premium': call_premium + put_premium,
                'max_profit': call_premium + put_premium,
                'upper_breakeven': call_strike + call_premium + put_premium,
                'lower_breakeven': put_strike - call_premium - put_premium
            }
            
        return recommendations
        
    def calculate_position_greeks(self, positions: List[Dict]) -> Dict:
        """
        Calculate aggregate Greeks for a portfolio of options.
        
        positions: List of dicts with keys: strike, days_to_expiry, option_type, quantity
        """
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for pos in positions:
            greeks = self.calculate_greeks(
                pos['strike'],
                pos['days_to_expiry'],
                pos['option_type']
            )
            qty = pos.get('quantity', 1)
            
            total_delta += greeks.delta * qty
            total_gamma += greeks.gamma * qty
            total_theta += greeks.theta * qty
            total_vega += greeks.vega * qty
            
        return {
            'net_delta': total_delta,
            'net_gamma': total_gamma,
            'net_theta': total_theta,
            'net_vega': total_vega,
            'delta_exposure': total_delta * self.current_price if self.current_price else 0
        }
        
    def plot_payoff_diagram(self, strategy_details: Dict, save_path='option_payoff.png'):
        """Plot payoff diagram for option strategy"""
        if self.current_price is None:
            self.fetch_underlying_data()
            
        S = self.current_price
        
        # Price range for payoff
        prices = np.linspace(S * 0.8, S * 1.2, 100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'type' in strategy_details:
            # Single option
            strike = strategy_details['strike']
            premium = strategy_details['premium']
            opt_type = strategy_details['type']
            
            if opt_type == 'CALL':
                payoff = np.maximum(prices - strike, 0) - premium
            else:
                payoff = np.maximum(strike - prices, 0) - premium
                
            ax.plot(prices, payoff, 'b-', linewidth=2, label=f'{opt_type} @ {strike}')
            
        elif 'buy_strike' in strategy_details and 'sell_strike' in strategy_details:
            # Spread
            buy_strike = strategy_details['buy_strike']
            sell_strike = strategy_details['sell_strike']
            net_premium = strategy_details['net_premium']
            
            if buy_strike < sell_strike:  # Bull call spread
                payoff = np.maximum(prices - buy_strike, 0) - np.maximum(prices - sell_strike, 0) - net_premium
            else:  # Bear put spread
                payoff = np.maximum(buy_strike - prices, 0) - np.maximum(sell_strike - prices, 0) - net_premium
                
            ax.plot(prices, payoff, 'b-', linewidth=2, label='Spread Payoff')
            
        elif 'call_strike' in strategy_details and 'put_strike' in strategy_details:
            # Iron condor / strangle
            call_strike = strategy_details['call_strike']
            put_strike = strategy_details['put_strike']
            total_premium = strategy_details['total_premium']
            
            call_payoff = -np.maximum(prices - call_strike, 0)
            put_payoff = -np.maximum(put_strike - prices, 0)
            payoff = call_payoff + put_payoff + total_premium
            
            ax.plot(prices, payoff, 'b-', linewidth=2, label='Short Strangle')
            
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=S, color='red', linestyle='--', alpha=0.5, label=f'Current Price: ₹{S:.0f}')
        ax.fill_between(prices, payoff, 0, where=(payoff > 0), alpha=0.3, color='green')
        ax.fill_between(prices, payoff, 0, where=(payoff < 0), alpha=0.3, color='red')
        
        ax.set_title('Option Strategy Payoff Diagram')
        ax.set_xlabel('MCX Stock Price (₹)')
        ax.set_ylabel('Profit/Loss (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Payoff diagram saved to: {save_path}")
        plt.show()
        
        return fig
        
    def print_option_chain(self, chain: pd.DataFrame):
        """Pretty print option chain"""
        print("\n" + "=" * 100)
        print(f"📊 MCX OPTIONS CHAIN (Current Price: ₹{self.current_price:.2f})")
        print("=" * 100)
        
        print(f"\n{'Strike':>10} │ {'Call ₹':>10} {'C.Delta':>8} {'C.Theta':>8} │ "
              f"{'Put ₹':>10} {'P.Delta':>8} {'P.Theta':>8} │ {'IV%':>6}")
        print("-" * 100)
        
        for _, row in chain.iterrows():
            atm = " ◄" if abs(row['Strike'] - self.current_price) < self.current_price * 0.025 else ""
            print(f"{row['Strike']:>10.0f} │ {row['Call_Price']:>10.2f} {row['Call_Delta']:>8.3f} "
                  f"{row['Call_Theta']:>8.3f} │ {row['Put_Price']:>10.2f} {row['Put_Delta']:>8.3f} "
                  f"{row['Put_Theta']:>8.3f} │ {row['IV']:>6.1f}{atm}")


def main():
    """Demo the options analyzer"""
    print("=" * 60)
    print("📈 MCX OPTIONS GREEKS ANALYZER")
    print("=" * 60)
    
    # Initialize
    analyzer = MCXOptionsAnalyzer()
    
    # Fetch underlying data
    analyzer.fetch_underlying_data()
    
    # Generate option chain
    print("\n🔄 Generating Option Chain (30 days to expiry)...")
    chain = analyzer.generate_option_chain(days_to_expiry=30)
    analyzer.print_option_chain(chain)
    
    # Simulate divergence signal from sync analysis
    divergence_z = 1.65  # From previous sync analysis
    
    print(f"\n🎯 SYNC-BASED STRATEGY RECOMMENDATION")
    print(f"   Current Divergence Z-Score: {divergence_z}")
    print("-" * 50)
    
    recommendation = analyzer.sync_based_option_strategy(divergence_z, days_to_expiry=30)
    
    print(f"\n   📊 Signal: {recommendation['signal']}")
    print(f"   📋 Strategy: {recommendation['strategy']}")
    print(f"\n   Details:")
    for key, value in recommendation['details'].items():
        if isinstance(value, float):
            print(f"      • {key}: ₹{value:.2f}")
        else:
            print(f"      • {key}: {value}")
            
    # Plot payoff
    if recommendation['details']:
        analyzer.plot_payoff_diagram(recommendation['details'])
    
    return analyzer, chain, recommendation


if __name__ == "__main__":
    analyzer, chain, recommendation = main()
