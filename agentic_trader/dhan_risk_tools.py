"""
DHAN RISK TOOLS — DhanHQ Data API for risk management & decision intelligence

Uses DhanHQ APIs that Kite doesn't offer:
  1. Kill Switch — server-side emergency trading halt 
  2. P&L Based Auto-Exit — auto-flatten at loss/profit threshold
  3. Margin Calculator — pre-validate IC/spread margin (with hedge benefits)
  4. Fund Limit — check available balance before orders
  5. Historical Data with OI — candle data including open interest

Kite remains the primary execution engine. Dhan provides supplementary
intelligence and safety nets.

Usage:
    from dhan_risk_tools import DhanRiskTools
    drt = DhanRiskTools()
    
    # Safety nets (set at market open)
    drt.set_pnl_exit(profit=5000, loss=3000)
    
    # Pre-validate margin for IC
    ok, margin_info = drt.check_ic_margin(ce_sell_id, ce_buy_id, pe_sell_id, pe_buy_id, qty)
    
    # Emergency halt
    drt.activate_kill_switch()
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

logger = logging.getLogger('dhan_risk')

# DhanHQ API base
DHAN_API = "https://api.dhan.co/v2"

# Load config — credentials from .env only

# Instrument segment codes
SEGMENTS = {
    'NSE_EQ': 'NSE_EQ',
    'NSE_FNO': 'NSE_FNO',
    'BSE_EQ': 'BSE_EQ',
    'BSE_FNO': 'BSE_FNO',
    'MCX_COMM': 'MCX_COMM',
    'IDX_I': 'IDX_I',
}


def _load_config() -> dict:
    """Load DhanHQ credentials from .env."""
    env_client = os.environ.get('DHAN_CLIENT_ID', '')
    env_token = os.environ.get('DHAN_ACCESS_TOKEN', '')
    if env_client and env_token:
        return {'client_id': env_client, 'access_token': env_token}
    return {}


class DhanRiskTools:
    """DhanHQ Data API client for risk management and decision intelligence."""

    def __init__(self, config_path: str = None):
        cfg = _load_config() if not config_path else json.load(open(config_path))
        self.client_id = cfg.get('client_id', '')
        self.access_token = cfg.get('access_token', '')
        self.ready = bool(self.client_id and self.access_token)
        
        if not self.ready:
            logger.warning("DhanRiskTools: No credentials found in .env (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)")
        
        self._headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'access-token': self.access_token,
            'client-id': self.client_id,
        }
        
        # Rate limiter
        self._last_call = 0
        self._min_interval = 1.0  # 1 req/sec for most endpoints
        
        # Scrip ID lookup (reuse from dhan_oi_fetcher)
        self._scrip_map = None

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _request(self, method: str, endpoint: str, json_body: dict = None,
                 params: dict = None) -> Tuple[int, dict]:
        """Make authenticated DhanHQ API call."""
        if not self.ready:
            return 0, {'error': 'DhanHQ not configured'}
        
        self._rate_limit()
        url = f"{DHAN_API}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                resp = requests.post(url, headers=self._headers, json=json_body, timeout=10)
            elif method.upper() == 'DELETE':
                resp = requests.delete(url, headers=self._headers, timeout=10)
            elif method.upper() == 'PUT':
                resp = requests.put(url, headers=self._headers, json=json_body, timeout=10)
            else:
                return 0, {'error': f'Unknown method: {method}'}
            
            try:
                data = resp.json()
            except Exception:
                data = {'raw': resp.text}
            
            return resp.status_code, data
        
        except requests.exceptions.Timeout:
            logger.warning(f"DhanHQ timeout: {endpoint}")
            return 0, {'error': 'timeout'}
        except Exception as e:
            logger.warning(f"DhanHQ error: {endpoint}: {e}")
            return 0, {'error': str(e)}

    # ========================================================================
    # 1. KILL SWITCH
    # ========================================================================
    
    def activate_kill_switch(self) -> Tuple[bool, str]:
        """Activate kill switch — disables ALL trading for the day.
        
        NOTE: All positions must be closed and no pending orders before activating.
        
        Returns:
            (success, message)
        """
        status, data = self._request('POST', '/killswitch?killSwitchStatus=ACTIVATE')
        if status == 200:
            msg = data.get('killSwitchStatus', 'Kill switch activated')
            logger.info(f"🛑 DhanHQ Kill Switch ACTIVATED: {msg}")
            return True, msg
        return False, data.get('killSwitchStatus', data.get('error', f'HTTP {status}'))
    
    def deactivate_kill_switch(self) -> Tuple[bool, str]:
        """Deactivate kill switch — re-enable trading."""
        status, data = self._request('POST', '/killswitch?killSwitchStatus=DEACTIVATE')
        if status == 200:
            msg = data.get('killSwitchStatus', 'Kill switch deactivated')
            logger.info(f"✅ DhanHQ Kill Switch DEACTIVATED: {msg}")
            return True, msg
        return False, data.get('killSwitchStatus', data.get('error', f'HTTP {status}'))
    
    def get_kill_switch_status(self) -> str:
        """Check kill switch status. Returns 'ACTIVATE' or 'DEACTIVATE'."""
        status, data = self._request('GET', '/killswitch')
        if status == 200:
            return data.get('killSwitchStatus', 'UNKNOWN')
        return 'UNKNOWN'

    # ========================================================================
    # 2. P&L BASED AUTO-EXIT
    # ========================================================================
    
    def set_pnl_exit(self, profit: float = 5000, loss: float = 3000,
                     product_types: list = None, enable_kill_switch: bool = True) -> Tuple[bool, str]:
        """Set P&L-based auto-exit for the day.
        
        When cumulative P&L breaches these thresholds, Dhan automatically
        exits all positions. This is a SERVER-SIDE safety net — works even
        if Titan crashes.
        
        Args:
            profit: Auto-exit when profit exceeds this (in ₹)
            loss: Auto-exit when loss exceeds this (in ₹)
            product_types: ['INTRADAY', 'DELIVERY'] — which product types to monitor
            enable_kill_switch: Also activate kill switch after exit
        
        Returns:
            (success, message)
        """
        if product_types is None:
            product_types = ['INTRADAY']
        
        body = {
            'dhanClientId': self.client_id,
            'profitValue': str(abs(profit)),
            'lossValue': str(-abs(loss)),
            'productType': product_types,
            'enableKillSwitch': enable_kill_switch,
        }
        
        status, data = self._request('POST', '/pnlExit', json_body=body)
        if status == 200:
            msg = data.get('message', 'P&L exit configured')
            logger.info(f"🛡️ DhanHQ P&L Exit set: profit=₹{profit}, loss=₹{loss}")
            return True, msg
        return False, data.get('message', data.get('error', f'HTTP {status}'))
    
    def stop_pnl_exit(self) -> Tuple[bool, str]:
        """Disable P&L-based auto-exit."""
        status, data = self._request('DELETE', '/pnlExit')
        if status == 200:
            return True, data.get('message', 'P&L exit stopped')
        return False, data.get('message', data.get('error', f'HTTP {status}'))
    
    def get_pnl_exit_status(self) -> dict:
        """Get current P&L exit configuration."""
        status, data = self._request('GET', '/pnlExit')
        if status == 200:
            return data
        return {'pnlExitStatus': 'UNKNOWN'}

    # ========================================================================
    # 3. MARGIN CALCULATOR
    # ========================================================================
    
    def check_margin(self, segment: str, transaction_type: str, quantity: int,
                     security_id: str, price: float, product_type: str = 'INTRADAY',
                     trigger_price: float = 0) -> dict:
        """Check margin requirement for a single order.
        
        Returns:
            dict with totalMargin, spanMargin, exposureMargin, availableBalance,
            insufficientBalance, brokerage, leverage
        """
        body = {
            'dhanClientId': self.client_id,
            'exchangeSegment': segment,
            'transactionType': transaction_type,
            'quantity': quantity,
            'productType': product_type,
            'securityId': str(security_id),
            'price': price,
        }
        if trigger_price > 0:
            body['triggerPrice'] = trigger_price
        
        status, data = self._request('POST', '/margincalculator', json_body=body)
        if status == 200:
            return data
        return {'error': data.get('error', f'HTTP {status}'), 'totalMargin': 0}
    
    def check_multi_margin(self, scripts: list, include_positions: bool = True,
                           include_orders: bool = True) -> dict:
        """Check margin for multiple orders (with hedge benefit).
        
        This is critical for IC/spreads — it computes the NET margin after
        hedge benefits from offsetting legs.
        
        Args:
            scripts: list of dicts, each with:
                exchangeSegment, transactionType, quantity, productType,
                securityId, price, [triggerPrice]
            include_positions: Account for existing positions
            include_orders: Account for pending orders
        
        Returns:
            dict with total_margin, span_margin, hedge_benefit, etc.
        """
        body = {
            'includePosition': include_positions,
            'includeOrders': include_orders,
            'dhanClientId': self.client_id,
            'scripts': scripts,
        }
        
        status, data = self._request('POST', '/margincalculator/multi', json_body=body)
        if status == 200:
            return data
        return {'error': data.get('error', f'HTTP {status}'), 'total_margin': '0'}
    
    def check_ic_margin(self, ce_sell_id: str, ce_buy_id: str,
                        pe_sell_id: str, pe_buy_id: str,
                        quantity: int, ce_sell_price: float = 0,
                        ce_buy_price: float = 0, pe_sell_price: float = 0,
                        pe_buy_price: float = 0) -> Tuple[bool, dict]:
        """Pre-validate margin for an Iron Condor (4 legs).
        
        Returns:
            (has_sufficient_margin, margin_details_dict)
        """
        scripts = [
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': 'SELL',
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(ce_sell_id),
                'price': ce_sell_price,
            },
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': 'BUY',
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(ce_buy_id),
                'price': ce_buy_price,
            },
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': 'SELL',
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(pe_sell_id),
                'price': pe_sell_price,
            },
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': 'BUY',
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(pe_buy_id),
                'price': pe_buy_price,
            },
        ]
        
        result = self.check_multi_margin(scripts)
        
        if 'error' in result:
            return False, result
        
        try:
            total_margin = float(result.get('total_margin', result.get('totalMargin', 0)))
            available = self.get_fund_limit().get('availabelBalance', 0)
            sufficient = available >= total_margin
            result['available_balance'] = available
            result['sufficient'] = sufficient
            if not sufficient:
                result['shortfall'] = total_margin - available
            return sufficient, result
        except Exception as e:
            return False, {'error': str(e)}
    
    def check_spread_margin(self, sell_id: str, buy_id: str, quantity: int,
                            sell_price: float = 0, buy_price: float = 0,
                            spread_type: str = 'CREDIT') -> Tuple[bool, dict]:
        """Pre-validate margin for a credit/debit spread (2 legs).
        
        Returns:
            (has_sufficient_margin, margin_details_dict)
        """
        sell_txn = 'SELL' if spread_type == 'CREDIT' else 'BUY'
        buy_txn = 'BUY' if spread_type == 'CREDIT' else 'SELL'
        
        scripts = [
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': sell_txn,
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(sell_id),
                'price': sell_price,
            },
            {
                'exchangeSegment': 'NSE_FNO',
                'transactionType': buy_txn,
                'quantity': quantity,
                'productType': 'INTRADAY',
                'securityId': str(buy_id),
                'price': buy_price,
            },
        ]
        
        result = self.check_multi_margin(scripts)
        
        if 'error' in result:
            return False, result
        
        try:
            total_margin = float(result.get('total_margin', result.get('totalMargin', 0)))
            available = self.get_fund_limit().get('availabelBalance', 0)
            sufficient = available >= total_margin
            result['available_balance'] = available
            result['sufficient'] = sufficient
            if not sufficient:
                result['shortfall'] = total_margin - available
            return sufficient, result
        except Exception as e:
            return False, {'error': str(e)}

    # ========================================================================
    # 4. FUND LIMIT
    # ========================================================================
    
    def get_fund_limit(self) -> dict:
        """Get available fund balance.
        
        Returns:
            dict with availabelBalance, sodLimit, utilizedAmount, etc.
        """
        status, data = self._request('GET', '/fundlimit')
        if status == 200:
            return data
        return {'availabelBalance': 0, 'error': data.get('error', f'HTTP {status}')}

    # ========================================================================
    # 5. HISTORICAL DATA WITH OI
    # ========================================================================
    
    def fetch_daily_candles(self, security_id: str, from_date: str, to_date: str,
                           segment: str = 'NSE_EQ', instrument: str = 'EQUITY',
                           include_oi: bool = False) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV candles from DhanHQ (inception lookback).
        
        Args:
            security_id: DhanHQ security ID (e.g., '3045' for SBIN)
            from_date: Start date 'YYYY-MM-DD'
            to_date: End date 'YYYY-MM-DD' (non-inclusive)
            segment: Exchange segment
            instrument: Instrument type (EQUITY, FUTIDX, FUTSTOCK, OPTIDX, OPTSTK)
            include_oi: Include open interest column
        
        Returns:
            DataFrame with date, open, high, low, close, volume, [oi]
        """
        body = {
            'securityId': str(security_id),
            'exchangeSegment': segment,
            'instrument': instrument,
            'expiryCode': 0,
            'oi': include_oi,
            'fromDate': from_date,
            'toDate': to_date,
        }
        
        status, data = self._request('POST', '/charts/historical', json_body=body)
        
        if status != 200 or not data.get('timestamp'):
            return None
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['timestamp'], unit='s'),
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
        })
        
        if include_oi and 'open_interest' in data:
            df['oi'] = data['open_interest']
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def fetch_intraday_candles(self, security_id: str, from_date: str, to_date: str,
                              interval: int = 5, segment: str = 'NSE_EQ',
                              instrument: str = 'EQUITY',
                              include_oi: bool = False) -> Optional[pd.DataFrame]:
        """Fetch intraday candles from DhanHQ (5 years lookback, 90 days per request).
        
        Args:
            security_id: DhanHQ security ID
            from_date: Start datetime 'YYYY-MM-DD HH:MM:SS'
            to_date: End datetime 'YYYY-MM-DD HH:MM:SS'
            interval: Candle interval in minutes (1, 5, 15, 25, 60)
            segment: Exchange segment
            instrument: Instrument type
            include_oi: Include open interest column
        
        Returns:
            DataFrame with date, open, high, low, close, volume, [oi]
        """
        body = {
            'securityId': str(security_id),
            'exchangeSegment': segment,
            'instrument': instrument,
            'interval': str(interval),
            'oi': include_oi,
            'fromDate': from_date,
            'toDate': to_date,
        }
        
        status, data = self._request('POST', '/charts/intraday', json_body=body)
        
        if status != 200 or not data.get('timestamp'):
            return None
        
        df = pd.DataFrame({
            'date': pd.to_datetime(data['timestamp'], unit='s'),
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
        })
        
        if include_oi and 'open_interest' in data:
            df['oi'] = data['open_interest']
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    def fetch_intraday_with_oi_bulk(self, security_id: str, days_back: int = 90,
                                    interval: int = 5, segment: str = 'NSE_EQ',
                                    instrument: str = 'EQUITY') -> Optional[pd.DataFrame]:
        """Fetch multiple 90-day chunks of intraday data with OI.
        
        DhanHQ allows max 90 days per request but 5 years total.
        This method auto-chunks.
        
        Args:
            security_id: DhanHQ security ID
            days_back: Total days of history to fetch (max ~1800)
            interval: Candle interval in minutes
            segment: Exchange segment
            instrument: Instrument type
        
        Returns:
            Combined DataFrame
        """
        all_dfs = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        current_start = start_date
        chunk_size = 85  # Slightly under 90 to be safe
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_size), end_date)
            
            from_str = current_start.strftime('%Y-%m-%d 09:15:00')
            to_str = current_end.strftime('%Y-%m-%d 15:30:00')
            
            df = self.fetch_intraday_candles(
                security_id, from_str, to_str,
                interval=interval, segment=segment,
                instrument=instrument, include_oi=True
            )
            
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.debug(f"DhanHQ: Fetched {len(df)} candles from {from_str[:10]} to {to_str[:10]}")
            
            current_start = current_end
            time.sleep(0.5)  # Be gentle on API
        
        if not all_dfs:
            return None
        
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        return combined

    # ========================================================================
    # UTILITY: Scrip ID lookup
    # ========================================================================
    
    def _get_scrip_map(self) -> dict:
        """Get scrip ID map (lazy load from dhan_oi_fetcher)."""
        if self._scrip_map is None:
            try:
                from dhan_oi_fetcher import DHAN_SCRIP_MAP
                self._scrip_map = DHAN_SCRIP_MAP
            except Exception:
                self._scrip_map = {}
        return self._scrip_map
    
    def get_security_id(self, symbol: str) -> Optional[str]:
        """Look up DhanHQ security ID for a symbol."""
        scrip_map = self._get_scrip_map()
        info = scrip_map.get(symbol.upper())
        if info:
            return str(info.get('scrip_id', ''))
        return None
    
    def get_segment(self, symbol: str) -> str:
        """Look up DhanHQ exchange segment for a symbol."""
        scrip_map = self._get_scrip_map()
        info = scrip_map.get(symbol.upper())
        if info:
            return info.get('segment', 'NSE_FNO')
        return 'NSE_FNO'

    # ========================================================================
    # HIGH-LEVEL INTEGRATION HELPERS
    # ========================================================================
    
    def setup_daily_safety(self, max_loss: float, max_profit: float = 0,
                           starting_capital: float = 500000) -> dict:
        """Set up daily safety nets at market open.
        
        Call this in Titan's initialization before trading starts.
        
        Args:
            max_loss: Maximum loss in ₹ before auto-exit (e.g., 30000 = 6% of 5L)
            max_profit: Maximum profit target in ₹ (0 = no profit cap)
            starting_capital: For percentage calculation
        
        Returns:
            dict with kill_switch_status and pnl_exit_status
        """
        result = {}
        
        # Check kill switch isn't already active
        ks_status = self.get_kill_switch_status()
        result['kill_switch'] = ks_status
        if ks_status == 'ACTIVATE':
            logger.warning("Kill switch is ACTIVE from previous session — deactivating")
            self.deactivate_kill_switch()
            result['kill_switch'] = 'DEACTIVATED'
        
        # Set P&L exit
        if max_profit <= 0:
            max_profit = max_loss * 3  # Default: 3:1 reward cap
        
        ok, msg = self.set_pnl_exit(
            profit=max_profit,
            loss=max_loss,
            product_types=['INTRADAY'],
            enable_kill_switch=True,
        )
        result['pnl_exit'] = {'success': ok, 'message': msg,
                              'profit_cap': max_profit, 'loss_cap': max_loss}
        
        if ok:
            logger.info(f"🛡️ Daily safety nets active: loss_cap=₹{max_loss:,.0f}, profit_cap=₹{max_profit:,.0f}")
        
        return result
    
    def emergency_halt(self) -> dict:
        """EMERGENCY: Activate kill switch immediately.
        
        Call from risk_governor.trigger_circuit_breaker() as a server-side backup.
        """
        result = {}
        ok, msg = self.activate_kill_switch()
        result['kill_switch'] = {'success': ok, 'message': msg}
        
        if ok:
            print(f"\n🚨🚨🚨 DHAN KILL SWITCH ACTIVATED — ALL TRADING DISABLED 🚨🚨🚨")
        else:
            print(f"\n⚠️ Kill switch activation failed: {msg}")
            print("  (This may be because positions are still open)")
        
        return result
    
    def get_status_summary(self) -> str:
        """Get a one-line summary for logging."""
        try:
            fund = self.get_fund_limit()
            ks = self.get_kill_switch_status()
            pnl = self.get_pnl_exit_status()
            
            bal = fund.get('availabelBalance', 0)
            pnl_status = pnl.get('pnlExitStatus', 'INACTIVE')
            
            return (f"Dhan: Balance=₹{bal:,.0f} | "
                    f"KillSwitch={ks} | "
                    f"P&L_Exit={pnl_status}")
        except Exception:
            return "Dhan: status unavailable"


# Singleton access
_instance = None

def get_dhan_risk_tools() -> DhanRiskTools:
    """Get or create singleton DhanRiskTools instance."""
    global _instance
    if _instance is None:
        _instance = DhanRiskTools()
    return _instance


if __name__ == '__main__':
    # Quick test
    drt = DhanRiskTools()
    
    if not drt.ready:
        print("❌ DhanHQ not configured. Check DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in .env")
        exit(1)
    
    print("=== DhanHQ Risk Tools Test ===\n")
    
    # 1. Kill Switch Status
    ks = drt.get_kill_switch_status()
    print(f"Kill Switch: {ks}")
    
    # 2. Fund Limit
    fund = drt.get_fund_limit()
    print(f"Available Balance: ₹{fund.get('availabelBalance', 0):,.2f}")
    print(f"Utilized: ₹{fund.get('utilizedAmount', 0):,.2f}")
    
    # 3. P&L Exit Status
    pnl = drt.get_pnl_exit_status()
    print(f"P&L Exit: {pnl.get('pnlExitStatus', 'N/A')}")
    
    # 4. Single margin check (SBIN)
    margin = drt.check_margin(
        segment='NSE_FNO',
        transaction_type='SELL',
        quantity=1500,
        security_id='3045',
        price=10.0,
        product_type='INTRADAY',
    )
    print(f"\nMargin for SBIN SELL 1500 @ ₹10:")
    for k, v in margin.items():
        print(f"  {k}: {v}")
    
    # 5. Status summary
    print(f"\n{drt.get_status_summary()}")
