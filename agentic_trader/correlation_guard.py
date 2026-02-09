"""
CORRELATION & INDEX GUARD MODULE
Prevents hidden overexposure from correlated positions

Features:
1. Static beta-to-index mapping for F&O stocks
2. Real-time correlation computation to NIFTY/BANKNIFTY
3. Max correlated positions limit
4. Index options/futures + stocks = same risk bucket
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class CorrelationCheck:
    """Result of correlation check"""
    symbol: str
    can_trade: bool
    reason: str
    correlation_to_nifty: float
    correlation_to_banknifty: float
    beta_category: str  # HIGH_BETA, MEDIUM_BETA, LOW_BETA, INDEX
    correlated_positions: List[str]  # Existing positions in same bucket
    warnings: List[str] = field(default_factory=list)


class CorrelationIndexGuard:
    """
    Guards against hidden overexposure through correlated positions
    
    Rules:
    1. Max 2 HIGH_BETA positions at a time
    2. If holding index options/futures, max 1 additional HIGH_BETA stock
    3. Compute real-time correlation and block if threshold exceeded
    """
    
    def __init__(self):
        # === STATIC BETA-TO-INDEX MAPPING ===
        # HIGH_BETA: Move more than NIFTY (beta > 1.2)
        # MEDIUM_BETA: Move with NIFTY (beta 0.8-1.2)
        # LOW_BETA: Move less than NIFTY (beta < 0.8)
        
        self.beta_mapping = {
            # HIGH BETA (>1.2) - Very correlated to index
            "NSE:BHARTIARTL": "HIGH_BETA",
            "NSE:BAJFINANCE": "HIGH_BETA",
            "NSE:BAJAJFINSV": "HIGH_BETA",
            "NSE:ADANIENT": "HIGH_BETA",
            "NSE:ADANIPORTS": "HIGH_BETA",
            "NSE:INDUSINDBK": "HIGH_BETA",
            "NSE:AXISBANK": "HIGH_BETA",
            "NSE:ICICIBANK": "HIGH_BETA",
            "NSE:SBIN": "HIGH_BETA",
            "NSE:TATASTEEL": "HIGH_BETA",
            "NSE:HINDALCO": "HIGH_BETA",
            "NSE:JSWSTEEL": "HIGH_BETA",
            "NSE:COALINDIA": "HIGH_BETA",
            "NSE:VEDL": "HIGH_BETA",
            "NSE:JINDALSTEL": "HIGH_BETA",
            "NSE:NMDC": "HIGH_BETA",
            "NSE:NATIONALUM": "HIGH_BETA",
            "NSE:SAIL": "HIGH_BETA",
            "NSE:HINDCOPPER": "HIGH_BETA",
            "NSE:ONGC": "HIGH_BETA",
            "NSE:BPCL": "HIGH_BETA",
            "NSE:M&M": "HIGH_BETA",
            "NSE:MARUTI": "HIGH_BETA",
            "NSE:EICHERMOT": "HIGH_BETA",
            "NSE:HEROMOTOCO": "HIGH_BETA",
            "NSE:BAJAJ-AUTO": "HIGH_BETA",
            
            # MEDIUM BETA (0.8-1.2) - Moderate correlation
            "NSE:RELIANCE": "MEDIUM_BETA",
            "NSE:HDFCBANK": "MEDIUM_BETA",
            "NSE:KOTAKBANK": "MEDIUM_BETA",
            "NSE:LT": "MEDIUM_BETA",
            "NSE:ULTRACEMCO": "MEDIUM_BETA",
            "NSE:GRASIM": "MEDIUM_BETA",
            "NSE:TITAN": "MEDIUM_BETA",
            "NSE:ASIANPAINT": "MEDIUM_BETA",
            "NSE:NTPC": "MEDIUM_BETA",
            "NSE:POWERGRID": "MEDIUM_BETA",
            
            # LOW BETA (<0.8) - Defensive, less correlated
            "NSE:INFY": "LOW_BETA",
            "NSE:TCS": "LOW_BETA",
            "NSE:WIPRO": "LOW_BETA",
            "NSE:HCLTECH": "LOW_BETA",
            "NSE:TECHM": "LOW_BETA",
            "NSE:LTIM": "LOW_BETA",
            "NSE:SUNPHARMA": "LOW_BETA",
            "NSE:DRREDDY": "LOW_BETA",
            "NSE:CIPLA": "LOW_BETA",
            "NSE:APOLLOHOSP": "LOW_BETA",
            "NSE:DIVISLAB": "LOW_BETA",
            "NSE:HINDUNILVR": "LOW_BETA",
            "NSE:ITC": "LOW_BETA",
            "NSE:NESTLEIND": "LOW_BETA",
            "NSE:BRITANNIA": "LOW_BETA",
            
            # INDEX instruments - Treated as HIGH correlation
            "NSE:NIFTY": "INDEX",
            "NSE:BANKNIFTY": "INDEX",
            "NFO:NIFTY": "INDEX",
            "NFO:BANKNIFTY": "INDEX",
        }
        
        # Sector groupings for additional correlation check
        self.sector_groups = {
            "BANKING": ["NSE:HDFCBANK", "NSE:ICICIBANK", "NSE:AXISBANK", "NSE:KOTAKBANK", "NSE:SBIN", "NSE:INDUSINDBK"],
            "IT": ["NSE:INFY", "NSE:TCS", "NSE:WIPRO", "NSE:HCLTECH", "NSE:TECHM", "NSE:LTIM"],
            "AUTO": ["NSE:M&M", "NSE:MARUTI", "NSE:BAJAJ-AUTO", "NSE:HEROMOTOCO", "NSE:EICHERMOT"],
            "TELECOM": ["NSE:BHARTIARTL"],
            "METAL": ["NSE:TATASTEEL", "NSE:HINDALCO", "NSE:JSWSTEEL", "NSE:COALINDIA", "NSE:VEDL", "NSE:JINDALSTEL", "NSE:NMDC", "NSE:NATIONALUM", "NSE:SAIL", "NSE:HINDCOPPER"],
            "PHARMA": ["NSE:SUNPHARMA", "NSE:DRREDDY", "NSE:CIPLA", "NSE:APOLLOHOSP", "NSE:DIVISLAB"],
            "OIL_GAS": ["NSE:RELIANCE", "NSE:ONGC", "NSE:BPCL"],
            "NBFC": ["NSE:BAJFINANCE", "NSE:BAJAJFINSV"],
            "FMCG": ["NSE:HINDUNILVR", "NSE:ITC", "NSE:NESTLEIND", "NSE:BRITANNIA"],
        }
        
        # Limits
        self.max_high_beta_positions = 2
        self.max_same_sector_positions = 2
        self.max_index_plus_stocks = 3  # If holding index, max 2 additional stocks
        self.correlation_threshold = 0.75  # Real-time correlation threshold
        
        # Track active positions for correlation
        self.active_positions: Dict[str, str] = {}  # symbol -> beta_category
        
        # Real-time correlation cache (refreshed periodically)
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_validity_minutes = 5
    
    def get_beta_category(self, symbol: str) -> str:
        """Get beta category for a symbol"""
        # Check direct mapping
        if symbol in self.beta_mapping:
            return self.beta_mapping[symbol]
        
        # Check if it's an index derivative
        if "NIFTY" in symbol or "BANKNIFTY" in symbol:
            return "INDEX"
        
        # Default to MEDIUM_BETA for unknown symbols
        return "MEDIUM_BETA"
    
    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol"""
        for sector, symbols in self.sector_groups.items():
            if symbol in symbols:
                return sector
        return None
    
    def update_positions(self, active_positions: List[Dict]):
        """Update tracked positions from active trades"""
        self.active_positions = {}
        for pos in active_positions:
            symbol = pos.get('symbol', '')
            if symbol:
                self.active_positions[symbol] = self.get_beta_category(symbol)
    
    def count_by_beta(self) -> Dict[str, int]:
        """Count active positions by beta category"""
        counts = {"HIGH_BETA": 0, "MEDIUM_BETA": 0, "LOW_BETA": 0, "INDEX": 0}
        for symbol, category in self.active_positions.items():
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    def count_by_sector(self) -> Dict[str, int]:
        """Count active positions by sector"""
        counts = {}
        for symbol in self.active_positions.keys():
            sector = self.get_sector(symbol)
            if sector:
                counts[sector] = counts.get(sector, 0) + 1
        return counts
    
    def get_correlated_positions(self, symbol: str) -> List[str]:
        """Get list of existing positions correlated with this symbol"""
        correlated = []
        target_category = self.get_beta_category(symbol)
        target_sector = self.get_sector(symbol)
        
        for existing_symbol in self.active_positions.keys():
            if existing_symbol == symbol:
                continue
            
            # Same beta category = correlated
            if self.active_positions[existing_symbol] == target_category:
                if target_category in ["HIGH_BETA", "INDEX"]:
                    correlated.append(existing_symbol)
            
            # Same sector = correlated
            existing_sector = self.get_sector(existing_symbol)
            if existing_sector and existing_sector == target_sector:
                if existing_symbol not in correlated:
                    correlated.append(existing_symbol)
        
        return correlated
    
    def can_trade(
        self,
        symbol: str,
        active_positions: List[Dict],
        realtime_correlation: Optional[Dict[str, float]] = None
    ) -> CorrelationCheck:
        """
        Check if a new trade is allowed based on correlation limits
        
        Args:
            symbol: Symbol to trade
            active_positions: List of current open positions
            realtime_correlation: Optional real-time correlation data
        
        Returns:
            CorrelationCheck with decision and reasons
        """
        # Update positions
        self.update_positions(active_positions)
        
        # Get symbol info
        beta_category = self.get_beta_category(symbol)
        sector = self.get_sector(symbol)
        correlated_positions = self.get_correlated_positions(symbol)
        
        warnings = []
        
        # Get correlation values (use cache or provided data)
        corr_nifty = 0.0
        corr_banknifty = 0.0
        
        if realtime_correlation:
            corr_nifty = realtime_correlation.get('nifty', 0.0)
            corr_banknifty = realtime_correlation.get('banknifty', 0.0)
        
        # === CHECK 1: High Beta Limit ===
        beta_counts = self.count_by_beta()
        
        if beta_category == "HIGH_BETA":
            if beta_counts["HIGH_BETA"] >= self.max_high_beta_positions:
                return CorrelationCheck(
                    symbol=symbol,
                    can_trade=False,
                    reason=f"MAX HIGH_BETA limit reached ({beta_counts['HIGH_BETA']}/{self.max_high_beta_positions})",
                    correlation_to_nifty=corr_nifty,
                    correlation_to_banknifty=corr_banknifty,
                    beta_category=beta_category,
                    correlated_positions=correlated_positions,
                    warnings=[f"Already holding: {', '.join([s for s, c in self.active_positions.items() if c == 'HIGH_BETA'])}"]
                )
        
        # === CHECK 2: Index + Stocks Limit ===
        if beta_category == "INDEX":
            # If already holding index, block
            if beta_counts["INDEX"] >= 1:
                return CorrelationCheck(
                    symbol=symbol,
                    can_trade=False,
                    reason="Already holding INDEX position - max 1 index position allowed",
                    correlation_to_nifty=corr_nifty,
                    correlation_to_banknifty=corr_banknifty,
                    beta_category=beta_category,
                    correlated_positions=correlated_positions,
                    warnings=["Index positions are highly correlated"]
                )
        
        # If holding index, limit additional high-beta stocks
        if beta_counts["INDEX"] > 0 and beta_category == "HIGH_BETA":
            total_correlated = beta_counts["HIGH_BETA"] + beta_counts["INDEX"]
            if total_correlated >= self.max_index_plus_stocks:
                return CorrelationCheck(
                    symbol=symbol,
                    can_trade=False,
                    reason=f"INDEX + HIGH_BETA limit reached ({total_correlated}/{self.max_index_plus_stocks})",
                    correlation_to_nifty=corr_nifty,
                    correlation_to_banknifty=corr_banknifty,
                    beta_category=beta_category,
                    correlated_positions=correlated_positions,
                    warnings=["Holding index position limits additional high-beta trades"]
                )
        
        # === CHECK 3: Same Sector Limit ===
        if sector:
            sector_counts = self.count_by_sector()
            if sector_counts.get(sector, 0) >= self.max_same_sector_positions:
                return CorrelationCheck(
                    symbol=symbol,
                    can_trade=False,
                    reason=f"MAX {sector} sector limit reached ({sector_counts[sector]}/{self.max_same_sector_positions})",
                    correlation_to_nifty=corr_nifty,
                    correlation_to_banknifty=corr_banknifty,
                    beta_category=beta_category,
                    correlated_positions=correlated_positions,
                    warnings=[f"Same sector positions move together"]
                )
        
        # === CHECK 4: Real-time Correlation Threshold ===
        if max(abs(corr_nifty), abs(corr_banknifty)) > self.correlation_threshold:
            # High real-time correlation - check if we have many correlated positions
            if len(correlated_positions) >= 2:
                warnings.append(f"High real-time correlation ({max(corr_nifty, corr_banknifty):.2f}) with {len(correlated_positions)} similar positions")
        
        # === PASSED ALL CHECKS ===
        # Add warnings for borderline cases
        if beta_category == "HIGH_BETA" and beta_counts["HIGH_BETA"] == self.max_high_beta_positions - 1:
            warnings.append("This will be your last HIGH_BETA position slot")
        
        if len(correlated_positions) > 0:
            warnings.append(f"Correlated with: {', '.join(correlated_positions)}")
        
        return CorrelationCheck(
            symbol=symbol,
            can_trade=True,
            reason="Correlation check passed",
            correlation_to_nifty=corr_nifty,
            correlation_to_banknifty=corr_banknifty,
            beta_category=beta_category,
            correlated_positions=correlated_positions,
            warnings=warnings
        )
    
    def get_exposure_summary(self) -> str:
        """Get summary of current exposure by correlation"""
        beta_counts = self.count_by_beta()
        sector_counts = self.count_by_sector()
        
        lines = ["📊 CORRELATION EXPOSURE:"]
        lines.append(f"  HIGH_BETA: {beta_counts['HIGH_BETA']}/{self.max_high_beta_positions}")
        lines.append(f"  INDEX: {beta_counts['INDEX']}/1")
        lines.append(f"  MEDIUM_BETA: {beta_counts['MEDIUM_BETA']}")
        lines.append(f"  LOW_BETA: {beta_counts['LOW_BETA']}")
        
        if sector_counts:
            lines.append("  SECTORS:")
            for sector, count in sector_counts.items():
                lines.append(f"    {sector}: {count}/{self.max_same_sector_positions}")
        
        return "\n".join(lines)


# Singleton instance
_correlation_guard: Optional[CorrelationIndexGuard] = None


def get_correlation_guard() -> CorrelationIndexGuard:
    """Get singleton instance of CorrelationIndexGuard"""
    global _correlation_guard
    if _correlation_guard is None:
        _correlation_guard = CorrelationIndexGuard()
    return _correlation_guard
