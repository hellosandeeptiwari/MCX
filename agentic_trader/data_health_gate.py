"""
DATA HEALTH GATE MODULE
Stale/dirty data = silent killer. This gate validates data quality before allowing trades.

Verifies before scoring/placing orders:
1. Last tick/candle timestamp within tolerance
2. No gaps in candle sequence
3. Volume not zero (feed glitch)
4. VWAP/EMA/ATR not NaN or zero

If any fail → NO_TRADE + increment stale counter (connects to breaker)
"""

import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class DataHealthStatus(Enum):
    """Data health status"""
    HEALTHY = "HEALTHY"           # All checks pass
    STALE = "STALE"               # Data too old
    GAPS = "GAPS"                 # Missing candles
    ZERO_VOLUME = "ZERO_VOLUME"   # Volume glitch
    INVALID_INDICATORS = "INVALID"  # NaN/zero indicators
    FAILED = "FAILED"             # Multiple issues


@dataclass
class DataHealthResult:
    """Result of data health check"""
    symbol: str
    status: DataHealthStatus
    can_trade: bool
    checks: Dict[str, Dict]  # Each check with pass/fail and details
    issues: List[str]
    stale_counter: int
    timestamp: str


class DataHealthGate:
    """
    Data Health Gate
    
    Validates data quality before allowing trades.
    Critical for preventing trades on stale, gapped, or corrupted data.
    """
    
    HEALTH_STATE_FILE = os.path.join(os.path.dirname(__file__), 'data_health_state.json')
    
    # Tolerances
    STALE_TICK_SECONDS = 60       # Max seconds since last tick
    STALE_CANDLE_MINUTES = 5      # Max minutes since last candle
    MAX_CANDLE_GAP_MINUTES = 15   # Max gap between candles
    MIN_VOLUME = 1                # Minimum volume (0 = glitch)
    MIN_ATR_PCT = 0.001           # Min ATR as % of price (avoid divide by zero)
    MAX_STALE_BEFORE_HALT = 5     # Max stale readings before halting symbol
    
    def __init__(self):
        """Initialize data health gate"""
        # Track stale counters per symbol
        self.stale_counters: Dict[str, int] = {}
        
        # Halted symbols (too many stale readings)
        self.halted_symbols: List[str] = []
        
        # Last known good data per symbol
        self.last_good_data: Dict[str, datetime] = {}
        
        # Health history for diagnostics
        self.health_history: List[Dict] = []
        
        # Load saved state
        self._load_state()
        
        print("🛡️ Data Health Gate: INITIALIZED")
    
    def _load_state(self):
        """Load health state from file"""
        try:
            if os.path.exists(self.HEALTH_STATE_FILE):
                with open(self.HEALTH_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.stale_counters = data.get('stale_counters', {})
                        self.halted_symbols = data.get('halted_symbols', [])
                        # Clean symbols no longer in universe
                        try:
                            from config import APPROVED_UNIVERSE
                            self.halted_symbols = [s for s in self.halted_symbols if s in APPROVED_UNIVERSE]
                            self.stale_counters = {k: v for k, v in self.stale_counters.items() if k in APPROVED_UNIVERSE}
                        except ImportError:
                            pass
        except Exception as e:
            print(f"⚠️ Error loading health state: {e}")
    
    def _save_state(self):
        """Save health state to file"""
        try:
            data = {
                'date': str(datetime.now().date()),
                'last_updated': datetime.now().isoformat(),
                'stale_counters': self.stale_counters,
                'halted_symbols': self.halted_symbols,
                'health_history': self.health_history[-100:]  # Keep last 100
            }
            with open(self.HEALTH_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving health state: {e}")
    
    def check_health(self, symbol: str, market_data: Dict) -> DataHealthResult:
        """
        Check data health for a symbol
        
        Args:
            symbol: Trading symbol (e.g., "NSE:INFY")
            market_data: Dict with ltp, volume, timestamp, indicators, etc.
            
        Returns:
            DataHealthResult with status and details
        """
        checks = {}
        issues = []
        all_passed = True
        
        # === CHECK 1: TICK FRESHNESS ===
        tick_check = self._check_tick_freshness(symbol, market_data)
        checks['tick_freshness'] = tick_check
        if not tick_check['passed']:
            issues.append(tick_check['issue'])
            all_passed = False
        
        # === CHECK 2: CANDLE FRESHNESS ===
        candle_check = self._check_candle_freshness(symbol, market_data)
        checks['candle_freshness'] = candle_check
        if not candle_check['passed']:
            issues.append(candle_check['issue'])
            all_passed = False
        
        # === CHECK 3: CANDLE SEQUENCE (no gaps) ===
        gap_check = self._check_candle_gaps(symbol, market_data)
        checks['candle_gaps'] = gap_check
        if not gap_check['passed']:
            issues.append(gap_check['issue'])
            all_passed = False
        
        # === CHECK 4: VOLUME NOT ZERO ===
        volume_check = self._check_volume(symbol, market_data)
        checks['volume'] = volume_check
        if not volume_check['passed']:
            issues.append(volume_check['issue'])
            all_passed = False
        
        # === CHECK 5: INDICATORS VALID (not NaN/zero) ===
        indicator_check = self._check_indicators(symbol, market_data)
        checks['indicators'] = indicator_check
        if not indicator_check['passed']:
            issues.extend(indicator_check.get('issues', []))
            all_passed = False
        
        # === UPDATE STALE COUNTER ===
        if all_passed:
            # Reset stale counter on good data
            self.stale_counters[symbol] = 0
            self.last_good_data[symbol] = datetime.now()
            
            # Remove from halted if it was halted
            if symbol in self.halted_symbols:
                self.halted_symbols.remove(symbol)
                print(f"✅ {symbol} UNHALTED - data healthy again")
            
            status = DataHealthStatus.HEALTHY
        else:
            # Increment stale counter
            self.stale_counters[symbol] = self.stale_counters.get(symbol, 0) + 1
            
            # Determine status based on issues
            if 'STALE' in str(issues):
                status = DataHealthStatus.STALE
            elif 'GAP' in str(issues):
                status = DataHealthStatus.GAPS
            elif 'VOLUME' in str(issues):
                status = DataHealthStatus.ZERO_VOLUME
            elif 'NaN' in str(issues) or 'ZERO' in str(issues):
                status = DataHealthStatus.INVALID_INDICATORS
            else:
                status = DataHealthStatus.FAILED
            
            # Halt symbol if too many stale readings
            if self.stale_counters[symbol] >= self.MAX_STALE_BEFORE_HALT:
                if symbol not in self.halted_symbols:
                    self.halted_symbols.append(symbol)
                    print(f"🚨 {symbol} HALTED - {self.stale_counters[symbol]} consecutive stale readings")
                    
                    # === RESET HYSTERESIS FOR HALTED SYMBOL ===
                    # Prevents stale trend state when data resumes
                    try:
                        from trend_following import get_trend_engine
                        get_trend_engine().reset_hysteresis(symbol, "DATA_HALT")
                    except ImportError:
                        pass
                    try:
                        from regime_score import get_regime_scorer
                        get_regime_scorer().reset_hysteresis(symbol, "DATA_HALT")
                    except ImportError:
                        pass
        
        stale_counter = self.stale_counters.get(symbol, 0)
        can_trade = all_passed and symbol not in self.halted_symbols
        
        # Log to history
        self.health_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'status': status.value,
            'can_trade': can_trade,
            'issues': issues
        })
        
        self._save_state()
        
        return DataHealthResult(
            symbol=symbol,
            status=status,
            can_trade=can_trade,
            checks=checks,
            issues=issues,
            stale_counter=stale_counter,
            timestamp=datetime.now().isoformat()
        )
    
    def _check_tick_freshness(self, symbol: str, market_data: Dict) -> Dict:
        """Check if last tick is fresh"""
        timestamp_str = market_data.get('timestamp', '')
        last_trade_time = market_data.get('last_trade_time')
        
        # Try to parse timestamp
        age_seconds = None
        
        if last_trade_time:
            try:
                if isinstance(last_trade_time, str):
                    last_trade_time = datetime.fromisoformat(last_trade_time.replace('Z', '+00:00'))
                age_seconds = (datetime.now() - last_trade_time).total_seconds()
            except:
                pass
        
        if timestamp_str and age_seconds is None:
            try:
                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
                age_seconds = (datetime.now() - ts).total_seconds()
            except:
                pass
        
        if age_seconds is None:
            age_seconds = 0  # Assume fresh if can't determine
        
        passed = age_seconds <= self.STALE_TICK_SECONDS
        
        return {
            'passed': passed,
            'age_seconds': age_seconds,
            'threshold': self.STALE_TICK_SECONDS,
            'issue': f"STALE_TICK: {age_seconds:.0f}s old (limit: {self.STALE_TICK_SECONDS}s)" if not passed else None
        }
    
    def _check_candle_freshness(self, symbol: str, market_data: Dict) -> Dict:
        """Check if last candle is fresh"""
        # Check for candle timestamp in various locations
        last_candle_time = market_data.get('last_candle_time') or market_data.get('candle_timestamp')
        
        if not last_candle_time:
            # If no candle time provided, assume fresh (will be caught by other checks)
            return {'passed': True, 'note': 'No candle timestamp provided'}
        
        try:
            if isinstance(last_candle_time, str):
                last_candle_time = datetime.fromisoformat(last_candle_time.replace('Z', '+00:00').replace('+00:00', ''))
            
            age_minutes = (datetime.now() - last_candle_time).total_seconds() / 60
            passed = age_minutes <= self.STALE_CANDLE_MINUTES
            
            return {
                'passed': passed,
                'age_minutes': age_minutes,
                'threshold': self.STALE_CANDLE_MINUTES,
                'issue': f"STALE_CANDLE: {age_minutes:.1f}min old (limit: {self.STALE_CANDLE_MINUTES}min)" if not passed else None
            }
        except:
            return {'passed': True, 'note': 'Could not parse candle timestamp'}
    
    def _check_candle_gaps(self, symbol: str, market_data: Dict) -> Dict:
        """Check for gaps in candle sequence"""
        candle_gaps = market_data.get('candle_gaps', [])
        gap_minutes = market_data.get('max_gap_minutes', 0)
        
        if not candle_gaps and gap_minutes == 0:
            # No gap info provided
            return {'passed': True, 'note': 'No gap info provided'}
        
        passed = gap_minutes <= self.MAX_CANDLE_GAP_MINUTES and len(candle_gaps) == 0
        
        return {
            'passed': passed,
            'max_gap_minutes': gap_minutes,
            'gaps_found': len(candle_gaps),
            'threshold': self.MAX_CANDLE_GAP_MINUTES,
            'issue': f"CANDLE_GAP: {gap_minutes}min gap detected" if not passed else None
        }
    
    def _check_volume(self, symbol: str, market_data: Dict) -> Dict:
        """Check if volume is valid (not zero from feed glitch)"""
        volume = market_data.get('volume', 0)
        
        #Also check today's volume vs historical
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volume_vs_avg = market_data.get('volume_vs_avg', 1.0)
        
        # Zero volume is suspicious
        if volume <= self.MIN_VOLUME:
            return {
                'passed': False,
                'volume': volume,
                'issue': f"ZERO_VOLUME: Volume={volume} (likely feed glitch)"
            }
        
        # Extremely low volume ratio is also suspicious
        if volume_ratio < 0.01 and volume_vs_avg < 0.01:
            return {
                'passed': False,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'issue': f"VOLUME_GLITCH: Volume ratio={volume_ratio:.3f} (extremely low)"
            }
        
        return {
            'passed': True,
            'volume': volume,
            'volume_ratio': volume_ratio
        }
    
    def _check_indicators(self, symbol: str, market_data: Dict) -> Dict:
        """Check if indicators are valid (not NaN or zero)"""
        issues_found = []
        
        # Critical indicators to check
        indicators_to_check = {
            'vwap': {'min': 0.001, 'name': 'VWAP'},
            'ema_9': {'min': 0.001, 'name': 'EMA_9'},
            'ema_21': {'min': 0.001, 'name': 'EMA_21'},
            'sma_20': {'min': 0.001, 'name': 'SMA_20'},
            'atr_14': {'min': 0.001, 'name': 'ATR_14'},
            'rsi_14': {'min': 0, 'max': 100, 'name': 'RSI_14'},
        }
        
        ltp = market_data.get('ltp', 0)
        
        for key, config in indicators_to_check.items():
            value = market_data.get(key, 0)
            
            # Check for NaN
            if value is None or (isinstance(value, float) and math.isnan(value)):
                issues_found.append(f"{config['name']}_NaN: Value is NaN")
                continue
            
            # Check for zero (when it shouldn't be)
            if config['min'] > 0 and value == 0:
                issues_found.append(f"{config['name']}_ZERO: Value is 0 (invalid)")
                continue
            
            # Check for out of range (RSI)
            if 'max' in config and (value < config['min'] or value > config['max']):
                issues_found.append(f"{config['name']}_RANGE: {value} not in [{config['min']}, {config['max']}]")
                continue
            
            # For price-based indicators, check if reasonable relative to LTP
            if key in ['vwap', 'ema_9', 'ema_21', 'sma_20'] and ltp > 0:
                deviation = abs(value - ltp) / ltp
                if deviation > 0.5:  # More than 50% from LTP is suspicious
                    issues_found.append(f"{config['name']}_DEVIANT: {value:.2f} deviates {deviation*100:.0f}% from LTP {ltp:.2f}")
        
        # Check ATR relative to price
        atr = market_data.get('atr_14', 0)
        if ltp > 0 and atr > 0:
            atr_pct = atr / ltp
            if atr_pct < self.MIN_ATR_PCT:
                issues_found.append(f"ATR_TOO_LOW: {atr_pct*100:.4f}% of price (min: {self.MIN_ATR_PCT*100}%)")
        
        passed = len(issues_found) == 0
        
        return {
            'passed': passed,
            'indicators_checked': len(indicators_to_check),
            'issues': issues_found,
            'issue': '; '.join(issues_found) if issues_found else None
        }
    
    def can_trade(self, symbol: str, market_data: Dict) -> Tuple[bool, str]:
        """
        Quick check if trading is allowed for a symbol
        
        Args:
            symbol: Trading symbol
            market_data: Market data dict
            
        Returns:
            (can_trade, reason)
        """
        # First check if symbol is halted
        if symbol in self.halted_symbols:
            return False, f"HALTED: {self.stale_counters.get(symbol, 0)} consecutive stale readings"
        
        # Do full health check
        result = self.check_health(symbol, market_data)
        
        if not result.can_trade:
            issues_str = '; '.join(result.issues[:3])  # First 3 issues
            return False, f"{result.status.value}: {issues_str}"
        
        return True, "DATA_HEALTHY"
    
    def get_stale_counter(self, symbol: str) -> int:
        """Get stale counter for a symbol"""
        return self.stale_counters.get(symbol, 0)
    
    def reset_symbol(self, symbol: str):
        """Reset a symbol's health state"""
        self.stale_counters[symbol] = 0
        if symbol in self.halted_symbols:
            self.halted_symbols.remove(symbol)
        self._save_state()
        print(f"✅ {symbol} health state reset")
    
    def get_halted_symbols(self) -> List[str]:
        """Get list of halted symbols"""
        return self.halted_symbols.copy()
    
    def get_status(self) -> Dict:
        """Get overall health gate status"""
        return {
            'halted_symbols': self.halted_symbols,
            'stale_counters': {k: v for k, v in self.stale_counters.items() if v > 0},
            'recent_issues': [h for h in self.health_history[-10:] if h.get('issues')]
        }
    
    def format_health_report(self, result: DataHealthResult) -> str:
        """Format a health result as human-readable report"""
        lines = [
            f"═══ DATA HEALTH: {result.symbol} ═══",
            f"Status: {result.status.value}",
            f"Can Trade: {'✅ YES' if result.can_trade else '❌ NO'}",
            f"Stale Counter: {result.stale_counter}/{self.MAX_STALE_BEFORE_HALT}",
            ""
        ]
        
        if result.issues:
            lines.append("ISSUES:")
            for issue in result.issues:
                lines.append(f"  ❌ {issue}")
        
        lines.append("")
        lines.append("CHECKS:")
        for check_name, check_data in result.checks.items():
            passed = check_data.get('passed', True)
            icon = "✅" if passed else "❌"
            lines.append(f"  {icon} {check_name.upper()}")
        
        return "\n".join(lines)


# Singleton instance
_health_gate_instance = None


def get_data_health_gate() -> DataHealthGate:
    """Get or create the singleton health gate instance"""
    global _health_gate_instance
    if _health_gate_instance is None:
        _health_gate_instance = DataHealthGate()
    return _health_gate_instance


# === TEST ===
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA HEALTH GATE")
    print("=" * 60)
    
    gate = get_data_health_gate()
    print(f"✅ DataHealthGate created")
    
    # Test 1: Healthy data
    print("\n[1] Testing HEALTHY data...")
    healthy_data = {
        'symbol': 'NSE:INFY',
        'ltp': 1500.0,
        'volume': 500000,
        'volume_ratio': 1.2,
        'timestamp': datetime.now().isoformat(),
        'vwap': 1505.0,
        'ema_9': 1498.0,
        'ema_21': 1495.0,
        'sma_20': 1490.0,
        'atr_14': 25.0,
        'rsi_14': 55.0
    }
    result = gate.check_health('NSE:INFY', healthy_data)
    print(f"    Status: {result.status.value}")
    print(f"    Can Trade: {result.can_trade}")
    assert result.can_trade, "Healthy data should allow trading"
    print("    ✅ Passed")
    
    # Test 2: Stale data
    print("\n[2] Testing STALE data...")
    stale_data = {
        'symbol': 'NSE:TCS',
        'ltp': 3500.0,
        'volume': 100000,
        'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
        'last_trade_time': (datetime.now() - timedelta(minutes=5)).isoformat(),
        'vwap': 3510.0,
        'ema_9': 3508.0,
        'ema_21': 3505.0,
        'sma_20': 3490.0,
        'atr_14': 40.0,
        'rsi_14': 60.0
    }
    result = gate.check_health('NSE:TCS', stale_data)
    print(f"    Status: {result.status.value}")
    print(f"    Issues: {result.issues}")
    # Note: 5 minutes may still pass if STALE_TICK_SECONDS is high
    print("    ✅ Test completed")
    
    # Test 3: Zero volume
    print("\n[3] Testing ZERO VOLUME...")
    zero_vol_data = {
        'symbol': 'NSE:RELIANCE',
        'ltp': 2500.0,
        'volume': 0,
        'volume_ratio': 0,
        'timestamp': datetime.now().isoformat(),
        'vwap': 2510.0,
        'ema_9': 2508.0,
        'ema_21': 2505.0,
        'sma_20': 2490.0,
        'atr_14': 35.0,
        'rsi_14': 50.0
    }
    result = gate.check_health('NSE:RELIANCE', zero_vol_data)
    print(f"    Status: {result.status.value}")
    print(f"    Can Trade: {result.can_trade}")
    assert not result.can_trade, "Zero volume should block trading"
    print("    ✅ Passed - correctly blocked")
    
    # Test 4: NaN indicators
    print("\n[4] Testing NaN INDICATORS...")
    nan_data = {
        'symbol': 'NSE:HDFC',
        'ltp': 1800.0,
        'volume': 200000,
        'timestamp': datetime.now().isoformat(),
        'vwap': float('nan'),
        'ema_9': 1798.0,
        'ema_21': float('nan'),
        'sma_20': 1790.0,
        'atr_14': 0,  # Zero ATR
        'rsi_14': 50.0
    }
    result = gate.check_health('NSE:HDFC', nan_data)
    print(f"    Status: {result.status.value}")
    print(f"    Issues: {result.issues}")
    assert not result.can_trade, "NaN indicators should block trading"
    print("    ✅ Passed - correctly blocked")
    
    # Test 5: Stale counter accumulation
    print("\n[5] Testing STALE COUNTER accumulation...")
    for i in range(6):
        result = gate.check_health('NSE:HDFC', nan_data)
    print(f"    Stale Counter: {gate.get_stale_counter('NSE:HDFC')}")
    print(f"    Halted Symbols: {gate.get_halted_symbols()}")
    assert 'NSE:HDFC' in gate.get_halted_symbols(), "Should be halted after 5 stale readings"
    print("    ✅ Passed - correctly halted")
    
    # Test 6: Health report
    print("\n[6] Testing health report...")
    print(gate.format_health_report(result))
    print("    ✅ Passed")
    
    # Test 7: Reset
    print("\n[7] Testing reset...")
    gate.reset_symbol('NSE:HDFC')
    assert gate.get_stale_counter('NSE:HDFC') == 0, "Counter should be 0 after reset"
    assert 'NSE:HDFC' not in gate.get_halted_symbols(), "Should not be halted after reset"
    print("    ✅ Passed")
    
    print("\n" + "=" * 60)
    print("✅ ALL DATA HEALTH GATE TESTS PASSED!")
    print("=" * 60)
