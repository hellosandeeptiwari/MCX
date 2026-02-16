"""
OPTIONS FLOW ANALYZER — Real-Time Directional Bias from Options Data

Standalone module. Titan runs perfectly without it.
Uses Kite's live options chain to provide directional signals
that OHLCV technical indicators cannot provide.

Signals:
  - Put-Call Ratio (PCR by OI) → Market sentiment
  - IV Skew (Put IV vs Call IV) → Smart money positioning
  - OI Concentration → Support/resistance levels from option writers
  - Max Pain → Price magnet from option writer pain

FAIL-SAFE DESIGN:
  - Returns NEUTRAL on ANY error
  - Never blocks trading
  - If Kite API fails, returns neutral → Titan continues unchanged
  - Aggressive caching (2-min TTL) to minimize API calls
  - Only analyze top-scoring stocks (not all 80+)

Usage:
    from options_flow_analyzer import get_options_flow_analyzer
    
    analyzer = get_options_flow_analyzer(chain_fetcher)
    result = analyzer.analyze("NSE:SBIN")
    # result = {
    #   'flow_bias': 'BULLISH',
    #   'flow_confidence': 0.65,
    #   'pcr_oi': 1.35,
    #   'iv_skew': -2.1,
    #   'max_pain': 780.0,
    #   'flow_score_boost': 2,
    #   'flow_gpt_line': '📊OI:BULLISH(PCR:1.35,IVskew:-2.1%)',
    # }
"""

from datetime import datetime
from typing import Dict, Optional


class OptionsFlowAnalyzer:
    """Real-time options flow analysis for directional bias.
    
    FAIL-SAFE: Returns NEUTRAL on ANY error. Never blocks trading.
    If chain_fetcher is None, analyzer is permanently disabled.
    
    Now with OPTIONAL NSE enrichment:
    - NSE provides OI CHANGE per strike (Kite doesn't)
    - OI buildup detection (LONG_BUILDUP, SHORT_BUILDUP, etc.)
    - Volume PCR, OI change PCR
    """
    
    _CACHE: Dict[str, tuple] = {}
    _CACHE_TTL = 120  # 2 minutes
    
    def __init__(self, chain_fetcher=None):
        """
        Args:
            chain_fetcher: OptionChainFetcher instance (from options_trader.py).
                          If None, analyzer is disabled (returns NEUTRAL always).
        """
        self.chain_fetcher = chain_fetcher
        self.ready = chain_fetcher is not None
        
        # DhanHQ OI fetcher (primary enrichment — has Greeks, bid/ask)
        self._dhan_fetcher = None
        self._dhan_ready = False
        try:
            from dhan_oi_fetcher import get_dhan_oi_fetcher
            self._dhan_fetcher = get_dhan_oi_fetcher()
            if self._dhan_fetcher.ready:
                self._dhan_ready = True
        except Exception:
            pass  # DhanHQ enrichment is optional
        
        # NSE OI fetcher (fallback enrichment layer)
        self._nse_fetcher = None
        self._nse_ready = False
        try:
            from nse_oi_fetcher import get_nse_oi_fetcher
            self._nse_fetcher = get_nse_oi_fetcher()
            self._nse_ready = True
        except Exception:
            pass  # NSE enrichment is optional
    
    def analyze(self, underlying: str) -> dict:
        """Analyze options flow for directional bias.
        
        FAIL-SAFE: Returns NEUTRAL on any error. Never blocks.
        
        Args:
            underlying: e.g., "NSE:RELIANCE"
            
        Returns:
            dict with:
            - flow_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            - flow_confidence: 0.0 to 1.0
            - pcr_oi: put-call ratio by OI
            - iv_skew: ATM put IV - call IV (positive = bearish fear)
            - max_pain: max pain strike price
            - call_resistance: highest call OI strike (resistance)
            - put_support: highest put OI strike (support)
            - spot_price: current spot
            - flow_score_boost: -4 to +4 points for scoring
            - flow_gpt_line: one-line summary for GPT
        """
        try:
            if not self.ready:
                return self._neutral()
            
            # Check cache
            cache_key = underlying
            if cache_key in self._CACHE:
                cached_time, cached_result = self._CACHE[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self._CACHE_TTL:
                    return cached_result
            
            # Fetch chain (uses chain_fetcher's own 60s cache)
            chain = self.chain_fetcher.fetch_option_chain(underlying)
            if not chain or not chain.contracts:
                return self._neutral()
            
            # === Compute PCR by OI ===
            total_call_oi = sum(c.oi for c in chain.contracts if c.option_type.value == 'CE')
            total_put_oi = sum(c.oi for c in chain.contracts if c.option_type.value == 'PE')
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            
            # === Compute IV Skew (ATM) ===
            atm_strike = chain.get_atm_strike()
            iv_skew = 0.0
            if atm_strike:
                atm_ce = chain.get_contract(atm_strike, _get_option_type('CE'))
                atm_pe = chain.get_contract(atm_strike, _get_option_type('PE'))
                if atm_ce and atm_pe and atm_ce.iv > 0 and atm_pe.iv > 0:
                    iv_skew = (atm_pe.iv - atm_ce.iv) * 100  # positive = puts more expensive = bearish
            
            # === Compute Max Pain ===
            max_pain = self._calc_max_pain(chain)
            
            # === OI Concentration (resistance/support) ===
            call_strikes = [(c.strike, c.oi) for c in chain.contracts if c.option_type.value == 'CE' and c.oi > 0]
            put_strikes = [(c.strike, c.oi) for c in chain.contracts if c.option_type.value == 'PE' and c.oi > 0]
            
            call_resistance = max(call_strikes, key=lambda x: x[1], default=(0, 0))[0] if call_strikes else 0
            put_support = max(put_strikes, key=lambda x: x[1], default=(0, 0))[0] if put_strikes else 0
            
            # === Determine Bias ===
            bias = 'NEUTRAL'
            confidence = 0.5
            boost = 0
            
            # PCR signal (contrarian interpretation):
            # High PCR (>1.2) = too many puts = potential bullish (short squeeze)
            # Low PCR (<0.7) = too many calls = potential bearish (overconfidence)
            if pcr_oi >= 1.5:
                bias = 'BULLISH'
                confidence = min(0.8, 0.5 + (pcr_oi - 1.0) * 0.15)
                boost = 3
            elif pcr_oi >= 1.2:
                bias = 'BULLISH'
                confidence = 0.6
                boost = 2
            elif pcr_oi <= 0.5:
                bias = 'BEARISH'
                confidence = min(0.8, 0.5 + (1.0 - pcr_oi) * 0.15)
                boost = -3
            elif pcr_oi <= 0.7:
                bias = 'BEARISH'
                confidence = 0.6
                boost = -2
            
            # IV skew adjustment
            if iv_skew > 5:  # Put IV much higher = bearish fear
                if bias == 'NEUTRAL':
                    bias = 'BEARISH'
                confidence = min(0.9, confidence + 0.1)
                boost -= 1
            elif iv_skew < -5:  # Call IV higher = bullish greed
                if bias == 'NEUTRAL':
                    bias = 'BULLISH'
                confidence = min(0.9, confidence + 0.1)
                boost += 1
            
            # Clamp boost
            boost = max(-4, min(4, boost))
            
            # === Spot vs Max Pain context ===
            spot = chain.spot_price
            mp_distance_pct = ((spot - max_pain) / max_pain * 100) if max_pain > 0 else 0
            
            # GPT summary
            gpt_parts = [f"PCR:{pcr_oi:.2f}"]
            if abs(iv_skew) > 2:
                gpt_parts.append(f"IVskew:{iv_skew:+.1f}%")
            if max_pain > 0:
                gpt_parts.append(f"MaxPain:{max_pain:.0f}")
            gpt_line = f"📊OI:{bias}({','.join(gpt_parts)})" if bias != 'NEUTRAL' else f"📊OI:NEUTRAL({','.join(gpt_parts)})"
            
            result = {
                'flow_bias': bias,
                'flow_confidence': round(confidence, 3),
                'pcr_oi': round(pcr_oi, 3),
                'iv_skew': round(iv_skew, 2),
                'max_pain': max_pain,
                'call_resistance': call_resistance,
                'put_support': put_support,
                'spot_price': spot,
                'spot_vs_max_pain_pct': round(mp_distance_pct, 2),
                'flow_score_boost': boost,
                'flow_gpt_line': gpt_line,
            }
            
            # === OI ENRICHMENT: DhanHQ (primary) → NSE (fallback) ===
            result = self._enrich_with_dhan(underlying, result)
            if not result.get('nse_enriched'):
                result = self._enrich_with_nse(underlying, result)
            
            # Cache
            self._CACHE[cache_key] = (datetime.now(), result)
            return result
            
        except Exception:
            return self._neutral()
    
    def _enrich_with_dhan(self, underlying: str, result: dict) -> dict:
        """Enrich Kite-based result with DhanHQ OI + Greeks data.
        
        FAIL-SAFE: Returns result unchanged on any error.
        DhanHQ adds: OI change, buildup signal, full Greeks, bid/ask.
        Takes priority over NSE (richer data, more reliable).
        """
        try:
            if not self._dhan_ready or not self._dhan_fetcher:
                return result
            
            dhan_data = self._dhan_fetcher.fetch(underlying)
            if not dhan_data:
                return result
            
            # Add OI change fields (same keys as NSE for compatibility)
            result['nse_total_call_oi_change'] = dhan_data.get('total_call_oi_change', 0)
            result['nse_total_put_oi_change'] = dhan_data.get('total_put_oi_change', 0)
            result['nse_pcr_volume'] = dhan_data.get('pcr_volume', 1.0)
            result['nse_pcr_oi_change'] = dhan_data.get('pcr_oi_change', 0.0)
            result['nse_oi_buildup'] = dhan_data.get('oi_buildup_signal', 'NEUTRAL')
            result['nse_oi_buildup_strength'] = dhan_data.get('oi_buildup_strength', 0.0)
            result['nse_top_call_oi_change'] = dhan_data.get('top_call_oi_change_strikes', [])[:3]
            result['nse_top_put_oi_change'] = dhan_data.get('top_put_oi_change_strikes', [])[:3]
            result['nse_enriched'] = True
            result['oi_source'] = 'DHAN'
            
            # DhanHQ exclusive: ATM Greeks
            atm_g = dhan_data.get('atm_greeks', {})
            if atm_g:
                result['atm_greeks'] = atm_g
            
            # Upgrade bias if DhanHQ buildup is strong and Kite was NEUTRAL
            buildup = dhan_data.get('oi_buildup_signal', 'NEUTRAL')
            strength = dhan_data.get('oi_buildup_strength', 0.0)
            
            if result['flow_bias'] == 'NEUTRAL' and buildup != 'NEUTRAL' and strength >= 0.4:
                if buildup in ('LONG_BUILDUP', 'SHORT_COVERING'):
                    result['flow_bias'] = 'BULLISH'
                    result['flow_confidence'] = max(result['flow_confidence'], 0.55 + strength * 0.15)
                    result['flow_score_boost'] = max(result['flow_score_boost'], min(3, int(strength * 4)))
                elif buildup in ('SHORT_BUILDUP', 'LONG_UNWINDING'):
                    result['flow_bias'] = 'BEARISH'
                    result['flow_confidence'] = max(result['flow_confidence'], 0.55 + strength * 0.15)
                    result['flow_score_boost'] = min(result['flow_score_boost'], max(-3, -int(strength * 4)))
            
            # Upgrade GPT line with DhanHQ data
            if buildup != 'NEUTRAL':
                result['flow_gpt_line'] += f"|Dhan:{buildup}({strength:.0%})"
            
            ce_chg = dhan_data.get('total_call_oi_change', 0)
            pe_chg = dhan_data.get('total_put_oi_change', 0)
            if ce_chg != 0 or pe_chg != 0:
                result['flow_gpt_line'] += f"|ΔOI:CE{ce_chg:+,}/PE{pe_chg:+,}"
            
            # Add ATM Greeks to GPT line
            if atm_g.get('ce_delta'):
                result['flow_gpt_line'] += f"|δ:{atm_g['ce_delta']:.2f}"
            
            return result
            
        except Exception:
            return result
    
    def _enrich_with_nse(self, underlying: str, result: dict) -> dict:
        """Enrich Kite-based result with NSE OI change data (FALLBACK).
        
        FAIL-SAFE: Returns result unchanged on any error.
        Only called if DhanHQ enrichment didn't happen.
        NSE adds: OI change, buildup signal, volume PCR — fields Kite lacks.
        """
        try:
            if not self._nse_ready or not self._nse_fetcher:
                return result
            
            nse_data = self._nse_fetcher.fetch(underlying)
            if not nse_data:
                return result
            
            # Add NSE-exclusive fields
            result['nse_total_call_oi_change'] = nse_data.get('total_call_oi_change', 0)
            result['nse_total_put_oi_change'] = nse_data.get('total_put_oi_change', 0)
            result['nse_pcr_volume'] = nse_data.get('pcr_volume', 1.0)
            result['nse_pcr_oi_change'] = nse_data.get('pcr_oi_change', 0.0)
            result['nse_oi_buildup'] = nse_data.get('oi_buildup_signal', 'NEUTRAL')
            result['nse_oi_buildup_strength'] = nse_data.get('oi_buildup_strength', 0.0)
            result['nse_top_call_oi_change'] = nse_data.get('top_call_oi_change_strikes', [])[:3]
            result['nse_top_put_oi_change'] = nse_data.get('top_put_oi_change_strikes', [])[:3]
            result['nse_enriched'] = True
            
            # Upgrade bias if NSE buildup is strong and Kite was NEUTRAL
            buildup = nse_data.get('oi_buildup_signal', 'NEUTRAL')
            strength = nse_data.get('oi_buildup_strength', 0.0)
            
            if result['flow_bias'] == 'NEUTRAL' and buildup != 'NEUTRAL' and strength >= 0.4:
                if buildup in ('LONG_BUILDUP', 'SHORT_COVERING'):
                    result['flow_bias'] = 'BULLISH'
                    result['flow_confidence'] = max(result['flow_confidence'], 0.55 + strength * 0.15)
                    result['flow_score_boost'] = max(result['flow_score_boost'], min(3, int(strength * 4)))
                elif buildup in ('SHORT_BUILDUP', 'LONG_UNWINDING'):
                    result['flow_bias'] = 'BEARISH'
                    result['flow_confidence'] = max(result['flow_confidence'], 0.55 + strength * 0.15)
                    result['flow_score_boost'] = min(result['flow_score_boost'], max(-3, -int(strength * 4)))
            
            # Upgrade GPT line with NSE buildup info
            if buildup != 'NEUTRAL':
                result['flow_gpt_line'] += f"|Buildup:{buildup}({strength:.0%})"
            
            ce_chg = nse_data.get('total_call_oi_change', 0)
            pe_chg = nse_data.get('total_put_oi_change', 0)
            if ce_chg != 0 or pe_chg != 0:
                result['flow_gpt_line'] += f"|ΔOI:CE{ce_chg:+,}/PE{pe_chg:+,}"
            
            return result
            
        except Exception:
            return result
    
    def get_nse_snapshot(self, underlying: str) -> dict:
        """Get NSE OI snapshot for logging (richer than Kite-based snapshots).
        
        FAIL-SAFE: Returns empty dict on error.
        """
        try:
            if not self._nse_ready or not self._nse_fetcher:
                return {}
            return self._nse_fetcher.get_snapshot_for_logging(underlying)
        except Exception:
            return {}
    
    def get_dhan_snapshot(self, underlying: str) -> dict:
        """Get DhanHQ OI snapshot for logging (richest: has Greeks + bid/ask).
        
        FAIL-SAFE: Returns empty dict on error.
        """
        try:
            if not self._dhan_ready or not self._dhan_fetcher:
                return {}
            return self._dhan_fetcher.get_snapshot_for_logging(underlying)
        except Exception:
            return {}
    
    def analyze_batch(self, symbols: list, max_symbols: int = 15) -> dict:
        """Analyze multiple symbols. Only analyzes top N to limit API calls.
        
        FAIL-SAFE: Returns empty dict on error.
        
        Args:
            symbols: List of underlying symbols (e.g., ["NSE:SBIN", ...])
            max_symbols: Max symbols to analyze (limits API calls)
            
        Returns:
            {symbol: analysis_dict, ...}
        """
        results = {}
        try:
            for sym in symbols[:max_symbols]:
                r = self.analyze(sym)
                if r.get('flow_bias') != 'NEUTRAL' or r.get('pcr_oi', 1.0) != 1.0:
                    results[sym] = r
        except Exception:
            pass
        return results
    
    def _calc_max_pain(self, chain) -> float:
        """Calculate max pain strike (price where option writers have minimum payout)."""
        try:
            strikes = sorted(set(c.strike for c in chain.contracts))
            if not strikes:
                return 0.0
            
            min_pain = float('inf')
            max_pain_strike = 0.0
            
            for test_strike in strikes:
                total_pain = 0
                for c in chain.contracts:
                    if c.option_type.value == 'CE':
                        # Call buyers profit if price > strike
                        if test_strike > c.strike:
                            total_pain += (test_strike - c.strike) * c.oi
                    else:
                        # Put buyers profit if price < strike
                        if test_strike < c.strike:
                            total_pain += (c.strike - test_strike) * c.oi
                
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = test_strike
            
            return max_pain_strike
        except Exception:
            return 0.0
    
    def _neutral(self) -> dict:
        """Safe neutral defaults — zero effect on Titan."""
        return {
            'flow_bias': 'NEUTRAL',
            'flow_confidence': 0.0,
            'pcr_oi': 1.0,
            'iv_skew': 0.0,
            'max_pain': 0.0,
            'call_resistance': 0,
            'put_support': 0,
            'spot_price': 0.0,
            'spot_vs_max_pain_pct': 0.0,
            'flow_score_boost': 0,
            'flow_gpt_line': '',
        }


def _get_option_type(type_str: str):
    """Safely import OptionType enum. Returns None on failure."""
    try:
        from options_trader import OptionType
        return OptionType[type_str]
    except Exception:
        return None


# === Singleton (lazy loaded) ===
_analyzer_instance = None


def get_options_flow_analyzer(chain_fetcher=None) -> Optional[OptionsFlowAnalyzer]:
    """Get or create singleton analyzer.
    
    Call with chain_fetcher on first use. Subsequent calls return cached instance.
    If chain_fetcher is never provided, returns None.
    """
    global _analyzer_instance
    if _analyzer_instance is None and chain_fetcher is not None:
        _analyzer_instance = OptionsFlowAnalyzer(chain_fetcher)
    return _analyzer_instance


if __name__ == '__main__':
    print("Options Flow Analyzer — standalone module")
    print("Usage: from options_flow_analyzer import get_options_flow_analyzer")
    print("analyzer = get_options_flow_analyzer(chain_fetcher)")
    print("result = analyzer.analyze('NSE:SBIN')")
    print("\nFAIL-SAFE: Returns NEUTRAL on any error. Never blocks Titan.")
