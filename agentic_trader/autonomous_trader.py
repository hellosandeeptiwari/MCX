"""
AUTONOMOUS TRADING BOT
Fully autonomous - makes decisions and executes without approval

⚠️ WARNING: This bot trades REAL MONEY automatically!
- It will place orders without asking
- It can lose your entire capital
- Past performance doesn't guarantee future results

USE AT YOUR OWN RISK!
"""

import time
import json
import threading
from datetime import datetime, timedelta
import schedule
import sys
import os
from safe_io import atomic_json_save

# Fix Windows cp1252 encoding for emoji output
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure CWD is the script's directory so relative file paths (risk_state.json, active_trades.json) resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from config import HARD_RULES, APPROVED_UNIVERSE, TRADING_HOURS, FNO_CONFIG, TIER_1_OPTIONS, TIER_2_OPTIONS, FULL_FNO_SCAN, calc_brokerage
from llm_agent import TradingAgent
from zerodha_tools import get_tools, reset_tools
from market_scanner import get_market_scanner
from options_trader import update_fno_lot_sizes
from exit_manager import get_exit_manager, calculate_structure_sl
from execution_guard import get_execution_guard
from risk_governor import get_risk_governor, SystemState
from correlation_guard import get_correlation_guard
from regime_score import get_regime_scorer
from position_reconciliation import get_position_reconciliation
from data_health_gate import get_data_health_gate
from trade_ledger import get_trade_ledger


class AutonomousTrader:
    """
    Fully autonomous trading bot that:
    1. Scans market every 5 minutes
    2. Identifies opportunities using GPT
    3. Executes trades automatically
    4. Manages risk and exits
    """
    
    def __init__(self, capital: float = 10000, paper_mode: bool = True):
        """
        Args:
            capital: Starting capital
            paper_mode: If True, simulates trades (no real orders)
        """
        self.capital = capital
        self.paper_mode = paper_mode
        self.start_capital = capital
        self.daily_pnl = 0
        
        # Set module-level PAPER_MODE in config for brokerage calculation
        import config as _cfg
        _cfg.PAPER_MODE = paper_mode
        self._pnl_lock = threading.Lock()  # Thread-safe P&L updates
        self.trades_today = []
        self.positions = []
        
        # Initialize agent with auto_execute=True
        print("\n" + "="*60)
        print("🤖 AUTONOMOUS TRADING BOT")
        print("="*60)
        print(f"\n  Capital: ₹{capital:,}")
        print(f"  Mode: {'📝 PAPER TRADING' if paper_mode else '💰 LIVE TRADING'}")
        print(f"  Risk per trade: {HARD_RULES['RISK_PER_TRADE']*100}%")
        print(f"  Max daily loss: {HARD_RULES['MAX_DAILY_LOSS']*100}%")
        print(f"  Max positions: {HARD_RULES['MAX_POSITIONS']}")
        print(f"\n  Universe: {len(APPROVED_UNIVERSE)} stocks ({len(TIER_1_OPTIONS)} Tier-1 + {len(TIER_2_OPTIONS)} Tier-2)")
        print(f"  Scanner: ALL F&O stocks (~200) scanned each cycle for wild-card movers")
        
        if not paper_mode:
            print("\n  ⚠️  LIVE MODE - Real orders will be placed!")
            confirm = input("  Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("  Aborted.")
                sys.exit(0)
        
        # Reset tools singleton to use new configuration
        reset_tools()
        # Auto-execute is ON for both paper and live trading
        self.agent = TradingAgent(auto_execute=True, paper_mode=paper_mode, paper_capital=capital)
        self.tools = get_tools(paper_mode=paper_mode, paper_capital=capital)
        
        # === LIVE MODE: Validate token before proceeding ===
        if not paper_mode:
            try:
                profile = self.tools.kite.profile()
                print(f"  ✅ Kite token valid — logged in as: {profile.get('user_name', 'Unknown')}")
                print(f"  💰 Live capital: ₹{self.tools.paper_capital:,.0f}")
            except Exception as e:
                print(f"\n  🚨 CRITICAL: Kite access token is invalid/expired!")
                print(f"     Error: {e}")
                print(f"     Run 'python agentic_trader/quick_auth.py' to refresh the token.")
                print(f"     Cannot start LIVE mode without a valid token.")
                sys.exit(1)
        
        # === RESTORE daily P&L from persisted realized P&L ===
        # On restart, zerodha_tools loads paper_pnl from active_trades.json.
        # Sync self.daily_pnl and self.capital so the display is correct.
        persisted_pnl = getattr(self.tools, 'paper_pnl', 0) or 0
        if persisted_pnl != 0:
            with self._pnl_lock:
                self.daily_pnl = persisted_pnl
                self.capital = capital + persisted_pnl
            print(f"  📊 Restored daily P&L: ₹{persisted_pnl:+,.0f} | Capital: ₹{self.capital:,.0f}")
        
        # Real-time monitoring
        self.monitor_running = False
        self.monitor_thread = None
        self.monitor_interval = 3  # Check every 3 seconds
        
        # ORB trade tracking - once per direction per symbol per day
        # Persisted to disk so mid-day restarts don't allow duplicate ORB entries
        self._orb_state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'orb_trades_state.json')
        self.orb_trades_today, self.orb_tracking_date = self._load_orb_state()
        
        # Exit Manager for consistent exits
        self.exit_manager = get_exit_manager()
        
        # Execution Guard for order quality
        self.execution_guard = get_execution_guard()
        
        # Risk Governor for account-level risk
        self.risk_governor = get_risk_governor(capital)
        
        # === DhanHQ Daily Safety Nets ===
        self._dhan_risk = None
        try:
            from dhan_risk_tools import get_dhan_risk_tools
            self._dhan_risk = get_dhan_risk_tools()
            if self._dhan_risk.ready:
                max_loss = capital * (self.risk_governor.limits.max_daily_loss_pct / 100)  # 6% = ₹30K on ₹5L
                max_profit = max_loss * 3  # ₹90K profit cap
                safety = self._dhan_risk.setup_daily_safety(
                    max_loss=max_loss,
                    max_profit=max_profit,
                    starting_capital=capital,
                )
                print(f"  🛡️ DhanHQ Safety: loss_cap=₹{max_loss:,.0f} profit_cap=₹{max_profit:,.0f}")
                print(f"     Kill Switch: {safety.get('kill_switch', 'N/A')}")
                pnl_exit = safety.get('pnl_exit', {})
                if pnl_exit.get('success'):
                    print(f"     P&L Auto-Exit: ACTIVE")
                else:
                    print(f"     P&L Auto-Exit: {pnl_exit.get('message', 'failed')}")
            else:
                print("  🛡️ DhanHQ Safety: DISABLED (not configured)")
        except Exception as e:
            print(f"  🛡️ DhanHQ Safety: DISABLED ({e})")
        
        # Correlation Guard for hidden overexposure
        self.correlation_guard = get_correlation_guard()
        
        # Regime Scorer for trade quality confidence
        self.regime_scorer = get_regime_scorer()
        
        # Position Reconciliation for broker sync (critical for live)
        self.position_recon = get_position_reconciliation(
            kite=self.tools.kite if hasattr(self.tools, 'kite') else None,
            paper_mode=paper_mode,
            check_interval=10
        )
        
        # Data Health Gate for data quality validation
        self.data_health_gate = get_data_health_gate()
        
        # Market Scanner for dynamic F&O stock discovery
        self.market_scanner = get_market_scanner(kite=self.tools.kite)
        self._wildcard_symbols = []  # Wild-card symbols from last scan
        self._wildcard_scores = {}   # Scanner scores for gate bypass
        self._wildcard_change = {}   # % change for momentum detection
        
        # Pre-populate lot sizes from Kite API at startup
        try:
            update_fno_lot_sizes(self.market_scanner.get_lot_map())
        except Exception as e:
            print(f"⚠️ Dynamic lot size fetch failed at startup (will retry on scan): {e}")
        
        # Start reconciliation loop
        self.position_recon.start()
        
        # Sync exit manager with existing positions (crash recovery)
        self._sync_exit_manager_with_positions()
        
        # === GTT ORPHAN CLEANUP (crash recovery) ===
        # If Titan crashed previously, some GTTs may still be active on Zerodha
        # for positions that were already closed. Clean them up.
        try:
            from config import GTT_CONFIG
            if GTT_CONFIG.get('cleanup_on_startup', True) and not paper_mode:
                self.tools._cleanup_orphaned_gtts()
        except Exception as e:
            print(f"  ⚠️ GTT startup cleanup skipped: {e}")
        
        print("\n  ✅ Bot initialized!")
        print("  🟢 Auto-execution: ON")
        print("  ⚡ Real-time monitoring: ENABLED (every 3 sec)")
        print("  📊 Exit Manager: ACTIVE")
        print("  🛡️ Execution Guard: ACTIVE")
        print("  ⚖️ Risk Governor: ACTIVE")
        print("  🔗 Correlation Guard: ACTIVE")
        print("  📈 Regime Scorer: ACTIVE")
        print("  🔄 Position Reconciliation: ACTIVE (every 10s)")
        print("  🛡️ Data Health Gate: ACTIVE")
        print("  📊 Options Trading: ACTIVE (F&O stocks)")
        print("  🛡️ GTT Safety Net: ACTIVE (server-side SL+target)")
        print("  📦 Autoslice: ACTIVE (freeze qty protection)")
        print("  ⚡ IOC Validity: ACTIVE (spread legs)")
        
        # === NEW: Decision Log + Elite Auto-Fire + Adaptive Scan ===
        from config import DECISION_LOG, ELITE_AUTO_FIRE, ADAPTIVE_SCAN, DYNAMIC_MAX_PICKS, DOWN_RISK_GATING
        self._decision_log_cfg = DECISION_LOG
        self._elite_auto_fire_cfg = ELITE_AUTO_FIRE
        self._adaptive_scan_cfg = ADAPTIVE_SCAN
        self._dynamic_max_picks_cfg = DYNAMIC_MAX_PICKS
        self._auto_fired_this_session = set()   # Symbols auto-fired today (no re-fire)
        self._last_signal_quality = 'normal'     # Track signal quality for adaptive scan
        
        if ELITE_AUTO_FIRE.get('enabled'): print("  🎯 Elite Auto-Fire: ACTIVE (score ≥78 → instant execution)")
        if DYNAMIC_MAX_PICKS.get('enabled'): print(f"  📊 Dynamic Max Picks: ACTIVE (3→5 when signals are hot)")
        if ADAPTIVE_SCAN.get('enabled'): print(f"  ⏱️ Adaptive Scan: ACTIVE ({ADAPTIVE_SCAN['fast_interval_minutes']}min/{ADAPTIVE_SCAN['normal_interval_minutes']}min/{ADAPTIVE_SCAN['slow_interval_minutes']}min)")
        if DECISION_LOG.get('enabled'): print(f"  📝 Decision Log: ACTIVE ({DECISION_LOG['file']})")
        
        # === DOWN-RISK SOFT SCORING: Reward safe / penalise risky ===
        self._down_risk_cfg = DOWN_RISK_GATING
        # Model-tracker state: 3 exclusive model-only trades per day
        self._model_tracker_trades_today = 0
        self._model_tracker_date = datetime.now().date()
        self._model_tracker_symbols = set()
        # GMM Sniper state: 1 high-conviction trade per scan cycle
        from config import GMM_SNIPER
        self._gmm_sniper_cfg = GMM_SNIPER
        self._gmm_sniper_trades_today = 0
        self._gmm_sniper_date = datetime.now().date()
        self._gmm_sniper_symbols = set()
        # GMM Contrarian (DR_FLIP) state
        self._dr_flip_symbols = set()
        self._dr_flip_trades_today = 0
        # ML_OVERRIDE gates: loaded from config.ML_OVERRIDE_GATES
        try:
            from config import ML_OVERRIDE_GATES
            self._ml_ovr_cfg = ML_OVERRIDE_GATES
        except ImportError:
            self._ml_ovr_cfg = {'min_move_prob': 0.58, 'max_dr_score': 0.12,
                                'min_directional_prob': 0.40, 'max_concurrent_open': 3}
        # _gmm_flip_count_today removed — GMM now used as veto/boost, not flip
        # === SNIPER STRATEGIES: OI Unwinding + PCR Extreme ===
        try:
            from config import SNIPER_OI_UNWINDING, SNIPER_PCR_EXTREME
            from sniper_strategies import SniperStrategies
            self._sniper_engine = SniperStrategies(
                oi_unwinding_cfg=SNIPER_OI_UNWINDING,
                pcr_extreme_cfg=SNIPER_PCR_EXTREME,
            )
        except Exception as _snp_init_err:
            print(f"  ⚠️ Sniper Strategies init error: {_snp_init_err}")
            self._sniper_engine = None
        if DOWN_RISK_GATING.get('enabled'):
            print(f"  🛡️ Down-Risk Graduated Scoring: ACTIVE (boost +{DOWN_RISK_GATING.get('clean_boost', 8)} / caution −{DOWN_RISK_GATING.get('mid_risk_penalty', 8)} / block −{DOWN_RISK_GATING.get('high_risk_penalty', 15)})")
            print(f"  📊 Model Tracker: ACTIVE ({DOWN_RISK_GATING.get('model_tracker_trades', 7)} exclusive trades/day for model evaluation)")
        if GMM_SNIPER.get('enabled'):
            print(f"  🎯 GMM Sniper: ACTIVE ({GMM_SNIPER.get('max_sniper_trades_per_day', 5)}/day, {GMM_SNIPER.get('lot_multiplier', 2.0)}x lots, dr<{GMM_SNIPER.get('max_dr_score', 0.10)})")
        if getattr(self, '_sniper_engine', None):
            _se_status = self._sniper_engine.get_status()
            if _se_status['oi_unwinding']['enabled']:
                print(f"  🔫 Sniper-OIUnwinding: ACTIVE ({_se_status['oi_unwinding']['max_per_day']}/day, OI reversal at S/R)")
            if _se_status['pcr_extreme']['enabled']:
                print(f"  🔫 Sniper-PCRExtreme: ACTIVE ({_se_status['pcr_extreme']['max_per_day']}/day, PCR contrarian fade)")
        
        # === ML MOVE PREDICTOR ===
        self._ml_predictor = None
        try:
            from ml_models.predictor import MovePredictor
            self._ml_predictor = MovePredictor()
            if self._ml_predictor.ready:
                print("  🧠 ML Move Predictor: ACTIVE (XGBoost score booster)")
            else:
                print("  🧠 ML Move Predictor: STANDBY (no trained model yet)")
                self._ml_predictor = None
        except Exception as e:
            print(f"  🧠 ML Move Predictor: DISABLED ({e})")
        
        # === ML HISTORICAL 5-MIN CANDLE BASE ===
        # Load stored 5-min parquets so ML always has proper intraday features,
        # even at 9:15-9:30 when live intraday cache has only a few candles.
        self._hist_5min_cache = {}  # symbol (e.g. 'RELIANCE') -> DataFrame
        self._nifty_5min_df = None
        self._nifty_daily_df = None
        try:
            import pandas as _pd_hist
            _ml_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data')
            _5m_dir = os.path.join(_ml_data_dir, 'candles_5min')
            _daily_dir = os.path.join(_ml_data_dir, 'candles_daily')
            if os.path.isdir(_5m_dir):
                for _f in os.listdir(_5m_dir):
                    if _f.endswith('.parquet') and not _f.startswith('SECTOR_'):
                        _sym_name = _f.replace('.parquet', '')
                        try:
                            _df = _pd_hist.read_parquet(os.path.join(_5m_dir, _f))
                            _df['date'] = _pd_hist.to_datetime(_df['date'])
                            if _sym_name == 'NIFTY50':
                                self._nifty_5min_df = _df
                            else:
                                self._hist_5min_cache[_sym_name] = _df
                        except Exception:
                            pass
                print(f"  📊 Historical 5-min candles: {len(self._hist_5min_cache)} stocks + {'✓' if self._nifty_5min_df is not None else '✗'} NIFTY50")
            # Load NIFTY50 daily candles
            _nifty_daily_path = os.path.join(_daily_dir, 'NIFTY50.parquet')
            if os.path.exists(_nifty_daily_path):
                self._nifty_daily_df = _pd_hist.read_parquet(_nifty_daily_path)
                self._nifty_daily_df['date'] = _pd_hist.to_datetime(self._nifty_daily_df['date'])
                print(f"  📊 NIFTY50 daily context: ✓ ({len(self._nifty_daily_df)} days)")
        except Exception as _hist_e:
            print(f"  ⚠ Historical 5-min load: {_hist_e}")
        
        # === AUTO-REFRESH STALE 5-MIN PARQUETS ===
        # If the historical 5-min parquets are >3 calendar days old, incrementally
        # update them using Kite historical data API. This prevents the gap between
        # stored data and live intraday candles from biasing ML features.
        try:
            import pandas as _pd_5m_refresh
            _5m_dir_refresh = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'candles_5min')
            if self._hist_5min_cache:
                # Check staleness from any cached DataFrame
                _sample_sym = next(iter(self._hist_5min_cache))
                _sample_df = self._hist_5min_cache[_sample_sym]
                _last_date = _sample_df['date'].max()
                if hasattr(_last_date, 'tz') and _last_date.tz is not None:
                    _last_date = _last_date.tz_localize(None)
                _now_ts = _pd_5m_refresh.Timestamp.now()
                _gap = (_now_ts - _last_date).days
                if _gap > 3:
                    print(f"  ⚠️ 5-min parquets stale ({_gap}d old: last={_last_date.date()}) — auto-refreshing...")
                    try:
                        from ml_models.data_fetcher import fetch_candles, get_instrument_token
                        _kite = self.tools.kite
                        _refreshed = 0
                        _all_syms = list(self._hist_5min_cache.keys())
                        if self._nifty_5min_df is not None:
                            _all_syms.append('NIFTY50')
                        for _rs in _all_syms:
                            try:
                                _existing = self._hist_5min_cache.get(_rs) if _rs != 'NIFTY50' else self._nifty_5min_df
                                if _existing is None:
                                    continue
                                _ex_last = _existing['date'].max()
                                if hasattr(_ex_last, 'tz') and _ex_last.tz is not None:
                                    _ex_last = _ex_last.tz_localize(None)
                                _days_to_fetch = (_now_ts - _ex_last).days + 1
                                if _days_to_fetch <= 1:
                                    continue
                                # Fetch only the missing days (cap at 30 to be safe)
                                _kite_sym = 'NIFTY 50' if _rs == 'NIFTY50' else _rs
                                _new_df = fetch_candles(_kite, _kite_sym, days=min(_days_to_fetch, 30), interval='5minute')
                                if len(_new_df) > 0:
                                    _new_df['date'] = _pd_5m_refresh.to_datetime(_new_df['date'])
                                    # Merge: keep existing + append new non-overlapping
                                    _ex_copy = _existing.copy()
                                    if _ex_copy['date'].dt.tz is not None:
                                        _ex_copy['date'] = _ex_copy['date'].dt.tz_localize(None)
                                    if _new_df['date'].dt.tz is not None:
                                        _new_df['date'] = _new_df['date'].dt.tz_localize(None)
                                    _combined = _pd_5m_refresh.concat([_ex_copy, _new_df], ignore_index=True)
                                    _combined = _combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                                    # Save updated parquet
                                    _pq_path = os.path.join(_5m_dir_refresh, f'{_rs}.parquet')
                                    _combined.to_parquet(_pq_path, index=False)
                                    # Update in-memory cache
                                    if _rs == 'NIFTY50':
                                        self._nifty_5min_df = _combined
                                    else:
                                        self._hist_5min_cache[_rs] = _combined
                                    _refreshed += 1
                            except Exception as _rs_err:
                                if _rs == 'NIFTY50':
                                    print(f"  ⚠️ NIFTY50 5-min refresh failed: {_rs_err}")
                        if _refreshed > 0:
                            _new_last = next(iter(self._hist_5min_cache.values()))['date'].max()
                            print(f"  ✅ 5-min parquets refreshed: {_refreshed}/{len(_all_syms)} stocks (now up to {str(_new_last)[:10]})")
                        else:
                            print(f"  ⚠️ 5-min parquet refresh: 0 stocks updated (Kite API may be unavailable)")
                        
                        # Also refresh NIFTY50 daily parquet
                        try:
                            _nifty_daily_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'candles_daily', 'NIFTY50.parquet')
                            if self._nifty_daily_df is not None:
                                _nd_last = self._nifty_daily_df['date'].max()
                                if hasattr(_nd_last, 'tz') and _nd_last.tz is not None:
                                    _nd_last = _nd_last.tz_localize(None)
                                _nd_gap = (_now_ts - _nd_last).days
                                if _nd_gap > 1:
                                    _new_daily = fetch_candles(_kite, 'NIFTY 50', days=min(_nd_gap + 1, 60), interval='day')
                                    if len(_new_daily) > 0:
                                        _new_daily['date'] = _pd_5m_refresh.to_datetime(_new_daily['date'])
                                        _nd_copy = self._nifty_daily_df.copy()
                                        if _nd_copy['date'].dt.tz is not None:
                                            _nd_copy['date'] = _nd_copy['date'].dt.tz_localize(None)
                                        if _new_daily['date'].dt.tz is not None:
                                            _new_daily['date'] = _new_daily['date'].dt.tz_localize(None)
                                        _nd_combined = _pd_5m_refresh.concat([_nd_copy, _new_daily], ignore_index=True)
                                        _nd_combined = _nd_combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                                        _nd_combined.to_parquet(_nifty_daily_path, index=False)
                                        self._nifty_daily_df = _nd_combined
                                        print(f"  ✅ NIFTY daily refreshed: {len(_nd_combined)} days (last={_nd_combined['date'].max()})")
                        except Exception as _nd_err:
                            print(f"  ⚠️ NIFTY daily refresh: {_nd_err}")
                    except ImportError:
                        print("  ⚠️ data_fetcher not available — 5-min refresh skipped")
                    except Exception as _5m_err:
                        print(f"  ⚠️ 5-min parquet refresh error: {_5m_err}")
                else:
                    print(f"  ✅ 5-min parquets: fresh (last={_last_date.date()}, {_gap}d ago)")
        except Exception as _5m_chk_e:
            print(f"  ⚠ 5-min freshness check: {_5m_chk_e}")
        
        # === AUTO-REFRESH STALE OI DATA ===
        # If futures OI parquets are >3 days old, refresh them proactively.
        try:
            import pandas as _pd_oi_chk
            _oi_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'futures_oi')
            _sample_files = [f for f in os.listdir(_oi_dir) if f.endswith('_futures_oi.parquet')][:3]
            _oi_stale = False
            if not _sample_files:
                _oi_stale = True
            else:
                for _sf in _sample_files:
                    _sdf = _pd_oi_chk.read_parquet(os.path.join(_oi_dir, _sf))
                    _last = _pd_oi_chk.Timestamp(_sdf['date'].max())
                    if (_pd_oi_chk.Timestamp.now() - _last).days > 3:
                        _oi_stale = True
                        break
            if _oi_stale:
                print("  ⚠️ Futures OI data stale (>3 days) — auto-refreshing...")
                from dhan_futures_oi import FuturesOIFetcher
                _oi_fetcher = FuturesOIFetcher()
                if _oi_fetcher.ready:
                    _oi_fetcher.backfill_all(months_back=1)
                    print("  ✅ OI data refreshed")
                else:
                    print("  ⚠️ DhanHQ not ready — OI refresh skipped (check DHAN_ACCESS_TOKEN in .env)")
            else:
                print("  ✅ Futures OI data: fresh")
        except Exception as _oi_chk_e:
            print(f"  ⚠ OI freshness check: {_oi_chk_e}")
        
        # === OI FLOW ANALYZER (post-ML overlay using live options chain) ===
        self._oi_analyzer = None
        try:
            from options_flow_analyzer import get_options_flow_analyzer
            from options_trader import get_options_trader
            _ot_init = get_options_trader(
                kite=self.tools.kite,
                capital=getattr(self.tools, 'paper_capital', 500000),
                paper_mode=getattr(self.tools, 'paper_mode', True)
            )
            self._oi_analyzer = get_options_flow_analyzer(_ot_init.chain_fetcher)
            if self._oi_analyzer and self._oi_analyzer.ready:
                print("  📊 OI Flow Analyzer: ACTIVE (real-time PCR/IV/MaxPain overlay)")
            else:
                print("  📊 OI Flow Analyzer: STANDBY (no chain fetcher)")
                self._oi_analyzer = None
        except Exception as e:
            print(f"  📊 OI Flow Analyzer: DISABLED ({e})")
        
        print("="*60)
    
    # ========== ML_OVERRIDE_WGMM GATE CHECK ==========
    def _ml_override_allowed(self, sym_clean: str, ml: dict, dr_score: float,
                             path: str = 'MODEL_TRACKER') -> tuple:
        """Check tightened ML_OVERRIDE_WGMM gates. Returns (allowed: bool, reason: str).
        Call BEFORE flipping direction.
        
        DR SCORE INTERPRETATION (correct):
          High dr in DOWN regime = stock WILL go DOWN (confirms DOWN)
          High dr in UP regime = stock WILL go UP (confirms UP)
          → ML_OVERRIDE fires when XGB opposes AND GMM confirms XGB via high dr
        
        Gates:
          1. Concurrent open position limit
          2. ml_move_prob minimum (higher bar than general gate)
          3. dr_score MINIMUM — need HIGH dr to confirm XGB's opposing direction
          4. XGB directional probability minimum
        """
        cfg = self._ml_ovr_cfg
        
        # Gate 1: Max concurrent open ML_OVERRIDE_WGMM positions
        _max_concurrent = cfg.get('max_concurrent_open', 3)
        try:
            _active = getattr(self.tools, 'paper_positions', []) or []
            _open_ovr = sum(1 for t in _active if t.get('setup_type') == 'ML_OVERRIDE_WGMM' and not t.get('closed'))
            if _open_ovr >= _max_concurrent:
                return False, f"concurrent open {_open_ovr}/{_max_concurrent}"
        except Exception:
            pass
        
        # Gate 2: ml_move_prob minimum
        _min_move = cfg.get('min_move_prob', 0.58)
        _move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
        if _move_prob < _min_move:
            return False, f"move_prob {_move_prob:.2f} < {_min_move}"
        
        # Gate 3: dr_score MINIMUM — high dr confirms XGB's direction is real
        _min_dr = cfg.get('min_dr_score', 0.15)
        if dr_score < _min_dr:
            return False, f"dr_score {dr_score:.3f} < {_min_dr} (need high dr to confirm XGB)"
        
        # Gate 4: XGB directional probability
        _min_dir_prob = cfg.get('min_directional_prob', 0.40)
        _xgb_signal = ml.get('ml_signal', 'UNKNOWN')
        if _xgb_signal == 'UP':
            _dir_prob = ml.get('ml_prob_up', 0.0)
        elif _xgb_signal == 'DOWN':
            _dir_prob = ml.get('ml_prob_down', 0.0)
        else:
            _dir_prob = 0.0
        if _dir_prob < _min_dir_prob:
            return False, f"dir_prob({_xgb_signal}) {_dir_prob:.2f} < {_min_dir_prob}"
        
        return True, "all gates passed"
    
    # ========== DECISION LOG ==========
    def _log_decision(self, cycle_time: str, symbol: str, score: float, outcome: str,
                      reason: str = '', setup: str = '', direction: str = '', extra: dict = None):
        """Log a scanning decision to centralized Trade Ledger"""
        if not self._decision_log_cfg.get('enabled', False):
            return
        try:
            from trade_ledger import get_trade_ledger
            get_trade_ledger().log_scan_decision(
                symbol=symbol,
                score=score,
                outcome=outcome,
                reason=reason,
                direction=direction,
                cycle=cycle_time,
                extra=extra,
            )
        except Exception:
            pass  # Silent — decision log should never break trading
    
    # ========== ELITE AUTO-FIRE ==========
    def _elite_auto_fire(self, pre_scores: dict, cycle_decisions: dict,
                         sorted_data: list, market_data: dict, fno_nfo_verified: set,
                         cycle_time: str) -> list:
        """Auto-execute elite-scored stocks (≥78) without waiting for GPT.
        
        Returns list of auto-fired symbols (to exclude from GPT prompt).
        """
        cfg = self._elite_auto_fire_cfg
        if not cfg.get('enabled', False):
            return []
        
        threshold = cfg.get('elite_threshold', 78)
        max_fires = cfg.get('max_auto_fires_per_cycle', 3)
        require_setup = cfg.get('require_setup', True)
        
        # Find elite candidates
        elite_candidates = []
        for sym, score in sorted(pre_scores.items(), key=lambda x: x[1], reverse=True):
            if score < threshold:
                break
            # Skip if already holding
            if self.tools.is_symbol_in_active_trades(sym):
                continue
            # Skip if already auto-fired this session
            if sym in self._auto_fired_this_session:
                continue
            # Skip if not F&O eligible
            if sym not in fno_nfo_verified:
                continue
            # Determine direction from cached decision
            cached = cycle_decisions.get(sym, {})
            decision = cached.get('decision')
            if not decision:
                continue
            # Get direction from the decision's signal analysis
            direction = None
            if hasattr(decision, 'direction') and decision.direction:
                direction = decision.direction
            else:
                # Fallback: infer from market data
                data = market_data.get(sym, {})
                if isinstance(data, dict):
                    chg = data.get('change_pct', 0)
                    direction = 'BUY' if chg > 0 else 'SELL'
            
            if not direction:
                continue
            
            # Check setup exists if required
            data = market_data.get(sym, {})
            if require_setup and isinstance(data, dict):
                orb = data.get('orb_signal', 'INSIDE_ORB')
                vwap = data.get('price_vs_vwap', 'AT_VWAP')
                vol = data.get('volume_regime', 'NORMAL')
                ema = data.get('ema_regime', 'NORMAL')
                has_setup = (
                    orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') or
                    (vwap in ('ABOVE_VWAP', 'BELOW_VWAP') and vol in ('HIGH', 'EXPLOSIVE')) or
                    ema == 'COMPRESSED' or
                    data.get('rsi_14', 50) < 30 or data.get('rsi_14', 50) > 70
                )
                if not has_setup:
                    self._log_decision(cycle_time, sym, score, 'ELITE_BLOCKED',
                                      reason='No valid setup despite high score', direction=direction)
                    continue
            
            # === FOLLOW-THROUGH GATE (prevents auto-firing dead momentum) ===
            # Pro trader rule: don't auto-fire if breakout has ZERO follow-through.
            # 8 TIME_STOP losses had MaxR=0 — stock never moved after entry.
            # Require at least 1 follow-through candle for auto-fire confidence.
            if isinstance(data, dict):
                ft_candles = data.get('follow_through_candles', 0)
                adx_val = data.get('adx', 20)
                oi_signal = data.get('oi_signal', 'NEUTRAL')
                
                # Block auto-fire if: zero follow-through AND not a fresh breakout
                # (orb_hold_candles > 2 means breakout happened a while ago)
                orb_hold = data.get('orb_hold_candles', 0)
                if ft_candles == 0 and orb_hold > 2:
                    self._log_decision(cycle_time, sym, score, 'ELITE_NO_FOLLOWTHROUGH',
                                      reason=f'FT=0, ORB hold={orb_hold} candles — stale breakout, no confirmation',
                                      direction=direction)
                    continue
                
                # Block auto-fire if ADX < 25 (no trend strength)
                if adx_val < 25:
                    self._log_decision(cycle_time, sym, score, 'ELITE_WEAK_ADX',
                                      reason=f'ADX={adx_val:.0f} < 25 — no trend strength for auto-fire',
                                      direction=direction)
                    continue
                
                # Block auto-fire if OI conflicts with direction
                if direction == 'BUY' and oi_signal == 'SHORT_BUILDUP':
                    self._log_decision(cycle_time, sym, score, 'ELITE_OI_CONFLICT',
                                      reason=f'BUY direction but OI={oi_signal} — institutions selling',
                                      direction=direction)
                    continue
                if direction == 'SELL' and oi_signal == 'LONG_BUILDUP':
                    self._log_decision(cycle_time, sym, score, 'ELITE_OI_CONFLICT',
                                      reason=f'SELL direction but OI={oi_signal} — institutions buying',
                                      direction=direction)
                    continue
            
            # ML soft gate: if ML strongly says FLAT, skip auto-fire
            # (GPT can still pick this stock — just don't auto-fire it)
            # FAIL-SAFE: if ML check crashes, proceed with auto-fire as before
            try:
                _cached_ml = cycle_decisions.get(sym, {}).get('ml_prediction', {})
                if _cached_ml.get('ml_elite_ok') is False:
                    _ml_flat_p = _cached_ml.get('ml_prob_flat', 0)
                    self._log_decision(cycle_time, sym, score, 'ELITE_ML_SKIP',
                                      reason=f"ML: FLAT ({_ml_flat_p:.0%} flat prob) — GPT can still pick",
                                      direction=direction)
                    continue
            except Exception:
                pass  # ML check failed — proceed with auto-fire
            
            elite_candidates.append((sym, score, direction, data))
        
        # Execute top N elite candidates
        auto_fired = []
        for sym, score, direction, data in elite_candidates[:max_fires]:
            # Check position limits (regime-aware: fewer positions in MIXED market)
            active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            _elite_breadth = data.get('market_breadth', 'MIXED') if isinstance(data, dict) else 'MIXED'
            if _elite_breadth == 'MIXED':
                _max_pos = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)
            elif _elite_breadth in ('BULLISH', 'BEARISH'):
                _max_pos = HARD_RULES.get('MAX_POSITIONS_TRENDING', 12)
            else:
                _max_pos = HARD_RULES['MAX_POSITIONS']
            if len(active_positions) >= _max_pos:
                print(f"   ⛔ POSITION LIMIT: {len(active_positions)} >= {_max_pos} ({_elite_breadth} regime)")
                break
            
            # XGB Direction override REMOVED — direction comes from Titan scorer only.
            # GMM veto/boost is applied in model-tracker, not in elite auto-fire.
            
            print(f"\n   🎯 ELITE AUTO-FIRE: {sym} score={score:.0f} direction={direction}")
            
            try:
                # Build ML data for elite auto-fire trades
                _elite_ml = getattr(self, '_cycle_ml_results', {}).get(sym, {})
                _elite_ml_data = {
                    'smart_score': score,
                    'p_score': score,
                    'dr_score': _elite_ml.get('ml_down_risk_score', 0),
                    'ml_move_prob': _elite_ml.get('ml_move_prob', 0),
                    'ml_confidence': _elite_ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': _elite_ml.get('ml_signal', 'UNKNOWN'),
                        'move_prob': _elite_ml.get('ml_move_prob', 0),
                        'prob_up': _elite_ml.get('ml_prob_up', 0),
                        'prob_down': _elite_ml.get('ml_prob_down', 0),
                        'prob_flat': _elite_ml.get('ml_prob_flat', 0),
                        'direction_bias': _elite_ml.get('ml_direction_bias', 0),
                        'confidence': _elite_ml.get('ml_confidence', 0),
                        'score_boost': _elite_ml.get('ml_score_boost', 0),
                        'direction_hint': _elite_ml.get('ml_direction_hint', 'NEUTRAL'),
                        'model_type': _elite_ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': _elite_ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': _elite_ml.get('ml_down_risk_score', 0),
                        'down_risk_flag': _elite_ml.get('ml_down_risk_flag', False),
                        'down_risk_bucket': _elite_ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': _elite_ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': _elite_ml.get('ml_gmm_regime_used', None),
                        'gmm_action': 'ELITE_AUTO',
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': False,
                } if _elite_ml else {}
                result = self.tools.place_option_order(
                    underlying=sym,
                    direction=direction,
                    strike_selection="ATM",
                    rationale=f"ELITE AUTO-FIRE: Score {score:.0f} (threshold {threshold}) — bypassing GPT for immediate execution",
                    ml_data=_elite_ml_data
                )
                if result and result.get('success'):
                    print(f"   ✅ ELITE AUTO-FIRED: {sym} ({direction}) score={score:.0f}")
                    auto_fired.append(sym)
                    self._auto_fired_this_session.add(sym)
                    self._log_decision(cycle_time, sym, score, 'AUTO_FIRED',
                                      reason=f'Elite score {score:.0f} ≥ {threshold}',
                                      direction=direction, setup='ELITE_AUTO')
                else:
                    error = result.get('error', 'unknown') if result else 'no result'
                    print(f"   ⚠️ Elite auto-fire failed for {sym}: {error}")
                    self._log_decision(cycle_time, sym, score, 'AUTO_FIRE_FAILED',
                                      reason=f'Execution failed: {str(error)[:80]}',
                                      direction=direction)
            except Exception as e:
                print(f"   ❌ Elite auto-fire error for {sym}: {e}")
                self._log_decision(cycle_time, sym, score, 'AUTO_FIRE_ERROR',
                                  reason=str(e)[:100], direction=direction)
        
        if auto_fired:
            print(f"\n   🎯 ELITE AUTO-FIRED: {len(auto_fired)} trades — {', '.join(auto_fired)}")
        
        return auto_fired
    
    # ========== DYNAMIC MAX PICKS ==========
    def _compute_max_picks(self, pre_scores: dict, breadth: str) -> int:
        """Compute dynamic max picks for GPT based on signal quality."""
        cfg = self._dynamic_max_picks_cfg
        if not cfg.get('enabled', False):
            return 3  # Default
        
        default_max = cfg.get('default_max', 3)
        bonus_max = cfg.get('elite_bonus_max', 5)
        min_score = cfg.get('min_score_for_bonus', 70)
        min_count = cfg.get('min_count_for_bonus', 3)
        choppy_max = cfg.get('choppy_max', 2)
        
        # Count high-scoring stocks
        high_score_count = sum(1 for s in pre_scores.values() if s >= min_score)
        medium_score_count = sum(1 for s in pre_scores.values() if s >= 60)
        
        # Bonus: many high-scoring setups → allow more picks
        if high_score_count >= min_count:
            return bonus_max
        
        # Restriction: choppy market with few setups → restrict
        if breadth == 'MIXED' and medium_score_count < 3:
            return choppy_max
        
        return default_max
    
    # ========== DOWN-RISK SOFT SCORING + MODEL TRACKER ==========
    def _reset_model_tracker_if_new_day(self):
        """Reset model-tracker trade counter at start of new trading day."""
        today = datetime.now().date()
        if today != self._model_tracker_date:
            self._model_tracker_trades_today = 0
            self._model_tracker_date = today
            self._model_tracker_symbols = set()
            self._dr_flip_symbols = set()
            self._dr_flip_trades_today = 0
            print(f"📅 New trading day — model-tracker counter reset")
    
    def _apply_down_risk_soft_scores(self, ml_results: dict, pre_scores: dict):
        """Apply DIRECTION-AWARE graduated score adjustments based on GMM anomaly.
        
        GMM dr_score meaning depends on regime + trade direction:
          UP regime  (detects hidden crash):
            high dr + BUY  → crash hurts CE  → PENALIZE
            high dr + SELL → crash helps PE  → BOOST  (dr CONFIRMS put)
          DOWN regime (detects bear trap / hidden rally):
            high dr + SELL → bear trap hurts PE → PENALIZE
            high dr + BUY  → bear trap = rally helps CE → BOOST (dr CONFIRMS call)
        
        In short: if dr agrees with trade direction → BOOST, if opposes → PENALIZE.
        Clean dr (low score) in aligned regime → genuine pattern → BOOST.
        
        Does NOT hard-block any trades — only nudges scores for natural prioritisation.
        Called once per scan cycle after ML predictions are merged.
        """
        if not self._down_risk_cfg.get('enabled', False):
            return
        
        high_thresh = self._down_risk_cfg.get('high_risk_threshold', 0.40)
        high_penalty = self._down_risk_cfg.get('high_risk_penalty', 15)
        mid_penalty = self._down_risk_cfg.get('mid_risk_penalty', 8)
        clean_thresh = self._down_risk_cfg.get('clean_threshold', 0.15)
        clean_boost = self._down_risk_cfg.get('clean_boost', 8)
        adjustments = []
        
        # Get cached directions from IntradayScorer (available at this point)
        _cycle_decs = getattr(self.tools, '_cached_cycle_decisions', {})
        
        for sym in list(pre_scores.keys()):
            ml = ml_results.get(sym, {})
            dr_flag = ml.get('ml_down_risk_flag', None)
            dr_score = ml.get('ml_down_risk_score', None)
            
            if dr_score is None:
                continue  # No down-risk prediction for this symbol
            
            gmm_regime = ml.get('ml_gmm_regime_used', 'UP')
            
            # Get trade direction from IntradayScorer (if available)
            _cd = _cycle_decs.get(sym, {})
            _decision = _cd.get('decision')
            direction = None
            if _decision and hasattr(_decision, 'recommended_direction'):
                direction = _decision.recommended_direction
            
            # Determine if dr signal OPPOSES or CONFIRMS trade direction
            # UP regime: high dr = crash likely → opposes BUY, confirms SELL
            # DOWN regime: high dr = bear trap (rally) → opposes SELL, confirms BUY
            dr_opposes_trade = True  # default: treat as opposing (conservative)
            if direction and direction != 'HOLD':
                if gmm_regime == 'UP':
                    # UP regime detects hidden crash risk
                    dr_opposes_trade = (direction == 'BUY')   # crash hurts CE
                elif gmm_regime == 'DOWN':
                    # DOWN regime detects bear trap (hidden rally)
                    dr_opposes_trade = (direction == 'SELL')   # rally hurts PE
            
            _sym_diag = sym.replace('NSE:', '')
            _dir_tag = direction or '?'
            
            # Score-graduated response — direction-aware
            if dr_score > high_thresh:
                if dr_opposes_trade:
                    # High dr OPPOSES trade → strong penalty (crash hurts CE / trap hurts PE)
                    pre_scores[sym] -= high_penalty
                    adjustments.append(f"🔴 {_sym_diag} −{high_penalty} (HIGH_{gmm_regime} vs {_dir_tag} dr={dr_score:.3f})")
                else:
                    # High dr CONFIRMS trade → BOOST (crash helps PE / trap helps CE)
                    pre_scores[sym] += clean_boost
                    adjustments.append(f"🟣 {_sym_diag} +{clean_boost} (DR_CONFIRMS_{_dir_tag} dr={dr_score:.3f})")
            elif dr_flag:
                if dr_opposes_trade:
                    # Flagged + opposes → moderate penalty
                    pre_scores[sym] -= mid_penalty
                    adjustments.append(f"🟠 {_sym_diag} −{mid_penalty} (CAUTION_{gmm_regime} vs {_dir_tag} dr={dr_score:.3f})")
                else:
                    # Flagged but confirms trade direction → small boost
                    _small_boost = clean_boost // 2
                    pre_scores[sym] += _small_boost
                    adjustments.append(f"🔵 {_sym_diag} +{_small_boost} (FLAG_CONFIRMS_{_dir_tag} dr={dr_score:.3f})")
            elif dr_score < clean_thresh:
                # Low dr = genuine clean pattern in this regime
                # Clean + aligned regime → boost (stock moving genuinely in scored direction)
                pre_scores[sym] += clean_boost
                adjustments.append(f"🟢 {_sym_diag} +{clean_boost} (CLEAN_{gmm_regime}_{_dir_tag} dr={dr_score:.3f})")
            # else: neutral zone — no adjustment
        
        if adjustments:
            print(f"   🛡️ GMM GRADUATED SCORE: {len(adjustments)} adjusted — {' | '.join(adjustments[:8])}")
            if len(adjustments) > 8:
                print(f"      ... and {len(adjustments) - 8} more")
    
    def _place_model_tracker_trades(self, ml_results: dict, pre_scores: dict, market_data: dict, cycle_time: str):
        """Place up to N exclusive model-tracker trades using GMM veto/boost selection.
        
        These are SEPARATE from the main trading workflow. The smart selector
        combines Titan's technical direction (from IntradayScorer) with GMM
        anomaly detection as a veto/boost filter.
        
        Dual-Regime GMM Veto/Boost Logic:
          Titan scans top stocks, scores them, determines direction (CE/PE).
          Two GMM models detect anomalous patterns:
          
          UP regime GMM (for CE/BUY candidates):
            GMM HIGH (dr_flag=True) + CE → BLOCK (hidden crash risk, CE dies)
            GMM LOW  (dr_flag=False) + CE → ALLOW (genuine UP, safe for calls)
          
          DOWN regime GMM (for PE/SELL candidates):
            GMM HIGH (dr_flag=True) + PE → BLOCK (bear trap, hidden UP risk)
            GMM LOW  (dr_flag=False) + PE → BOOST (genuine DOWN, PE profits)
        
        Smart Scoring (composite score):
          1. Gate P(MOVE) × pre_score quality (conviction)
          2. GMM anomaly-aware safety (direction-aware)
          3. Technical strength from pre_score
          4. Move probability bonus
        
        Diversification Rules:
          - Max 2 stocks per sector (avoid sector concentration)
          - No HOLD direction (need conviction)
        """
        if not self._down_risk_cfg.get('enabled', False):
            return []
        
        self._reset_model_tracker_if_new_day()
        
        max_tracker = self._down_risk_cfg.get('model_tracker_trades', 7)
        budget = max_tracker - self._model_tracker_trades_today
        if budget <= 0:
            return []
        
        # Import sector mapping for diversification
        try:
            from ml_models.feature_engineering import get_sector_for_symbol as _get_sector_mt
        except Exception:
            _get_sector_mt = lambda s: ''
        
        # ── Step 1: Build candidate pool with GMM veto/boost on Titan direction ──
        # Direction comes from Titan's IntradayScorer (technicals), NOT XGB Direction model.
        # GMM anomaly score decides: block, allow, or boost.
        raw_candidates = []
        _cycle_decs = getattr(self.tools, '_cached_cycle_decisions', {})
        
        for sym in pre_scores:
            ml = ml_results.get(sym, {})
            dr_score = ml.get('ml_down_risk_score', None)
            dr_flag = ml.get('ml_down_risk_flag', None)
            if dr_score is None or dr_flag is None:
                continue
            if sym in self._model_tracker_symbols:
                continue  # Already tracked today
            
            # ── Get Titan's technical direction from IntradayScorer ──
            _cd = _cycle_decs.get(sym, {})
            _decision = _cd.get('decision')
            if _decision and hasattr(_decision, 'recommended_direction') and _decision.recommended_direction:
                direction = _decision.recommended_direction
            else:
                continue  # No direction from scorer → skip
            
            if direction == 'HOLD':
                continue  # Scorer couldn't determine direction → skip
            
            # ── DUAL-REGIME GMM VETO/BOOST LOGIC ──
            # CRITICAL: We must check the ACTUAL XGB signal (ml_signal), not gmm_regime.
            # Reason: FLAT signals (very common — ~30-50% of stocks) get routed to
            # UP regime by default in predictor. Using gmm_regime would incorrectly
            # treat all FLAT signals as "XGB says UP", creating a massive BUY bias.
            #
            # DR SCORE INTERPRETATION:
            #   High dr in DOWN regime = stock WILL go DOWN (confirms DOWN direction)
            #   High dr in UP regime = stock WILL go UP (confirms UP direction)
            #   Low dr = uncertain, no strong GMM signal
            #
            # Logic:
            #   XGB=FLAT/UNKNOWN → no directional opinion → ALLOW (no boost, no block)
            #   XGB aligns + high dr (flag) → GMM confirms aligned direction → ALL_AGREE BOOST
            #   XGB aligns + low dr (clean) → GMM uncertain → just ALLOW as MODEL_TRACKER
            #   XGB opposes + high dr (flag) → GMM confirms XGB's direction → ML_OVERRIDE FLIP
            #   XGB opposes + low dr (clean) → GMM uncertain → XGB alone opposes → ALLOW Titan
            trade_type = 'MODEL_TRACKER'
            gmm_action = 'ALLOW'  # default
            _xgb_signal = ml.get('ml_signal', 'UNKNOWN')
            gmm_regime = ml.get('ml_gmm_regime_used', 'UP')
            
            if _xgb_signal in ('FLAT', 'UNKNOWN'):
                # XGB has no directional opinion → GMM regime is just default routing
                # Cannot use GMM to BOOST or BLOCK directionally → ALLOW
                gmm_action = 'ALLOW'
            elif (_xgb_signal == 'UP' and direction == 'BUY') or (_xgb_signal == 'DOWN' and direction == 'SELL'):
                # XGB signal ALIGNS with trade direction
                if dr_flag:
                    # HIGH dr confirms the aligned direction → ALL 3 systems agree
                    # XGB says direction + GMM confirms with high dr + Titan agrees → BOOST
                    gmm_action = 'BOOST'
                    trade_type = 'ALL_AGREE'
                else:
                    # LOW dr = GMM uncertain about direction → only XGB + Titan agree
                    # Still proceed but as standard MODEL_TRACKER (no extra boost)
                    gmm_action = 'ALLOW'
            else:
                # XGB signal OPPOSES trade direction (UP vs SELL, DOWN vs BUY)
                if dr_flag:
                    # HIGH dr confirms XGB's opposing direction → XGB + GMM both oppose Titan
                    # This is the STRONGEST opposition signal → FLIP to XGB direction
                    sym_clean_tmp = sym.replace('NSE:', '')
                    _ovr_ok, _ovr_reason = self._ml_override_allowed(sym_clean_tmp, ml, dr_score, path='MODEL_TRACKER')
                    if not _ovr_ok:
                        print(f"      🚫 ML_OVR BLOCKED: {sym_clean_tmp} — {_ovr_reason}")
                        continue
                    # XGB + GMM confirm opposing direction → FLIP
                    old_direction = direction
                    direction = 'BUY' if _xgb_signal == 'UP' else 'SELL'
                    gmm_action = 'BOOST'
                    trade_type = 'ML_OVERRIDE_WGMM'
                    print(f"      🔄 ML_OVERRIDE_WGMM: {sym_clean_tmp} — XGB={_xgb_signal} + GMM confirms "
                          f"(dr={dr_score:.3f}) → FLIPPED {old_direction}→{direction}")
                    self._log_decision(cycle_time, sym, pre_scores.get(sym, 0), 'ML_OVERRIDE_WGMM',
                                      reason=f"XGB={_xgb_signal} + GMM dr={dr_score:.3f} confirms → flipped {old_direction}→{direction}",
                                      direction=direction)
                else:
                    # LOW dr = GMM uncertain → XGB alone opposes → weak opposition
                    # Let Titan's direction stand, soft scoring will penalize for XGB opposition
                    gmm_action = 'ALLOW'
                    sym_clean_tmp = sym.replace('NSE:', '')
                    print(f"      ⚠️ XGB opposes ({_xgb_signal}) but GMM uncertain (dr={dr_score:.3f}) "
                          f"→ ALLOW Titan {direction} (soft penalty via scoring)")
            
            # Gather available ML metrics (Gate model still runs)
            ml_confidence = ml.get('ml_confidence', 0.0)  # Gate confidence
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))  # P(MOVE) from Gate
            p_score = pre_scores.get(sym, 0)
            sym_clean = sym.replace('NSE:', '')
            sector = _get_sector_mt(sym_clean)
            
            # ── HARD DR_SCORE GATE → GMM CONTRARIAN FLIP (Feb 24 fix) ──
            # Instead of just blocking high-dr trades, FLIP direction:
            #   BUY + high dr = "fake rally, crash likely" → SELL (buy PUT)
            #   SELL + high dr = "bear trap, bounce likely" → BUY (buy CALL)
            # The GMM anomaly IS the signal.
            _mt_max_dr = self._down_risk_cfg.get('max_dr_score', 0.12)
            if dr_score > _mt_max_dr:
                # Check if GMM_CONTRARIAN (DR_FLIP) is enabled and conditions met
                from config import GMM_CONTRARIAN as _dr_flip_cfg
                _flip_ok = False
                if _dr_flip_cfg.get('enabled', False) and dr_score >= _dr_flip_cfg.get('min_dr_score', 0.15):
                    # Check daily limit
                    _flip_today = getattr(self, '_dr_flip_trades_today', 0)
                    _flip_max_day = _dr_flip_cfg.get('max_trades_per_day', 4)
                    # Check concurrent open limit
                    _flip_open = sum(1 for t in (getattr(self.tools, 'paper_positions', []) or [])
                                    if t.get('setup_type') == 'DR_FLIP' and t.get('status', 'OPEN') == 'OPEN')
                    _flip_max_concurrent = _dr_flip_cfg.get('max_concurrent_open', 3)
                    # Check Gate P(MOVE)
                    _flip_min_gate = _dr_flip_cfg.get('min_gate_prob', 0.50)
                    
                    if _flip_today >= _flip_max_day:
                        print(f"      🚫 DR_FLIP DAILY LIMIT: {sym_clean} — {_flip_today}/{_flip_max_day} flips today")
                    elif _flip_open >= _flip_max_concurrent:
                        print(f"      🚫 DR_FLIP CONCURRENT LIMIT: {sym_clean} — {_flip_open}/{_flip_max_concurrent} open")
                    elif ml_move_prob < _flip_min_gate:
                        print(f"      🚫 DR_FLIP GATE LOW: {sym_clean} — P(MOVE)={ml_move_prob:.2f} < {_flip_min_gate}")
                    else:
                        # XGB safety: don't flip INTO a direction XGB strongly opposes
                        _flip_dir = 'SELL' if direction == 'BUY' else 'BUY'
                        _xgb_blocks_flip = False
                        if _flip_dir == 'BUY' and _xgb_signal == 'DOWN' and ml.get('ml_prob_down', 0) > 0.55:
                            _xgb_blocks_flip = True
                        elif _flip_dir == 'SELL' and _xgb_signal == 'UP' and ml.get('ml_prob_up', 0) > 0.55:
                            _xgb_blocks_flip = True
                        
                        if _xgb_blocks_flip:
                            print(f"      🚫 DR_FLIP XGB CONFLICT: {sym_clean} — XGB={_xgb_signal} strongly opposes flip to {_flip_dir}")
                        else:
                            _flip_ok = True
                            old_dir = direction
                            direction = _flip_dir
                            trade_type = 'DR_FLIP'
                            gmm_action = 'CONTRARIAN'
                            if not hasattr(self, '_dr_flip_symbols'):
                                self._dr_flip_symbols = set()
                            self._dr_flip_symbols.add(sym)
                            print(f"      🔀 GMM CONTRARIAN FLIP: {sym_clean} — dr={dr_score:.3f} (HIGH) → "
                                  f"FLIPPED {old_dir}→{direction} | P(MOVE)={ml_move_prob:.2f}")
                            self._log_decision(cycle_time, sym, p_score, 'DR_FLIP',
                                              reason=f"GMM dr={dr_score:.3f} too high for {old_dir} → contrarian {direction}",
                                              direction=direction)
                
                if not _flip_ok:
                    # Still blocked if flip didn't fire
                    # Skip ALL_AGREE — high dr was already handled (GMM + XGB + Titan aligned)
                    # Skip ML_OVERRIDE_WGMM — high dr was used to confirm the flip
                    if dr_score > _mt_max_dr and trade_type not in ('DR_FLIP', 'ALL_AGREE', 'ML_OVERRIDE_WGMM'):
                        print(f"      🚫 MODEL-TRACKER DR GATE: {sym_clean} — dr={dr_score:.3f} > {_mt_max_dr} → BLOCKED")
                        continue
            
            # ── Gate floor: P(MOVE) must be meaningful ──
            if ml_move_prob < 0.40:
                continue  # Gate says stock unlikely to move — skip
            
            # ── ML DIRECTION CONFLICT FILTER ──
            # Check if XGBoost ML signal disagrees with scored direction
            # SKIP for DR_FLIP trades — contrarian direction naturally opposes XGB
            from config import ML_DIRECTION_CONFLICT
            _dir_conflict_cfg = ML_DIRECTION_CONFLICT
            _xgb_disagrees = False
            if _dir_conflict_cfg.get('enabled', False) and trade_type not in ('DR_FLIP', 'ML_OVERRIDE_WGMM'):
                _ml_signal = ml.get('ml_signal', 'UNKNOWN')
                if direction == 'BUY' and _ml_signal == 'DOWN' and ml_move_prob >= _dir_conflict_cfg.get('min_xgb_confidence', 0.55):
                    _xgb_disagrees = True
                elif direction == 'SELL' and _ml_signal == 'UP' and ml_move_prob >= _dir_conflict_cfg.get('min_xgb_confidence', 0.55):
                    _xgb_disagrees = True
                
                if _xgb_disagrees:
                    _gmm_caution = dr_score > _dir_conflict_cfg.get('gmm_caution_threshold', 0.15)
                    if _gmm_caution:
                        # HARD BLOCK: Both XGBoost AND GMM disagree with scored direction
                        print(f"      🚫 DIRECTION CONFLICT BLOCK: {sym_clean} — XGB={_ml_signal} vs scored={direction}, "
                              f"GMM dr={dr_score:.3f} > {_dir_conflict_cfg.get('gmm_caution_threshold', 0.15)} → BOTH disagree")
                        self._log_decision(cycle_time, sym, p_score, 'ML_DIRECTION_BLOCK',
                                          reason=f"XGB={_ml_signal} + GMM dr={dr_score:.3f} vs scored={direction}",
                                          direction=direction)
                        continue
                    else:
                        # SOFT PENALTY: Only XGB disagrees, GMM is clean — penalize score
                        _xgb_penalty = _dir_conflict_cfg.get('xgb_penalty', 15)
                        print(f"      ⚠️ XGB DIRECTION CONFLICT: {sym_clean} — XGB={_ml_signal} vs scored={direction}, "
                              f"GMM clean (dr={dr_score:.3f}) → −{_xgb_penalty} penalty")
            
            # ── Composite smart score (dual-regime GMM-aware) ──
            # Weight 1: Conviction = Gate P(MOVE) × ML confidence (0-40 pts)
            # NOTE: Uses ml_confidence (not p_score) to avoid double-counting
            # p_score already represented in Weight 3 (technical)
            conviction = ml_move_prob * max(ml_confidence, 0.01) * 40.0
            
            # Weight 2: Safety / GMM-awareness (0-25 pts)
            # Both CE and PE get BOOST when GMM confirms — symmetric treatment
            # Lower anomaly score = more genuine pattern = higher safety score
            safety = (1.0 - min(dr_score, 1.0)) * 20.0 + 5.0  # up to 25 pts (boosted)
            
            # Weight 3: Technical strength = pre_score normalized (0-100 → 0-20 pts)
            technical = min(p_score, 100) * 0.20
            
            # Weight 4: Move probability bonus (0-1 → 0-15 pts)
            # Increased weight since we no longer have direction model agreement signal
            move_bonus = ml_move_prob * 15.0
            
            smart_score = conviction + safety + technical + move_bonus
            
            # Apply XGB direction conflict penalty (soft gate — only XGB disagrees, GMM clean)
            if _dir_conflict_cfg.get('enabled', False) and _xgb_disagrees:
                smart_score -= _dir_conflict_cfg.get('xgb_penalty', 15)
            
            raw_candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'trade_type': trade_type,
                'gmm_action': gmm_action,
                'sector': sector or 'OTHER',
                'smart_score': round(smart_score, 2),
                'dr_score': dr_score,
                'dr_flag': dr_flag,
                'ml_confidence': ml_confidence,
                'ml_move_prob': ml_move_prob,
                'p_score': p_score,
                # === FULL ML DATA for trade record ===
                'ml_data': {
                    'smart_score': round(smart_score, 2),
                    'p_score': p_score,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'ml_confidence': ml_confidence,
                    'xgb_model': {
                        'signal': ml.get('ml_signal', 'UNKNOWN'),
                        'move_prob': ml.get('ml_move_prob', 0),
                        'prob_up': ml.get('ml_prob_up', 0),
                        'prob_down': ml.get('ml_prob_down', 0),
                        'prob_flat': ml.get('ml_prob_flat', 0),
                        'direction_bias': ml.get('ml_direction_bias', 0),
                        'confidence': ml.get('ml_confidence', 0),
                        'score_boost': ml.get('ml_score_boost', 0),
                        'direction_hint': ml.get('ml_direction_hint', 'NEUTRAL'),
                        'model_type': ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': dr_score,
                        'down_risk_flag': dr_flag,
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', None),
                        'gmm_action': gmm_action,
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': _xgb_disagrees,
                },
            })
        
        if not raw_candidates:
            return []
        
        # ── Step 2: Sort by smart_score descending ──
        raw_candidates.sort(key=lambda c: c['smart_score'], reverse=True)
        
        # ── Step 3: Diversified selection with sector cap ──
        MAX_PER_SECTOR = 3
        sector_counts = {}
        selected = []
        
        for cand in raw_candidates:
            if len(selected) >= budget:
                break
            sec = cand['sector']
            if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                continue  # Sector full — skip to next
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            selected.append(cand)
        
        # Print selection summary
        if selected:
            buy_count = sum(1 for c in selected if c['direction'] == 'BUY')
            sell_count = len(selected) - buy_count
            boost_count = sum(1 for c in selected if c['trade_type'] == 'ALL_AGREE')
            sectors_used = set(c['sector'] for c in selected)
            print(f"\n   🧠 MODEL-TRACKER SMART SELECT: {len(selected)} picks from {len(raw_candidates)} candidates")
            print(f"      Direction mix: {buy_count} BUY(CALL) / {sell_count} SELL(PUT) | ALL_AGREE: {boost_count}")
            print(f"      Sectors: {', '.join(sorted(sectors_used))}")
            for i, c in enumerate(selected, 1):
                _type_tag = f" [{c['trade_type']}]" if c['trade_type'] != 'MODEL_TRACKER' else ''
                _gmm_tag = f" dr={c['dr_score']:.3f}"
                print(f"      #{i} {c['sym_clean']:<12s} {c['direction']:<4s} smart={c['smart_score']:5.1f} "
                      f"(gate={c['ml_move_prob']:.2f}, dr={c['dr_score']:.3f}, "
                      f"tech={c['p_score']:.0f}) [{c['sector']}]{_type_tag}{_gmm_tag}")
        
        # ── Step 4: Place trades ──
        placed = []
        for cand in selected:
            sym = cand['sym']
            direction = cand['direction']
            trade_type = cand['trade_type']
            try:
                _type_label = f" [{trade_type}]" if trade_type != 'MODEL_TRACKER' else ''
                print(f"\n   📊 MODEL-TRACKER TRADE: {cand['sym_clean']} ({direction}, smart={cand['smart_score']:.1f}){_type_label}")
                
                # TODO: Wire score_tier_override='premium' for ALL_AGREE to increase lots
                # For now, ALL_AGREE gets higher smart_score → naturally higher entry score
                
                result = self.tools.place_option_order(
                    underlying=sym,
                    direction=direction,
                    strike_selection="ATM",
                    use_intraday_scoring=False,  # Bypass scorer — model-tracker evaluates independently
                    rationale=(f"{trade_type} smart-pick #{selected.index(cand)+1}: "
                              f"smart={cand['smart_score']:.1f}, dr={cand['dr_score']:.3f}, "
                              f"gate={cand['ml_move_prob']:.2f}, gmm_action={cand['gmm_action']}, "
                              f"sector={cand['sector']}"),
                    setup_type=trade_type,
                    ml_data=cand.get('ml_data', {}),
                    sector=cand.get('sector', ''),
                )
                if result and result.get('success'):
                    print(f"   ✅ MODEL-TRACKER PLACED: {cand['sym_clean']} ({direction}) [smart={cand['smart_score']:.1f}]{_type_label}")
                    self._model_tracker_symbols.add(sym)
                    self._model_tracker_trades_today += 1
                    # Track DR_FLIP trades separately
                    if trade_type == 'DR_FLIP':
                        self._dr_flip_trades_today += 1
                    placed.append(cand['sym_clean'])
                    self._log_decision(cycle_time, sym, cand['p_score'], f'{trade_type}_PLACED',
                                      reason=(f"Smart pick: score={cand['smart_score']:.1f}, "
                                             f"dr={cand['dr_score']:.3f}, gate={cand['ml_move_prob']:.2f}, "
                                             f"gmm_action={cand['gmm_action']}, sector={cand['sector']}"),
                                      direction=direction, setup=trade_type)
                else:
                    error = result.get('error', 'unknown') if result else 'no result'
                    print(f"   ⚠️ Model-tracker failed for {cand['sym_clean']}: {error}")
            except Exception as e:
                print(f"   ❌ Model-tracker error for {cand['sym_clean']}: {e}")
        
        if placed:
            remaining = max_tracker - self._model_tracker_trades_today
            print(f"\n   📊 MODEL-TRACKER: {len(placed)} placed today ({self._model_tracker_trades_today}/{max_tracker}) — {', '.join(placed)}")
        
        return placed
    
    # ========== GMM SNIPER TRADE (1 per cycle, 2x lots) ==========
    def _place_gmm_sniper_trade(self, ml_results: dict, pre_scores: dict, market_data: dict, cycle_time: str):
        """Place 1 high-conviction GMM sniper trade per scan cycle.
        
        Picks the single CLEANEST GMM candidate (lowest down_risk score) that
        also passes strict quality gates. Placed with 2x lot size for maximum
        conviction. Separate budget from model-tracker trades.
        
        Selection criteria (all must pass):
          1. GMM down_risk_score < max_dr_score (very clean, stricter than 0.15)
          2. Smart score >= min_smart_score
          3. XGB gate P(move) >= min_gate_prob (higher floor than model-tracker)
          4. GMM confirms direction (not flagged)
          5. Not already in model-tracker or sniper set today
          
        The candidate with the LOWEST dr_score wins (most confident CLEAN signal).
        """
        cfg = self._gmm_sniper_cfg
        if not cfg.get('enabled', False):
            return None
        
        # Reset on new day
        today = datetime.now().date()
        if today != self._gmm_sniper_date:
            self._gmm_sniper_trades_today = 0
            self._gmm_sniper_date = today
            self._gmm_sniper_symbols = set()
        
        max_per_day = cfg.get('max_sniper_trades_per_day', 5)
        if self._gmm_sniper_trades_today >= max_per_day:
            print(f"   🎯 SNIPER: Daily cap reached ({self._gmm_sniper_trades_today}/{max_per_day})")
            return None
        
        lot_multiplier = cfg.get('lot_multiplier', 2.0)
        min_smart = cfg.get('min_smart_score', 55)
        max_dr = cfg.get('max_dr_score', 0.10)
        min_gate = cfg.get('min_gate_prob', 0.55)
        score_tier = cfg.get('score_tier', 'premium')
        
        print(f"\n   🎯 SNIPER SCAN: gates → dr<{max_dr}, smart>={min_smart}, gate>={min_gate} | {self._gmm_sniper_trades_today}/{max_per_day} used")
        
        # Import sector mapping
        try:
            from ml_models.feature_engineering import get_sector_for_symbol as _get_sector_sniper
        except Exception:
            _get_sector_sniper = lambda s: 'OTHER'
        
        # Build candidate pool — same logic as model-tracker but stricter thresholds
        _cycle_decs = getattr(self.tools, '_cached_cycle_decisions', {})
        candidates = []
        _sniper_reject_counts = {'no_ml': 0, 'already_traded': 0, 'active_pos': 0, 'dr_high': 0, 'dr_flagged': 0, 'no_direction': 0, 'hold': 0, 'gate_low': 0, 'smart_low': 0, 'xgb_block': 0, 'dir_block': 0}
        _sniper_near_misses = []  # Symbols close to passing
        
        for sym in pre_scores:
            ml = ml_results.get(sym, {})
            dr_score = ml.get('ml_down_risk_score', None)
            dr_flag = ml.get('ml_down_risk_flag', None)
            
            sym_diag = sym.replace('NSE:', '')
            if dr_score is None or dr_flag is None:
                _sniper_reject_counts['no_ml'] += 1
                continue
            
            # Skip if already traded as sniper or model-tracker today
            if sym in self._gmm_sniper_symbols or sym in self._model_tracker_symbols:
                _sniper_reject_counts['already_traded'] += 1
                continue
            
            # Skip if already in active positions
            active_syms = {p.get('underlying', '') for p in getattr(self.tools, 'paper_positions', []) 
                          if p.get('status', 'OPEN') == 'OPEN'}
            if sym in active_syms:
                _sniper_reject_counts['active_pos'] += 1
                continue
            
            # Strict GMM cleanliness gate
            if dr_score > max_dr:
                _sniper_reject_counts['dr_high'] += 1
                if dr_score <= max_dr + 0.05:  # Near miss
                    _sniper_near_misses.append(f"{sym_diag}(dr={dr_score:.3f}>max {max_dr})")
                continue
            if dr_flag:
                _sniper_reject_counts['dr_flagged'] += 1
                _sniper_near_misses.append(f"{sym_diag}(dr={dr_score:.3f} FLAGGED)")
                continue  # Must not be flagged
            
            # Get direction from IntradayScorer
            _cd = _cycle_decs.get(sym, {})
            _decision = _cd.get('decision')
            if _decision and hasattr(_decision, 'recommended_direction') and _decision.recommended_direction:
                direction = _decision.recommended_direction
            else:
                _sniper_reject_counts['no_direction'] += 1
                continue
            
            if direction == 'HOLD':
                _sniper_reject_counts['hold'] += 1
                continue
            
            # ── XGB DIRECTION ALIGNMENT CHECK (Sniper) ──
            # Check actual XGB signal, not gmm_regime. FLAT signals default to
            # UP regime routing in predictor — treating that as "XGB says UP"
            # would incorrectly block all SELL snipers when XGB is indecisive.
            _sniper_was_flipped = False
            _xgb_signal = ml.get('ml_signal', 'UNKNOWN')
            if _xgb_signal in ('FLAT', 'UNKNOWN'):
                pass  # No XGB directional opinion — proceed on GMM cleanliness + IntradayScorer
            elif (_xgb_signal == 'UP' and direction == 'SELL') or (_xgb_signal == 'DOWN' and direction == 'BUY'):
                # XGB actively opposes trade direction
                # DR INTERPRETATION: high dr confirms XGB direction is real
                sym_clean_tmp = sym.replace('NSE:', '')
                _sniper_dr_flag = ml.get('ml_down_risk_flag', False)
                if _sniper_dr_flag:
                    # HIGH dr → GMM confirms XGB opposing direction → check ML_OVERRIDE gates → FLIP
                    _ovr_ok, _ovr_reason = self._ml_override_allowed(sym_clean_tmp, ml, dr_score, path='SNIPER')
                    if not _ovr_ok:
                        print(f"      🚫 SNIPER ML_OVR BLOCKED: {sym_clean_tmp} — {_ovr_reason} (dr={dr_score:.3f})")
                        _sniper_reject_counts['xgb_block'] += 1
                        continue
                    old_direction = direction
                    direction = 'BUY' if _xgb_signal == 'UP' else 'SELL'
                    _sniper_was_flipped = True
                    print(f"      🔄 SNIPER ML_OVERRIDE_WGMM: {sym_clean_tmp} — XGB={_xgb_signal} + GMM confirms "
                          f"(dr={dr_score:.4f}) → FLIPPED {old_direction}→{direction}")
                else:
                    # LOW dr → GMM uncertain about XGB's opposing direction → weak opposition
                    # Sniper needs full alignment — block on unconfirmed XGB opposition
                    print(f"      🚫 SNIPER XGB_OPPOSE: {sym_clean_tmp} — XGB={_xgb_signal} vs {direction}, "
                          f"GMM uncertain (dr={dr_score:.3f}) → BLOCK")
                    _sniper_reject_counts['xgb_block'] += 1
                    continue
            
            # XGB gate probability floor
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
            if ml_move_prob < min_gate:
                _sniper_reject_counts['gate_low'] += 1
                _sniper_near_misses.append(f"{sym_diag}(gate={ml_move_prob:.2f}<{min_gate}, dr={dr_score:.3f})")
                continue
            
            # ── ML DIRECTION CONFLICT FILTER (Sniper) ──
            from config import ML_DIRECTION_CONFLICT
            _dir_conflict_cfg = ML_DIRECTION_CONFLICT
            _xgb_disagrees = False
            if _dir_conflict_cfg.get('enabled', False):
                _ml_signal = ml.get('ml_signal', 'UNKNOWN')
                sym_clean_chk = sym.replace('NSE:', '')
                if direction == 'BUY' and _ml_signal == 'DOWN' and ml_move_prob >= _dir_conflict_cfg.get('min_xgb_confidence', 0.55):
                    _xgb_disagrees = True
                elif direction == 'SELL' and _ml_signal == 'UP' and ml_move_prob >= _dir_conflict_cfg.get('min_xgb_confidence', 0.55):
                    _xgb_disagrees = True
                if _xgb_disagrees:
                    # For sniper, any XGB disagreement is a hard block (high-conviction trades only)
                    _gmm_caution = dr_score > _dir_conflict_cfg.get('gmm_caution_threshold', 0.15)
                    if _gmm_caution:
                        print(f"      🚫 SNIPER DIRECTION BLOCK: {sym_clean_chk} — XGB={_ml_signal} vs scored={direction}, "
                              f"GMM dr={dr_score:.3f} → BOTH disagree")
                        _sniper_reject_counts['dir_block'] += 1
                        continue
                    else:
                        print(f"      ⚠️ SNIPER XGB CONFLICT: {sym_clean_chk} — XGB={_ml_signal} vs scored={direction}, "
                              f"GMM clean (dr={dr_score:.3f}) → −{_dir_conflict_cfg.get('xgb_penalty', 15)} penalty")
            
            # Compute smart score (same formula as model-tracker)
            p_score = pre_scores.get(sym, 0)
            conviction = ml_move_prob * min(p_score / 100.0, 1.0) * 40.0
            safety = (1.0 - min(dr_score, 1.0)) * 20.0 + 5.0
            technical = min(p_score, 100) * 0.20
            move_bonus = ml_move_prob * 15.0
            smart_score = conviction + safety + technical + move_bonus
            
            # Apply XGB direction conflict penalty for sniper
            if _dir_conflict_cfg.get('enabled', False) and _xgb_disagrees:
                smart_score -= _dir_conflict_cfg.get('xgb_penalty', 15)
            
            if smart_score < min_smart:
                _sniper_reject_counts['smart_low'] += 1
                _sniper_near_misses.append(f"{sym_diag}(smart={smart_score:.1f}<{min_smart}, dr={dr_score:.3f}, gate={ml_move_prob:.2f})")
                continue
            
            sym_clean = sym.replace('NSE:', '')
            sector = _get_sector_sniper(sym_clean) or 'OTHER'
            
            candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'sector': sector,
                'smart_score': round(smart_score, 2),
                'dr_score': dr_score,
                'ml_move_prob': ml_move_prob,
                'p_score': p_score,
                'was_flipped': _sniper_was_flipped,
                # === FULL ML DATA for trade record ===
                'ml_data': {
                    'smart_score': round(smart_score, 2),
                    'p_score': p_score,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'ml_confidence': ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': ml.get('ml_signal', 'UNKNOWN'),
                        'move_prob': ml.get('ml_move_prob', 0),
                        'prob_up': ml.get('ml_prob_up', 0),
                        'prob_down': ml.get('ml_prob_down', 0),
                        'prob_flat': ml.get('ml_prob_flat', 0),
                        'direction_bias': ml.get('ml_direction_bias', 0),
                        'confidence': ml.get('ml_confidence', 0),
                        'score_boost': ml.get('ml_score_boost', 0),
                        'direction_hint': ml.get('ml_direction_hint', 'NEUTRAL'),
                        'model_type': ml.get('ml_model_type', 'unknown'),
                        'sizing_factor': ml.get('ml_sizing_factor', 1.0),
                    },
                    'gmm_model': {
                        'down_risk_score': dr_score,
                        'down_risk_flag': ml.get('ml_down_risk_flag', False),
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', None),
                        'gmm_action': 'BOOST',
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': _xgb_disagrees,
                },
            })
        
        # --- SNIPER DIAGNOSTIC SUMMARY ---
        _total_syms = len(pre_scores)
        _passed = len(candidates)
        _reject_str = ', '.join(f"{k}={v}" for k, v in _sniper_reject_counts.items() if v > 0)
        print(f"   🎯 SNIPER RESULT: {_passed} candidates from {_total_syms} symbols | Rejected: {_reject_str or 'none'}")
        if candidates:
            for i, c in enumerate(candidates[:5]):
                print(f"      #{i+1} {c['sym_clean']} {c['direction']} | dr={c['dr_score']:.4f} smart={c['smart_score']:.1f} gate={c['ml_move_prob']:.2f} | {c['sector']}")
        if _sniper_near_misses and not candidates:
            print(f"      Near misses: {' | '.join(_sniper_near_misses[:6])}")
        
        if not candidates:
            return None
        
        # Pick the ONE with lowest dr_score (most confident CLEAN signal)
        # Tiebreak by highest smart_score
        candidates.sort(key=lambda c: (c['dr_score'], -c['smart_score']))
        pick = candidates[0]
        
        print(f"\n   🎯 GMM SNIPER: {pick['sym_clean']} ({pick['direction']}) "
              f"| dr={pick['dr_score']:.4f} | smart={pick['smart_score']:.1f} "
              f"| gate={pick['ml_move_prob']:.2f} | {pick['sector']} "
              f"| {lot_multiplier}x lots")
        if len(candidates) > 1:
            runner_up = candidates[1]
            print(f"      Runner-up: {runner_up['sym_clean']} (dr={runner_up['dr_score']:.4f}, smart={runner_up['smart_score']:.1f})")
        
        # Place with lot_multiplier
        _sniper_setup = 'ML_OVERRIDE_WGMM' if pick.get('was_flipped') else 'GMM_SNIPER'
        # ML_OVERRIDE_WGMM gates already checked before flip in candidate building
        
        try:
            result = self.tools.place_option_order(
                underlying=pick['sym'],
                direction=pick['direction'],
                strike_selection="ATM",
                use_intraday_scoring=False,
                lot_multiplier=lot_multiplier,
                rationale=(f"{_sniper_setup}: dr={pick['dr_score']:.4f}, smart={pick['smart_score']:.1f}, "
                          f"gate={pick['ml_move_prob']:.2f}, sector={pick['sector']}, "
                          f"lots={lot_multiplier}x"),
                setup_type=_sniper_setup,
                ml_data=pick.get('ml_data', {}),
                sector=pick.get('sector', ''),
            )
            
            if result and result.get('success'):
                self._gmm_sniper_trades_today += 1
                self._gmm_sniper_symbols.add(pick['sym'])
                print(f"   ✅ GMM SNIPER PLACED: {pick['sym_clean']} ({pick['direction']}) "
                      f"[{lot_multiplier}x lots] dr={pick['dr_score']:.4f}")
                
                self._log_decision(cycle_time, pick['sym'], pick['p_score'], 'GMM_SNIPER_PLACED',
                                  reason=(f"Sniper: dr={pick['dr_score']:.4f}, smart={pick['smart_score']:.1f}, "
                                         f"gate={pick['ml_move_prob']:.2f}, lots={lot_multiplier}x"),
                                  direction=pick['direction'], setup='GMM_SNIPER')
                
                remaining = max_per_day - self._gmm_sniper_trades_today
                print(f"   📊 GMM SNIPER: {self._gmm_sniper_trades_today}/{max_per_day} today | {remaining} remaining")
                return pick['sym_clean']
            else:
                error = result.get('error', 'unknown') if result else 'no result'
                print(f"   ⚠️ GMM Sniper failed for {pick['sym_clean']}: {error}")
        except Exception as e:
            print(f"   ❌ GMM Sniper error for {pick['sym_clean']}: {e}")
        
        return None

    # ========== ADAPTIVE SCAN INTERVAL ==========
    def _adapt_scan_interval(self, pre_scores: dict):
        """Adjust the next scan interval based on current signal quality."""
        cfg = self._adaptive_scan_cfg
        if not cfg.get('enabled', False):
            return
        
        # Already in early session mode — don't override
        if not getattr(self, '_switched_to_normal', True):
            return
        
        fast_trigger = cfg.get('fast_trigger_signals', 3)
        slow_trigger = cfg.get('slow_trigger_signals', 0)
        
        hot_count = sum(1 for s in pre_scores.values() if s >= 65)
        warm_count = sum(1 for s in pre_scores.values() if s >= 55)
        
        if hot_count >= fast_trigger:
            new_interval = cfg.get('fast_interval_minutes', 3)
            quality = 'fast'
        elif warm_count <= slow_trigger:
            new_interval = cfg.get('slow_interval_minutes', 7)
            quality = 'slow'
        else:
            new_interval = cfg.get('normal_interval_minutes', 5)
            quality = 'normal'
        
        # Only reschedule if quality changed
        if quality != self._last_signal_quality:
            self._last_signal_quality = quality
            self._normal_interval = new_interval
            try:
                import schedule
                schedule.clear()
                schedule.every(new_interval).minutes.do(self.scan_and_trade)
                _icons = {'fast': '🔥', 'normal': '⏱️', 'slow': '💤'}
                print(f"\n   {_icons.get(quality, '⏱️')} ADAPTIVE SCAN: Switched to {new_interval}min interval ({quality.upper()}) — {hot_count} hot signals, {warm_count} warm signals")
            except Exception:
                pass
    
    # ========== LOG FULL CYCLE DECISIONS ==========
    def _log_cycle_decisions(self, cycle_time: str, pre_scores: dict, 
                             fno_opportunities: list, auto_fired: list,
                             sorted_data: list, market_data: dict):
        """Log all scored stocks from this cycle to the decision log."""
        if not self._decision_log_cfg.get('enabled', False):
            return
        
        # Build set of symbols that became F&O opportunities
        fno_syms = set()
        for opp in fno_opportunities:
            import re
            m = re.search(r'underlying="(NSE:\w+)"', opp)
            if m:
                fno_syms.add(m.group(1))
        
        for sym, score in sorted(pre_scores.items(), key=lambda x: x[1], reverse=True):
            if sym in auto_fired:
                continue  # Already logged during auto-fire
            
            data = market_data.get(sym, {})
            chg = data.get('change_pct', 0) if isinstance(data, dict) else 0
            
            if sym in fno_syms:
                outcome = 'FNO_OPPORTUNITY'
            elif score >= 49:
                outcome = 'SCORED_PASS'
            elif score >= 40:
                outcome = 'SCORED_MARGINAL'
            else:
                outcome = 'SCORED_LOW'
            
            self._log_decision(cycle_time, sym, score, outcome,
                              reason=f'chg={chg:+.2f}%',
                              extra={'change_pct': round(chg, 2)})
    
    def _save_orb_state(self):
        """Persist ORB trade tracking to disk for restart safety"""
        try:
            state = {
                'date': str(self.orb_tracking_date),
                'trades': self.orb_trades_today
            }
            atomic_json_save(self._orb_state_file, state, indent=0)
        except Exception as e:
            print(f"⚠️ Failed to save ORB state: {e}")

    def _load_orb_state(self):
        """Load persisted ORB state; returns empty if file missing or stale date"""
        today = datetime.now().date()
        try:
            if os.path.exists(self._orb_state_file):
                with open(self._orb_state_file, 'r') as f:
                    state = json.load(f)
                saved_date = state.get('date', '')
                if saved_date == str(today):
                    trades = state.get('trades', {})
                    count = sum(1 for s in trades.values() for d, v in s.items() if v)
                    if count:
                        print(f"📊 ORB state restored: {count} direction(s) already traded today")
                    return trades, today
        except Exception as e:
            print(f"⚠️ Failed to load ORB state: {e}")
        return {}, today

    def _reset_orb_tracker_if_new_day(self):
        """Reset ORB tracker at start of new trading day"""
        today = datetime.now().date()
        if today != self.orb_tracking_date:
            self.orb_trades_today = {}
            self.orb_tracking_date = today
            self._save_orb_state()
            print(f"📅 New trading day - ORB tracker reset")
    
    def _is_orb_trade_allowed(self, symbol: str, direction: str) -> bool:
        """Check if ORB trade is allowed (once per direction per symbol per day)"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        return not self.orb_trades_today[symbol].get(direction, False)
    
    def _sync_exit_manager_with_positions(self):
        """
        Sync exit manager with paper_positions on startup.
        If exit_manager_state.json is missing/empty but we have active positions,
        re-register them so exit logic (breakeven, trailing SL, time stop) works.
        """
        active = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        active_symbols = {t.get('symbol', '') for t in active}
        
        # Clean ghost positions from exit manager (in state file but not in paper_positions)
        ghost_symbols = [sym for sym in list(self.exit_manager.trade_states.keys()) if sym not in active_symbols]
        for sym in ghost_symbols:
            self.exit_manager.remove_trade(sym)
            print(f"🧹 Exit Manager: Removed ghost position {sym} (not in active trades)")
        
        if not active:
            return
        
        already_tracked = set(self.exit_manager.trade_states.keys())
        registered = 0
        
        for trade in active:
            # Skip IC trades — they have their own dedicated monitoring
            if trade.get('is_iron_condor', False):
                continue
            symbol = trade.get('symbol', '')
            if symbol in already_tracked:
                # Estimate candle count from entry time if still 0
                state = self.exit_manager.trade_states[symbol]
                if state.candles_since_entry == 0 and state.entry_time:
                    elapsed_min = (datetime.now() - state.entry_time).total_seconds() / 60
                    estimated_candles = max(0, int(elapsed_min / 5))  # 5-min candles
                    state.candles_since_entry = estimated_candles
                continue  # Already restored from state file
            
            entry = trade.get('avg_price', trade.get('entry_price', 0))
            sl = trade.get('stop_loss', 0)
            target = trade.get('target', 0)
            qty = trade.get('quantity', 0)
            side = trade.get('side', 'BUY')
            
            if entry > 0 and sl > 0 and target > 0:
                self.exit_manager.register_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    stop_loss=sl,
                    target=target,
                    quantity=qty
                )
                # Mark credit spread fields on the trade state
                if trade.get('is_credit_spread'):
                    state = self.exit_manager.trade_states.get(symbol)
                    if state:
                        state.is_credit_spread = True
                        state.net_credit = trade.get('net_credit', 0)
                        state.spread_width = trade.get('spread_width', 0)
                # Mark debit spread fields on the trade state
                if trade.get('is_debit_spread'):
                    state = self.exit_manager.trade_states.get(symbol)
                    if state:
                        state.is_debit_spread = True
                        state.net_debit = trade.get('net_debit', 0)
                        state.spread_width = trade.get('spread_width', 0)
                        # THP: sync hedged_from_tie flag for extended time stop
                        if trade.get('hedged_from_tie', False):
                            state.hedged_from_tie = True
                # Populate Greeks/Expiry fields for dynamic SL/target
                if trade.get('is_option'):
                    state = self.exit_manager.trade_states.get(symbol)
                    if state:
                        state.strike = trade.get('strike', 0)
                        state.option_type_str = trade.get('option_type', '')
                        state.expiry_str = trade.get('expiry', '')
                        state.underlying_symbol = trade.get('underlying', '')
                        state.last_delta = trade.get('delta', 0)
                        state.score_tier = trade.get('score_tier', 'standard')
                # Estimate candles from trade entry timestamp
                ts = trade.get('timestamp', '')
                if ts:
                    try:
                        entry_time = datetime.fromisoformat(ts)
                        elapsed_min = (datetime.now() - entry_time).total_seconds() / 60
                        estimated_candles = max(0, int(elapsed_min / 5))
                        state = self.exit_manager.trade_states.get(symbol)
                        if state:
                            state.candles_since_entry = estimated_candles
                            state.entry_time = entry_time
                    except Exception:
                        pass
                registered += 1
        
        if registered:
            print(f"📊 Exit Manager: Synced {registered} existing positions for exit management")
    
    def _mark_orb_trade_taken(self, symbol: str, direction: str):
        """Mark ORB direction as used for symbol today"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        self.orb_trades_today[symbol][direction] = True
        self._save_orb_state()
        print(f"📊 ORB {direction} marked as taken for {symbol} today")
    
    def start_realtime_monitor(self):
        """Start the real-time position monitor in a separate thread"""
        if self.monitor_running:
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._realtime_monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("⚡ Real-time monitor started")
    
    def stop_realtime_monitor(self):
        """Stop the real-time monitor"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("⚡ Real-time monitor stopped")
    
    def _proactive_loss_hedge_check(self):
        """Proactive loss-based hedge: every 60s check open naked options.
        If any is down >= loss_trigger_pct, convert to debit spread immediately.
        Runs in the realtime monitor thread (not tied to scan cycle).
        """
        from config import PROACTIVE_HEDGE_CONFIG as _plh_cfg
        if not _plh_cfg.get('enabled', False):
            return

        loss_trigger = _plh_cfg.get('loss_trigger_pct', 8)
        max_loss = _plh_cfg.get('max_hedge_loss_pct', 20)
        cooldown = _plh_cfg.get('cooldown_seconds', 300)
        log_checks = _plh_cfg.get('log_checks', False)

        # Snapshot open naked options
        with self.tools._positions_lock:
            all_open = [t.copy() for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        naked_opts = [
            t for t in all_open
            if t.get('is_option', False)
            and not t.get('is_debit_spread', False)
            and not t.get('is_credit_spread', False)
            and not t.get('is_iron_condor', False)
            and not t.get('hedged_from_tie', False)
        ]
        if not naked_opts:
            return

        # Cooldown tracking (per-underlying)
        if not hasattr(self, '_proactive_hedge_cooldowns'):
            self._proactive_hedge_cooldowns = {}
        _now_ts = time.time()

        # Fetch LTPs
        symbols = [t['symbol'] for t in naked_opts]
        try:
            if self.tools.ticker and self.tools.ticker.connected:
                ltp_data = self.tools.ticker.get_ltp_batch(symbols)
                quotes = {sym: ltp for sym, ltp in ltp_data.items()}
                missing = [s for s in symbols if s not in quotes]
                if missing:
                    rest = self.tools.kite.ltp(missing)
                    for s, v in rest.items():
                        quotes[s] = v.get('last_price', 0) if isinstance(v, dict) else 0
            else:
                raw = self.tools.kite.ltp(symbols)
                quotes = {s: v.get('last_price', 0) if isinstance(v, dict) else 0 for s, v in raw.items()}
        except Exception as e:
            if log_checks:
                print(f"   [PLH] Price fetch error: {e}")
            return

        hedged_this_cycle = []
        for trade in naked_opts:
            symbol = trade['symbol']
            entry = trade.get('avg_price', 0)
            ltp = quotes.get(symbol, 0)
            if entry <= 0 or ltp <= 0:
                continue

            loss_pct = ((entry - ltp) / entry) * 100  # +ve = loss
            underlying = trade.get('underlying', '')

            # Cooldown check
            if underlying in self._proactive_hedge_cooldowns:
                if _now_ts - self._proactive_hedge_cooldowns[underlying] < cooldown:
                    continue

            if loss_pct >= loss_trigger:
                if loss_pct > max_loss:
                    print(f"\n   [PLH] {symbol} loss {loss_pct:.1f}% > {max_loss}% cap -- too deep, skipping")
                    continue

                print(f"\n🛡️ PROACTIVE HEDGE: {symbol} down {loss_pct:.1f}% (trigger {loss_trigger}%) -- converting to spread")
                try:
                    hedge_result = self.tools.convert_naked_to_spread(trade, tie_check="PROACTIVE_LOSS_HEDGE")
                    if hedge_result.get('success'):
                        # Update ExitManager
                        em_state = self.exit_manager.get_trade_state(symbol)
                        if em_state:
                            new_symbol = hedge_result['symbol']
                            em_state.symbol = new_symbol
                            em_state.is_debit_spread = True
                            em_state.hedged_from_tie = True
                            em_state.net_debit = hedge_result['net_debit']
                            em_state.spread_width = hedge_result['spread_width']
                            em_state.current_sl = hedge_result['hedged_sl']
                            em_state.target = hedge_result['hedged_target']
                            em_state.initial_sl = hedge_result['hedged_sl']
                            em_state.highest_price = hedge_result['net_debit']
                            em_state.trailing_active = False
                            em_state.breakeven_applied = False
                            em_state.candles_since_entry = 0  # Fresh window
                            if new_symbol != symbol:
                                self.exit_manager.trade_states[new_symbol] = em_state
                                if symbol in self.exit_manager.trade_states:
                                    del self.exit_manager.trade_states[symbol]
                            self.exit_manager._persist_state()
                        print(f"   ✅ PLH: {symbol} hedged -> {hedge_result['symbol']}")
                        print(f"      Sell: {hedge_result['sell_symbol']} @ Rs{hedge_result['sell_premium']:.2f}")
                        print(f"      Net debit: Rs{hedge_result['net_debit']:.2f} | Width: {hedge_result['spread_width']}")
                        hedged_this_cycle.append(symbol)
                        # Set cooldown for this underlying
                        self._proactive_hedge_cooldowns[underlying] = _now_ts
                    else:
                        print(f"   ⚠️ PLH hedge failed: {hedge_result.get('error', 'unknown')}")
                except Exception as e:
                    print(f"   ⚠️ PLH exception: {e}")
            elif log_checks and loss_pct > 3:  # Log positions approaching trigger
                print(f"   [PLH] {symbol} loss {loss_pct:.1f}% (trigger at {loss_trigger}%)")

        if hedged_this_cycle:
            print(f"\n🛡️ PROACTIVE HEDGE CYCLE: converted {len(hedged_this_cycle)} positions")

    def _realtime_monitor_loop(self):
        """Continuous loop that checks positions every few seconds"""
        candle_timer = 0  # Track time for candle increment
        _proactive_hedge_timer = 0  # Track time for proactive hedge check
        while self.monitor_running:
            try:
                if self.is_trading_hours():
                    self._check_positions_realtime()
                    self._check_eod_exit()  # Check if need to exit before close
                    
                    # === PROACTIVE LOSS HEDGE (every 60s) ===
                    _proactive_hedge_timer += self.monitor_interval
                    from config import PROACTIVE_HEDGE_CONFIG as _plh_interval_cfg
                    _plh_check_s = _plh_interval_cfg.get('check_interval_seconds', 60)
                    if _proactive_hedge_timer >= _plh_check_s:
                        _proactive_hedge_timer = 0
                        try:
                            self._proactive_loss_hedge_check()
                        except Exception as _plh_err:
                            print(f"   ⚠️ Proactive hedge check error: {_plh_err}")
                    
                    # Increment candle counter every ~5 minutes (300s / monitor_interval)
                    candle_timer += self.monitor_interval
                    if candle_timer >= 300:  # 5 minutes = 1 candle
                        candle_timer = 0
                        for state in self.exit_manager.get_all_states():
                            self.exit_manager.increment_candles(state.symbol)
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _check_eod_exit(self):
        """Exit all positions before market close (3:30 PM)"""
        now = datetime.now().time()
        eod_exit_time = datetime.strptime("15:22", "%H:%M").time()  # Exit 8 mins before 3:30
        
        if now >= eod_exit_time:
            with self.tools._positions_lock:
                active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            if active_trades:
                print(f"\n⏰ END OF DAY - Closing all positions...")
                
                # Separate trade types
                regular_trades = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False)]
                spread_trades_eod = [t for t in active_trades if t.get('is_credit_spread', False)]
                debit_spread_trades_eod = [t for t in active_trades if t.get('is_debit_spread', False)]
                ic_trades_eod = [t for t in active_trades if t.get('is_iron_condor', False)]
                
                # Collect all symbols needed for quotes
                all_symbols = [t['symbol'] for t in regular_trades]
                for st in spread_trades_eod:
                    sold_sym = st.get('sold_symbol', '')
                    hedge_sym = st.get('hedge_symbol', '')
                    if sold_sym: all_symbols.append(sold_sym)
                    if hedge_sym: all_symbols.append(hedge_sym)
                for dt in debit_spread_trades_eod:
                    buy_sym = dt.get('buy_symbol', '')
                    sell_sym = dt.get('sell_symbol', '')
                    if buy_sym: all_symbols.append(buy_sym)
                    if sell_sym: all_symbols.append(sell_sym)
                for ict in ic_trades_eod:
                    for leg in ['sold_ce_symbol', 'hedge_ce_symbol', 'sold_pe_symbol', 'hedge_pe_symbol']:
                        sym = ict.get(leg, '')
                        if sym: all_symbols.append(sym)
                
                if not all_symbols:
                    return
                    
                try:
                    # Use ticker cache for EOD pricing if available
                    if self.tools.ticker and self.tools.ticker.connected:
                        ltp_data = self.tools.ticker.get_ltp_batch(all_symbols)
                        quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                        missing = [s for s in all_symbols if s not in quotes]
                        if missing:
                            quotes.update(self.tools.kite.ltp(missing))
                    else:
                        quotes = self.tools.kite.ltp(all_symbols)
                except Exception:
                    return
                
                # --- Close regular trades ---
                for trade in regular_trades:
                    symbol = trade['symbol']
                    if symbol not in quotes:
                        continue
                    
                    ltp = quotes[symbol]['last_price']
                    entry = trade['avg_price']
                    qty = trade['quantity']
                    side = trade['side']
                    
                    if side == 'BUY':
                        pnl = (ltp - entry) * qty
                    else:
                        pnl = (entry - ltp) * qty
                    pnl -= calc_brokerage(entry, ltp, qty)
                    
                    print(f"   🚪 EOD EXIT: {symbol} @ ₹{ltp:.2f}")
                    print(f"      P&L: ₹{pnl:+,.2f}")
                    
                    # Grab exit manager state for this trade
                    eod_exit_detail = {'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'}
                    try:
                        em_st = self.exit_manager.get_trade_state(symbol)
                        if em_st:
                            eod_exit_detail['candles_held'] = em_st.candles_since_entry
                            eod_exit_detail['r_multiple_achieved'] = round(em_st.max_favorable_move, 3)
                            eod_exit_detail['max_favorable_excursion'] = round(em_st.highest_price, 2)
                    except Exception:
                        pass
                    
                    self.tools.update_trade_status(symbol, 'EOD_EXIT', ltp, pnl, exit_detail=eod_exit_detail)
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close credit spreads (both legs) ---
                for trade in spread_trades_eod:
                    spread_id = trade.get('spread_id', trade['symbol'])
                    sold_sym = trade.get('sold_symbol', '')
                    hedge_sym = trade.get('hedge_symbol', '')
                    qty = trade['quantity']
                    net_credit = trade.get('net_credit', 0)
                    
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp  # Cost to close the spread
                    pnl = (net_credit - current_debit) * qty
                    pnl -= calc_brokerage(net_credit, current_debit, qty)
                    
                    print(f"   🚪 EOD EXIT SPREAD: {spread_id}")
                    print(f"      Sold leg: {sold_sym} @ ₹{sold_ltp:.2f} | Hedge: {hedge_sym} @ ₹{hedge_ltp:.2f}")
                    print(f"      Credit: ₹{net_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_debit, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close debit spreads (both legs) ---
                for trade in debit_spread_trades_eod:
                    spread_id = trade.get('spread_id', trade['symbol'])
                    buy_sym = trade.get('buy_symbol', '')
                    sell_sym = trade.get('sell_symbol', '')
                    qty = trade['quantity']
                    net_debit = trade.get('net_debit', 0)
                    
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    pnl = (current_value - net_debit) * qty
                    pnl -= calc_brokerage(net_debit, current_value, qty)
                    
                    print(f"   🚪 EOD EXIT DEBIT SPREAD: {spread_id}")
                    print(f"      Buy leg: {buy_sym} @ ₹{buy_ltp:.2f} | Sell: {sell_sym} @ ₹{sell_ltp:.2f}")
                    print(f"      Debit: ₹{net_debit:.2f} → Value: ₹{current_value:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_value, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                
                # --- Close Iron Condors (all 4 legs) ---
                for trade in ic_trades_eod:
                    condor_id = trade.get('condor_id', trade['symbol'])
                    sold_ce_sym = trade.get('sold_ce_symbol', '')
                    hedge_ce_sym = trade.get('hedge_ce_symbol', '')
                    sold_pe_sym = trade.get('sold_pe_symbol', '')
                    hedge_pe_sym = trade.get('hedge_pe_symbol', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    
                    sold_ce_ltp = quotes.get(sold_ce_sym, {}).get('last_price', 0) if sold_ce_sym else 0
                    hedge_ce_ltp = quotes.get(hedge_ce_sym, {}).get('last_price', 0) if hedge_ce_sym else 0
                    sold_pe_ltp = quotes.get(sold_pe_sym, {}).get('last_price', 0) if sold_pe_sym else 0
                    hedge_pe_ltp = quotes.get(hedge_pe_sym, {}).get('last_price', 0) if hedge_pe_sym else 0
                    
                    # Current debit to close all 4 legs
                    current_debit = (sold_ce_ltp - hedge_ce_ltp) + (sold_pe_ltp - hedge_pe_ltp)
                    pnl = (total_credit - current_debit) * qty
                    pnl -= calc_brokerage(total_credit, current_debit, qty)
                    
                    print(f"   🚪 EOD EXIT IRON CONDOR: {condor_id}")
                    print(f"      CE wing: Sold ₹{sold_ce_ltp:.2f} / Hedge ₹{hedge_ce_ltp:.2f}")
                    print(f"      PE wing: Sold ₹{sold_pe_ltp:.2f} / Hedge ₹{hedge_pe_ltp:.2f}")
                    print(f"      Credit: ₹{total_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    self.tools.update_trade_status(trade['symbol'], 'EOD_EXIT', current_debit, pnl,
                                                   exit_detail={'exit_reason': 'EOD_AUTO_CLOSE', 'exit_type': 'EOD_EXIT', 'strategy': 'IRON_CONDOR'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    
                    # Record EOD IC close with risk governor
                    open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != trade['symbol']]
                    unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                    self.risk_governor.record_trade_result(trade['symbol'], pnl, pnl > 0, unrealized_pnl=unrealized)
                    self.risk_governor.update_capital(self.capital)
    
    def _check_positions_realtime(self):
        """Check all positions for target/stoploss hits using Exit Manager"""
        # Snapshot under lock to prevent race with trading thread
        with self.tools._positions_lock:
            active_trades = [t.copy() for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # === SYNC: Register any new trades the agent placed since last check ===
        self._sync_exit_manager_with_positions()
        
        # Separate trade types
        equity_trades = [t for t in active_trades if not t.get('is_option', False)]
        option_trades = [t for t in active_trades if t.get('is_option', False) and not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False)]
        spread_trades = [t for t in active_trades if t.get('is_credit_spread', False)]
        debit_spread_trades = [t for t in active_trades if t.get('is_debit_spread', False)]
        ic_trades = [t for t in active_trades if t.get('is_iron_condor', False)]
        
        # === GET CURRENT PRICES ===
        # For spreads (credit & debit), we need both leg symbols
        # For iron condors, we need all 4 leg symbols
        # For TIE: also fetch underlying symbols for option trades
        all_symbols = set()
        _option_underlying_map = {}  # option_symbol → underlying_symbol (for TIE)
        for t in active_trades:
            if t.get('is_iron_condor'):
                all_symbols.add(t.get('sold_ce_symbol', ''))
                all_symbols.add(t.get('hedge_ce_symbol', ''))
                all_symbols.add(t.get('sold_pe_symbol', ''))
                all_symbols.add(t.get('hedge_pe_symbol', ''))
            elif t.get('is_credit_spread'):
                all_symbols.add(t.get('sold_symbol', ''))
                all_symbols.add(t.get('hedge_symbol', ''))
            elif t.get('is_debit_spread'):
                all_symbols.add(t.get('buy_symbol', ''))
                all_symbols.add(t.get('sell_symbol', ''))
            else:
                all_symbols.add(t['symbol'])
            # TIE: collect underlying symbols for option positions
            ul = t.get('underlying', '')
            if ul and t.get('is_option'):
                all_symbols.add(ul)
                _option_underlying_map[t['symbol']] = ul
        all_symbols.discard('')
        all_symbols = list(all_symbols)
        if all_symbols:
            try:
                # Use WebSocket cache first (zero API calls), fallback to REST
                if self.tools.ticker and self.tools.ticker.connected:
                    ltp_data = self.tools.ticker.get_ltp_batch(all_symbols)
                    quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                    # For any symbols not in cache, fall back to REST
                    missing = [s for s in all_symbols if s not in quotes]
                    if missing:
                        rest_data = self.tools.kite.ltp(missing)
                        quotes.update(rest_data)
                else:
                    quotes = self.tools.kite.ltp(all_symbols)
            except Exception as e:
                # If mixed exchange query fails, try separately
                quotes = {}
                if equity_trades:
                    try:
                        eq_syms = [t['symbol'] for t in equity_trades]
                        quotes.update(self.tools.kite.ltp(eq_syms))
                    except Exception:
                        pass
                if option_trades:
                    try:
                        opt_syms = [t['symbol'] for t in option_trades]
                        quotes.update(self.tools.kite.ltp(opt_syms))
                    except Exception:
                        pass
        else:
            quotes = {}
        
        # Print position status every 30 seconds (every 10th check)
        if not hasattr(self, '_monitor_count'):
            self._monitor_count = 0
        self._monitor_count += 1
        
        # Suppress dashboard while scan is running to reduce noise
        if getattr(self, '_scanning', False):
            show_status = False
        else:
            show_status = (self._monitor_count % 10 == 0)  # Every 30 seconds
        
        # === CHECK OPTION EXITS (Greeks-based) ===
        # Only check naked option trades — credit spreads use exit_manager
        if option_trades:
            try:
                option_exits = self.tools.check_option_exits()
                for exit_signal in option_exits:
                    symbol = exit_signal['symbol']
                    # Skip credit spread symbols (contain '|')
                    if '|' in symbol:
                        continue
                    # Only act on signals that actually require exit
                    if not exit_signal.get('should_exit', False):
                        continue
                    reason = exit_signal.get('reason', 'Greeks exit')
                    
                    # Find the option trade
                    opt_trade = next((t for t in option_trades if t['symbol'] == symbol), None)
                    if opt_trade:
                        # Guard: re-check status to prevent double-close from main thread
                        if opt_trade.get('status', 'OPEN') != 'OPEN':
                            continue
                        entry = opt_trade['avg_price']
                        qty = opt_trade['quantity']
                        # Try to get current price
                        try:
                            opt_quote = self.tools.kite.ltp([symbol])
                            ltp = opt_quote[symbol]['last_price']
                        except Exception:
                            # WARN: Quote failure — skip exit to avoid masking real P&L
                            print(f"   ⚠️ Quote failure for {symbol} — skipping option exit (will retry)")
                            continue
                        
                        pnl = (ltp - entry) * qty
                        pnl -= calc_brokerage(entry, ltp, qty)
                        print(f"\n📊 OPTION EXIT: {symbol}")
                        print(f"   Reason: {reason}")
                        print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                        print(f"   P&L: ₹{pnl:+,.0f}")
                        
                        self.tools.update_trade_status(symbol, exit_signal.get('exit_type', 'GREEKS_EXIT'), ltp, pnl)
                        with self._pnl_lock:
                            self.daily_pnl += pnl
                            self.capital += pnl
                        
                        # Record with Risk Governor (include unrealized P&L from open positions)
                        open_pos = [t for t in active_trades if t.get('symbol') != symbol]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol, pnl, pnl > 0, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                        
                        # Notify scorer for per-symbol re-entry prevention
                        try:
                            from options_trader import get_intraday_scorer
                            scorer = get_intraday_scorer()
                            if pnl > 0:
                                scorer.record_symbol_win(symbol)
                            else:
                                scorer.record_symbol_loss(symbol)
                        except Exception:
                            pass
            except Exception as e:
                if show_status:
                    print(f"   ⚠️ Option exit check error: {e}")
        
        # === EXIT MANAGER INTEGRATION ===
        # Check all trades via exit manager (equity AND options AND spreads)
        # Filter out zero-price entries (quote failures) to prevent phantom SL triggers
        price_dict = {}
        for t in active_trades:
            if t.get('is_credit_spread'):
                # For credit spreads: compute current net debit (cost to close)
                sold_sym = t.get('sold_symbol', '')
                hedge_sym = t.get('hedge_symbol', '')
                sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0)
                hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0)
                if sold_ltp > 0 and hedge_ltp > 0:
                    # Current debit to close = sold_ltp - hedge_ltp (buy back sold, sell hedge)
                    current_debit = sold_ltp - hedge_ltp
                    price_dict[t['symbol']] = max(0, current_debit)  # Debit can't be negative
            elif t.get('is_debit_spread'):
                # For debit spreads: compute current spread value
                buy_sym = t.get('buy_symbol', '')
                sell_sym = t.get('sell_symbol', '')
                buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0)
                sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0)
                if buy_ltp > 0 and sell_ltp > 0:
                    # Current value = buy_ltp - sell_ltp (what we'd receive if closing)
                    current_value = buy_ltp - sell_ltp
                    price_dict[t['symbol']] = max(0, current_value)
                    
                    # ═══════════════════════════════════════════════════════
                    # HEDGE UNWIND PROTOCOL — Restore full upside on recovery
                    # If THP-hedged spread's buy leg recovers past entry price,
                    # buy back the sold leg so upside is no longer capped.
                    # ═══════════════════════════════════════════════════════
                    if t.get('hedged_from_tie', False):
                        from config import THESIS_HEDGE_CONFIG as _unwind_cfg
                        _unwind_enabled = _unwind_cfg.get('unwind_enabled', False)
                        _recovery_pct = _unwind_cfg.get('unwind_buy_leg_recovery_pct', 100)
                        _min_profit = _unwind_cfg.get('unwind_min_profit_after_cost', 2)
                        _buy_entry = t.get('buy_premium', 0)
                        
                        if _unwind_enabled and _buy_entry > 0:
                            _recovery_threshold = _buy_entry * (_recovery_pct / 100)
                            # Check 1: Buy leg has recovered past threshold
                            # Check 2: After buying back sold leg, we still have net profit
                            _sell_entry = t.get('sell_premium', 0)
                            _hedge_leg_cost = sell_ltp - _sell_entry  # +ve = loss on hedge leg
                            _buy_leg_profit = buy_ltp - _buy_entry    # +ve = gain on buy leg
                            _net_after_unwind = _buy_leg_profit - max(0, _hedge_leg_cost)
                            
                            if buy_ltp >= _recovery_threshold and _net_after_unwind >= _min_profit:
                                print(f"\n🔓 HEDGE UNWIND: {t['symbol']} — buy leg ₹{buy_ltp:.2f} recovered past ₹{_recovery_threshold:.2f}")
                                print(f"   Buy leg P&L: ₹{_buy_leg_profit:+.2f} | Hedge leg cost: ₹{_hedge_leg_cost:+.2f} | Net: ₹{_net_after_unwind:+.2f}")
                                try:
                                    _old_symbol = t['symbol']
                                    unwind_result = self.tools.unwind_hedge(t, sell_ltp)
                                    if unwind_result.get('success'):
                                        _new_sym = unwind_result['symbol']
                                        # Update ExitManager: revert to naked option mode
                                        em_state = self.exit_manager.get_trade_state(_old_symbol)
                                        if em_state:
                                            em_state.symbol = _new_sym
                                            em_state.is_debit_spread = False
                                            em_state.hedged_from_tie = False
                                            em_state.entry_price = unwind_result['buy_entry_price']
                                            em_state.current_sl = unwind_result['buy_entry_price'] * 0.85
                                            em_state.initial_sl = unwind_result['buy_entry_price'] * 0.85
                                            em_state.target = unwind_result['buy_entry_price'] * 1.40
                                            em_state.highest_price = buy_ltp  # Current high
                                            em_state.trailing_active = False
                                            em_state.breakeven_applied = False
                                            em_state.net_debit = 0
                                            em_state.spread_width = 0
                                            # Move state to new key
                                            if _new_sym != _old_symbol:
                                                self.exit_manager.trade_states[_new_sym] = em_state
                                                if _old_symbol in self.exit_manager.trade_states:
                                                    del self.exit_manager.trade_states[_old_symbol]
                                            self.exit_manager._persist_state()
                                        # Update price_dict with naked option price
                                        price_dict[_new_sym] = buy_ltp
                                        if _old_symbol in price_dict:
                                            del price_dict[_old_symbol]
                                        print(f"   ✅ UNWIND DONE: {_new_sym} — naked option with full upside")
                                    else:
                                        print(f"   ⚠️ Unwind failed: {unwind_result.get('error', 'unknown')}")
                                except Exception as e:
                                    print(f"   ⚠️ Unwind exception: {e}")
            else:
                ltp_val = quotes.get(t['symbol'], {}).get('last_price', 0)
                if ltp_val and ltp_val > 0:
                    price_dict[t['symbol']] = ltp_val
        # Build underlying price map for TIE (option_symbol → underlying LTP)
        underlying_prices = {}
        for opt_sym, ul_sym in _option_underlying_map.items():
            ul_ltp = quotes.get(ul_sym, {}).get('last_price', 0)
            if ul_ltp > 0:
                underlying_prices[opt_sym] = ul_ltp
        exit_signals = self.exit_manager.check_all_exits(price_dict, underlying_prices=underlying_prices)
        
        # Process exit signals first (highest priority)
        for signal in exit_signals:
            if signal.should_exit:
                trade = next((t for t in active_trades if t['symbol'] == signal.symbol), None)
                if trade:
                    symbol = signal.symbol
                    # Guard: re-check status to prevent double-close from main thread
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    
                    # ═══════════════════════════════════════════════════════
                    # THESIS HEDGE PROTOCOL (THP) — INTERCEPT TIE SIGNALS
                    # If TIE fires a hedgeable check on a naked option,
                    # convert to debit spread instead of exiting.
                    # ═══════════════════════════════════════════════════════
                    if signal.exit_type.startswith('THESIS_INVALID_'):
                        tie_check_name = signal.exit_type.replace('THESIS_INVALID_', '')
                        is_naked_option = (
                            trade.get('is_option', False) and
                            not trade.get('is_debit_spread', False) and
                            not trade.get('is_credit_spread', False) and
                            not trade.get('hedged_from_tie', False)
                        )
                        if is_naked_option:
                            from thesis_validator import ThesisResult, should_hedge_instead_of_exit
                            from config import THESIS_HEDGE_CONFIG as _thp_tie_cfg
                            _tie_result = ThesisResult(
                                is_invalid=True,
                                check_name=tie_check_name,
                                reason=signal.reason,
                                details={},
                            )
                            # Unified loss cap: don't hedge if already too deep
                            _tie_ltp = price_dict.get(symbol, 0) or signal.exit_price
                            _tie_entry = trade.get('avg_price', 0)
                            _tie_loss_pct = ((_tie_entry - _tie_ltp) / _tie_entry * 100) if _tie_entry > 0 and _tie_ltp > 0 else 999
                            _max_hedge_loss = _thp_tie_cfg.get('max_hedge_loss_pct', 12)
                            if _tie_loss_pct > _max_hedge_loss:
                                print(f"\n📉 THP: {symbol} — loss {_tie_loss_pct:.1f}% > {_max_hedge_loss}% cap — too deep to hedge, exiting")
                            elif should_hedge_instead_of_exit(_tie_result):
                                print(f"\n🛡️ THP INTERCEPT: {symbol} — {tie_check_name} is hedgeable ({_tie_loss_pct:.1f}% loss), attempting spread conversion")
                                try:
                                    hedge_result = self.tools.convert_naked_to_spread(trade, tie_check=tie_check_name)
                                    if hedge_result.get('success'):
                                        # Hedge succeeded — update ExitManager state to debit spread mode
                                        em_state = self.exit_manager.get_trade_state(symbol)
                                        if em_state:
                                            new_symbol = hedge_result['symbol']
                                            # Re-key ExitManager state under new spread symbol
                                            em_state.symbol = new_symbol
                                            em_state.is_debit_spread = True
                                            em_state.hedged_from_tie = True
                                            em_state.net_debit = hedge_result['net_debit']
                                            em_state.spread_width = hedge_result['spread_width']
                                            em_state.current_sl = hedge_result['hedged_sl']
                                            em_state.target = hedge_result['hedged_target']
                                            em_state.initial_sl = hedge_result['hedged_sl']
                                            em_state.highest_price = hedge_result['net_debit']  # Reset high tracking
                                            em_state.trailing_active = False
                                            em_state.breakeven_applied = False
                                            # Move state to new key
                                            if new_symbol != symbol:
                                                self.exit_manager.trade_states[new_symbol] = em_state
                                                if symbol in self.exit_manager.trade_states:
                                                    del self.exit_manager.trade_states[symbol]
                                            self.exit_manager._persist_state()
                                        print(f"   ✅ THP: {symbol} hedged → {hedge_result['symbol']}")
                                        print(f"      Sell: {hedge_result['sell_symbol']} @ ₹{hedge_result['sell_premium']:.2f}")
                                        print(f"      Net debit: ₹{hedge_result['net_debit']:.2f} | Width: {hedge_result['spread_width']}")
                                        continue  # SKIP exit — position is now a hedged spread
                                    else:
                                        print(f"   ⚠️ THP hedge failed: {hedge_result.get('error', 'unknown')} — proceeding with exit")
                                except Exception as e:
                                    print(f"   ⚠️ THP exception: {e} — proceeding with exit")
                            else:
                                print(f"   🔴 THP: {tie_check_name} is non-hedgeable — immediate exit")
                    
                    # ═══════════════════════════════════════════════════════
                    # THP — INTERCEPT TIME_STOP ON NAKED OPTIONS
                    # Dead trade at candle 10 with moderate loss? Convert to
                    # spread so if momentum resumes (candle 11-20) we capture it.
                    # ═══════════════════════════════════════════════════════
                    if signal.exit_type == 'TIME_STOP':
                        from config import THESIS_HEDGE_CONFIG as _thp_cfg
                        _is_naked_opt = (
                            trade.get('is_option', False) and
                            not trade.get('is_debit_spread', False) and
                            not trade.get('is_credit_spread', False) and
                            not trade.get('hedged_from_tie', False)
                        )
                        _thp_enabled = _thp_cfg.get('enabled', False) and _thp_cfg.get('hedge_time_stop', False)
                        if _is_naked_opt and _thp_enabled:
                            # Check current loss is moderate enough to justify hedging
                            _ts_ltp = price_dict.get(symbol, 0) or signal.exit_price
                            _ts_entry = trade.get('avg_price', 0)
                            _ts_loss_pct = ((_ts_entry - _ts_ltp) / _ts_entry * 100) if _ts_entry > 0 and _ts_ltp > 0 else 999
                            _max_loss_for_hedge = _thp_cfg.get('max_hedge_loss_pct', 12)
                            if _ts_loss_pct <= _max_loss_for_hedge:
                                print(f"\n🛡️ THP TIME_STOP INTERCEPT: {symbol} — dead trade ({_ts_loss_pct:.1f}% loss), hedging instead of exiting")
                                try:
                                    hedge_result = self.tools.convert_naked_to_spread(trade, tie_check="TIME_STOP_HEDGE")
                                    if hedge_result.get('success'):
                                        em_state = self.exit_manager.get_trade_state(symbol)
                                        if em_state:
                                            new_symbol = hedge_result['symbol']
                                            em_state.symbol = new_symbol
                                            em_state.is_debit_spread = True
                                            em_state.hedged_from_tie = True
                                            em_state.net_debit = hedge_result['net_debit']
                                            em_state.spread_width = hedge_result['spread_width']
                                            em_state.current_sl = hedge_result['hedged_sl']
                                            em_state.target = hedge_result['hedged_target']
                                            em_state.initial_sl = hedge_result['hedged_sl']
                                            em_state.highest_price = hedge_result['net_debit']
                                            em_state.trailing_active = False
                                            em_state.breakeven_applied = False
                                            em_state.candles_since_entry = 0  # Reset candle count — fresh 20-candle window
                                            if new_symbol != symbol:
                                                self.exit_manager.trade_states[new_symbol] = em_state
                                                if symbol in self.exit_manager.trade_states:
                                                    del self.exit_manager.trade_states[symbol]
                                            self.exit_manager._persist_state()
                                        print(f"   ✅ THP: {symbol} rescued → {hedge_result['symbol']} (20 candle window)")
                                        print(f"      Sell: {hedge_result['sell_symbol']} @ ₹{hedge_result['sell_premium']:.2f}")
                                        print(f"      Net debit: ₹{hedge_result['net_debit']:.2f} | Width: {hedge_result['spread_width']}")
                                        continue  # SKIP exit — position hedged, fresh window
                                    else:
                                        print(f"   ⚠️ THP time_stop hedge failed: {hedge_result.get('error', 'unknown')} — exiting")
                                except Exception as e:
                                    print(f"   ⚠️ THP time_stop exception: {e} — exiting")
                            else:
                                print(f"   📉 THP: TIME_STOP loss {_ts_loss_pct:.1f}% > {_max_loss_for_hedge}% cap — too deep to hedge, exiting")
                    
                    # ═══════════════════════════════════════════════════════
                    # THP — INTERCEPT SL_HIT ON NAKED OPTIONS
                    # When hard SL fires on a naked option (e.g. 8-20% loss),
                    # convert to debit spread instead of closing outright.
                    # Caps max loss and gives the trade a second chance.
                    # ═══════════════════════════════════════════════════════
                    if signal.exit_type == 'SL_HIT':
                        from config import THESIS_HEDGE_CONFIG as _thp_sl_cfg
                        _is_naked_opt_sl = (
                            trade.get('is_option', False) and
                            not trade.get('is_debit_spread', False) and
                            not trade.get('is_credit_spread', False) and
                            not trade.get('hedged_from_tie', False)
                        )
                        _thp_sl_enabled = _thp_sl_cfg.get('enabled', False)
                        if _is_naked_opt_sl and _thp_sl_enabled:
                            _sl_ltp = price_dict.get(symbol, 0) or signal.exit_price
                            _sl_entry = trade.get('avg_price', 0)
                            _sl_loss_pct = ((_sl_entry - _sl_ltp) / _sl_entry * 100) if _sl_entry > 0 and _sl_ltp > 0 else 999
                            _max_sl_hedge_loss = _thp_sl_cfg.get('max_hedge_loss_pct', 20)
                            if _sl_loss_pct <= _max_sl_hedge_loss:
                                print(f"\n🛡️ THP SL_HIT INTERCEPT: {symbol} — SL hit ({_sl_loss_pct:.1f}% loss), converting to spread instead of exiting")
                                try:
                                    hedge_result = self.tools.convert_naked_to_spread(trade, tie_check="SL_HIT_HEDGE")
                                    if hedge_result.get('success'):
                                        em_state = self.exit_manager.get_trade_state(symbol)
                                        if em_state:
                                            new_symbol = hedge_result['symbol']
                                            em_state.symbol = new_symbol
                                            em_state.is_debit_spread = True
                                            em_state.hedged_from_tie = True
                                            em_state.net_debit = hedge_result['net_debit']
                                            em_state.spread_width = hedge_result['spread_width']
                                            em_state.current_sl = hedge_result['hedged_sl']
                                            em_state.target = hedge_result['hedged_target']
                                            em_state.initial_sl = hedge_result['hedged_sl']
                                            em_state.highest_price = hedge_result['net_debit']
                                            em_state.trailing_active = False
                                            em_state.breakeven_applied = False
                                            em_state.candles_since_entry = 0  # Reset candle count
                                            if new_symbol != symbol:
                                                self.exit_manager.trade_states[new_symbol] = em_state
                                                if symbol in self.exit_manager.trade_states:
                                                    del self.exit_manager.trade_states[symbol]
                                            self.exit_manager._persist_state()
                                        print(f"   ✅ THP: {symbol} hedged at SL → {hedge_result['symbol']}")
                                        print(f"      Sell: {hedge_result['sell_symbol']} @ ₹{hedge_result['sell_premium']:.2f}")
                                        print(f"      Net debit: ₹{hedge_result['net_debit']:.2f} | Width: {hedge_result['spread_width']}")
                                        print(f"      Max loss now capped | 20 candle window for recovery")
                                        continue  # SKIP exit — position is now a hedged spread
                                    else:
                                        print(f"   ⚠️ THP SL hedge failed: {hedge_result.get('error', 'unknown')} — proceeding with SL exit")
                                except Exception as e:
                                    print(f"   ⚠️ THP SL exception: {e} — proceeding with SL exit")
                            else:
                                print(f"   📉 THP: SL_HIT loss {_sl_loss_pct:.1f}% > {_max_sl_hedge_loss}% cap — too deep, exiting at SL")
                    
                    ltp = price_dict.get(symbol, 0) or signal.exit_price
                    # Guard: never exit at price 0 (quote failure)
                    if ltp <= 0:
                        print(f"   ⚠️ Skipping {symbol} exit — LTP is 0 (quote failure)")
                        continue
                    entry = trade['avg_price']
                    qty = trade['quantity']
                    side = trade['side']
                    
                    # === PARTIAL PROFIT EXIT ===
                    if signal.exit_type == "PARTIAL_PROFIT" and signal.partial_pct > 0:
                        # Get lot size to ensure we exit in whole lots
                        lot_size = trade.get('quantity', qty) // max(1, trade.get('lots', 1))  # shares per lot
                        total_lots = qty // lot_size if lot_size > 0 else 1
                        exit_lots = total_lots // 2  # Exit half the lots
                        exit_qty = exit_lots * lot_size
                        
                        if exit_qty <= 0 or exit_lots < 1:
                            continue
                        # Calculate P&L on partial quantity only
                        if side == 'BUY':
                            partial_pnl = (ltp - entry) * exit_qty
                        else:
                            partial_pnl = (entry - ltp) * exit_qty
                        partial_pnl -= calc_brokerage(entry, ltp, exit_qty)
                        
                        remaining_qty = qty - exit_qty
                        pnl_pct = partial_pnl / (entry * exit_qty) * 100
                        
                        print(f"\n💰 PARTIAL_PROFIT! {symbol}")
                        print(f"   Reason: {signal.reason}")
                        print(f"   Booked {exit_qty}/{qty} @ ₹{ltp:.2f}")
                        print(f"   Partial P&L: ₹{partial_pnl:+,.2f} ({pnl_pct:+.2f}%)")
                        print(f"   Remaining: {remaining_qty} qty, SL moved to entry ₹{entry:.2f}")
                        
                        # Update trade quantity to remaining
                        self.tools.partial_exit_trade(symbol, exit_qty, ltp, partial_pnl)
                        with self._pnl_lock:
                            self.daily_pnl += partial_pnl
                            self.capital += partial_pnl
                        
                        # Update exit manager state with reduced quantity
                        em_state = self.exit_manager.get_trade_state(symbol)
                        if em_state:
                            em_state.quantity = remaining_qty
                            self.exit_manager._persist_state()
                        
                        # Record partial win with risk governor
                        open_pos = [t for t in active_trades if t.get('symbol') != symbol]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol + "_PARTIAL", partial_pnl, True, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                        continue
                    
                    # === FULL EXIT ===
                    if trade.get('is_credit_spread'):
                        # Credit spread P&L = (net_credit - current_debit) × quantity
                        credit = trade.get('net_credit', 0)
                        current_debit = ltp  # For spreads, ltp stores current net debit
                        pnl = (credit - current_debit) * qty
                        pnl -= calc_brokerage(credit, current_debit, qty)
                        pnl_pct = (credit - current_debit) / credit * 100 if credit > 0 else 0
                    elif trade.get('is_debit_spread'):
                        # Debit spread P&L = (current_value - net_debit) × quantity
                        net_debit = trade.get('net_debit', 0)
                        current_value = ltp  # For debit spreads, ltp stores current spread value
                        pnl = (current_value - net_debit) * qty
                        pnl -= calc_brokerage(net_debit, current_value, qty)
                        pnl_pct = (current_value - net_debit) / net_debit * 100 if net_debit > 0 else 0
                    elif side == 'BUY':
                        pnl = (ltp - entry) * qty
                        pnl -= calc_brokerage(entry, ltp, qty)
                    else:
                        pnl = (entry - ltp) * qty
                        pnl -= calc_brokerage(entry, ltp, qty)
                    
                    pnl_pct = pnl / (entry * qty) * 100 if not trade.get('is_debit_spread') else pnl_pct
                    was_win = pnl > 0
                    
                    # Exit based on signal type
                    emoji = {
                        'SL_HIT': '❌',
                        'TARGET_HIT': '🎯',
                        'SESSION_CUTOFF': '⏰',
                        'TIME_STOP': '⏱️',
                        'TRAILING_SL': '📈',
                        'PARTIAL_PROFIT': '💰',
                        'OPTION_SPEED_GATE': '🚀',
                        'DEBIT_SPREAD_SL': '❌',
                        'DEBIT_SPREAD_TARGET': '🎯',
                        'DEBIT_SPREAD_TIME_EXIT': '⏰',
                        'DEBIT_SPREAD_TRAIL_SL': '📈',
                        'DEBIT_SPREAD_MAX_PROFIT': '💰',
                        'THESIS_INVALID_R_COLLAPSE': '🔴',
                        'THESIS_INVALID_NEVER_SHOWED_LIFE': '🔴',
                        'THESIS_INVALID_IV_CRUSH': '🔴',
                        'THESIS_INVALID_UNDERLYING_BOS': '🔴',
                        'THESIS_INVALID_MAX_PAIN_CEILING': '🔴',
                    }.get(signal.exit_type, '🚪')
                    
                    print(f"\n{emoji} {signal.exit_type}! {symbol}")
                    print(f"   Reason: {signal.reason}")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{ltp:.2f}")
                    print(f"   P&L: ₹{pnl:+,.2f} ({pnl_pct:+.2f}%)")
                    
                    # Grab exit manager state BEFORE removing trade (has candles, R-multiple, etc.)
                    exit_detail = {}
                    try:
                        em_state = self.exit_manager.get_trade_state(symbol)
                        if em_state:
                            exit_detail = {
                                'candles_held': em_state.candles_since_entry,
                                'r_multiple_achieved': round(em_state.max_favorable_move, 3),
                                'max_favorable_excursion': round(em_state.highest_price, 2) if em_state.highest_price else 0,
                                'exit_reason': signal.reason,
                                'exit_type': signal.exit_type,
                                'breakeven_applied': em_state.breakeven_applied,
                                'trailing_active': em_state.trailing_active,
                                'partial_booked': em_state.partial_booked,
                                'current_sl_at_exit': round(em_state.current_sl, 2),
                            }
                            # TIE: enrich exit_detail with thesis invalidation metadata
                            if signal.exit_type.startswith('THESIS_INVALID_'):
                                exit_detail['thesis_check'] = signal.exit_type.replace('THESIS_INVALID_', '')
                                exit_detail['thesis_reason'] = signal.reason
                                ul_sym = _option_underlying_map.get(symbol, '')
                                if ul_sym:
                                    exit_detail['underlying_ltp_at_exit'] = underlying_prices.get(symbol, 0)
                            print(f"   📋 Exit context: {em_state.candles_since_entry} candles | MaxR: {em_state.max_favorable_move:.2f} | BE: {em_state.breakeven_applied} | Trail: {em_state.trailing_active}")
                    except Exception as e:
                        print(f"   ⚠️ Could not capture exit detail: {e}")
                    
                    self.tools.update_trade_status(symbol, signal.exit_type, ltp, pnl, exit_detail=exit_detail)
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    
                    # Record with Risk Governor (include unrealized P&L from open positions)
                    open_pos = [t for t in active_trades if t.get('symbol') != symbol]
                    unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                    self.risk_governor.record_trade_result(symbol, pnl, was_win, unrealized_pnl=unrealized)
                    self.risk_governor.update_capital(self.capital)
                    
                    # Notify scorer for per-symbol re-entry prevention
                    try:
                        from options_trader import get_intraday_scorer
                        scorer = get_intraday_scorer()
                        if was_win:
                            scorer.record_symbol_win(symbol)
                        else:
                            scorer.record_symbol_loss(symbol)
                    except Exception:
                        pass
                    
                    # Record slippage if we have expected price
                    if hasattr(trade, 'expected_exit') and getattr(trade, 'expected_exit', None):
                        self.execution_guard.record_slippage(
                            symbol=symbol,
                            side=side,
                            expected_price=trade.expected_exit,
                            actual_price=ltp,
                            volume_regime=trade.get('volume_regime', 'NORMAL'),
                            order_type='MARKET'
                        )
                    
                    # Remove from exit manager
                    self.exit_manager.remove_trade(symbol)
                    continue  # Skip further processing for this trade
        
        # Update exit manager with current SL (sync back from trade state)
        for trade in active_trades:
            symbol = trade['symbol']
            state = self.exit_manager.get_trade_state(symbol)
            if state:
                # Sync trailing SL back to trade
                if state.current_sl != trade['stop_loss']:
                    old_sl = trade['stop_loss']
                    trade['stop_loss'] = state.current_sl
                    self.tools._save_active_trades()
                    
                    # LIVE MODE: Modify the SL order on the exchange
                    if not self.tools.paper_mode:
                        sl_order_id = trade.get('sl_order_id')
                        if sl_order_id and not str(sl_order_id).startswith('PAPER_'):
                            try:
                                new_trigger = state.current_sl * (0.999 if trade['side'] == 'BUY' else 1.001)
                                self.tools.kite.modify_order(
                                    variety=self.tools.kite.VARIETY_REGULAR,
                                    order_id=sl_order_id,
                                    trigger_price=round(new_trigger, 1)
                                )
                                print(f"   🔄 LIVE SL modified on exchange: {symbol} ₹{old_sl:.2f} → ₹{state.current_sl:.2f}")
                            except Exception as e:
                                print(f"   ⚠️ Failed to modify SL order on exchange for {symbol}: {e}")
                        elif sl_order_id is None and trade.get('is_live') and trade.get('is_option'):
                            # Option trade in LIVE mode without broker SL — place one now
                            try:
                                exchange, tradingsymbol = symbol.split(':')
                                sl_trigger = state.current_sl * 0.999
                                new_sl_oid = self.tools._place_order_autoslice(
                                    variety=self.tools.kite.VARIETY_REGULAR,
                                    exchange=exchange,
                                    tradingsymbol=tradingsymbol,
                                    transaction_type=self.tools.kite.TRANSACTION_TYPE_SELL,
                                    quantity=trade.get('quantity', 0),
                                    product=self.tools.kite.PRODUCT_MIS,
                                    order_type=self.tools.kite.ORDER_TYPE_SLM,
                                    trigger_price=round(sl_trigger, 1),
                                    validity=self.tools.kite.VALIDITY_DAY,
                                    tag='TITAN_OPT_SL'
                                )
                                trade['sl_order_id'] = str(new_sl_oid)
                                self.tools._save_active_trades()
                                print(f"   🛡️ LIVE SL order placed for option {symbol}: trigger ₹{sl_trigger:.2f}")
                            except Exception as e:
                                print(f"   ⚠️ Failed to place SL order for option {symbol}: {e}")
        
        # === IRON CONDOR MONITORING ===
        # Dedicated exit logic: target capture, SL, time-based auto-exit, breakout
        if ic_trades:
            try:
                from config import IRON_CONDOR_CONFIG
                ic_auto_exit_str = IRON_CONDOR_CONFIG.get('auto_exit_time', '14:50')
                ic_auto_exit_time = datetime.strptime(ic_auto_exit_str, '%H:%M').time()
                now_time = datetime.now().time()
                
                for trade in ic_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    
                    condor_id = trade.get('condor_id', trade['symbol'])
                    underlying = trade.get('underlying', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    total_credit_amount = trade.get('total_credit_amount', 0)
                    target_buyback = trade.get('target', 0)
                    stop_loss_debit = trade.get('stop_loss', 0)
                    upper_be = trade.get('upper_breakeven', 0)
                    lower_be = trade.get('lower_breakeven', 0)
                    sold_ce_strike = trade.get('sold_ce_strike', 0)
                    sold_pe_strike = trade.get('sold_pe_strike', 0)
                    
                    # Get current LTPs for all 4 legs
                    sold_ce_ltp = quotes.get(trade.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    hedge_ce_ltp = quotes.get(trade.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    sold_pe_ltp = quotes.get(trade.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    hedge_pe_ltp = quotes.get(trade.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                    
                    # Skip if we can't get prices for all legs
                    if not all([sold_ce_ltp, hedge_ce_ltp, sold_pe_ltp, hedge_pe_ltp]):
                        continue
                    
                    # Current cost to close = buy back sold legs, sell hedge legs
                    # P&L = credit collected - current debit to close
                    ce_debit = sold_ce_ltp - hedge_ce_ltp  # Cost to close CE wing
                    pe_debit = sold_pe_ltp - hedge_pe_ltp  # Cost to close PE wing
                    current_debit = ce_debit + pe_debit     # Total cost to close
                    ic_pnl = (total_credit - current_debit) * qty
                    ic_pnl -= calc_brokerage(total_credit, current_debit, qty)
                    ic_pnl_pct = (total_credit - current_debit) / total_credit * 100 if total_credit > 0 else 0
                    
                    # Get underlying spot price for breakout check
                    underlying_ltp = 0
                    try:
                        ul_quote = quotes.get(underlying, {})
                        if ul_quote:
                            underlying_ltp = ul_quote.get('last_price', 0)
                        if not underlying_ltp:
                            ul_data = self.tools.kite.ltp([underlying])
                            underlying_ltp = ul_data[underlying]['last_price']
                    except Exception:
                        pass
                    
                    exit_reason = None
                    exit_type = None
                    
                    # === EXIT CHECK 1: TIME-BASED AUTO-EXIT (2:50 PM) ===
                    if now_time >= ic_auto_exit_time:
                        exit_reason = f"IC auto-exit at {ic_auto_exit_str} (before EOD vol spike)"
                        exit_type = "IC_TIME_EXIT"
                    
                    # === EXIT CHECK 2: TARGET HIT (credit captured %) ===
                    # If current debit < target_buyback, we've captured enough premium
                    elif current_debit <= target_buyback and target_buyback > 0:
                        exit_reason = f"IC TARGET HIT: debit ₹{current_debit:.2f} ≤ target ₹{target_buyback:.2f} ({ic_pnl_pct:.0f}% captured)"
                        exit_type = "IC_TARGET_HIT"
                    
                    # === EXIT CHECK 3: STOP LOSS (loss exceeds multiplier) ===
                    elif current_debit >= stop_loss_debit and stop_loss_debit > 0:
                        exit_reason = f"IC STOP LOSS: debit ₹{current_debit:.2f} ≥ SL ₹{stop_loss_debit:.2f}"
                        exit_type = "IC_SL_HIT"
                    
                    # === EXIT CHECK 4: BREAKOUT WARNING (price approaching sold strikes) ===
                    elif underlying_ltp > 0 and IRON_CONDOR_CONFIG.get('breakout_exit', True):
                        buffer_pct = IRON_CONDOR_CONFIG.get('breakout_buffer_pct', 0.3) / 100
                        ce_danger = sold_ce_strike * (1 - buffer_pct)
                        pe_danger = sold_pe_strike * (1 + buffer_pct)
                        
                        if underlying_ltp >= ce_danger:
                            exit_reason = f"IC BREAKOUT UP: {underlying} ₹{underlying_ltp:.0f} approaching sold CE strike ₹{sold_ce_strike:.0f}"
                            exit_type = "IC_BREAKOUT_EXIT"
                        elif underlying_ltp <= pe_danger:
                            exit_reason = f"IC BREAKOUT DOWN: {underlying} ₹{underlying_ltp:.0f} approaching sold PE strike ₹{sold_pe_strike:.0f}"
                            exit_type = "IC_BREAKOUT_EXIT"
                    
                    if exit_reason:
                        emoji = "🎯" if exit_type == "IC_TARGET_HIT" else "❌" if exit_type == "IC_SL_HIT" else "⏰" if exit_type == "IC_TIME_EXIT" else "🚨"
                        print(f"\n{emoji} {exit_type}: {condor_id}")
                        print(f"   {exit_reason}")
                        print(f"   CE wing: Sold ₹{sold_ce_ltp:.2f} / Hedge ₹{hedge_ce_ltp:.2f}")
                        print(f"   PE wing: Sold ₹{sold_pe_ltp:.2f} / Hedge ₹{hedge_pe_ltp:.2f}")
                        print(f"   Credit: ₹{total_credit:.2f} → Debit: ₹{current_debit:.2f} | P&L: ₹{ic_pnl:+,.2f} ({ic_pnl_pct:+.1f}%)")
                        
                        self.tools.update_trade_status(
                            trade['symbol'], exit_type, current_debit, ic_pnl,
                            exit_detail={
                                'exit_reason': exit_reason,
                                'exit_type': exit_type,
                                'strategy': 'IRON_CONDOR',
                                'credit': total_credit,
                                'debit_at_exit': current_debit,
                                'underlying_at_exit': underlying_ltp,
                            }
                        )
                        with self._pnl_lock:
                            self.daily_pnl += ic_pnl
                            self.capital += ic_pnl
                        
                        # Record with risk governor
                        open_pos = [t for t in active_trades if t.get('symbol') != trade['symbol']]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(trade['symbol'], ic_pnl, ic_pnl > 0, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                    
                    elif show_status:
                        status = "🟢" if ic_pnl > 0 else "🔴"
                        spot_info = f" | Spot: ₹{underlying_ltp:.0f}" if underlying_ltp > 0 else ""
                        print(f"   {status} 🦅 IC {underlying}: Credit ₹{total_credit:.2f} → Debit ₹{current_debit:.2f} | P&L: ₹{ic_pnl:+,.0f} ({ic_pnl_pct:+.1f}%){spot_info}")
                        print(f"      Zone: ₹{lower_be:.0f} — ₹{upper_be:.0f} | SL: ₹{stop_loss_debit:.2f} | TGT: ₹{target_buyback:.2f}")
            except Exception as e:
                print(f"   ⚠️ IC monitoring error: {e}")
        
        if show_status and active_trades:
            print(f"\n👁️ LIVE POSITIONS [{datetime.now().strftime('%H:%M:%S')}]:")
            
            # --- Show credit spreads first ---
            if spread_trades:
                print(f"{'Spread':30} {'Credit':>10} {'Debit':>10} {'MaxRisk':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in spread_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    spread_id = trade.get('spread_id', trade['symbol'])
                    sold_sym = trade.get('sold_symbol', '')
                    hedge_sym = trade.get('hedge_symbol', '')
                    qty = trade['quantity']
                    net_credit = trade.get('net_credit', 0)
                    max_risk = trade.get('max_risk', (trade.get('spread_width', 0) - net_credit) * qty)
                    
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp
                    pnl = (net_credit - current_debit) * qty
                    pnl_pct = (net_credit - current_debit) / net_credit * 100 if net_credit > 0 else 0
                    
                    state = self.exit_manager.get_trade_state(trade['symbol'])
                    status_flags = "θ+ "
                    if state:
                        target_cr = getattr(state, 'target_credit', net_credit * 0.35)
                        sl_debit = getattr(state, 'stop_loss_debit', net_credit * 2)
                        status_flags += f"TGT:{target_cr:.1f} SL:{sl_debit:.1f}"
                    
                    status = "🟢" if pnl > 0 else "🔴"
                    # Shorten spread_id for display
                    display_name = spread_id[:28] if len(spread_id) > 28 else spread_id
                    print(f"{status} {display_name:28} ₹{net_credit:>9.2f} ₹{current_debit:>9.2f} ₹{max_risk:>9,.0f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ SELL {sold_sym} @ ₹{sold_ltp:.2f} | HEDGE {hedge_sym} @ ₹{hedge_ltp:.2f}")
                print()
            
            # --- Show debit spreads ---
            if debit_spread_trades:
                print(f"{'DebitSpread':30} {'Entry':>10} {'Value':>10} {'MaxProfit':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in debit_spread_trades:
                    if trade.get('status', 'OPEN') != 'OPEN':
                        continue
                    spread_id = trade.get('spread_id', trade['symbol'])
                    buy_sym = trade.get('buy_symbol', '')
                    sell_sym = trade.get('sell_symbol', '')
                    qty = trade['quantity']
                    net_debit = trade.get('net_debit', 0)
                    max_profit = trade.get('max_profit', 0)
                    
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    pnl = (current_value - net_debit) * qty
                    pnl_pct = (current_value - net_debit) / net_debit * 100 if net_debit > 0 else 0
                    
                    state = self.exit_manager.get_trade_state(trade['symbol'])
                    status_flags = "Δ+ "
                    if state:
                        if state.trailing_active:
                            status_flags += "📈TRAIL "
                        if state.breakeven_applied:
                            status_flags += "🔒BE "
                        status_flags += f"TGT:{state.target:.1f} SL:{state.current_sl:.1f}"
                    
                    status = "🟢" if pnl > 0 else "🔴"
                    display_name = spread_id[:28] if len(spread_id) > 28 else spread_id
                    print(f"{status} {display_name:28} ₹{net_debit:>9.2f} ₹{current_value:>9.2f} ₹{max_profit:>9,.0f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {status_flags}")
                    print(f"   └─ BUY {buy_sym} @ ₹{buy_ltp:.2f} | SELL {sell_sym} @ ₹{sell_ltp:.2f}")
                print()
            
            # --- Show iron condors ---
            ic_open = [t for t in ic_trades if t.get('status', 'OPEN') == 'OPEN']
            if ic_open:
                print(f"{'🦅 Iron Condor':30} {'Credit':>10} {'Debit':>10} {'MaxRisk':>10} {'P&L':>12} {'Status'}")
                print("-" * 95)
                for trade in ic_open:
                    underlying = trade.get('underlying', '')
                    qty = trade['quantity']
                    total_credit = trade.get('total_credit', 0)
                    total_credit_amount = trade.get('total_credit_amount', 0)
                    max_risk = trade.get('max_risk', 0)
                    upper_be = trade.get('upper_breakeven', 0)
                    lower_be = trade.get('lower_breakeven', 0)
                    target_buyback = trade.get('target', 0)
                    stop_loss_debit = trade.get('stop_loss', 0)

                    s_ce = quotes.get(trade.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    h_ce = quotes.get(trade.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    s_pe = quotes.get(trade.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    h_pe = quotes.get(trade.get('hedge_pe_symbol', ''), {}).get('last_price', 0)

                    if all([s_ce, h_ce, s_pe, h_pe]):
                        current_debit = (s_ce - h_ce) + (s_pe - h_pe)
                        ic_pnl = (total_credit - current_debit) * qty
                        ic_pnl_pct = (total_credit - current_debit) / total_credit * 100 if total_credit > 0 else 0
                    else:
                        current_debit = 0
                        ic_pnl = 0
                        ic_pnl_pct = 0

                    # Get underlying spot
                    ul_ltp = quotes.get(underlying, {}).get('last_price', 0)
                    spot_str = f" Spot:₹{ul_ltp:.0f}" if ul_ltp > 0 else ""

                    status = "🟢" if ic_pnl >= 0 else "🔴"
                    display_name = f"IC {underlying}"[:28]
                    print(f"{status} {display_name:28} ₹{total_credit:>9.2f} ₹{current_debit:>9.2f} ₹{max_risk:>9,.0f} ₹{ic_pnl:>+10,.0f} ({ic_pnl_pct:+.1f}%) θ+{spot_str}")
                    print(f"   └─ Zone: ₹{lower_be:.0f}—₹{upper_be:.0f} | TGT: ₹{target_buyback:.2f} | SL: ₹{stop_loss_debit:.2f}")
                    print(f"   └─ CE: SELL {trade.get('sold_ce_symbol','')} @₹{s_ce:.2f} | PE: SELL {trade.get('sold_pe_symbol','')} @₹{s_pe:.2f}")
                print()

            # --- Show regular trades ---
            regular_open = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False) and t.get('status', 'OPEN') == 'OPEN']
            if regular_open:
                print(f"{'Symbol':15} {'Side':6} {'Entry':>10} {'LTP':>10} {'SL':>10} {'Target':>10} {'P&L':>12} {'Type':10} {'Status'}")
                print("-" * 110)
        
        for trade in active_trades:
            if trade.get('is_credit_spread', False) or trade.get('is_debit_spread', False) or trade.get('is_iron_condor', False):
                continue  # Already displayed above
            symbol = trade['symbol']
            if symbol not in quotes:
                continue
            if trade.get('status', 'OPEN') != 'OPEN':
                continue  # Already exited
            
            ltp = quotes[symbol]['last_price']
            entry = trade['avg_price']
            sl = trade['stop_loss']
            target = trade.get('target', entry * (1.02 if trade['side'] == 'BUY' else 0.98))
            qty = trade['quantity']
            side = trade['side']
            
            # Get exit manager state for status
            state = self.exit_manager.get_trade_state(symbol)
            status_flags = ""
            if state:
                if state.breakeven_applied:
                    status_flags += "🔒BE "
                if state.trailing_active:
                    status_flags += "📈TRAIL "
            
            # Get trade type tag for display
            _setup = trade.get('setup_type', '')
            _type_tag = ''
            if _setup == 'GMM_SNIPER':
                _type_tag = '🎯SNIPER'
            elif _setup == 'ML_OVERRIDE_WGMM':
                _type_tag = '🔄ML_GMM'
            elif _setup == 'ALL_AGREE':
                _type_tag = '🧬ALL3'
            elif _setup == 'MODEL_TRACKER':
                _type_tag = '🧠SCORE'
            elif _setup in ('', 'MANUAL', 'GPT') or not _setup:
                _type_tag = '🤖GPT'
            else:
                _type_tag = _setup[:8]
            
            # Calculate current P&L
            if side == 'BUY':
                pnl = (ltp - entry) * qty
                pnl_pct = (ltp - entry) / entry * 100
                
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (ltp - sl) / ltp * 100
                    tgt_dist = (target - ltp) / ltp * 100
                    print(f"{status} {symbol:13} {'BUY':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {_type_tag:10} {status_flags}")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
            
            else:  # SHORT position
                pnl = (entry - ltp) * qty
                pnl_pct = (entry - ltp) / entry * 100
                
                if show_status:
                    status = "🟢" if pnl > 0 else "🔴"
                    sl_dist = (sl - ltp) / ltp * 100
                    tgt_dist = (ltp - target) / ltp * 100
                    print(f"{status} {symbol:13} {'SHORT':6} ₹{entry:>9.2f} ₹{ltp:>9.2f} ₹{sl:>9.2f} ₹{target:>9.2f} ₹{pnl:>+10,.0f} ({pnl_pct:+.1f}%) {_type_tag:10} {status_flags}")
                    print(f"   └─ SL: {sl_dist:.1f}% away | Target: {tgt_dist:.1f}% away")
        
        # Print summary after all positions
        if show_status and active_trades:
            # Calculate total P&L including all spread types
            total_pnl = 0
            for t in active_trades:
                if t.get('status', 'OPEN') != 'OPEN':
                    continue
                if t.get('is_credit_spread', False):
                    sold_sym = t.get('sold_symbol', '')
                    hedge_sym = t.get('hedge_symbol', '')
                    sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
                    hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
                    current_debit = sold_ltp - hedge_ltp
                    total_pnl += (t.get('net_credit', 0) - current_debit) * t['quantity']
                elif t.get('is_debit_spread', False):
                    buy_sym = t.get('buy_symbol', '')
                    sell_sym = t.get('sell_symbol', '')
                    buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
                    sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
                    current_value = buy_ltp - sell_ltp
                    total_pnl += (current_value - t.get('net_debit', 0)) * t['quantity']
                elif t.get('is_iron_condor', False):
                    s_ce = quotes.get(t.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    h_ce = quotes.get(t.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    s_pe = quotes.get(t.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    h_pe = quotes.get(t.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                    if all([s_ce, h_ce, s_pe, h_pe]):
                        ic_debit = (s_ce - h_ce) + (s_pe - h_pe)
                        total_pnl += (t.get('total_credit', 0) - ic_debit) * t['quantity']
                elif t['symbol'] in quotes:
                    if t['side'] == 'BUY':
                        total_pnl += (quotes[t['symbol']]['last_price'] - t['avg_price']) * t['quantity']
                    else:
                        total_pnl += (t['avg_price'] - quotes[t['symbol']]['last_price']) * t['quantity']
            print("-" * 95)
            print(f"📊 TOTAL UNREALIZED P&L: ₹{total_pnl:+,.0f} | Capital: ₹{self.capital:,.0f} | Daily P&L: ₹{self.daily_pnl:+,.0f}")
            # Print exit manager status
            print(self.exit_manager.get_status_summary())
    
    def reset_agent(self):
        """Reset agent to clear conversation history - but KEEP positions"""
        # DON'T reset tools - preserve paper positions!
        # Just create new agent with fresh conversation
        from llm_agent import TradingAgent
        self.agent = TradingAgent(auto_execute=True, paper_mode=self.paper_mode, paper_capital=self.capital)
    
    def _format_watchlist_for_prompt(self) -> str:
        """Format hot watchlist for GPT prompt — these are 55+ stocks that need re-evaluation"""
        try:
            from options_trader import get_hot_watchlist
            wl = get_hot_watchlist()
            if not wl:
                return 'No watchlist stocks — all prior attempts either succeeded or scored too low'
            lines = []
            for sym, entry in sorted(wl.items(), key=lambda x: x[1].get('score', 0), reverse=True):
                lines.append(
                    f"  🔥 {sym}: Score {entry.get('score', 0):.0f}/100 | Dir: {entry.get('direction', '?')} | "
                    f"Conviction: {entry.get('directional_strength', 0):.0f}/8 needed | "
                    f"Seen {entry.get('cycle_count', 1)}x — TRY AGAIN with place_option_order()"
                )
            return chr(10).join(lines)
        except Exception:
            return 'Watchlist unavailable'
    
    def is_trading_hours(self) -> bool:
        """Check if within trading hours"""
        now = datetime.now().time()
        start = datetime.strptime(TRADING_HOURS["start"], "%H:%M").time()
        end = datetime.strptime(TRADING_HOURS["end"], "%H:%M").time()
        return start <= now <= end
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit hit"""
        max_loss = self.start_capital * HARD_RULES["MAX_DAILY_LOSS"]
        if self.daily_pnl <= -max_loss:
            print(f"❌ Daily loss limit hit! P&L: ₹{self.daily_pnl:,.0f}")
            return False
        return True
    
    def _detect_expiry_day(self) -> dict:
        """
        Auto-detect if today is an expiry day for stocks or indices.
        Uses the NFO instrument cache (already loaded by kite_ticker/options_trader).
        Automatically sets EXPIRY_SHIELD_CONFIG['is_monthly_expiry'] for downstream use.
        
        Returns dict with:
          is_stock_expiry: bool - True if any stock option expires today
          is_index_expiry: bool - True if any index option expires today
          stock_count: int - how many stocks expire today
          index_names: list - which indices expire today
          next_stock_expiry: str - next stock expiry date (if not today)
          days_to_stock_expiry: int - days until next stock expiry
        """
        from datetime import date as _date_type
        today = _date_type.today()
        
        INDICES = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX', 'BANKEX', 'NIFTYNXT50'}
        
        result = {
            'is_stock_expiry': False,
            'is_index_expiry': False,
            'stock_count': 0,
            'index_names': [],
            'next_stock_expiry': None,
            'days_to_stock_expiry': None,
        }
        
        # Use cached expiry info if available (avoid re-querying every 5 min)
        if hasattr(self, '_expiry_day_cache') and self._expiry_day_cache.get('date') == str(today):
            return self._expiry_day_cache['result']
        
        try:
            # Get NFO instruments (uses cached data from kite_ticker startup)
            instruments = self.tools.kite.instruments("NFO") if hasattr(self.tools, 'kite') and self.tools.kite else []
            if not instruments:
                # Fallback: try options_trader's cache
                from options_trader import get_options_trader
                _ot = get_options_trader()
                if _ot and hasattr(_ot, 'chain_fetcher'):
                    instruments = _ot.chain_fetcher._get_nfo_instruments()
        except Exception:
            instruments = []
        
        if not instruments:
            return result
        
        # Collect unique (name, expiry_date) pairs for options
        stocks_expiring_today = set()
        indices_expiring_today = set()
        stock_future_expiries = set()  # For computing next expiry
        
        for inst in instruments:
            name = inst.get('name', '')
            inst_type = inst.get('instrument_type', '')
            expiry = inst.get('expiry')
            if not name or not expiry or inst_type not in ('CE', 'PE'):
                continue
            
            exp_date = expiry if isinstance(expiry, _date_type) and not isinstance(expiry, datetime) else (
                expiry.date() if hasattr(expiry, 'date') else None)
            if exp_date is None:
                continue
            
            if exp_date == today:
                if name in INDICES:
                    indices_expiring_today.add(name)
                else:
                    stocks_expiring_today.add(name)
            elif exp_date > today and name not in INDICES:
                stock_future_expiries.add(exp_date)
        
        result['is_stock_expiry'] = len(stocks_expiring_today) > 0
        result['stock_count'] = len(stocks_expiring_today)
        result['is_index_expiry'] = len(indices_expiring_today) > 0
        result['index_names'] = sorted(indices_expiring_today)
        
        if stock_future_expiries:
            next_exp = min(stock_future_expiries)
            result['next_stock_expiry'] = str(next_exp)
            result['days_to_stock_expiry'] = (next_exp - today).days
        
        # Auto-set EXPIRY_SHIELD for downstream use (replaces manual toggle)
        try:
            import config
            if result['is_stock_expiry']:
                config.EXPIRY_SHIELD_CONFIG['is_monthly_expiry'] = True
                config.EXPIRY_SHIELD_CONFIG['enabled'] = True
            else:
                config.EXPIRY_SHIELD_CONFIG['is_monthly_expiry'] = False
        except Exception:
            pass
        
        # Cache for the rest of the day
        self._expiry_day_cache = {'date': str(today), 'result': result}
        return result

    def scan_and_trade(self):
        """Main trading loop - scan market and execute trades"""
        import traceback as _scan_tb
        _dbg_ts = datetime.now().strftime('%H:%M:%S')
        _debug_log = 'bot_debug.log'
        def _scan_dbg(msg):
            _ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            _line = f"[{_ts}] {msg}"
            print(_line)
            try:
                with open(_debug_log, 'a', encoding='utf-8') as _f:
                    _f.write(_line + '\n')
            except Exception:
                pass
        
        _scan_dbg("SCAN: scan_and_trade() ENTER")
        
        if not self.is_trading_hours():
            _scan_dbg(f"SCAN: EXIT - outside trading hours ({datetime.now().strftime('%H:%M')})")
            return
        
        _scan_dbg("SCAN: passed is_trading_hours()")
        
        if not self.check_daily_loss_limit():
            _scan_dbg("SCAN: EXIT - daily loss limit")
            return
        
        _scan_dbg("SCAN: passed check_daily_loss_limit()")
        
        # === RISK GOVERNOR CHECK ===
        _rg_allowed = self.risk_governor.is_trading_allowed()
        _scan_dbg(f"SCAN: risk_governor.is_trading_allowed() = {_rg_allowed}")
        if not _rg_allowed:
            _scan_dbg(f"SCAN: EXIT - risk governor blocked")
            print(self.risk_governor.get_status())
            return
        
        _scan_dbg("SCAN: all pre-checks passed, proceeding to scan...")
        
        self._scanning = True  # Suppress real-time dashboard during scan
        _cycle_start = time.time()
        print(f"\n{'='*80}")
        print(f"🔍 SCAN CYCLE @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        # === EXPIRY DAY AUTO-DETECTION (Feb 24 fix) ===
        # Automatically detects if today is stock/index expiry using NFO instrument cache.
        # Prints prominent warning + auto-enables EXPIRY_SHIELD so _recommend_expiry()
        # doesn't need manual config toggle.
        try:
            _expiry_info = self._detect_expiry_day()
            if _expiry_info['is_stock_expiry'] or _expiry_info['is_index_expiry']:
                _exp_tags = []
                if _expiry_info['is_stock_expiry']:
                    _exp_tags.append(f"STOCK MONTHLY ({_expiry_info['stock_count']} stocks)")
                if _expiry_info['is_index_expiry']:
                    _exp_tags.append(f"INDEX ({', '.join(_expiry_info['index_names'])})")
                _exp_label = ' + '.join(_exp_tags)
                print(f"⚠️⚠️⚠️  TODAY IS EXPIRY DAY: {_exp_label}  ⚠️⚠️⚠️")
                print(f"   → Premium crush risk HIGH | Theta decay accelerated | Gamma spikes possible")
                if _expiry_info['is_stock_expiry']:
                    print(f"   → Stock options expire TODAY — cheap premiums will decay to zero")
                    print(f"   → Auto-enabled: EXPIRY_SHIELD + NEXT_MONTH expiry selection")
                if _expiry_info.get('next_stock_expiry'):
                    print(f"   → Next stock expiry: {_expiry_info['next_stock_expiry']}")
            else:
                # Not expiry day — show days to next expiry
                if _expiry_info.get('days_to_stock_expiry') is not None:
                    _dte = _expiry_info['days_to_stock_expiry']
                    _next = _expiry_info.get('next_stock_expiry', '?')
                    print(f"📅 Stock expiry in {_dte} day{'s' if _dte != 1 else ''} ({_next})")
        except Exception as _exp_err:
            print(f"   ⚠️ Expiry detection error: {_exp_err}")
        
        print(self.risk_governor.get_status())
        self._rejected_this_cycle = set()  # Reset rejected symbols for new scan
        
        # === HOT WATCHLIST: Clean stale entries & display ===
        try:
            from options_trader import get_hot_watchlist, cleanup_stale_watchlist
            cleanup_stale_watchlist(max_age_minutes=20)
            _watchlist = get_hot_watchlist()
            if _watchlist:
                print(f"\n🔥 HOT WATCHLIST ({len(_watchlist)} stocks warming up):")
                for _ws, _wd in sorted(_watchlist.items(), key=lambda x: x[1].get('score', 0), reverse=True):
                    print(f"   🔥 {_ws}: Score {_wd.get('score', 0):.0f} | {_wd.get('direction', '?')} | Conviction {_wd.get('directional_strength', 0):.0f}/8 | Seen {_wd.get('cycle_count', 1)}x")
        except Exception as e:
            print(f"   ⚠️ Watchlist check error: {e}")
        
        # CHECK AND UPDATE EXISTING TRADES (target/stoploss hits)
        trade_updates = self.tools.check_and_update_trades()
        if trade_updates:
            print(f"\n📊 TRADE UPDATES:")
            for update in trade_updates:
                emoji = "✅" if update['result'] == 'TARGET_HIT' else "❌"
                print(f"   {emoji} {update['symbol']}: {update['result']}")
                print(f"      Entry: ₹{update['entry']:.2f} → Exit: ₹{update['exit']:.2f}")
                print(f"      P&L: ₹{update['pnl']:+,.2f}")
                with self._pnl_lock:
                    self.daily_pnl += update['pnl']
                    self.capital += update['pnl']  # Also update capital (was missing)
        
        # === LIVE MODE: Sync with broker positions ===
        if not self.paper_mode:
            self._sync_broker_positions()
        
        # Show current active positions
        active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        if active_trades:
            print(f"\n📂 ACTIVE POSITIONS ({len(active_trades)}):")
            for t in active_trades:
                print(f"   • {t['symbol']}: {t['side']} {t['quantity']} @ ₹{t['avg_price']:.2f}")
                print(f"     SL: ₹{t['stop_loss']:.2f} | Target: ₹{t.get('target', 0):.2f}")
        
        # Reset agent for fresh analysis each time
        self.reset_agent()
        
        try:
            # === MARKET SCANNER: Discover movers outside fixed universe ===
            try:
                scan_result = self.market_scanner.scan(existing_universe=APPROVED_UNIVERSE)
                print(self.market_scanner.format_scan_summary(scan_result))
                # Collect wild-card symbols to merge into data pipeline
                self._wildcard_symbols = [f"NSE:{w.symbol}" for w in scan_result.wildcards]
                # Store scanner scores for wildcard gate bypass
                self._wildcard_scores = {f"NSE:{w.symbol}": w.score for w in scan_result.wildcards}
                self._wildcard_change = {f"NSE:{w.symbol}": w.change_pct for w in scan_result.wildcards}
                # === DYNAMIC LOT SIZES: ensure every F&O stock has correct lot size ===
                update_fno_lot_sizes(self.market_scanner.get_lot_map())
            except Exception as e:
                print(f"⚠️ Scanner error (non-fatal): {e}")
                scan_result = None
                self._wildcard_symbols = []
                self._wildcard_scores = {}
                self._wildcard_change = {}
            
            # Merge wild-cards into scan universe for this cycle
            # With FULL_FNO_SCAN enabled, we scan ALL F&O stocks (not just curated + wildcards)
            # Pre-filter by change% to only run expensive indicators on movers
            scan_universe = list(APPROVED_UNIVERSE)
            _full_scan_mode = FULL_FNO_SCAN.get('enabled', False)
            
            if _full_scan_mode:
                # === FULL F&O UNIVERSE SCAN (NO FIXED LIST BIAS) ===
                # All F&O stocks compete equally — top N by composite rank are scanned
                try:
                    _all_fo = self.market_scanner.get_all_fo_symbols()
                    _all_fo_syms = set(_all_fo)
                    
                    # Pre-filter: use scanner results to only include stocks with meaningful movement
                    _min_change = FULL_FNO_SCAN.get('min_change_pct_filter', 0.3)
                    _max_indicator_stocks = FULL_FNO_SCAN.get('max_indicator_stocks', 50)
                    
                    # ALL F&O stocks compete equally — no curated set gets guaranteed inclusion
                    _full_universe = set()
                    
                    # === QUALITY GATE: Only pass stocks with scorer-relevant signals ===
                    # Scanner's _all_results has ~50-60 stocks above 0.5% change, but most
                    # are mid-range with no setups and score 20-35. This gate keeps only stocks
                    # that match what the scorer actually rewards (trend/ORB/volume).
                    # NOTE: Gate D (scanner category) was REMOVED — it passed everything since
                    # scanner assigns score>0 to all categorized stocks, defeating the gate.
                    if scan_result and hasattr(self.market_scanner, '_all_results'):
                        _quality_candidates = []
                        _all_raw = self.market_scanner._all_results
                        _gate_a_count = 0  # strong movers (≥1.0%)
                        _gate_b_count = 0  # near day extreme
                        _gate_c_count = 0  # volume top 25%
                        _below_min_change = 0  # filtered by min_change
                        
                        # First: compute volume percentile for relative volume check
                        _all_vols = sorted([r.volume for r in _all_raw if r.volume > 0])
                        _vol_p75 = _all_vols[int(len(_all_vols) * 0.75)] if _all_vols else 0
                        
                        print(f"   🔍 Quality gate input: {len(_all_raw)} raw stocks, min_change={_min_change}%, vol_p75={_vol_p75:,.0f}")
                        
                        for r in _all_raw:
                            if abs(r.change_pct) < _min_change:
                                _below_min_change += 1
                                continue
                            
                            # Day extremity: is price in top/bottom 15% of day range?
                            _dr = r.day_high - r.day_low
                            _near_extreme = False
                            if _dr > 0 and r.ltp > 0:
                                _pos = (r.ltp - r.day_low) / _dr  # 0=day low, 1=day high
                                _near_extreme = _pos >= 0.85 or _pos <= 0.15
                            
                            # Volume surge: top 25% of all scanned stocks
                            _vol_surge = r.volume >= _vol_p75 if _vol_p75 > 0 else False
                            
                            # ---- QUALITY GATE: must pass at least ONE of A/B/C ----
                            # A) Strong mover:  ≥1.0% change = likely has trend + ORB signal
                            # B) Day extreme:   near high/low = likely ORB breakout (scorer +20)
                            # C) Volume surge:  top 25% volume = likely HIGH/EXPLOSIVE vol (scorer +12-15)
                            _is_a = abs(r.change_pct) >= 1.0
                            _is_b = _near_extreme
                            _is_c = _vol_surge
                            
                            if _is_a: _gate_a_count += 1
                            if _is_b: _gate_b_count += 1
                            if _is_c: _gate_c_count += 1
                            
                            if _is_a or _is_b or _is_c:
                                _quality_candidates.append(r)
                        
                        _after_min = len(_all_raw) - _below_min_change
                        _dropped = _after_min - len(_quality_candidates)
                        
                        # ALWAYS print gate stats for diagnostics
                        print(f"   🔍 Quality gate: {_after_min} above min_change → {len(_quality_candidates)} passed, {_dropped} dropped")
                        print(f"      Gate A (≥1.0% chg): {_gate_a_count} | Gate B (day extreme): {_gate_b_count} | Gate C (vol top25): {_gate_c_count}")
                        
                        if _quality_candidates:
                            # Rank by scorer-aligned composite (matters when >50 quality candidates)
                            _vols = sorted(set(c.volume for c in _quality_candidates))
                            _vol_rank = {v: i / max(len(_vols) - 1, 1) for i, v in enumerate(_vols)}
                            
                            def _rank(r):
                                chg_score = min(abs(r.change_pct) / 3.0, 1.0)
                                vol_score = _vol_rank.get(r.volume, 0)
                                scan_score = min(r.score / 200.0, 1.0) if r.score else 0
                                _dr = r.day_high - r.day_low
                                if _dr > 0 and r.ltp > 0:
                                    _pos = (r.ltp - r.day_low) / _dr
                                    extremity = abs(_pos - 0.5) * 2
                                else:
                                    extremity = 0
                                _total_qty = (r.buy_qty or 0) + (r.sell_qty or 0)
                                imbalance = abs((r.buy_qty or 0) - (r.sell_qty or 0)) / _total_qty if _total_qty > 0 else 0
                                return (chg_score * 0.25 + vol_score * 0.20 + scan_score * 0.10
                                        + extremity * 0.30 + imbalance * 0.15)
                            
                            _quality_candidates.sort(key=_rank, reverse=True)
                            _selected = _quality_candidates[:_max_indicator_stocks]
                            for _r in _selected:
                                _full_universe.add(_r.nse_symbol)
                            
                            # Show top 5 selected with reasons
                            print(f"      Top 5 selected:")
                            for _i, _r in enumerate(_selected[:5]):
                                _dr2 = _r.day_high - _r.day_low
                                _pos2 = ((_r.ltp - _r.day_low) / _dr2 * 100) if _dr2 > 0 else 50
                                _gates = []
                                if abs(_r.change_pct) >= 1.0: _gates.append("A:chg")
                                if _dr2 > 0 and (_pos2 >= 85 or _pos2 <= 15): _gates.append("B:ext")
                                if _r.volume >= _vol_p75 and _vol_p75 > 0: _gates.append("C:vol")
                                print(f"        {_i+1}. {_r.symbol} chg={_r.change_pct:+.1f}% dayPos={_pos2:.0f}% vol={_r.volume:,.0f} gates=[{','.join(_gates)}]")
                    
                    # Always include stocks we already hold (need exit monitoring)
                    for _t in self.tools.paper_positions:
                        if _t.get('status', 'OPEN') == 'OPEN':
                            _held_sym = _t.get('symbol', '')
                            if _held_sym:
                                _full_universe.add(_held_sym)
                    
                    scan_universe = list(_full_universe)
                    _skipped = len(_all_fo) - len(scan_universe)
                    print(f"   📡 FULL F&O SCAN: {len(scan_universe)} stocks passing filter (skipped {_skipped} flat/illiquid from {len(_all_fo)} total)")
                    
                except Exception as _e:
                    _all_fo_syms = set()
                    print(f"   ⚠️ Full scan fallback to curated: {_e}")
                    scan_universe = list(APPROVED_UNIVERSE)
                    for ws in self._wildcard_symbols:
                        if ws not in scan_universe:
                            scan_universe.append(ws)
            else:
                # === CLASSIC MODE: curated + wildcards ===
                for ws in self._wildcard_symbols:
                    if ws not in scan_universe:
                        scan_universe.append(ws)
                
                try:
                    _all_fo = self.market_scanner.get_all_fo_symbols()
                    _all_fo_syms = set(_all_fo)
                    if len(self._wildcard_symbols) > 0:
                        print(f"   📡 Scan universe: {len(APPROVED_UNIVERSE)} curated + {len(self._wildcard_symbols)} wildcards = {len(scan_universe)} total (from {len(_all_fo)} F&O)")
                except Exception as _e:
                    _all_fo_syms = set()
                    print(f"   ⚠️ Could not get F&O universe: {_e}")
            
            # Get fresh market data for fixed universe + wild-cards
            market_data = self.tools.get_market_data(scan_universe)
            
            # Get volume analysis for EOD predictions
            volume_analysis = self.tools.get_volume_analysis(scan_universe)
            
            # Format ENHANCED data for prompt
            data_summary = []
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'ltp' in data:
                    # Get volume data for this symbol
                    vol_data = volume_analysis.get(symbol, {})
                    eod_pred = vol_data.get('eod_prediction', 'N/A')
                    order_flow = vol_data.get('order_flow', 'N/A')
                    buy_ratio = vol_data.get('buy_ratio', 50)
                    
                    line = f"""
{symbol}:
  Price: ₹{data['ltp']:.2f} | Change: {data.get('change_pct', 0):.2f}% | Trend: {data.get('trend', 'N/A')}
  RSI: {data.get('rsi_14', 50):.0f} | ATR: ₹{data.get('atr_14', 0):.2f} | ADX: {data.get('adx', 20):.0f}
  VWAP: ₹{data.get('vwap', 0):.2f} ({data.get('price_vs_vwap', 'N/A')}) Slope: {data.get('vwap_slope', 'FLAT')}
  EMA9: ₹{data.get('ema_9', 0):.2f} | EMA21: ₹{data.get('ema_21', 0):.2f} | Regime: {data.get('ema_regime', 'N/A')}
  ORB: H=₹{data.get('orb_high', 0):.2f} L=₹{data.get('orb_low', 0):.2f} → {data.get('orb_signal', 'N/A')} (Str:{data.get('orb_strength', 0):.1f}%) Hold:{data.get('orb_hold_candles', 0)}
  Volume: {data.get('volume_regime', 'N/A')} ({data.get('volume_vs_avg', 1.0):.1f}x avg) | Order Flow: {order_flow} | Buy%: {buy_ratio}%
  HTF: {data.get('htf_trend', 'N/A')} ({data.get('htf_alignment', 'N/A')}) | Chop: {'⚠️YES' if data.get('chop_zone', False) else 'NO'}
  Accel: FollowThru:{data.get('follow_through_candles', 0)} RangeExp:{data.get('range_expansion_ratio', 0):.1f} VWAPSteep:{'Y' if data.get('vwap_slope_steepening', False) else 'N'}
  Support: ₹{data.get('support_1', 0):.2f} / ₹{data.get('support_2', 0):.2f}
  Resistance: ₹{data.get('resistance_1', 0):.2f} / ₹{data.get('resistance_2', 0):.2f}
  EOD Prediction: {eod_pred}"""
                    data_summary.append(line)
            
            # Sort by absolute change to show active stocks first
            sorted_data = sorted(market_data.items(), 
                                key=lambda x: abs(x[1].get('change_pct', 0)) if isinstance(x[1], dict) else 0,
                                reverse=True)
            
            # === SCORE ALL F&O stocks (single-pass — cached for trade-time reuse) ===
            # Scores ALL stocks in sorted_data. Cached decisions are passed to
            # place_option_order / place_credit_spread / place_debit_spread so they
            # DON'T re-score. Only microstructure (bid-ask/OI) is fetched at trade time.
            _pre_scores = {}       # symbol → score (for display / GPT prompt)
            _cycle_decisions = {}  # symbol → {decision, direction, market_data}
            
            # Compute market breadth early so scorer can use it for regime-aware ORB weighting
            _pre_up = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) > 0.5)
            _pre_down = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) < -0.5)
            _pre_breadth = "BULLISH" if _pre_up > _pre_down * 1.5 else "BEARISH" if _pre_down > _pre_up * 1.5 else "MIXED"
            
            try:
                from options_trader import get_intraday_scorer, IntradaySignal
                _scorer = get_intraday_scorer()
                for _sym, _d in sorted_data:
                    if not isinstance(_d, dict) or 'ltp' not in _d:
                        continue
                    # Inject market breadth into market_data for regime-aware ORB scoring
                    _d['market_breadth'] = _pre_breadth
                    try:
                        _sig = IntradaySignal(
                            symbol=_sym,
                            orb_signal=_d.get('orb_signal', 'INSIDE_ORB'),
                            vwap_position=_d.get('price_vs_vwap', _d.get('vwap_position', 'AT_VWAP')),
                            vwap_trend=_d.get('vwap_slope', _d.get('vwap_trend', 'FLAT')),
                            ema_regime=_d.get('ema_regime', 'NORMAL'),
                            volume_regime=_d.get('volume_regime', 'NORMAL'),
                            rsi=_d.get('rsi_14', 50.0),
                            price_momentum=_d.get('momentum_15m', 0.0),
                            htf_alignment=_d.get('htf_alignment', 'NEUTRAL'),
                            chop_zone=_d.get('chop_zone', False),
                            follow_through_candles=_d.get('follow_through_candles', 0),
                            range_expansion_ratio=_d.get('range_expansion_ratio', 0.0),
                            vwap_slope_steepening=_d.get('vwap_slope_steepening', False),
                            atr=_d.get('atr_14', 0.0)
                        )
                        # S3 FIX: Don't bias scorer with mechanical change_pct direction
                        # Let the scorer determine direction from its own signal analysis
                        # change_pct is backward-looking noise, not an analytical view
                        _dir = None
                        _dec = _scorer.score_intraday_signal(_sig, market_data=_d, caller_direction=_dir)
                        _pre_scores[_sym] = _dec.confidence_score
                        # Cache full decision for trade-time reuse (no re-scoring)
                        _cycle_decisions[_sym] = {
                            'decision': _dec,
                            'direction': _dir,
                            'score': _dec.confidence_score,
                            'raw_score': _dec.confidence_score,  # Pre-ML baseline for fair re-score comparison
                        }
                    except Exception:
                        pass
                
                # Attach cached decisions to tools layer for trade-time reuse
                self.tools._cached_cycle_decisions = _cycle_decisions
                
                # Log score distribution
                _scored_above_49 = sum(1 for s in _pre_scores.values() if s >= 49)
                _scored_above_45 = sum(1 for s in _pre_scores.values() if s >= 45)
                print(f"   📊 SCORED {len(_pre_scores)} F&O stocks: {_scored_above_49} score ≥49, {_scored_above_45} score ≥45")
                _top10 = sorted(_pre_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                if _top10:
                    _top10_str = " | ".join(f"{s.replace('NSE:', '')}={v:.0f}" for s, v in _top10)
                    print(f"   🏆 TOP 10: {_top10_str}")
                
                # === ML MOVE PREDICTOR: Full Titan Integration (FAIL-SAFE) ===
                # All ML is wrapped in try/except. If model crashes, _ml_results
                # stays empty and Titan runs identically to pre-ML behavior.
                _ml_results = {}  # symbol -> full titan signals (always defined)
                try:
                    _has_candle = hasattr(self.tools, '_candle_cache')
                    _has_daily = hasattr(self.tools, '_daily_cache')
                    _candle_count = len(getattr(self.tools, '_candle_cache', {}))
                    _daily_count = len(getattr(self.tools, '_daily_cache', {}))
                    if self._ml_predictor and (_has_candle or _has_daily):
                        # Load futures OI data once per cycle (FAIL-SAFE)
                        _futures_oi_cache = {}
                        try:
                            # Reload OI data once per cycle (files may be
                            # refreshed by a background backfill job)
                            from dhan_futures_oi import load_all_futures_oi_daily
                            self._futures_oi_data = load_all_futures_oi_daily()
                            _futures_oi_cache = self._futures_oi_data or {}
                        except Exception:
                            pass
                        
                        _ml_boosted = 0
                        _candle_cache = getattr(self.tools, '_candle_cache', {})
                        _daily_cache = getattr(self.tools, '_daily_cache', {})
                        
                        # === PARALLEL ML PREDICTIONS ===
                        # Each stock's prediction is independent (no shared mutable state).
                        # Feature engineering + XGBoost inference is CPU-bound (~20-50ms/stock).
                        # Parallelising across threads overlaps pandas/numpy GIL-releasing ops.
                        _nifty_5min = getattr(self, '_nifty_5min_df', None)
                        _nifty_daily = getattr(self, '_nifty_daily_df', None)
                        
                        # Load sector index candles (cached, load once per session)
                        if not hasattr(self, '_sector_5min_cache'):
                            self._sector_5min_cache = {}
                            self._sector_daily_cache = {}
                            try:
                                import os
                                _ml_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data')
                                _sec_names = ['METAL', 'IT', 'BANK', 'AUTO', 'PHARMA', 'ENERGY', 'FMCG', 'REALTY', 'INFRA']
                                import pandas as _pd_sector
                                for _sn in _sec_names:
                                    _s5_path = os.path.join(_ml_data_dir, 'candles_5min', f'SECTOR_{_sn}.parquet')
                                    _sd_path = os.path.join(_ml_data_dir, 'candles_daily', f'SECTOR_{_sn}.parquet')
                                    if os.path.exists(_s5_path):
                                        _sdf = _pd_sector.read_parquet(_s5_path)
                                        _sdf['date'] = _pd_sector.to_datetime(_sdf['date'])
                                        self._sector_5min_cache[_sn] = _sdf
                                    if os.path.exists(_sd_path):
                                        _sdd = _pd_sector.read_parquet(_sd_path)
                                        _sdd['date'] = _pd_sector.to_datetime(_sdd['date'])
                                        self._sector_daily_cache[_sn] = _sdd
                                if self._sector_5min_cache:
                                    print(f"   📊 Sector indices loaded: {', '.join(sorted(self._sector_5min_cache.keys()))}")
                            except Exception as _sec_e:
                                print(f"   ⚠ Sector index load: {_sec_e}")
                        
                        # Import sector mapping
                        try:
                            from ml_models.feature_engineering import get_sector_for_symbol as _get_sector
                        except ImportError:
                            _get_sector = lambda s: ''
                        
                        def _predict_one(_sym):
                            """Run ML prediction for a single stock (thread-safe).
                            
                            Data strategy:
                            1. 5-min candles: historical parquet + today's live intraday (preferred)
                            2. daily_df: passed separately for daily context features
                            3. Falls back gracefully if no data available
                            """
                            import pandas as _pd_ml
                            _sym_clean = _sym.replace('NSE:', '')
                            _live_intraday = _candle_cache.get(_sym)  # Today's 5-min bars
                            _daily_df = _daily_cache.get(_sym)        # Historical daily bars
                            _hist_5min = self._hist_5min_cache.get(_sym_clean)  # Stored 5-min parquet
                            
                            # Build best possible 5-min candle series
                            _ml_candles = None
                            if _hist_5min is not None:
                                # Use historical 5-min as base
                                if _live_intraday is not None and len(_live_intraday) >= 2:
                                    # Append today's live candles to historical
                                    try:
                                        _live_copy = _live_intraday.copy()
                                        _live_copy['date'] = _pd_ml.to_datetime(_live_copy['date'])
                                        # Strip timezone if present
                                        if _live_copy['date'].dt.tz is not None:
                                            _live_copy['date'] = _live_copy['date'].dt.tz_localize(None)
                                        _hist_copy = _hist_5min.copy()
                                        if _hist_copy['date'].dt.tz is not None:
                                            _hist_copy['date'] = _hist_copy['date'].dt.tz_localize(None)
                                        # ── GAP CHECK: Don't concat if historical data is too old ──
                                        # If there's a multi-day gap between hist and live data,
                                        # concatenation creates feature artifacts (huge returns,
                                        # distorted momentum) that bias the direction model.
                                        _hist_last_date = _hist_copy['date'].max()
                                        _live_first_date = _live_copy['date'].min()
                                        _gap_days = (_live_first_date - _hist_last_date).days
                                        if _gap_days > 3:
                                            # Gap too large — use hist alone (contiguous features)
                                            _ml_candles = _hist_5min.tail(500)
                                        else:
                                            # Small gap (yesterday/weekend) — safe to concat
                                            _hist_tail = _hist_copy.tail(500)
                                            # Deduplicate: drop historical rows that overlap with live
                                            _live_start = _live_copy['date'].min()
                                            _hist_tail = _hist_tail[_hist_tail['date'] < _live_start]
                                            _common_cols = [c for c in ['date','open','high','low','close','volume'] if c in _hist_tail.columns and c in _live_copy.columns]
                                            _ml_candles = _pd_ml.concat([_hist_tail[_common_cols], _live_copy[_common_cols]], ignore_index=True)
                                    except Exception:
                                        _ml_candles = _hist_5min.tail(500)
                                else:
                                    _ml_candles = _hist_5min.tail(500)
                            elif _live_intraday is not None and len(_live_intraday) >= 50:
                                _ml_candles = _live_intraday
                            
                            if _ml_candles is None or len(_ml_candles) < 50:
                                return _sym, None
                            
                            try:
                                _fut_oi = _futures_oi_cache.get(_sym_clean)
                                _sec_name = _get_sector(_sym_clean)
                                _sec_5m = self._sector_5min_cache.get(_sec_name) if _sec_name else None
                                _sec_dl = self._sector_daily_cache.get(_sec_name) if _sec_name else None
                                _pred = self._ml_predictor.get_titan_signals(
                                    _ml_candles,
                                    daily_df=_daily_df,
                                    futures_oi_df=_fut_oi,
                                    nifty_5min_df=_nifty_5min,
                                    nifty_daily_df=_nifty_daily,
                                    sector_5min_df=_sec_5m,
                                    sector_daily_df=_sec_dl
                                )
                                return _sym, _pred
                            except Exception:
                                return _sym, None
                        
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        _ml_predictions = {}  # symbol -> prediction dict
                        # === PERF: Only run ML for stocks with score >= 40 ===
                        # Max ML boost is +10, pass threshold is 52. Score 40 + boost 10 = 50 (close).
                        # Stocks below 40 won't trade regardless of ML result.
                        _ML_SCORE_THRESHOLD = 40
                        _ml_eligible = [s for s in list(_pre_scores.keys()) if _pre_scores.get(s, 0) >= _ML_SCORE_THRESHOLD]
                        _ml_skipped = len(_pre_scores) - len(_ml_eligible)
                        with ThreadPoolExecutor(max_workers=8) as _ml_executor:
                            _ml_futures = {_ml_executor.submit(_predict_one, s): s for s in _ml_eligible}
                            for _fut in as_completed(_ml_futures):
                                _sym, _pred = _fut.result()
                                if _pred:
                                    _ml_predictions[_sym] = _pred
                        
                        # Merge predictions into scores (single-threaded — modifies shared state)
                        for _sym, _ml_pred in _ml_predictions.items():
                            if _ml_pred.get('ml_score_boost', 0) != 0:
                                _boost = _ml_pred['ml_score_boost']
                                _pre_scores[_sym] += _boost
                                _ml_boosted += 1
                            if _ml_pred.get('ml_signal') != 'UNKNOWN':
                                _ml_results[_sym] = _ml_pred
                                if _sym in _cycle_decisions:
                                    _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                                    _cycle_decisions[_sym]['ml_prediction'] = _ml_pred
                        if _ml_boosted > 0 or _ml_results:
                            _ml_up = sum(1 for r in _ml_results.values() if r.get('ml_signal') == 'UP')
                            _ml_down = sum(1 for r in _ml_results.values() if r.get('ml_signal') == 'DOWN')
                            _ml_flat = sum(1 for r in _ml_results.values() if r.get('ml_signal') == 'FLAT')
                            _ml_caution = sum(1 for r in _ml_results.values() if r.get('ml_entry_caution'))
                            _skip_note = f" | {_ml_skipped} skipped(score<{_ML_SCORE_THRESHOLD})" if _ml_skipped > 0 else ""
                            print(f"   🧠 ML: {len(_ml_results)} analyzed | {_ml_boosted} score-adjusted | {_ml_up} UP | {_ml_down} DOWN | {_ml_flat} FLAT | {_ml_caution} CAUTION{_skip_note}")
                        else:
                            _eligible = sum(1 for s in _pre_scores if (_candle_cache.get(s) is not None and len(_candle_cache.get(s, [])) >= 50) or (_daily_cache.get(s) is not None and len(_daily_cache.get(s, [])) >= 50))
                            print(f"   🧠 ML: NO predictions (candle_cache={_candle_count}, daily_cache={_daily_count}, eligible={_eligible}/{len(_pre_scores)})")
                except Exception as _ml_err:
                    print(f"   ⚠️ ML predictor error (non-fatal, continuing without ML): {_ml_err}")
                
                # Store ML results for downstream use (GPT prompt, sizing, etc.)
                self._cycle_ml_results = _ml_results
                
                # === OI FLOW OVERLAY: Adjust ML predictions with live options chain data ===
                # Only analyze top-scoring F&O stocks to minimize API calls (max 15)
                # FAIL-SAFE: If OI analyzer crashes, _ml_results stays unchanged
                _oi_results = {}
                try:
                    if self._oi_analyzer and _ml_results:
                        # Pick top stocks worth analyzing OI for (scored >=45 + have ML data)
                        _oi_candidates = [
                            sym for sym in list(_pre_scores.keys())
                            if _pre_scores.get(sym, 0) >= 45 and sym in _ml_results
                        ]
                        # Sort by score descending, limit to 15 to save API calls
                        _oi_candidates.sort(key=lambda s: _pre_scores.get(s, 0), reverse=True)
                        _oi_candidates = _oi_candidates[:15]
                        
                        _oi_adjusted_count = 0
                        
                        # === PARALLEL OI ANALYSIS ===
                        # Each stock's option chain fetch is independent network I/O.
                        # Parallelising overlaps network latency (~200-500ms/stock).
                        def _analyze_oi_one(_oi_sym):
                            """Fetch & analyze OI for one stock (thread-safe)."""
                            try:
                                _oi_data = self._oi_analyzer.analyze(_oi_sym)
                                return _oi_sym, _oi_data
                            except Exception:
                                return _oi_sym, None
                        
                        from concurrent.futures import ThreadPoolExecutor, as_completed
                        _oi_raw = {}
                        with ThreadPoolExecutor(max_workers=5) as _oi_executor:
                            _oi_futures = {_oi_executor.submit(_analyze_oi_one, s): s for s in _oi_candidates}
                            for _fut in as_completed(_oi_futures):
                                _sym, _data = _fut.result()
                                if _data:
                                    _oi_raw[_sym] = _data
                        
                        # Apply OI overlay to ML predictions (single-threaded — modifies shared dicts)
                        for _oi_sym, _oi_data in _oi_raw.items():
                            try:
                                if _oi_data.get('flow_bias') != 'NEUTRAL':
                                    _oi_results[_oi_sym] = _oi_data
                                if self._ml_predictor and _oi_sym in _ml_results:
                                    self._ml_predictor.apply_oi_overlay(_ml_results[_oi_sym], _oi_data)
                                    if _ml_results[_oi_sym].get('oi_adjusted'):
                                        _oi_adjusted_count += 1
                                        if _oi_sym in _cycle_decisions:
                                            _cycle_decisions[_oi_sym]['ml_prediction'] = _ml_results[_oi_sym]
                            except Exception:
                                pass
                        
                        if _oi_adjusted_count > 0 or _oi_results:
                            print(f"   📊 OI: {len(_oi_candidates)} analyzed | {len(_oi_results)} directional | {_oi_adjusted_count} ML-adjusted")
                except Exception as _oi_err:
                    print(f"   ⚠️ OI analyzer error (non-fatal): {_oi_err}")
                
                # === OI CROSS-VALIDATION ON SCORES ===
                # If OI says BEARISH but scored direction is BUY (or vice versa),
                # penalize score. OI flow (PCR, IV skew, MaxPain, buildup) reflects
                # institutional positioning — conflicting with it is risky.
                # JINDALSTEL lesson: scored BUY 62, OI was BEARISH — stock reversed.
                _oi_score_adjusted = 0
                for _oi_sym, _oi_data in _oi_results.items():
                    try:
                        _oi_bias = _oi_data.get('flow_bias', 'NEUTRAL')
                        _oi_conf = _oi_data.get('flow_confidence', 0.0)
                        if _oi_bias == 'NEUTRAL' or _oi_conf < 0.55:
                            continue
                        # Get scored direction from cycle decision
                        _cd = _cycle_decisions.get(_oi_sym)
                        if not _cd or not _cd.get('decision'):
                            continue
                        _scored_dir = _cd['decision'].recommended_direction
                        if _scored_dir == 'HOLD':
                            continue
                        # Cross-validate: OI direction vs scored direction
                        _oi_dir = 'BUY' if _oi_bias == 'BULLISH' else 'SELL'
                        if _scored_dir != _oi_dir:
                            # OI conflicts with scored direction → penalize
                            _oi_penalty = -5 if _oi_conf >= 0.70 else -3
                            _pre_scores[_oi_sym] += _oi_penalty
                            if _oi_sym in _cycle_decisions:
                                _cycle_decisions[_oi_sym]['score'] = _pre_scores[_oi_sym]
                            _oi_score_adjusted += 1
                    except Exception:
                        pass
                if _oi_score_adjusted > 0:
                    print(f"   📊 OI CROSS-VAL: {_oi_score_adjusted} stocks penalized for direction conflict")

                # === SECTOR INDEX CROSS-VALIDATION ON SCORES ===
                # If sector index (NIFTY METAL, NIFTY IT, etc.) is bearish but stock
                # is scored as BUY (CE), penalize. A single stock rarely sustains a
                # rally when its entire sector is falling.
                # Exception: if stock is outperforming sector by 2x+, skip penalty
                # (genuine rotation/divergence).
                _sector_stock_map = {
                    'METALS': {
                        'index': 'NSE:NIFTY METAL',
                        'stocks': {'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL',
                                   'NMDC', 'NATIONALUM', 'HINDZINC', 'SAIL', 'HINDCOPPER',
                                   'APLAPOLLO', 'RATNAMANI', 'WELCORP', 'COALINDIA'},
                    },
                    'IT': {
                        'index': 'NSE:NIFTY IT',
                        'stocks': {'INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM',
                                   'KPITTECH', 'COFORGE', 'MPHASIS', 'PERSISTENT'},
                    },
                    'BANKS': {
                        'index': 'NSE:NIFTY BANK',
                        'stocks': {'SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK',
                                   'BANKBARODA', 'PNB', 'IDFCFIRSTB', 'INDUSINDBK', 'FEDERALBNK',
                                   'RBLBANK', 'UNIONBANK', 'CANBK', 'AUBANK'},
                    },
                    'AUTO': {
                        'index': 'NSE:NIFTY AUTO',
                        'stocks': {'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO',
                                   'EICHERMOT', 'ASHOKLEY', 'BHARATFORG', 'MOTHERSON', 'BALKRISIND'},
                    },
                    'PHARMA': {
                        'index': 'NSE:NIFTY PHARMA',
                        'stocks': {'SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'AUROPHARMA',
                                   'BIOCON', 'LUPIN', 'APOLLOHOSP', 'MAXHEALTH', 'LALPATHLAB'},
                    },
                    'ENERGY': {
                        'index': 'NSE:NIFTY ENERGY',
                        'stocks': {'RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN',
                                   'TATAPOWER', 'ADANIENT', 'BPCL', 'IOC', 'GAIL'},
                    },
                    'FMCG': {
                        'index': 'NSE:NIFTY FMCG',
                        'stocks': {'ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR',
                                   'GODREJCP', 'MARICO', 'COLPAL', 'TATACONSUM', 'VBL'},
                    },
                    'REALTY': {
                        'index': 'NSE:NIFTY REALTY',
                        'stocks': {'DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD',
                                   'BRIGADE', 'LODHA', 'SOBHA'},
                    },
                    'INFRA': {
                        'index': 'NSE:NIFTY INFRA',
                        'stocks': {'LT', 'ADANIPORTS', 'ULTRACEMCO', 'GRASIM', 'SHREECEM',
                                   'AMBUJACEM', 'ACC', 'SIEMENS', 'ABB', 'BEL', 'HAL',
                                   'BHEL', 'INOXWIND'},
                    },
                }
                # Build reverse lookup: stock_name → sector info
                _stock_to_sector = {}
                for _sec_name, _sec_info in _sector_stock_map.items():
                    for _stk in _sec_info['stocks']:
                        _stock_to_sector[_stk] = (_sec_name, _sec_info['index'])

                # Fetch sector index change% from ticker
                _sector_index_changes = {}
                try:
                    _sec_idx_symbols = [v['index'] for v in _sector_stock_map.values()]
                    _sec_quotes = self.tools.ticker.get_quote_batch(_sec_idx_symbols) if self.tools.ticker else {}
                    for _sec_sym, _sec_q in _sec_quotes.items():
                        if _sec_q:
                            _sec_ohlc = _sec_q.get('ohlc', {})
                            _sec_prev = _sec_ohlc.get('close', 0)
                            _sec_ltp = _sec_q.get('last_price', 0)
                            if _sec_prev > 0 and _sec_ltp > 0:
                                _sector_index_changes[_sec_sym] = ((_sec_ltp - _sec_prev) / _sec_prev) * 100
                except Exception:
                    pass

                _sector_score_adjusted = 0
                _sector_penalized_details = []
                for _sym, _score in list(_pre_scores.items()):
                    try:
                        _stock_name = _sym.replace('NSE:', '')
                        _sec_match = _stock_to_sector.get(_stock_name)
                        if not _sec_match:
                            continue
                        _sec_name, _sec_index = _sec_match
                        _sec_chg = _sector_index_changes.get(_sec_index)
                        if _sec_chg is None:
                            continue

                        # Get scored direction
                        _cd = _cycle_decisions.get(_sym)
                        if not _cd or not _cd.get('decision'):
                            continue
                        _scored_dir = _cd['decision'].recommended_direction
                        if _scored_dir == 'HOLD':
                            continue

                        # Get stock's own change%
                        _stk_data = market_data.get(_sym, {})
                        _stk_chg = 0
                        if isinstance(_stk_data, dict):
                            _stk_chg = _stk_data.get('change_pct', 0)
                            if not _stk_chg:
                                _stk_ohlc = _stk_data.get('ohlc', {})
                                _stk_prev = _stk_ohlc.get('close', 0)
                                _stk_ltp = _stk_data.get('last_price', 0)
                                if _stk_prev > 0 and _stk_ltp > 0:
                                    _stk_chg = ((_stk_ltp - _stk_prev) / _stk_prev) * 100

                        # Sector BEARISH + stock scored BUY (CE)
                        if _sec_chg <= -1.0 and _scored_dir == 'BUY':
                            # Exception: stock outperforming sector by 2x+ → genuine divergence
                            if _stk_chg > 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                                continue
                            _sec_penalty = -5 if _sec_chg <= -2.0 else -3
                            _pre_scores[_sym] += _sec_penalty
                            if _sym in _cycle_decisions:
                                _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                            _sector_score_adjusted += 1
                            _sector_penalized_details.append(f"{_stock_name}({_sec_name}:{_sec_chg:+.1f}%→{_sec_penalty})")

                        # Sector BULLISH + stock scored SELL (PE)
                        elif _sec_chg >= 1.0 and _scored_dir == 'SELL':
                            if _stk_chg < 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                                continue
                            _sec_penalty = -5 if _sec_chg >= 2.0 else -3
                            _pre_scores[_sym] += _sec_penalty
                            if _sym in _cycle_decisions:
                                _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                            _sector_score_adjusted += 1
                            _sector_penalized_details.append(f"{_stock_name}({_sec_name}:{_sec_chg:+.1f}%→{_sec_penalty})")
                    except Exception:
                        pass

                if _sector_score_adjusted > 0:
                    print(f"   🏭 SECTOR CROSS-VAL: {_sector_score_adjusted} stocks penalized — {', '.join(_sector_penalized_details[:5])}")
                elif _sector_index_changes:
                    _bearish_secs = [f"{k.replace('NSE:NIFTY ', '')}:{v:+.1f}%" for k, v in _sector_index_changes.items() if v <= -1.0]
                    if _bearish_secs:
                        print(f"   🏭 SECTOR INDEX: Bearish sectors: {', '.join(_bearish_secs)} (no scored stocks conflicting)")

                # Cache sector index changes for GPT prompt section
                self._sector_index_changes_cache = _sector_index_changes

                # Store OI results for GPT prompt
                self._cycle_oi_results = _oi_results
                
                # === DOWN-RISK SOFT SCORING: Adjust pre_scores ±5 based on model ===
                # Must happen after ML predictions are merged so ml_down_risk_score is available.
                # Does NOT block any trades — just nudges scores for natural prioritisation.
                try:
                    self._apply_down_risk_soft_scores(_ml_results, _pre_scores)
                except Exception as _dr_err:
                    print(f"   ⚠️ Down-risk soft scoring error (non-fatal): {_dr_err}")
                
                # === MODEL-TRACKER TRADES: Place up to 7 smart-selected model-only trades ===
                # Independent of main workflow — purely for evaluating down-risk model.
                try:
                    _model_tracker_placed = self._place_model_tracker_trades(
                        _ml_results, _pre_scores, market_data, datetime.now().strftime('%H:%M:%S')
                    )
                except Exception as _mt_err:
                    print(f"   ⚠️ Model-tracker error (non-fatal): {_mt_err}")
                    _model_tracker_placed = []
                
                # === GMM SNIPER TRADE: 1 highest-conviction trade per cycle, 2x lots ===
                # Picks the cleanest GMM candidate (lowest dr_score) with strict gates.
                # Separate from model-tracker — this is the alpha trade.
                try:
                    _sniper_placed = self._place_gmm_sniper_trade(
                        _ml_results, _pre_scores, market_data, datetime.now().strftime('%H:%M:%S')
                    )
                except Exception as _snp_err:
                    print(f"   ⚠️ GMM Sniper error (non-fatal): {_snp_err}")
                    _sniper_placed = None
                
                # === SNIPER STRATEGIES: OI Unwinding + PCR Extreme ===
                # Scans OI data for high-edge reversal / contrarian setups.
                # Independent from GMM Sniper — these use OI flow signals + GMM confirmation.
                try:
                    if getattr(self, '_sniper_engine', None) and _oi_results and _ml_results:
                        _active_syms = {p.get('underlying', '') for p in getattr(self.tools, 'paper_positions', [])
                                       if p.get('status', 'OPEN') == 'OPEN'}
                        _cycle_time_now = datetime.now().strftime('%H:%M:%S')
                        
                        # --- Strategy 1: OI Unwinding Reversal ---
                        _oi_unwind_candidates = self._sniper_engine.scan_oi_unwinding(
                            oi_results=_oi_results, ml_results=_ml_results, pre_scores=_pre_scores,
                            market_data=market_data,
                            active_symbols=_active_syms, model_tracker_symbols=self._model_tracker_symbols,
                            gmm_sniper_symbols=self._gmm_sniper_symbols,
                        )
                        _oi_unwind_placed = set()
                        for _oiu_cand in _oi_unwind_candidates:
                            try:
                                _oiu_cfg = getattr(self._sniper_engine, '_oi_cfg', {})
                                _oiu_lot_mult = _oiu_cfg.get('lot_multiplier', 1.5)
                                print(f"\n   🔫 SNIPER-OIUnwinding: {_oiu_cand['sym_clean']} ({_oiu_cand['direction']}) "
                                      f"| {_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f} "
                                      f"| chg={_oiu_cand.get('change_pct', 0):+.1f}% RSI={_oiu_cand.get('rsi', 50):.0f} "
                                      f"| exhaust={_oiu_cand.get('exhaustion_score', 0):.2f} "
                                      f"| spot→SR {_oiu_cand['dist_from_sr_pct']:.1f}% "
                                      f"| dr={_oiu_cand['dr_score']:.4f} | smart={_oiu_cand['smart_score']:.1f} "
                                      f"| {_oiu_lot_mult}x lots")
                                _oiu_result = self.tools.place_option_order(
                                    underlying=_oiu_cand['sym'], direction=_oiu_cand['direction'],
                                    strike_selection="ATM", use_intraday_scoring=False,
                                    lot_multiplier=_oiu_lot_mult,
                                    rationale=(f"SNIPER_OI_UNWINDING: {_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f}, "
                                              f"spot→SR={_oiu_cand['dist_from_sr_pct']:.1f}%, dr={_oiu_cand['dr_score']:.4f}, "
                                              f"smart={_oiu_cand['smart_score']:.1f}, lots={_oiu_lot_mult}x"),
                                    setup_type='SNIPER_OI_UNWINDING',
                                    ml_data=_oiu_cand.get('ml_data', {}),
                                )
                                if _oiu_result and _oiu_result.get('success'):
                                    self._sniper_engine.record_oi_trade(_oiu_cand['sym'])
                                    _oi_unwind_placed.add(_oiu_cand['sym'])
                                    print(f"   ✅ SNIPER-OIUnwinding PLACED: {_oiu_cand['sym_clean']} ({_oiu_cand['direction']})")
                                    self._log_decision(_cycle_time_now, _oiu_cand['sym'], _oiu_cand['p_score'],
                                                      'SNIPER_OI_UNWINDING_PLACED',
                                                      reason=f"OI={_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f}, dr={_oiu_cand['dr_score']:.4f}",
                                                      direction=_oiu_cand['direction'], setup='SNIPER_OI_UNWINDING')
                                else:
                                    _oiu_err = _oiu_result.get('error', 'unknown') if _oiu_result else 'no result'
                                    print(f"   ⚠️ Sniper-OIUnwinding failed for {_oiu_cand['sym_clean']}: {_oiu_err}")
                            except Exception as _oiu_place_err:
                                print(f"   ❌ Sniper-OIUnwinding error for {_oiu_cand['sym_clean']}: {_oiu_place_err}")
                        
                        if _oi_unwind_placed:
                            _oiu_status = self._sniper_engine.get_status()['oi_unwinding']
                            print(f"   📊 Sniper-OIUnwinding: {_oiu_status['trades_today']}/{_oiu_status['max_per_day']} today")
                        
                        # --- Strategy 2: PCR Extreme Fade ---
                        _pcr_candidates = self._sniper_engine.scan_pcr_extreme(
                            oi_results=_oi_results, ml_results=_ml_results, pre_scores=_pre_scores,
                            market_data=market_data,
                            active_symbols=_active_syms, model_tracker_symbols=self._model_tracker_symbols,
                            gmm_sniper_symbols=self._gmm_sniper_symbols,
                            oi_unwinding_symbols=_oi_unwind_placed,
                        )
                        for _pcr_cand in _pcr_candidates:
                            try:
                                _pcr_cfg_val = getattr(self._sniper_engine, '_pcr_cfg', {})
                                _pcr_lot_mult = _pcr_cfg_val.get('lot_multiplier', 1.5)
                                _pcr_thr_info = f"thr={_pcr_cand.get('adaptive_oversold_thr', 1.35):.2f}/{_pcr_cand.get('adaptive_overbought_thr', 0.65):.2f}"
                                _pcr_idx_tag = ' IDX✓' if _pcr_cand.get('index_agrees') else ''
                                print(f"\n   🔫 SNIPER-PCRExtreme: {_pcr_cand['sym_clean']} ({_pcr_cand['direction']}) "
                                      f"| PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}) "
                                      f"| edge={_pcr_cand['pcr_edge']:.3f} {_pcr_thr_info}{_pcr_idx_tag} "
                                      f"| dr={_pcr_cand['dr_score']:.4f} | smart={_pcr_cand['smart_score']:.1f} "
                                      f"| {_pcr_lot_mult}x lots")
                                _pcr_result = self.tools.place_option_order(
                                    underlying=_pcr_cand['sym'], direction=_pcr_cand['direction'],
                                    strike_selection="ATM", use_intraday_scoring=False,
                                    lot_multiplier=_pcr_lot_mult,
                                    rationale=(f"SNIPER_PCR_EXTREME: PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}), "
                                              f"edge={_pcr_cand['pcr_edge']:.3f}, dr={_pcr_cand['dr_score']:.4f}, "
                                              f"smart={_pcr_cand['smart_score']:.1f}, lots={_pcr_lot_mult}x"),
                                    setup_type='SNIPER_PCR_EXTREME',
                                    ml_data=_pcr_cand.get('ml_data', {}),
                                )
                                if _pcr_result and _pcr_result.get('success'):
                                    self._sniper_engine.record_pcr_trade(_pcr_cand['sym'])
                                    print(f"   ✅ SNIPER-PCRExtreme PLACED: {_pcr_cand['sym_clean']} ({_pcr_cand['direction']}) "
                                          f"PCR={_pcr_cand['blended_pcr']:.3f}")
                                    self._log_decision(_cycle_time_now, _pcr_cand['sym'], _pcr_cand['p_score'],
                                                      'SNIPER_PCR_EXTREME_PLACED',
                                                      reason=f"PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}), dr={_pcr_cand['dr_score']:.4f}",
                                                      direction=_pcr_cand['direction'], setup='SNIPER_PCR_EXTREME')
                                else:
                                    _pcr_err = _pcr_result.get('error', 'unknown') if _pcr_result else 'no result'
                                    print(f"   ⚠️ Sniper-PCRExtreme failed for {_pcr_cand['sym_clean']}: {_pcr_err}")
                            except Exception as _pcr_place_err:
                                print(f"   ❌ Sniper-PCRExtreme error for {_pcr_cand['sym_clean']}: {_pcr_place_err}")
                        
                        _pcr_placed_count = len([c for c in _pcr_candidates if c['sym'] in getattr(self._sniper_engine, '_pcr_symbols', set())])
                        if _pcr_placed_count:
                            _pcr_status = self._sniper_engine.get_status()['pcr_extreme']
                            print(f"   📊 Sniper-PCRExtreme: {_pcr_status['trades_today']}/{_pcr_status['max_per_day']} today")
                except Exception as _snp_strat_err:
                    print(f"   ⚠️ Sniper Strategies error (non-fatal): {_snp_strat_err}")
                
                # === OI SNAPSHOT LOGGER: Collect data for future model retraining ===
                # Now includes NSE OI change data (richer than Kite-only snapshots)
                try:
                    if _oi_results:
                        import json
                        _oi_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'oi_snapshots')
                        os.makedirs(_oi_log_path, exist_ok=True)
                        _oi_log_file = os.path.join(_oi_log_path, f"oi_{datetime.now().strftime('%Y%m%d')}.jsonl")
                        with open(_oi_log_file, 'a', encoding='utf-8') as _oi_f:
                            for _oi_sym, _oi_d in _oi_results.items():
                                _oi_snapshot = {
                                    'timestamp': datetime.now().isoformat(),
                                    'symbol': _oi_sym,
                                    'pcr_oi': _oi_d.get('pcr_oi'),
                                    'iv_skew': _oi_d.get('iv_skew'),
                                    'max_pain': _oi_d.get('max_pain'),
                                    'spot_price': _oi_d.get('spot_price'),
                                    'flow_bias': _oi_d.get('flow_bias'),
                                    'flow_confidence': _oi_d.get('flow_confidence'),
                                    'call_resistance': _oi_d.get('call_resistance'),
                                    'put_support': _oi_d.get('put_support'),
                                    'ml_move_prob': _ml_results.get(_oi_sym, {}).get('ml_move_prob'),
                                    'ml_signal': _ml_results.get(_oi_sym, {}).get('ml_signal'),
                                    'ml_prob_up': _ml_results.get(_oi_sym, {}).get('ml_prob_up'),
                                    'ml_prob_down': _ml_results.get(_oi_sym, {}).get('ml_prob_down'),
                                    'ml_direction_hint': _ml_results.get(_oi_sym, {}).get('ml_direction_hint'),
                                    'score': _pre_scores.get(_oi_sym, 0),
                                    # NSE-exclusive fields (OI change — unavailable from Kite)
                                    'nse_call_oi_change': _oi_d.get('nse_total_call_oi_change', None),
                                    'nse_put_oi_change': _oi_d.get('nse_total_put_oi_change', None),
                                    'nse_pcr_volume': _oi_d.get('nse_pcr_volume', None),
                                    'nse_oi_buildup': _oi_d.get('nse_oi_buildup', None),
                                    'nse_oi_buildup_strength': _oi_d.get('nse_oi_buildup_strength', None),
                                    'nse_enriched': _oi_d.get('nse_enriched', False),
                                    'oi_source': _oi_d.get('oi_source', 'NSE'),
                                    # DhanHQ exclusive: ATM Greeks
                                    'atm_greeks': _oi_d.get('atm_greeks', None),
                                }
                                _oi_f.write(json.dumps(_oi_snapshot) + '\n')
                        
                        # Log dedicated DhanHQ snapshots (richest: OI + Greeks + bid/ask)
                        if self._oi_analyzer and hasattr(self._oi_analyzer, 'get_dhan_snapshot'):
                            _dhan_log_file = os.path.join(_oi_log_path, f"dhan_oi_{datetime.now().strftime('%Y%m%d')}.jsonl")
                            _dhan_logged = 0
                            with open(_dhan_log_file, 'a', encoding='utf-8') as _dhan_f:
                                for _oi_sym in _oi_results:
                                    _dhan_snap = self._oi_analyzer.get_dhan_snapshot(_oi_sym)
                                    if _dhan_snap:
                                        _dhan_snap['ml_move_prob'] = _ml_results.get(_oi_sym, {}).get('ml_move_prob')
                                        _dhan_snap['ml_signal'] = _ml_results.get(_oi_sym, {}).get('ml_signal')
                                        _dhan_snap['score'] = _pre_scores.get(_oi_sym, 0)
                                        _dhan_f.write(json.dumps(_dhan_snap) + '\n')
                                        _dhan_logged += 1
                            if _dhan_logged > 0:
                                print(f"   📋 DhanHQ OI snapshots logged: {_dhan_logged} stocks")
                        
                        # Also log dedicated NSE snapshots (full strike-level data)
                        if self._oi_analyzer and hasattr(self._oi_analyzer, 'get_nse_snapshot'):
                            _nse_log_file = os.path.join(_oi_log_path, f"nse_oi_{datetime.now().strftime('%Y%m%d')}.jsonl")
                            _nse_logged = 0
                            with open(_nse_log_file, 'a', encoding='utf-8') as _nse_f:
                                for _oi_sym in _oi_results:
                                    _nse_snap = self._oi_analyzer.get_nse_snapshot(_oi_sym)
                                    if _nse_snap:
                                        _nse_snap['ml_move_prob'] = _ml_results.get(_oi_sym, {}).get('ml_move_prob')
                                        _nse_snap['ml_signal'] = _ml_results.get(_oi_sym, {}).get('ml_signal')
                                        _nse_snap['score'] = _pre_scores.get(_oi_sym, 0)
                                        _nse_f.write(json.dumps(_nse_snap) + '\n')
                                        _nse_logged += 1
                            if _nse_logged > 0:
                                print(f"   📋 NSE OI snapshots logged: {_nse_logged} stocks")
                except Exception:
                    pass  # Logging failure is never fatal
                
                # === OI COLLECTOR: Structured parquet snapshots for ML training ===
                try:
                    if _oi_results:
                        from ml_models.oi_collector import OICollector
                        if not hasattr(self, '_oi_collector'):
                            self._oi_collector = OICollector()
                        for _oi_sym, _oi_d in _oi_results.items():
                            self._oi_collector.collect(_oi_sym.replace('NSE:', ''), _oi_d)
                except Exception:
                    pass  # OI collection failure is never fatal
                
            except Exception as _e:
                print(f"   ⚠️ Scoring failed: {_e}")
                self.tools._cached_cycle_decisions = {}
                _ml_results = {}  # Ensure defined even on scoring failure
            
            # === REVERSAL SNIPE MANAGEMENT: Trailing SL + Time Guard ===
            # Snipe trades target 10% with trailing SL. Time guard cuts at 12 min if < 3%.
            try:
                _snipe_positions = [t for t in self.tools.paper_positions 
                                   if t.get('status', 'OPEN') == 'OPEN' and t.get('is_reversal_snipe')]
                for snipe in _snipe_positions:
                    snipe_sym = snipe.get('symbol', '')
                    snipe_entry_price = snipe.get('avg_price', 0)
                    snipe_entry_time_str = snipe.get('snipe_entry_time', '')
                    if not snipe_entry_time_str or not snipe_entry_price:
                        continue
                    
                    try:
                        snipe_entry_dt = datetime.fromisoformat(snipe_entry_time_str)
                    except Exception:
                        continue
                    
                    held_minutes = (datetime.now() - snipe_entry_dt).total_seconds() / 60
                    
                    # Get current price
                    try:
                        _snipe_ltp_data = self.tools.kite.ltp([snipe_sym])
                        if snipe_sym not in _snipe_ltp_data:
                            continue
                        snipe_ltp = _snipe_ltp_data[snipe_sym]['last_price']
                    except Exception:
                        continue
                    
                    snipe_pnl_pct = ((snipe_ltp - snipe_entry_price) / snipe_entry_price) * 100
                    
                    # === TRAILING SL: Lock in profits as price moves up ===
                    # Activates at +6% gain, then trails giving back 60% of gains from high watermark.
                    # Example: entry ₹100, HWM reaches ₹108 (+8%) → trailing SL = ₹108 - (8*0.5)% = ₹104 (+4%)
                    # This ensures minimum +4% profit once trailing kicks in.
                    _hwm = snipe.get('snipe_high_watermark', snipe_entry_price)
                    if snipe_ltp > _hwm:
                        # New high watermark — update
                        with self.tools._positions_lock:
                            snipe['snipe_high_watermark'] = snipe_ltp
                        _hwm = snipe_ltp
                    
                    _hwm_gain_pct = ((_hwm - snipe_entry_price) / snipe_entry_price) * 100
                    
                    # Activate trailing once gain reaches 4%
                    if _hwm_gain_pct >= 6.0 and not snipe.get('snipe_trailing_active'):
                        with self.tools._positions_lock:
                            snipe['snipe_trailing_active'] = True
                        underlying_snipe = snipe.get('underlying', '')
                        print(f"   📈 SNIPE TRAILING ACTIVATED: {underlying_snipe} HWM +{_hwm_gain_pct:.1f}%")
                    
                    if snipe.get('snipe_trailing_active'):
                        # Trail at 50% of gains from high watermark
                        # i.e., give back half of peak profit before exiting
                        _trail_give_back = _hwm_gain_pct * 0.60  # Give back 60% — wider room (was 50%)
                        _trailing_sl_price = snipe_entry_price * (1 + (_hwm_gain_pct - _trail_give_back) / 100)
                        
                        # Trailing SL should never be below original SL (entry * 0.94)
                        _original_sl = snipe_entry_price * 0.94
                        _trailing_sl_price = max(_trailing_sl_price, _original_sl)
                        
                        # Update SL if trailing is higher than current SL
                        _current_sl = snipe.get('stop_loss', _original_sl)
                        if _trailing_sl_price > _current_sl:
                            with self.tools._positions_lock:
                                snipe['stop_loss'] = _trailing_sl_price
                        
                        # Check if price dropped below trailing SL
                        if snipe_ltp <= _trailing_sl_price:
                            snipe_qty = snipe.get('quantity', 0)
                            snipe_pnl = (snipe_ltp - snipe_entry_price) * snipe_qty
                            
                            from config import calc_brokerage
                            snipe_brokerage = calc_brokerage(snipe_entry_price, snipe_ltp, snipe_qty)
                            snipe_pnl -= snipe_brokerage
                            
                            underlying_snipe = snipe.get('underlying', '')
                            _locked_pct = ((_trailing_sl_price - snipe_entry_price) / snipe_entry_price) * 100
                            print(f"\n   📈 SNIPE TRAILING SL HIT: {underlying_snipe}")
                            print(f"      Entry: ₹{snipe_entry_price:.2f} → HWM: ₹{_hwm:.2f} (+{_hwm_gain_pct:.1f}%) → Exit: ₹{snipe_ltp:.2f}")
                            print(f"      Trailing SL: ₹{_trailing_sl_price:.2f} (locked +{_locked_pct:.1f}%) | P&L: ₹{snipe_pnl:+,.0f}")
                            
                            _snipe_exit_detail = {
                                'exit_type': 'SNIPE_TRAILING_SL',
                                'exit_reason': f'Trailing SL hit — HWM +{_hwm_gain_pct:.1f}%, trail locked +{_locked_pct:.1f}%',
                                'held_minutes': round(held_minutes, 1),
                                'pnl_pct_at_exit': round(snipe_pnl_pct, 2),
                                'high_watermark': round(_hwm, 2),
                                'hwm_gain_pct': round(_hwm_gain_pct, 2),
                                'trailing_sl_price': round(_trailing_sl_price, 2),
                                'brokerage': round(snipe_brokerage, 2),
                                'was_reversal_snipe': True,
                            }
                            
                            self.tools.update_trade_status(snipe_sym, 'SNIPE_TRAILING_SL', snipe_ltp, snipe_pnl, exit_detail=_snipe_exit_detail)
                            with self._pnl_lock:
                                self.daily_pnl += snipe_pnl
                                self.capital += snipe_pnl
                            
                            _snipe_open = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != snipe_sym]
                            _snipe_unreal = self.risk_governor._calc_unrealized_pnl(_snipe_open)
                            self.risk_governor.record_trade_result(snipe_sym, snipe_pnl, True, unrealized_pnl=_snipe_unreal)
                            self.risk_governor.update_capital(self.capital)
                            
                            self._log_decision(_cycle_time, underlying_snipe, 0, 'SNIPE_TRAILING_SL',
                                              reason=f'HWM +{_hwm_gain_pct:.1f}%, trail hit, P&L={snipe_pnl:+,.0f}',
                                              direction=snipe.get('direction', ''))
                            
                            print(f"      ✅ Snipe trailing SL exit | Profit: ₹{snipe_pnl:+,.0f}")
                            continue  # Don't also check time-guard
                    
                    # === TIME GUARD: 12 min with < 3% → cut ===
                    if held_minutes < 12:
                        continue
                    
                    # If profit ≥ 3%, let it ride to 10% target / trailing SL
                    if snipe_pnl_pct >= 3.0:
                        continue
                    
                    # Time's up and not at 3%+ — exit at market
                    snipe_qty = snipe.get('quantity', 0)
                    snipe_pnl = (snipe_ltp - snipe_entry_price) * snipe_qty
                    
                    from config import calc_brokerage
                    snipe_brokerage = calc_brokerage(snipe_entry_price, snipe_ltp, snipe_qty)
                    snipe_pnl -= snipe_brokerage
                    
                    underlying_snipe = snipe.get('underlying', '')
                    print(f"\n   ⏱️ SNIPE TIME-GUARD: {underlying_snipe} held {held_minutes:.0f}min, P&L {snipe_pnl_pct:+.1f}% (< 3%) — cutting")
                    print(f"      Entry: ₹{snipe_entry_price:.2f} → ₹{snipe_ltp:.2f} | P&L: ₹{snipe_pnl:+,.0f}")
                    
                    _snipe_exit_detail = {
                        'exit_type': 'SNIPE_TIME_GUARD',
                        'exit_reason': f'Reversal snipe held {held_minutes:.0f}min with only {snipe_pnl_pct:+.1f}% — time guard exit',
                        'held_minutes': round(held_minutes, 1),
                        'pnl_pct_at_exit': round(snipe_pnl_pct, 2),
                        'brokerage': round(snipe_brokerage, 2),
                        'was_reversal_snipe': True,
                    }
                    
                    self.tools.update_trade_status(snipe_sym, 'SNIPE_TIME_GUARD', snipe_ltp, snipe_pnl, exit_detail=_snipe_exit_detail)
                    with self._pnl_lock:
                        self.daily_pnl += snipe_pnl
                        self.capital += snipe_pnl
                    
                    _snipe_was_win = snipe_pnl > 0
                    _snipe_open = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != snipe_sym]
                    _snipe_unreal = self.risk_governor._calc_unrealized_pnl(_snipe_open)
                    self.risk_governor.record_trade_result(snipe_sym, snipe_pnl, _snipe_was_win, unrealized_pnl=_snipe_unreal)
                    self.risk_governor.update_capital(self.capital)
                    
                    self._log_decision(_cycle_time, underlying_snipe, 0, 'SNIPE_TIME_GUARD',
                                      reason=f'Snipe held {held_minutes:.0f}min, {snipe_pnl_pct:+.1f}%, P&L={snipe_pnl:+,.0f}',
                                      direction=snipe.get('direction', ''))
                    
                    print(f"      ✅ Snipe time-guard exit | {'Profit' if _snipe_was_win else 'Cut loss'}: ₹{snipe_pnl:+,.0f}")
                    
            except Exception as _snipe_guard_err:
                print(f"   ⚠️ Snipe management error (non-fatal): {_snipe_guard_err}")
            
            # === CONVICTION REVERSAL EXIT: Your own system says you're wrong ===
            # Pro trader rule: if your system flips direction on a stock you're holding
            # with HIGH conviction (score ≥ 70), exit IMMEDIATELY. Don't wait for SL.
            # GODREJCP: entered BUY at 11:15 score 81, system said SELL at 11:24 score 78.
            # Instead of waiting 23 more min for SL (-₹23K), exit NOW at smaller loss.
            try:
                active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
                for trade in active_trades:
                    # Only check naked options (not spreads/ICs)
                    if trade.get('is_credit_spread') or trade.get('is_debit_spread') or trade.get('is_iron_condor'):
                        continue
                    
                    underlying = trade.get('underlying', '')
                    if not underlying or underlying not in _cycle_decisions:
                        continue
                    
                    cached = _cycle_decisions[underlying]
                    new_decision = cached.get('decision')
                    if not new_decision:
                        continue
                    
                    new_direction = new_decision.recommended_direction
                    new_score = new_decision.confidence_score
                    trade_direction = trade.get('direction', '')
                    
                    # Check: is the system now saying the OPPOSITE with high conviction?
                    direction_flipped = (
                        (trade_direction == 'BUY' and new_direction == 'SELL') or
                        (trade_direction == 'SELL' and new_direction == 'BUY')
                    )
                    
                    if direction_flipped and new_score >= 70:
                        symbol = trade.get('symbol', '')
                        entry_price = trade.get('avg_price', 0)
                        
                        # Get current price
                        try:
                            ltp_data = self.tools.kite.ltp([symbol])
                            if symbol in ltp_data:
                                ltp = ltp_data[symbol]['last_price']
                            else:
                                continue  # Can't get price — skip
                        except Exception:
                            continue
                        
                        pnl = (ltp - entry_price) * trade.get('quantity', 0)
                        if trade.get('side') == 'SELL':
                            pnl = -pnl
                        
                        entry_score = trade.get('entry_score', 0)
                        
                        print(f"\n   🔄 CONVICTION REVERSAL EXIT: {underlying}")
                        print(f"      Held: {trade_direction} (entry score {entry_score:.0f})")
                        print(f"      Now: system says {new_direction} with score {new_score:.0f}")
                        print(f"      Entry: ₹{entry_price:.2f} → Current: ₹{ltp:.2f}")
                        print(f"      P&L: ₹{pnl:+,.0f} — EXITING NOW instead of waiting for SL")
                        
                        # Exit immediately
                        from config import calc_brokerage
                        brokerage = calc_brokerage(entry_price, ltp, trade.get('quantity', 0))
                        pnl -= brokerage
                        
                        exit_detail = {
                            'exit_type': 'CONVICTION_REVERSAL',
                            'exit_reason': f'System flipped to {new_direction} with score {new_score:.0f} — own conviction reversed',
                            'new_direction': new_direction,
                            'new_score': new_score,
                            'entry_score': entry_score,
                            'brokerage': round(brokerage, 2),
                        }
                        
                        self.tools.update_trade_status(symbol, 'CONVICTION_REVERSAL', ltp, pnl, exit_detail=exit_detail)
                        with self._pnl_lock:
                            self.daily_pnl += pnl
                            self.capital += pnl
                        
                        was_win = pnl > 0
                        open_pos = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN' and t.get('symbol') != symbol]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol, pnl, was_win, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)
                        
                        # Log for audit
                        self._log_decision(_cycle_time, underlying, new_score, 'CONVICTION_REVERSAL_EXIT',
                                          reason=f'Held {trade_direction}, system flipped {new_direction}@{new_score:.0f}, P&L={pnl:+,.0f}',
                                          direction=new_direction)
                        
                        print(f"      ✅ EXITED via conviction reversal | Saved from waiting for SL")
                        
                        # === REVERSAL SNIPE: Aggressive reverse trade for quick 10% profit ===
                        # Pro rule: when your system flips with conviction, the momentum is real.
                        # Ride the reversal for a quick scalp — tight target, tight SL, trailing SL.
                        # GODREJCP: exited LONG at 11:24, system says SHORT@78 → BUY PE, grab 10%.
                        try:
                            print(f"\n      🎯 REVERSAL SNIPE: Placing aggressive {new_direction} trade on {underlying}")
                            snipe_result = self.tools.place_option_order(
                                underlying=underlying,
                                direction=new_direction,
                                strike_selection="ATM",
                                rationale=f"REVERSAL_SNIPE: Conviction flipped from {trade_direction} to {new_direction} with score {new_score:.0f}. Quick 10% scalp on momentum reversal."
                            )
                            
                            if snipe_result and snipe_result.get('success'):
                                snipe_entry = snipe_result.get('entry_price', 0)
                                print(f"      ✅ REVERSAL SNIPE FIRED: {underlying} {new_direction} @ ₹{snipe_entry:.2f}")
                                
                                # Override SL/target for aggressive quick scalp
                                # Target: +10%  |  SL: -6%  |  Trailing SL activates at +6%
                                snipe_target = snipe_entry * 1.10  # 10% profit target
                                snipe_sl = snipe_entry * 0.94       # 6% SL
                                
                                # Find the new position and override SL/target + tag it
                                with self.tools._positions_lock:
                                    for pos in reversed(self.tools.paper_positions):
                                        if (pos.get('underlying') == underlying and 
                                            pos.get('status', 'OPEN') == 'OPEN'):
                                            pos['stop_loss'] = snipe_sl
                                            pos['target'] = snipe_target
                                            pos['is_reversal_snipe'] = True
                                            pos['snipe_entry_time'] = datetime.now().isoformat()
                                            pos['snipe_original_direction'] = trade_direction
                                            pos['snipe_trigger_score'] = new_score
                                            pos['snipe_high_watermark'] = snipe_entry  # For trailing SL
                                            pos['snipe_trailing_active'] = False        # Activates at +6%
                                            print(f"      📐 Snipe overrides: Target ₹{snipe_target:.2f} (+10%) | SL ₹{snipe_sl:.2f} (-6%)")
                                            print(f"      📈 Trailing SL: activates at +6%, trails giving back 60% of gains")
                                            print(f"      ⏱️ Time guard: auto-exit in 12 min if < 3% profit")
                                            break
                                
                                self._log_decision(_cycle_time, underlying, new_score, 'REVERSAL_SNIPE_FIRED',
                                                  reason=f'Reversed {trade_direction}→{new_direction}, target +10%, SL -6%, trailing@4%',
                                                  direction=new_direction, setup='REVERSAL_SNIPE')
                            else:
                                snipe_err = snipe_result.get('error', 'unknown') if snipe_result else 'no result'
                                print(f"      ⚠️ Reversal snipe blocked: {snipe_err}")
                                self._log_decision(_cycle_time, underlying, new_score, 'REVERSAL_SNIPE_BLOCKED',
                                                  reason=f'Snipe blocked: {str(snipe_err)[:80]}',
                                                  direction=new_direction)
                        except Exception as _snipe_err:
                            print(f"      ⚠️ Reversal snipe error (non-fatal): {_snipe_err}")
                        
            except Exception as _conv_err:
                print(f"   ⚠️ Conviction reversal check error (non-fatal): {_conv_err}")
            
            # === ELITE AUTO-FIRE: Execute top-scoring stocks BEFORE GPT ===
            _cycle_time = datetime.now().strftime('%H:%M:%S')
            _auto_fired_syms = self._elite_auto_fire(
                pre_scores=_pre_scores,
                cycle_decisions=_cycle_decisions,
                sorted_data=sorted_data,
                market_data=market_data,
                fno_nfo_verified=_all_fo_syms if '_all_fo_syms' in dir() else set(),
                cycle_time=_cycle_time
            )
            
            # Create a quick summary of all stocks for scanning with EOD predictions
            quick_scan = []
            eod_opportunities = []
            regime_signals = []
            
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data:
                    chg = data.get('change_pct', 0)
                    rsi = data.get('rsi_14', 50)
                    trend = data.get('trend', 'N/A')
                    
                    # Get regime detection signals
                    vwap_slope = data.get('vwap_slope', 'FLAT')
                    price_vs_vwap = data.get('price_vs_vwap', 'AT_VWAP')
                    ema_regime = data.get('ema_regime', 'NORMAL')
                    orb_signal = data.get('orb_signal', 'INSIDE_ORB')
                    orb_strength = data.get('orb_strength', 0)
                    volume_regime = data.get('volume_regime', 'NORMAL')
                    
                    # Get CHOP filter signals
                    chop_zone = data.get('chop_zone', False)
                    chop_reason = data.get('chop_reason', '')
                    
                    # Get HTF alignment signals
                    htf_trend = data.get('htf_trend', 'NEUTRAL')
                    htf_ema_slope = data.get('htf_ema_slope', 'FLAT')
                    htf_alignment = data.get('htf_alignment', 'NEUTRAL')
                    
                    # Get volume analysis
                    vol_data = volume_analysis.get(symbol, {})
                    order_flow = vol_data.get('order_flow', 'BALANCED')
                    eod_pred = vol_data.get('eod_prediction', 'NEUTRAL')
                    eod_conf = vol_data.get('eod_confidence', 'LOW')
                    trade_signal = vol_data.get('trade_signal', 'NO_SIGNAL')
                    
                    # Flag potential setups
                    setup = ""
                    
                    # ======= CHOP FILTER - BLOCK ALL TRADES IN CHOP ZONE =======
                    if chop_zone:
                        setup = f"⚠️CHOP-ZONE({chop_reason})"
                        if symbol in self._wildcard_symbols:
                            print(f"   ⭐ WILDCARD CHOP-BLOCKED: {symbol} — {chop_reason}")
                        # Decision log: chop rejection
                        self._log_decision(_cycle_time, symbol, _pre_scores.get(symbol, 0),
                                          'CHOP_FILTERED', reason=chop_reason)
                        quick_scan.append(f"{symbol}: {chg:+.2f}% RSI:{rsi:.0f} {trend} {setup}")
                        continue  # Skip further analysis for this symbol
                    
                    # ======= HTF ALIGNMENT CHECK =======
                    # Determine intended trade direction based on signals
                    intended_direction = None
                    htf_blocked = False
                    
                    # REGIME-BASED SETUPS (highest priority)
                    # ORB trades - only once per direction per symbol per day
                    if orb_signal == "BREAKOUT_UP" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "BUY"
                        # HTF Check: Block BUY if HTF is BEARISH (unless explosive volume)
                        if htf_trend == "BEARISH" and htf_ema_slope == "FALLING" and volume_regime != "EXPLOSIVE":
                            htf_blocked = True
                            setup = "⛔HTF-BEAR-BLOCKS-BUY"
                        elif self._is_orb_trade_allowed(symbol, "UP"):
                            setup = "🚀ORB-BREAKOUT-BUY"
                            regime_signals.append(f"  🚀 {symbol}: ORB↑ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BULL | HTF:{htf_trend}")
                        else:
                            setup = "⛔ORB-UP-ALREADY-TAKEN"
                    elif orb_signal == "BREAKOUT_DOWN" and volume_regime in ["HIGH", "EXPLOSIVE"] and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "SELL"
                        # HTF Check: Block SELL if HTF is BULLISH (unless explosive volume)
                        if htf_trend == "BULLISH" and htf_ema_slope == "RISING" and volume_regime != "EXPLOSIVE":
                            htf_blocked = True
                            setup = "⛔HTF-BULL-BLOCKS-SHORT"
                        elif self._is_orb_trade_allowed(symbol, "DOWN"):
                            setup = "🔻ORB-BREAKOUT-SHORT"
                            regime_signals.append(f"  🔻 {symbol}: ORB↓ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BEAR")
                        else:
                            setup = "⛔ORB-DOWN-ALREADY-TAKEN"
                    elif ema_regime == "COMPRESSED" and volume_regime in ["HIGH", "EXPLOSIVE"]:
                        setup = "💥SQUEEZE-PENDING"
                        regime_signals.append(f"  💥 {symbol}: EMA SQUEEZE + High Volume - BREAKOUT IMMINENT | HTF:{htf_trend}")
                    elif price_vs_vwap == "ABOVE_VWAP" and vwap_slope == "RISING" and rsi < 60:
                        # HTF check for VWAP trend buy
                        if htf_trend == "BEARISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-BUY-HTF-CONFLICT"
                        else:
                            setup = "📈VWAP-TREND-BUY"
                    elif price_vs_vwap == "BELOW_VWAP" and vwap_slope == "FALLING" and rsi > 40:
                        # HTF check for VWAP trend short
                        if htf_trend == "BULLISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-SHORT-HTF-CONFLICT"
                        else:
                            setup = "📉VWAP-TREND-SHORT"
                    # Standard RSI setups (also check HTF)
                    elif rsi < 30:
                        if htf_trend == "BEARISH":
                            setup = "⚠️OVERSOLD-HTF-BEAR"  # Weaker signal
                        else:
                            setup = "⚡OVERSOLD-BUY"
                    elif rsi > 70:
                        if htf_trend == "BULLISH":
                            setup = "⚠️OVERBOUGHT-HTF-BULL"  # Weaker signal
                        else:
                            setup = "⚡OVERBOUGHT-SHORT"
                    elif trade_signal == "BUY_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-BUY ({order_flow})"
                        eod_opportunities.append(f"  🟢 {symbol}: EOD↑ - {order_flow}, conf:{eod_conf} | HTF:{htf_trend}")
                    elif trade_signal == "SHORT_FOR_EOD" and eod_conf in ["MEDIUM", "HIGH"]:
                        setup = f"📊EOD-SHORT ({order_flow})"
                        eod_opportunities.append(f"  🔴 {symbol}: EOD↓ - {order_flow}, conf:{eod_conf} | HTF:{htf_trend}")
                    elif chg < -1.5 and rsi > 45:
                        setup = "📉WEAK-SHORT"
                    elif chg > 1 and rsi < 55:
                        setup = "📈STRONG-BUY"
                    elif chg < 0 and rsi < 35:
                        setup = "🔄BOUNCE-BUY"
                    
                    # Mark if already in active trades
                    if self.tools.is_symbol_in_active_trades(symbol):
                        setup = "🔒ALREADY HOLDING"
                    
                    # Include CHOP and HTF status in scan output
                    htf_icon = "🐂" if htf_trend == "BULLISH" else "🐻" if htf_trend == "BEARISH" else "➖"
                    fno_tag = "[F&O]" if symbol in FNO_CONFIG.get('prefer_options_for', []) else ""
                    # Include pre-computed intraday score for F&O stocks
                    _score_tag = ""
                    if symbol in _pre_scores:
                        _s = _pre_scores[symbol]
                        _score_tag = f" S:{_s:.0f}" + ("🔥" if _s >= 65 else "✅" if _s >= 52 else "⚠️" if _s >= 45 else "❌")
                    quick_scan.append(f"{symbol}{fno_tag}: {chg:+.2f}% RSI:{rsi:.0f} {trend} ORB:{orb_signal} Vol:{volume_regime} HTF:{htf_icon}{_score_tag} {setup}")
            
            # Print regime signals
            if regime_signals:
                print(f"\n🎯 REGIME SIGNALS (HIGH PRIORITY):")
                for sig in regime_signals[:5]:
                    print(sig)
            
            # Print EOD opportunities
            if eod_opportunities:
                print(f"\n📊 EOD VOLUME ANALYSIS:")
                for opp in eod_opportunities[:5]:  # Top 5
                    print(opp)
            
            # Get list of symbols already in positions
            active_symbols = [t['symbol'] for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            # Get risk governor status for prompt
            risk_status = self.risk_governor.state
            trades_remaining = self.risk_governor.limits.max_trades_per_day - risk_status.trades_today
            
            # Check trade permission before prompting agent
            active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            can_trade_check = self.risk_governor.can_trade_general(
                active_positions=active_positions
            )
            
            if not can_trade_check.allowed:
                print(f"\n⚠️ Trading blocked: {can_trade_check.reason}")
                if can_trade_check.warnings:
                    for w in can_trade_check.warnings:
                        print(f"   ⚠️ {w}")
                return
            
            # Update correlation guard with current positions
            self.correlation_guard.update_positions(active_positions)
            corr_exposure = self.correlation_guard.get_exposure_summary()
            print(f"\n{corr_exposure}")
            
            # Get reconciliation status
            recon_can_trade, recon_reason = self.position_recon.can_trade()
            recon_state = self.position_recon.state.value
            
            # Get data health status
            halted_symbols = self.data_health_gate.get_halted_symbols()
            
            # Display reconciliation and health status
            print(f"\n🔄 RECONCILIATION: {recon_state} - {'✅ Can Trade' if recon_can_trade else '❌ ' + recon_reason}")
            if halted_symbols:
                print(f"🛡️ DATA HEALTH: ⚠️ {len(halted_symbols)} symbols halted: {', '.join(halted_symbols[:5])}")
            else:
                print(f"🛡️ DATA HEALTH: ✅ All symbols healthy")
            
            # Build F&O opportunity list — ONLY stocks that actually exist in NFO instruments
            fno_prefer = FNO_CONFIG.get('prefer_options_for', [])
            fno_prefer_set = set(fno_prefer) | set(self._wildcard_symbols) | _all_fo_syms
            # Filter to ONLY NFO-verified stocks (scanner loaded from actual instruments)
            fno_nfo_verified = _all_fo_syms  # Only symbols actually in NFO instruments file
            fno_opportunities = []
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data and symbol in fno_prefer_set and symbol in fno_nfo_verified:
                    if self.tools.is_symbol_in_active_trades(symbol):
                        continue
                    # Skip auto-fired stocks (already executed this cycle)
                    if symbol in _auto_fired_syms:
                        continue
                    if data.get('chop_zone', False):
                        continue
                    
                    # All stocks that passed the scanner rank filter are eligible
                    # The intraday scorer handles quality gating (score threshold, microstructure, etc.)
                    # No separate Tier-2/wildcard gate needed — scorer is the single source of truth
                    is_wildcard = symbol in self._wildcard_symbols
                    
                    setup_type = None
                    direction = None
                    orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                    vol_reg = data.get('volume_regime', 'NORMAL')
                    ema_reg = data.get('ema_regime', 'NORMAL')
                    htf_a = data.get('htf_alignment', 'NEUTRAL')
                    rsi = data.get('rsi_14', 50)
                    pvw = data.get('price_vs_vwap', 'AT_VWAP')
                    vs = data.get('vwap_slope', 'FLAT')
                    
                    if orb_sig == "BREAKOUT_UP" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "ORB_BREAKOUT"
                        direction = "BUY"
                    elif orb_sig == "BREAKOUT_DOWN" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "ORB_BREAKOUT"
                        direction = "SELL"
                    elif pvw == "ABOVE_VWAP" and vs == "RISING" and ema_reg in ["EXPANDING", "COMPRESSED"]:
                        setup_type = "VWAP_TREND"
                        direction = "BUY"
                    elif pvw == "BELOW_VWAP" and vs == "FALLING" and ema_reg in ["EXPANDING", "COMPRESSED"]:
                        setup_type = "VWAP_TREND"
                        direction = "SELL"
                    elif pvw == "ABOVE_VWAP" and vs == "RISING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "MOMENTUM"
                        direction = "BUY"
                    elif pvw == "BELOW_VWAP" and vs == "FALLING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "MOMENTUM"
                        direction = "SELL"
                    elif ema_reg == "COMPRESSED" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                        setup_type = "EMA_SQUEEZE"
                        direction = "BUY" if data.get('change_pct', 0) > 0 else "SELL"
                    elif rsi < 30 and htf_a != "BEARISH_ALIGNED":
                        setup_type = "RSI_REVERSAL"
                        direction = "BUY"
                    elif rsi > 70 and htf_a != "BULLISH_ALIGNED":
                        setup_type = "RSI_REVERSAL"
                        direction = "SELL"
                    
                    # WILDCARD MOMENTUM SETUP: if no standard setup detected but wildcard has strong momentum
                    if not setup_type and is_wildcard:
                        wc_chg = self._wildcard_change.get(symbol, 0)
                        if abs(wc_chg) >= 2.0 and vol_reg in ['HIGH', 'EXPLOSIVE']:
                            setup_type = "WILDCARD_MOMENTUM"
                            direction = "BUY" if wc_chg > 0 else "SELL"
                    
                    # S4 FIX: VWAP GRIND SETUP — catches INSIDE_ORB stocks grinding with volume
                    # (e.g., HINDALCO -5.7% but never broke ORB range)
                    if not setup_type and orb_sig == "INSIDE_ORB" and vol_reg in ('HIGH', 'EXPLOSIVE'):
                        _grind_ltp = data.get('ltp', 0)
                        _grind_open = data.get('open', _grind_ltp)
                        _grind_vwap = data.get('vwap', 0)
                        _grind_move = abs((_grind_ltp - _grind_open) / _grind_open * 100) if _grind_open > 0 else 0
                        if _grind_move >= 0.8 and _grind_ltp < _grind_vwap and _grind_ltp < _grind_open:
                            setup_type = "VWAP_GRIND"
                            direction = "SELL"
                        elif _grind_move >= 0.8 and _grind_ltp > _grind_vwap and _grind_ltp > _grind_open:
                            setup_type = "VWAP_GRIND"
                            direction = "BUY"
                    
                    if setup_type and direction:
                        # XGB Direction override REMOVED — direction comes from technicals only.
                        # GMM veto/boost is applied in model-tracker, not in GPT path.
                        
                        opt_type = "CE" if direction == "BUY" else "PE"
                        wc_tag = " [⭐WILDCARD]" if is_wildcard else ""
                        _fno_score = _pre_scores.get(symbol, 0)
                        _fno_score_tag = f" [Score:{_fno_score:.0f}]" if _fno_score > 0 else ""
                        # S5 FIX: Pre-scores don't include microstructure (15pts) because
                        # option_data isn't fetched during bulk scanning. Offset threshold
                        # by 12 (average micro contribution for liquid F&O stocks) to avoid
                        # filtering out stocks that would pass at trade time.
                        _micro_absent_offset = 12
                        # GPT-selected minimum: score + micro offset must reach 60
                        # (raw score ~48+ since micro adds ~12)
                        if _fno_score > 0 and _fno_score + _micro_absent_offset < 60:
                            continue
                        # Append ML signal tag if available (fail-safe: empty string if not)
                        _ml_tag = ""
                        try:
                            _ml_info = _ml_results.get(symbol, {})
                            if _ml_info.get('ml_gpt_summary'):
                                _ml_tag = f" {_ml_info['ml_gpt_summary']}"
                        except Exception:
                            pass
                        # Append OI flow tag if available (fail-safe)
                        _oi_tag = ""
                        try:
                            _oi_info = getattr(self, '_cycle_oi_results', {}).get(symbol, {})
                            if _oi_info.get('flow_gpt_line'):
                                _oi_tag = f" {_oi_info['flow_gpt_line']}"
                        except Exception:
                            pass
                        
                        fno_opportunities.append(
                            f"  🎯 {symbol}: {setup_type} → place_option_order(underlying=\"{symbol}\", direction=\"{direction}\", strike_selection=\"ATM\") [{opt_type}]{_fno_score_tag}{_ml_tag}{_oi_tag}{wc_tag}"
                        )
            
            # === PROACTIVE DEBIT SPREAD SCANNER ===
            # Scan for debit spread opportunities INDEPENDENTLY from the cascade
            # This is the key fix: debit spreads are no longer just a fallback
            debit_spread_placed = []
            try:
                from config import DEBIT_SPREAD_CONFIG
                if DEBIT_SPREAD_CONFIG.get('proactive_scan', False) and DEBIT_SPREAD_CONFIG.get('enabled', False):
                    now_time = datetime.now().time()
                    debit_cutoff = datetime.strptime(DEBIT_SPREAD_CONFIG.get('no_entry_after', '14:30'), '%H:%M').time()
                    
                    if now_time < debit_cutoff:
                        proactive_min_score = DEBIT_SPREAD_CONFIG.get('proactive_scan_min_score', 68)
                        proactive_min_move = DEBIT_SPREAD_CONFIG.get('proactive_scan_min_move_pct', 1.5)
                        
                        # Find momentum setups that qualify for debit spreads
                        debit_candidates = []
                        for symbol, data in sorted_data:
                            if not isinstance(data, dict) or 'ltp' not in data:
                                continue
                            # Skip if already holding or in chop zone
                            if self.tools.is_symbol_in_active_trades(symbol):
                                continue
                            if data.get('chop_zone', False):
                                continue
                            # Skip if not F&O eligible
                            if symbol not in fno_prefer_set:
                                continue
                            
                            # Check move % — momentum plays need real movement
                            chg = abs(data.get('change_pct', 0))
                            if chg < proactive_min_move:
                                continue
                            
                            # Check follow-through (strongest winner signal)
                            ft = data.get('follow_through_candles', 0)
                            min_ft = DEBIT_SPREAD_CONFIG.get('min_follow_through_candles', 2)
                            if ft < min_ft:
                                continue
                            
                            # Check ADX
                            adx = data.get('adx_14', data.get('adx', 0))
                            min_adx = DEBIT_SPREAD_CONFIG.get('min_adx', 28)
                            if adx > 0 and adx < min_adx:
                                continue
                            
                            # Determine direction from price action
                            change_pct = data.get('change_pct', 0)
                            direction = "BUY" if change_pct > 0 else "SELL"
                            
                            # Check trend continuation — move must be in direction of trade
                            orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                            vol_reg = data.get('volume_regime', 'NORMAL')
                            
                            # Priority scoring for debit spread candidates
                            ds_priority = 0
                            ds_priority += ft * 10  # Follow-through is king
                            ds_priority += min(adx, 50)  # ADX contribution
                            ds_priority += chg * 5  # Bigger move = better
                            if orb_sig in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
                                ds_priority += 15
                            if vol_reg in ("HIGH", "EXPLOSIVE"):
                                ds_priority += 10
                            
                            # === ML DIRECTION PREDICTION BOOST (FAIL-SAFE) ===
                            # ML UP/DOWN = perfect for debit spreads (directional move expected)
                            # ML FLAT = soft skip (stock likely flat, debit spread loses)
                            try:
                                _ds_ml = getattr(self, '_cycle_ml_results', {}).get(symbol, {})
                                _ds_ml_signal = _ds_ml.get('ml_signal', 'UNKNOWN')
                                _ds_prob_up = _ds_ml.get('ml_prob_up', 0.33)
                                _ds_prob_down = _ds_ml.get('ml_prob_down', 0.33)
                                _ds_prob_flat = _ds_ml.get('ml_prob_flat', 0.34)
                                _ds_max_dir = max(_ds_prob_up, _ds_prob_down)
                                if _ds_ml_signal in ('UP', 'DOWN') and _ds_max_dir >= 0.50:
                                    ds_priority += 20  # Strong directional ML confirmation
                                    print(f"      🧠 ML BOOST: {symbol} {_ds_ml_signal} prob={_ds_max_dir:.0%} → +20 priority")
                                elif _ds_ml_signal in ('UP', 'DOWN') and _ds_max_dir >= 0.40:
                                    ds_priority += 10  # Moderate ML confirmation
                                elif _ds_ml_signal == 'FLAT' and _ds_prob_flat >= 0.70:
                                    print(f"      🧠 ML SKIP: {symbol} FLAT prob={_ds_prob_flat:.0%} → skipping debit spread")
                                    continue  # Soft skip — stock likely flat
                            except Exception:
                                pass  # ML crash → no impact, proceed normally
                            
                            debit_candidates.append((symbol, data, direction, ds_priority))
                        
                        # Sort by priority and try top candidates
                        debit_candidates.sort(key=lambda x: x[3], reverse=True)
                        
                        max_debit_entries = 3  # Max 3 proactive debit spreads per scan cycle (was 2)
                        for symbol, data, direction, priority in debit_candidates[:max_debit_entries]:
                            try:
                                print(f"\n   🎯 PROACTIVE DEBIT SPREAD: Trying {symbol} ({direction}) — Priority: {priority:.0f}")
                                
                                # === PRE-FLIGHT LIQUIDITY CHECK ===
                                # Verify option chain has adequate OI + tight bid-ask BEFORE
                                # spending API calls on full debit spread creation
                                try:
                                    from options_trader import get_options_trader
                                    _ot = get_options_trader(
                                        kite=self.tools.kite,
                                        capital=getattr(self.tools, 'paper_capital', 500000),
                                        paper_mode=getattr(self.tools, 'paper_mode', True)
                                    )
                                    is_liquid, liq_reason = _ot.chain_fetcher.quick_check_option_liquidity(
                                        underlying=symbol,
                                        min_oi=DEBIT_SPREAD_CONFIG.get('min_oi', 500),
                                        max_bid_ask_pct=DEBIT_SPREAD_CONFIG.get('max_spread_bid_ask_pct', 5.0)
                                    )
                                    if not is_liquid:
                                        print(f"   ❌ LIQUIDITY PRE-CHECK FAILED for {symbol}: {liq_reason}")
                                        continue
                                    print(f"   ✅ Liquidity OK: {liq_reason}")
                                except Exception as liq_e:
                                    print(f"   ⚠️ Liquidity check skipped (error: {liq_e}) — proceeding anyway")
                                
                                result = self.tools.place_debit_spread(
                                    underlying=symbol,
                                    direction=direction,
                                    rationale=f"Proactive debit spread: FT={data.get('follow_through_candles', 0)} ADX={data.get('adx_14', 0):.0f} Move={data.get('change_pct', 0):+.1f}%",
                                    pre_fetched_market_data=data
                                )
                                if result.get('success'):
                                    print(f"   ✅ PROACTIVE DEBIT SPREAD PLACED on {symbol}!")
                                    debit_spread_placed.append(symbol)
                                else:
                                    print(f"   ℹ️ Debit spread not viable for {symbol}: {result.get('error', 'unknown')}")
                            except Exception as e:
                                print(f"   ⚠️ Debit spread attempt failed for {symbol}: {e}")
                        
                        if debit_spread_placed:
                            print(f"\n   🚀 PROACTIVE DEBIT SPREADS PLACED: {', '.join(debit_spread_placed)}")
                        elif debit_candidates:
                            print(f"\n   ℹ️ {len(debit_candidates)} debit spread candidates found, none viable this cycle")
            except Exception as e:
                print(f"   ⚠️ Proactive debit spread scan error: {e}")
            
            # === PROACTIVE IRON CONDOR SCANNER (INDEX + STOCK) ===
            # Scan NIFTY/BANKNIFTY for IC opportunities (weekly expiry, 0DTE only)
            # Also check recently rejected stocks with low scores
            ic_placed = []
            try:
                from config import IRON_CONDOR_CONFIG, IC_INDEX_SYMBOLS
                if IRON_CONDOR_CONFIG.get('enabled', False) and IRON_CONDOR_CONFIG.get('proactive_scan', False):
                    
                    # === DTE PRE-CHECK (once per day): Skip IC scan if no expiry within DTE limits ===
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    if not hasattr(self, '_ic_eligible_date') or self._ic_eligible_date != today_str:
                        self._ic_eligible_today = False
                        self._ic_eligible_date = today_str
                        try:
                            from options_trader import get_options_trader as _get_ot, ExpirySelection
                            _ot_check = _get_ot(
                                kite=self.tools.kite,
                                capital=getattr(self.tools, 'paper_capital', 500000),
                                paper_mode=getattr(self.tools, 'paper_mode', True)
                            )
                            idx_mode = IRON_CONDOR_CONFIG.get('index_mode', {})
                            stk_mode = IRON_CONDOR_CONFIG.get('stock_mode', {})
                            idx_max_dte = idx_mode.get('max_dte', 2)
                            stk_max_dte = stk_mode.get('max_dte', 15)
                            today_date = datetime.now().date()
                            
                            for idx_sym in IC_INDEX_SYMBOLS:
                                try:
                                    idx_expiry_sel = ExpirySelection[idx_mode.get('prefer_expiry', 'CURRENT_WEEK')]
                                    idx_expiry = _ot_check.chain_fetcher.get_nearest_expiry(idx_sym, idx_expiry_sel)
                                    if idx_expiry:
                                        from datetime import date as _date
                                        exp_d = idx_expiry if isinstance(idx_expiry, _date) and not isinstance(idx_expiry, datetime) else idx_expiry.date() if hasattr(idx_expiry, 'date') else idx_expiry
                                        idx_dte = (exp_d - today_date).days
                                        if idx_dte <= idx_max_dte:
                                            self._ic_eligible_today = True
                                            print(f"   🦅 IC eligible today: {idx_sym} expiry {idx_expiry} (DTE={idx_dte}, max={idx_max_dte})")
                                            break
                                except Exception:
                                    pass
                            
                            if not self._ic_eligible_today:
                                try:
                                    stk_expiry_sel = ExpirySelection[stk_mode.get('prefer_expiry', 'CURRENT_MONTH')]
                                    stk_expiry = _ot_check.chain_fetcher.get_nearest_expiry("NSE:RELIANCE", stk_expiry_sel)
                                    if stk_expiry:
                                        from datetime import date as _date
                                        exp_d = stk_expiry if isinstance(stk_expiry, _date) and not isinstance(stk_expiry, datetime) else stk_expiry.date() if hasattr(stk_expiry, 'date') else stk_expiry
                                        stk_dte = (exp_d - today_date).days
                                        if stk_dte <= stk_max_dte:
                                            self._ic_eligible_today = True
                                            print(f"   🦅 IC eligible today: Stocks expiry {stk_expiry} (DTE={stk_dte}, max={stk_max_dte})")
                                except Exception:
                                    pass
                            
                            if not self._ic_eligible_today:
                                print(f"   🦅 IC SKIPPED TODAY: No expiry within DTE limits (index max={idx_max_dte}, stock max={stk_max_dte})")
                        except Exception as dte_err:
                            print(f"   ⚠️ IC DTE pre-check failed: {dte_err} — will skip IC scan")
                            self._ic_eligible_today = False
                    
                    # Only run IC scan if today has eligible expiry AND within time window
                    now_time = datetime.now().time()
                    ic_earliest = datetime.strptime(IRON_CONDOR_CONFIG.get('earliest_entry', '10:30'), '%H:%M').time()
                    ic_cutoff = datetime.strptime(IRON_CONDOR_CONFIG.get('no_entry_after', '12:30'), '%H:%M').time()
                    
                    if self._ic_eligible_today and ic_earliest <= now_time <= ic_cutoff:
                        from options_trader import get_options_trader
                        _ot = get_options_trader(
                            kite=self.tools.kite,
                            capital=getattr(self.tools, 'paper_capital', 500000),
                            paper_mode=getattr(self.tools, 'paper_mode', True)
                        )
                        
                        # --- INDEX IC SCAN (primary — weekly expiry, best profit) ---
                        for idx_symbol in IC_INDEX_SYMBOLS:
                            try:
                                if self.tools.is_symbol_in_active_trades(idx_symbol):
                                    continue
                                
                                # Fetch index market data
                                idx_data_raw = self.tools.get_market_data([idx_symbol])
                                idx_data = idx_data_raw.get(idx_symbol, {})
                                if not idx_data or not isinstance(idx_data, dict):
                                    continue
                                
                                # Check if index is range-bound
                                idx_change = abs(idx_data.get('change_pct', 99))
                                idx_rsi = idx_data.get('rsi_14', 50)
                                max_move = IRON_CONDOR_CONFIG.get('max_intraday_move_pct', 1.2)
                                rsi_range = IRON_CONDOR_CONFIG.get('prefer_rsi_range', [38, 62])
                                
                                if idx_change > max_move:
                                    print(f"   🦅 IC SKIP {idx_symbol}: moved {idx_change:.1f}% (>{max_move}%)")
                                    continue
                                
                                # Assign a synthetic "choppy" score for IC (lower = choppier = better for IC)
                                # Use: low change%, near-50 RSI, flat VWAP = lower score
                                ic_score = 30  # Default choppy score
                                if idx_change < 0.3:
                                    ic_score -= 5  # Very tight range
                                if 45 <= idx_rsi <= 55:
                                    ic_score -= 3  # Dead neutral RSI 
                                if idx_data.get('vwap_slope', 'FLAT') == 'FLAT':
                                    ic_score -= 2  # Flat VWAP
                                ic_score = max(15, min(45, ic_score))
                                
                                print(f"\n   🦅 INDEX IC SCAN: {idx_symbol} | Chg: {idx_change:.2f}% | RSI: {idx_rsi:.0f} | IC Score: {ic_score}")
                                
                                exec_result = self.tools.place_iron_condor(
                                    underlying=idx_symbol,
                                    rationale=f"Proactive index IC: flat range {idx_change:.1f}%, RSI {idx_rsi:.0f}",
                                    directional_score=ic_score,
                                    pre_fetched_market_data=idx_data
                                )
                                
                                if exec_result and exec_result.get('success'):
                                    print(f"   ✅ INDEX IC PLACED on {idx_symbol}!")
                                    ic_placed.append(idx_symbol)
                                else:
                                    print(f"   ℹ️ Index IC not viable for {idx_symbol}: {exec_result.get('error', 'creation failed')}")
                            except Exception as ie:
                                print(f"   ⚠️ Index IC scan error for {idx_symbol}: {ie}")
                        
                        # --- STOCK IC SCAN (SOPHISTICATED MULTI-FACTOR SCORING) ---
                        # Score each F&O stock on 7 IC-quality factors, rank, and pick best
                        if IRON_CONDOR_CONFIG.get('scan_rejected_stocks', True):
                            stock_ic_candidates = []
                            for symbol, data in sorted_data:
                                if not isinstance(data, dict) or 'ltp' not in data:
                                    continue
                                if self.tools.is_symbol_in_active_trades(symbol):
                                    continue
                                if symbol not in fno_prefer_set:
                                    continue
                                if symbol in IC_INDEX_SYMBOLS:
                                    continue  # Already handled above
                                
                                # === IC QUALITY SCORE (0-100) ===
                                # Higher = better IC candidate
                                ic_quality = 0
                                ic_reasons = []
                                
                                chg = abs(data.get('change_pct', 0))
                                rsi_val = data.get('rsi_14', 50)
                                adx_val = data.get('adx_14', data.get('adx', 25))
                                is_chop = data.get('chop_zone', False)
                                vwap_slope = data.get('vwap_slope', 'FLAT')
                                orb_sig = data.get('orb_signal', 'INSIDE_ORB')
                                vol_regime = data.get('volume_regime', 'NORMAL')
                                vol_ratio = data.get('volume_vs_avg', 1.0)
                                ema_regime = data.get('ema_regime', 'NORMAL')
                                atr_14 = data.get('atr_14', 0)
                                ltp = data.get('ltp', 0)
                                range_exp = data.get('range_expansion_ratio', 0.5)
                                
                                # HARD FILTERS — skip immediately
                                if chg > max_move:
                                    continue
                                if orb_sig in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and data.get('orb_strength', 0) > 50:
                                    continue  # Strong directional breakout — not IC material
                                
                                # FACTOR 1: RANGE TIGHTNESS (0-20 pts)
                                # Tighter intraday range = better for IC
                                if chg < 0.3:
                                    ic_quality += 20; ic_reasons.append(f"TIGHT_RANGE({chg:.1f}%)")
                                elif chg < 0.6:
                                    ic_quality += 15; ic_reasons.append(f"NARROW_RANGE({chg:.1f}%)")
                                elif chg < 1.0:
                                    ic_quality += 8; ic_reasons.append(f"MOD_RANGE({chg:.1f}%)")
                                else:
                                    ic_quality += 3  # Still within max_move but wide
                                
                                # FACTOR 2: RSI NEUTRALITY (0-15 pts)
                                # 45-55 = dead center = perfect; 40-60 = good; outside = risky
                                rsi_mid_dist = abs(rsi_val - 50)
                                if rsi_mid_dist <= 5:
                                    ic_quality += 15; ic_reasons.append(f"RSI_DEAD_CENTER({rsi_val:.0f})")
                                elif rsi_mid_dist <= 10:
                                    ic_quality += 10
                                elif rsi_mid_dist <= 15:
                                    ic_quality += 5
                                else:
                                    ic_quality -= 5  # RSI trending — danger for IC
                                
                                # FACTOR 3: ADX LOW = NO TREND (0-15 pts)
                                # ADX < 20 = no trend (ideal); < 25 = weak trend; > 30 = strong trend (bad)
                                if adx_val < 18:
                                    ic_quality += 15; ic_reasons.append(f"NO_TREND(ADX={adx_val:.0f})")
                                elif adx_val < 22:
                                    ic_quality += 12; ic_reasons.append(f"WEAK_TREND(ADX={adx_val:.0f})")
                                elif adx_val < 26:
                                    ic_quality += 8
                                elif adx_val < 30:
                                    ic_quality += 3
                                else:
                                    ic_quality -= 5  # Strong trend — IC is risky
                                    ic_reasons.append(f"STRONG_TREND_WARNING(ADX={adx_val:.0f})")
                                
                                # FACTOR 4: VWAP SLOPE FLAT (0-10 pts)
                                if vwap_slope == 'FLAT':
                                    ic_quality += 10; ic_reasons.append("VWAP_FLAT")
                                elif vwap_slope in ('RISING', 'FALLING'):
                                    ic_quality -= 3  # Trending — not ideal
                                
                                # FACTOR 5: VOLUME DECLINING (0-10 pts) 
                                # Declining volume = dying momentum = perfect for IC
                                if vol_regime == 'LOW' or vol_ratio < 0.7:
                                    ic_quality += 10; ic_reasons.append(f"LOW_VOL({vol_ratio:.1f}x)")
                                elif vol_regime == 'NORMAL' and vol_ratio < 1.2:
                                    ic_quality += 6
                                elif vol_regime == 'HIGH':
                                    ic_quality += 0  # High volume could mean breakout
                                elif vol_regime == 'EXPLOSIVE':
                                    ic_quality -= 8  # Explosive = breakout imminent
                                    ic_reasons.append("EXPLOSIVE_VOL_DANGER")
                                
                                # FACTOR 6: EMA COMPRESSION (0-10 pts)
                                # Compressed EMAs = coiling for a move or stuck in range
                                if ema_regime == 'COMPRESSED':
                                    ic_quality += 10; ic_reasons.append("EMA_COMPRESSED")
                                elif ema_regime == 'NORMAL':
                                    ic_quality += 5
                                elif ema_regime == 'EXPANDING':
                                    ic_quality -= 3  # Expanding = trending
                                
                                # FACTOR 7: RANGE EXPANSION LOW (0-10 pts)
                                # Low range expansion = price not going anywhere
                                if range_exp < 0.2:
                                    ic_quality += 10; ic_reasons.append(f"LOW_EXPANSION({range_exp:.2f})")
                                elif range_exp < 0.35:
                                    ic_quality += 7
                                elif range_exp < 0.5:
                                    ic_quality += 3
                                else:
                                    ic_quality -= 3  # High expansion = trending
                                
                                # FACTOR 8: CHOP ZONE confirmation (0-10 pts)
                                if is_chop:
                                    ic_quality += 10; ic_reasons.append("CHOP_CONFIRMED")
                                
                                # FACTOR 9: ML MOVE PREDICTION (0-15 pts bonus for FLAT, -10 penalty for UP/DOWN)
                                # ML FLAT = ideal for IC (stock predicted flat)
                                # ML UP/DOWN = dangerous for IC (breakout could blow through wings)
                                try:
                                    _ic_ml = getattr(self, '_cycle_ml_results', {}).get(symbol, {})
                                    _ic_move_prob = _ic_ml.get('ml_move_prob', 0.5)
                                    _ic_ml_signal = _ic_ml.get('ml_signal', 'UNKNOWN')
                                    if _ic_ml_signal == 'FLAT' and _ic_move_prob < 0.30:
                                        ic_quality += 15; ic_reasons.append(f"ML_FLAT({_ic_move_prob:.0%})")
                                    elif _ic_ml_signal == 'FLAT':
                                        ic_quality += 8; ic_reasons.append(f"ML_FLAT_MILD({_ic_move_prob:.0%})")
                                    elif _ic_ml_signal in ('UP', 'DOWN') and _ic_move_prob >= 0.60:
                                        ic_quality -= 15; ic_reasons.append(f"ML_MOVE_DANGER({_ic_move_prob:.0%})")
                                    elif _ic_ml_signal in ('UP', 'DOWN'):
                                        ic_quality -= 8; ic_reasons.append(f"ML_MOVE_WARN({_ic_move_prob:.0%})")
                                except Exception:
                                    pass  # ML crash → no impact on IC quality
                                
                                # === MINIMUM IC QUALITY GATE ===
                                min_ic_quality = IRON_CONDOR_CONFIG.get('min_ic_quality_score', 50)
                                if ic_quality < min_ic_quality:
                                    continue  # Not choppy enough for IC
                                
                                # Map ic_quality (0-100) to a directional_score (15-45) for IC
                                # Higher IC quality → lower directional score (more neutral)
                                mapped_dir_score = max(15, min(45, 45 - int((ic_quality - 50) * 0.6)))
                                
                                stock_ic_candidates.append((symbol, data, ic_quality, mapped_dir_score, ic_reasons))
                            
                            # Sort by IC quality score DESCENDING (best IC candidates first)
                            stock_ic_candidates.sort(key=lambda x: x[2], reverse=True)
                            
                            max_stock_ic = IRON_CONDOR_CONFIG.get('max_stock_ic_per_cycle', 2)
                            for symbol, data, ic_quality, mapped_score, reasons in stock_ic_candidates[:max_stock_ic]:
                                try:
                                    reasons_str = " | ".join(reasons[:5])
                                    print(f"\n   🦅 STOCK IC SCAN: {symbol} | Quality: {ic_quality}/100 | Chg: {abs(data.get('change_pct', 0)):.2f}% | RSI: {data.get('rsi_14', 50):.0f}")
                                    print(f"      IC Factors: {reasons_str}")
                                    
                                    exec_result = self.tools.place_iron_condor(
                                        underlying=symbol,
                                        rationale=f"IC Quality {ic_quality}/100: {reasons_str} | Move {abs(data.get('change_pct', 0)):.1f}% RSI {data.get('rsi_14', 50):.0f}",
                                        directional_score=mapped_score,
                                        pre_fetched_market_data=data
                                    )
                                    if exec_result and exec_result.get('success'):
                                        print(f"   ✅ STOCK IC PLACED on {symbol}! (Quality: {ic_quality})")
                                        ic_placed.append(symbol)
                                    else:
                                        print(f"   ℹ️ Stock IC not viable for {symbol}: {exec_result.get('error', 'creation failed')}")
                                except Exception as se:
                                    print(f"   ⚠️ Stock IC attempt failed for {symbol}: {se}")
                        
                        if ic_placed:
                            print(f"\n   🦅 IRON CONDORS PLACED: {', '.join(ic_placed)}")
                    else:
                        if now_time < ic_earliest:
                            pass  # Silent — too early, no need to print every cycle
                        else:
                            pass  # Silent — past cutoff
            except Exception as e:
                print(f"   ⚠️ Proactive IC scan error: {e}")
            
            # Select top detailed stocks for GPT (most active + those with setups)
            top_detail_symbols = [s for s, _ in sorted_data[:10] if isinstance(market_data.get(s), dict) and 'ltp' in market_data.get(s, {})]
            detailed_data = [line for line in data_summary if any(sym in line for sym in top_detail_symbols)]
            
            # Build scanner wild-card summary for GPT
            wildcard_info = ""
            if scan_result and scan_result.wildcards:
                wc_lines = []
                for w in scan_result.wildcards:
                    wc_lines.append(f"  ⭐ {w.nse_symbol}: {w.change_pct:+.2f}% ₹{w.ltp:.2f} [{w.category}] — OUTSIDE fixed universe, use place_option_order()")
                wildcard_info = "\n".join(wc_lines)
            
            # Build BROAD MARKET HEAT MAP from scanner (all 191 F&O stocks)
            broad_heat = ""
            try:
                broad_heat = self.market_scanner.get_broad_market_heat(
                    existing_universe=set(APPROVED_UNIVERSE)
                )
            except Exception:
                broad_heat = "Scanner heat map unavailable"
            
            # Ask agent to analyze market with FULL CONTEXT + GPT-5.2 reasoning
            # Compute market breadth for macro context
            _up_count = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) > 0.5)
            _down_count = sum(1 for s, d in sorted_data if isinstance(d, dict) and d.get('change_pct', 0) < -0.5)
            _flat_count = len(sorted_data) - _up_count - _down_count
            _breadth = "BULLISH" if _up_count > _down_count * 1.5 else "BEARISH" if _down_count > _up_count * 1.5 else "MIXED"
            
            # Compute sector summary — prefer real sector index data, fallback to stock-average
            _sector_perf = []
            _sector_idx_map = {
                'IT': 'NSE:NIFTY IT', 'BANKS': 'NSE:NIFTY BANK', 'METALS': 'NSE:NIFTY METAL',
                'PHARMA': 'NSE:NIFTY PHARMA', 'AUTO': 'NSE:NIFTY AUTO',
                'ENERGY': 'NSE:NIFTY ENERGY', 'FMCG': 'NSE:NIFTY FMCG',
            }
            _sector_map = {
                'IT': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'KPITTECH', 'COFORGE', 'MPHASIS', 'PERSISTENT'],
                'BANKS': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'BANKBARODA', 'PNB', 'IDFCFIRSTB', 'INDUSINDBK', 'FEDERALBNK'],
                'METALS': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL', 'NMDC', 'NATIONALUM', 'HINDZINC', 'SAIL'],
                'PHARMA': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'AUROPHARMA', 'BIOCON', 'LUPIN'],
                'AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY', 'BHARATFORG'],
                'ENERGY': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'ADANIGREEN', 'TATAPOWER', 'ADANIENT'],
                'FMCG': ['ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO']
            }
            # Use _sector_index_changes computed in cross-val block if available
            _sec_idx_chg = getattr(self, '_sector_index_changes_cache', {})
            for _sec, _syms in _sector_map.items():
                _idx_sym = _sector_idx_map.get(_sec, '')
                _idx_val = _sec_idx_chg.get(_idx_sym)
                if _idx_val is not None:
                    _sector_perf.append(f"  {_sec}: {_idx_val:+.2f}% (NIFTY {_sec} index)")
                else:
                    _changes = [market_data.get(f"NSE:{s}", {}).get('change_pct', 0) for s in _syms if isinstance(market_data.get(f"NSE:{s}", {}), dict)]
                    if _changes:
                        _avg = sum(_changes) / len(_changes)
                        _sector_perf.append(f"  {_sec}: {_avg:+.2f}% avg ({len([c for c in _changes if c > 0])}/{len(_changes)} ↑)")
            
            # === DYNAMIC MAX PICKS: Scale GPT picks with signal quality ===
            _dynamic_max = self._compute_max_picks(_pre_scores, _breadth)
            if _dynamic_max != 3:
                print(f"   📊 DYNAMIC PICKS: GPT allowed up to {_dynamic_max} trades (signal quality {'HIGH' if _dynamic_max > 3 else 'LOW'})")
            
            # === AUTO-FIRED MESSAGE: Tell GPT which stocks were already executed ===
            if _auto_fired_syms:
                _auto_fired_msg = f"\n⚡ ALREADY AUTO-FIRED (DO NOT re-pick): {', '.join(_auto_fired_syms)}"
            else:
                _auto_fired_msg = ""
            
            # === BUILD ML SUMMARY FOR GPT PROMPT (fail-safe: empty if ML unavailable) ===
            _ml_prompt_section = 'ML predictor not available this cycle'
            try:
                if _ml_results:
                    _ml_summary_lines = []
                    # Top movers (high move probability)
                    _ml_movers = [(s, r) for s, r in _ml_results.items() if r.get('ml_move_prob', 0) >= 0.40]
                    _ml_movers.sort(key=lambda x: x[1]['ml_move_prob'], reverse=True)
                    if _ml_movers:
                        for s, r in _ml_movers[:10]:
                            _ml_summary_lines.append(f"  {s.replace('NSE:','')}: {r['ml_gpt_summary']} (boost:{r['ml_score_boost']:+d}, size:{r.get('ml_sizing_factor', 1.0):.1f}x)")
                    # Flat/choppy stocks (ML caution)
                    _ml_flat = [(s, r) for s, r in _ml_results.items() if r.get('ml_entry_caution')]
                    if _ml_flat:
                        _flat_names = [s.replace('NSE:', '') for s, _ in _ml_flat[:10]]
                        _ml_summary_lines.append(f"  ⚠️ LIKELY FLAT (ML caution — avoid or size down): {', '.join(_flat_names)}")
                    # Summary stats
                    _total_ml = len(_ml_results)
                    _move_count = sum(1 for r in _ml_results.values() if r.get('ml_signal') == 'MOVE')
                    _ml_summary_lines.append(f"  Stats: {_total_ml} analyzed | {_move_count} MOVE signals | {_total_ml - _move_count} NO_MOVE")
                    _ml_prompt_section = chr(10).join(_ml_summary_lines)
            except Exception:
                pass  # ML summary failed — GPT continues without it
            
            # === BUILD OI FLOW SUMMARY FOR GPT PROMPT (fail-safe) ===
            _oi_prompt_section = 'OI flow data not available this cycle'
            try:
                _oi_results_gpt = getattr(self, '_cycle_oi_results', {})
                if _oi_results_gpt:
                    _oi_lines = []
                    # Sort by confidence descending
                    _oi_sorted = sorted(_oi_results_gpt.items(), key=lambda x: x[1].get('flow_confidence', 0), reverse=True)
                    for _os, _od in _oi_sorted[:12]:
                        _os_clean = _os.replace('NSE:', '')
                        _bias = _od.get('flow_bias', 'NEUTRAL')
                        _pcr = _od.get('pcr_oi', 1.0)
                        _ivs = _od.get('iv_skew', 0)
                        _mp = _od.get('max_pain', 0)
                        _spot = _od.get('spot_price', 0)
                        _mp_dist = _od.get('spot_vs_max_pain_pct', 0)
                        _cr = _od.get('call_resistance', 0)
                        _ps = _od.get('put_support', 0)
                        _oi_line = f"  {_os_clean}: {_bias} | PCR:{_pcr:.2f} | IVskew:{_ivs:+.1f}% | MaxPain:{_mp:.0f} ({_mp_dist:+.1f}% from spot) | CallRes:{_cr:.0f} PutSup:{_ps:.0f}"
                        # Add OI buildup signal (from DhanHQ or NSE)
                        _nse_buildup = _od.get('nse_oi_buildup')
                        _nse_strength = _od.get('nse_oi_buildup_strength', 0)
                        _oi_src = _od.get('oi_source', 'NSE')
                        if _nse_buildup and _nse_buildup != 'NEUTRAL':
                            _oi_line += f" | OI_Buildup[{_oi_src}]:{_nse_buildup}({_nse_strength:.0%})"
                        _nse_ce_chg = _od.get('nse_total_call_oi_change', 0)
                        _nse_pe_chg = _od.get('nse_total_put_oi_change', 0)
                        if _nse_ce_chg or _nse_pe_chg:
                            _oi_line += f" | ΔOI:CE{_nse_ce_chg:+,}/PE{_nse_pe_chg:+,}"
                        # DhanHQ exclusive: ATM Greeks
                        _atm_g = _od.get('atm_greeks', {})
                        if _atm_g and _atm_g.get('ce_delta'):
                            _oi_line += f" | δ:{_atm_g['ce_delta']:.2f} θ:{_atm_g.get('ce_theta', 0):.1f}"
                        _oi_lines.append(_oi_line)
                    _oi_prompt_section = chr(10).join(_oi_lines)
            except Exception:
                pass  # OI summary failed — GPT continues without it
            
            prompt = f"""ANALYZE → REASON → EXECUTE. Use your GPT-5.2 reasoning depth.

=== 🌐 MARKET BREADTH ===
Breadth: {_breadth} | Up: {_up_count} | Down: {_down_count} | Flat: {_flat_count}
Sector Performance:
{chr(10).join(_sector_perf) if _sector_perf else '  No sector data'}

=== ⚡ F&O READY SIGNALS (HIGHEST PRIORITY) ===
F&O stocks: {', '.join(fno_prefer)}
{chr(10).join(fno_opportunities) if fno_opportunities else 'No F&O setups right now - check CASH stocks below'}

=== 📡 WILD-CARD MOVERS ===
{wildcard_info if wildcard_info else 'No wild-card movers this cycle'}

=== 🌡️ BROAD MARKET HEAT MAP (Top 40 F&O Movers — ⭐NEW = outside curated list) ===
{broad_heat}

=== 📊 MARKET SCAN ({len(quick_scan)} stocks) ===
{chr(10).join(quick_scan[:50])}

=== 🔬 DETAILED TECHNICALS (Top 10 Movers) ===
{chr(10).join(detailed_data[:10])}

=== 🎯 REGIME SIGNALS ===
{chr(10).join(regime_signals) if regime_signals else 'No strong regime signals'}

=== 🧠 ML MOVE PREDICTIONS (XGBoost volatility model) ===
{_ml_prompt_section}

=== 📊 LIVE OI FLOW (Real-time PCR / IV Skew / Max Pain) ===
{_oi_prompt_section}

=== 📊 EOD PREDICTIONS ===
{chr(10).join(eod_opportunities) if eod_opportunities else 'No EOD signals'}

=== � HOT WATCHLIST (scored 55+ but conviction-blocked — RE-EVALUATE these!) ===
{self._format_watchlist_for_prompt()}

=== �🔒 SKIP (already holding) ===
{', '.join(active_symbols) if active_symbols else 'None'}

=== ⚖️ CORRELATION EXPOSURE ===
{corr_exposure}

=== 🏥 SYSTEM HEALTH ===
Reconciliation: {recon_state} {'(CAN TRADE)' if recon_can_trade else '(BLOCKED: ' + recon_reason + ')'}
Data Health: {len(halted_symbols)} halted | {'Halted: ' + ', '.join(halted_symbols[:5]) if halted_symbols else 'All healthy'}

=== 💰 ACCOUNT ===
Capital: Rs{self.capital:,.0f} | Daily P&L: Rs{risk_status.daily_pnl:+,.0f} ({risk_status.daily_pnl_pct:+.2f}%)
Trades: {risk_status.trades_today}/{self.risk_governor.limits.max_trades_per_day} (Remaining: {trades_remaining})
W/L: {risk_status.wins_today}/{risk_status.losses_today} | Consec Losses: {risk_status.consecutive_losses}/{self.risk_governor.limits.max_consecutive_losses}

=== 🧠 YOUR TASK ===
1. Assess MARKET REGIME first (trending/range/mixed day, sector rotation)
2. ONLY pick from the '⚡ F&O READY SIGNALS' section above — these are pre-validated. Do NOT invent your own picks.
3. Look at the [Score:XX] tags — ONLY pick stocks scoring ≥56. Stocks below 56 WILL BE BLOCKED by the scorer.
4. Identify TOP {_dynamic_max} setups from the listed opportunities using CONFLUENCE SCORING:
   - Score ≥70 = PREMIUM (high conviction, sized up)
   - Score 65-69 = STANDARD (normal conviction)
   - Score 56-64 = BASELINE (enters but smaller size)
   - Score <56 = BLOCKED (do NOT pick)
5. USE ML + OI DATA to validate your picks:
   - 🧠ML:STRONG_MOVE or MOVE → PREFER these stocks (confirmed volatility)
   - 🧠ML:FLAT → AVOID for directional trades (stock predicted dead)
   - OI BULLISH + BUY direction → strong confluence (PREFER)
   - OI BEARISH + SELL direction → strong confluence (PREFER)
   - OI contradicts direction → extra caution, mention in reasoning
6. Check CONTRARIAN risks (chasing? extended? volume divergence? ML says FLAT?)
7. EXECUTE via tools — place_option_order(underlying, direction) for F&O, place_order() for cash
8. State your reasoning briefly: Setup | Score | ML Signal | OI Bias | Why

⚠️ CRITICAL RULES:
- Do NOT pick stocks outside the F&O READY SIGNALS list. They are NOT tradeable.
- Do NOT pick stocks scoring below 56. The scorer WILL block them.
- If ML says FLAT (🧠ML:FLAT) on a stock, do NOT pick it for directional trades — it will likely not move.
- If ML says MOVE and OI confirms direction — this is the STRONGEST signal. Prioritize these.
- If no setups score ≥56 with ML confirmation, say 'NO TRADES' — do NOT force a trade.
{_auto_fired_msg}
RULES: F&O → place_option_order() | Cash → place_order() | Max {_dynamic_max} trades"""

            response = self.agent.run(prompt)
            print(f"\n📊 Agent response:\n{response[:300]}...")
            
            # === AUTO-RETRY: Detect trades mentioned but not placed ===
            # Collect scorer rejections from the tools layer
            if not hasattr(self, '_rejected_this_cycle'):
                self._rejected_this_cycle = set()
            if hasattr(self.tools, '_scorer_rejected_symbols'):
                # Don't permanently blacklist hot watchlist stocks — allow retry next cycle
                try:
                    from options_trader import get_hot_watchlist
                    _wl = get_hot_watchlist()
                    _wl_syms = {s.replace('NSE:', '') for s in _wl.keys()}
                    _hard_rejected = self.tools._scorer_rejected_symbols - _wl_syms
                    _soft_rejected = self.tools._scorer_rejected_symbols & _wl_syms
                    self._rejected_this_cycle.update(_hard_rejected)
                    if _hard_rejected:
                        print(f"   🚫 Scorer rejected (won't retry): {_hard_rejected}")
                    if _soft_rejected:
                        print(f"   🔥 On hot watchlist (will retry next cycle): {_soft_rejected}")
                except Exception:
                    self._rejected_this_cycle.update(self.tools._scorer_rejected_symbols)
                    if self.tools._scorer_rejected_symbols:
                        print(f"   🚫 Scorer rejected (won't retry): {self.tools._scorer_rejected_symbols}")
                self.tools._scorer_rejected_symbols.clear()
            
            if response and not self.agent.get_pending_approvals():
                # Check if response mentions trade symbols but no orders were placed
                import re
                mentioned_symbols = re.findall(r'NSE:([A-Z]+)', response)
                # Build active set from BOTH equity symbols (NSE:X) AND option underlying (NSE:X)
                active_symbols_set = set()
                for t in self.tools.paper_positions:
                    if t.get('status', 'OPEN') != 'OPEN':
                        continue
                    sym = t.get('symbol', '')
                    # Equity: NSE:INFY → INFY
                    if sym.startswith('NSE:'):
                        active_symbols_set.add(sym.replace('NSE:', ''))
                    # Options: NFO:ICICIBANK26FEB1410CE → check underlying field
                    underlying = t.get('underlying', '')
                    if underlying:
                        active_symbols_set.add(underlying.replace('NSE:', ''))
                # Include ALL F&O stocks in retry — scanner covers them all
                retry_eligible = set(APPROVED_UNIVERSE) | set(self._wildcard_symbols) | _all_fo_syms
                unplaced = [s for s in set(mentioned_symbols) 
                           if s not in active_symbols_set 
                           and f'NSE:{s}' in retry_eligible
                           and s not in self._rejected_this_cycle]
                
                if unplaced and len(unplaced) <= 5:
                    print(f"\n🔄 Detected {len(unplaced)} unplaced trades in response: {unplaced}")
                    fno_prefer_set = {s.replace('NSE:', '') for s in FNO_CONFIG.get('prefer_options_for', [])} | {s.replace('NSE:', '') for s in _all_fo_syms}                    
                    # Parse direction from GPT's response text for each symbol
                    def _parse_direction_from_response(text, symbol):
                        """Extract BUY/SELL direction for a symbol from GPT response text"""
                        import re
                        # Look for patterns like "NSE:PAYTM | ... | SELL" or "direction: SELL" near symbol
                        # Search within 200 chars of the symbol mention
                        patterns = [
                            rf'{symbol}[^{{}}]{{0,200}}(?:direction|side|action)[:\s]*["\']?(SELL|BUY|BEARISH|BULLISH)',
                            rf'{symbol}[^{{}}]{{0,200}}direction["\s:=]*["\']?(BUY|SELL)',
                            rf'{symbol}[^{{}}]{{0,100}}\]\s*\[?(CE|PE)\b',
                            rf'{symbol}[^{{}}]{{0,200}}(SELL|SHORT|BEARISH|PUT|Breakdown|breakdown|fade)',
                            rf'{symbol}[^{{}}]{{0,200}}(BUY|LONG|BULLISH|CALL|Breakout|breakout|momentum up)',
                            rf'(SELL|SHORT|BEARISH)[^{{}}]{{0,100}}{symbol}',
                            rf'(BUY|LONG|BULLISH)[^{{}}]{{0,100}}{symbol}',
                        ]
                        for i, pattern in enumerate(patterns):
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                found = match.group(1) if i > 0 else match.group(1)
                                if found.upper() in ('SELL', 'SHORT', 'BEARISH', 'PUT', 'BREAKDOWN', 'FADE', 'PE'):
                                    return 'SELL'
                                elif found.upper() in ('BUY', 'LONG', 'BULLISH', 'CALL', 'BREAKOUT', 'MOMENTUM UP', 'CE'):
                                    return 'BUY'
                        return None
                    
                    for sym in unplaced[:3]:  # Max 3 retries
                        direction = _parse_direction_from_response(response, sym)
                        # Fallback: use scored direction from cycle decisions
                        if not direction:
                            _cd = _cycle_decisions.get(f'NSE:{sym}')
                            if _cd and _cd.get('decision'):
                                _scored_dir = _cd['decision'].recommended_direction
                                if _scored_dir in ('BUY', 'SELL'):
                                    direction = _scored_dir
                                    print(f"   🔄 Using scored direction for {sym}: {direction} (GPT text unparseable)")
                        if not direction:
                            print(f"   ⚠️ Could not parse direction for {sym} from GPT response, skipping")
                            continue
                        
                        print(f"   🔄 Direct-placing {sym} ({direction}) — parsed from GPT analysis")
                        
                        # ── ML DIRECTION CONFLICT FILTER (GPT direct-place) ──
                        _gpt_was_flipped = False
                        from config import ML_DIRECTION_CONFLICT
                        _gpt_dir_cfg = ML_DIRECTION_CONFLICT
                        if _gpt_dir_cfg.get('enabled', False) and _gpt_dir_cfg.get('block_gpt_trades', True):
                            _gpt_ml = getattr(self, '_cycle_ml_results', {}).get(f'NSE:{sym}', {})
                            _gpt_ml_signal = _gpt_ml.get('ml_signal', 'UNKNOWN')
                            _gpt_move_prob = _gpt_ml.get('ml_move_prob', _gpt_ml.get('ml_p_move', 0.0))
                            _gpt_dr_score = _gpt_ml.get('ml_down_risk_score', 0.0)
                            _gpt_xgb_disagrees = False
                            if direction == 'BUY' and _gpt_ml_signal == 'DOWN' and _gpt_move_prob >= _gpt_dir_cfg.get('min_xgb_confidence', 0.55):
                                _gpt_xgb_disagrees = True
                            elif direction == 'SELL' and _gpt_ml_signal == 'UP' and _gpt_move_prob >= _gpt_dir_cfg.get('min_xgb_confidence', 0.55):
                                _gpt_xgb_disagrees = True
                            
                            if _gpt_xgb_disagrees:
                                # XGB actively opposes GPT trade direction.
                                # DR INTERPRETATION: high dr confirms XGB's direction is real
                                _gpt_dr_flag = _gpt_ml.get('ml_down_risk_flag', False)
                                if _gpt_dr_flag:
                                    # HIGH dr → GMM confirms XGB's opposing direction
                                    # XGB + GMM both oppose → either FLIP or BLOCK
                                    _ovr_ok, _ovr_reason = self._ml_override_allowed(sym, _gpt_ml, _gpt_dr_score, path='GPT')
                                    if _ovr_ok:
                                        # Gates passed → FLIP to XGB direction
                                        old_direction = direction
                                        direction = 'BUY' if _gpt_ml_signal == 'UP' else 'SELL'
                                        _gpt_was_flipped = True
                                        print(f"   🔄 GPT ML_OVERRIDE_WGMM: {sym} — XGB={_gpt_ml_signal} + GMM confirms "
                                              f"(dr={_gpt_dr_score:.3f}) → FLIPPED {old_direction}→{direction}")
                                    else:
                                        # Gates failed → BLOCK (XGB+GMM both oppose but gates not met)
                                        print(f"   🚫 GPT ML_OVR BLOCKED: {sym} — {_ovr_reason}")
                                        continue
                                else:
                                    # LOW dr → GMM uncertain → XGB alone opposes → allow with warning
                                    print(f"   ⚠️ GPT XGB CONFLICT: {sym} — XGB={_gpt_ml_signal} vs GPT={direction}, "
                                          f"GMM uncertain (dr={_gpt_dr_score:.3f}) → allowing with caution")
                        
                        try:
                            # Build ML data for GPT direct-place trades
                            _gpt_ml_full = getattr(self, '_cycle_ml_results', {}).get(f'NSE:{sym}', {})
                            _gpt_ml_data = {
                                'smart_score': None,
                                'p_score': getattr(self, '_cycle_pre_scores', {}).get(f'NSE:{sym}', 0),
                                'dr_score': _gpt_ml_full.get('ml_down_risk_score', 0),
                                'ml_move_prob': _gpt_ml_full.get('ml_move_prob', 0),
                                'ml_confidence': _gpt_ml_full.get('ml_confidence', 0),
                                'xgb_model': {
                                    'signal': _gpt_ml_full.get('ml_signal', 'UNKNOWN'),
                                    'move_prob': _gpt_ml_full.get('ml_move_prob', 0),
                                    'prob_up': _gpt_ml_full.get('ml_prob_up', 0),
                                    'prob_down': _gpt_ml_full.get('ml_prob_down', 0),
                                    'prob_flat': _gpt_ml_full.get('ml_prob_flat', 0),
                                    'direction_bias': _gpt_ml_full.get('ml_direction_bias', 0),
                                    'confidence': _gpt_ml_full.get('ml_confidence', 0),
                                    'score_boost': _gpt_ml_full.get('ml_score_boost', 0),
                                    'direction_hint': _gpt_ml_full.get('ml_direction_hint', 'NEUTRAL'),
                                    'model_type': _gpt_ml_full.get('ml_model_type', 'unknown'),
                                    'sizing_factor': _gpt_ml_full.get('ml_sizing_factor', 1.0),
                                },
                                'gmm_model': {
                                    'down_risk_score': _gpt_ml_full.get('ml_down_risk_score', 0),
                                    'down_risk_flag': _gpt_ml_full.get('ml_down_risk_flag', False),
                                    'down_risk_bucket': _gpt_ml_full.get('ml_down_risk_bucket', 'LOW'),
                                    'gmm_confirms_direction': _gpt_ml_full.get('ml_gmm_confirms_direction', False),
                                    'gmm_regime_used': _gpt_ml_full.get('ml_gmm_regime_used', None),
                                    'gmm_action': 'GPT_DIRECT',
                                },
                                'scored_direction': direction,
                                'xgb_disagrees': _gpt_xgb_disagrees if '_gpt_xgb_disagrees' in dir() else False,
                            } if _gpt_ml_full else {}
                            _gpt_setup_type = 'ML_OVERRIDE_WGMM' if _gpt_was_flipped else 'GPT'
                            # ML_OVERRIDE gates already checked before flip above
                            
                            if sym in fno_prefer_set:
                                result = self.tools.place_option_order(
                                    underlying=f"NSE:{sym}",
                                    direction=direction,
                                    strike_selection="ATM",
                                    rationale=f"GPT identified {sym} as top setup ({direction})" + (" [ML-FLIPPED]" if _gpt_was_flipped else ""),
                                    setup_type=_gpt_setup_type,
                                    ml_data=_gpt_ml_data
                                )
                            else:
                                # For non-F&O, try option first then cash
                                result = self.tools.place_option_order(
                                    underlying=f"NSE:{sym}",
                                    direction=direction,
                                    strike_selection="ATM",
                                    rationale=f"GPT identified {sym} as top setup ({direction})" + (" [ML-FLIPPED]" if _gpt_was_flipped else ""),
                                    setup_type=_gpt_setup_type,
                                    ml_data=_gpt_ml_data
                                )
                            if result:
                                status = "PLACED" if result.get('success') else result.get('error', 'unknown')[:80]
                                print(f"   ✅ Direct-place {sym}: {status}")
                                # If not F&O eligible, blacklist for this session
                                if not result.get('success') and 'not F&O eligible' in result.get('error', ''):
                                    self._rejected_this_cycle.add(sym)
                                    print(f"   🚫 {sym} blacklisted: not F&O eligible")
                            else:
                                print(f"   ❌ Direct-place {sym}: No result returned")
                        except Exception as e:
                            print(f"   ❌ Direct-place {sym} failed: {str(e)[:100]}")
            
            # Check if agent created any trades
            pending = self.agent.get_pending_approvals()
            if pending:
                for trade in pending:
                    self._record_trade(trade)
            
            # Mark ORB trades as taken to prevent re-entry
            # Check active trades for symbols that had ORB signals
            for symbol, data in sorted_data:
                if isinstance(data, dict) and 'ltp' in data:
                    orb_signal = data.get('orb_signal', 'INSIDE_ORB')
                    if self.tools.is_symbol_in_active_trades(symbol):
                        if orb_signal == "BREAKOUT_UP":
                            self._mark_orb_trade_taken(symbol, "UP")
                        elif orb_signal == "BREAKOUT_DOWN":
                            self._mark_orb_trade_taken(symbol, "DOWN")
            
            # === DECISION DASHBOARD: Compact summary of what happened this cycle ===
            _cycle_elapsed = time.time() - _cycle_start
            _active_now = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            _options_now = [t for t in _active_now if t.get('is_option')]
            _spreads_now = [t for t in _active_now if t.get('is_credit_spread') or t.get('is_debit_spread')]
            _ics_now = [t for t in _active_now if t.get('is_iron_condor')]
            
            print(f"\n{'='*80}")
            print(f"📋 CYCLE SUMMARY @ {datetime.now().strftime('%H:%M:%S')} ({_cycle_elapsed:.0f}s)")
            print(f"{'='*80}")
            
            # Market Regime (safe access)
            _d_breadth = _breadth if '_breadth' in dir() else 'N/A'
            _d_up = _up_count if '_up_count' in dir() else '?'
            _d_down = _down_count if '_down_count' in dir() else '?'
            _d_flat = _flat_count if '_flat_count' in dir() else '?'
            print(f"🌐 Market: {_d_breadth} | Up:{_d_up} Down:{_d_down} Flat:{_d_flat}")
            
            # Scorer summary (safe access)
            _d_scores = _pre_scores if '_pre_scores' in dir() else {}
            _d_fno = fno_opportunities if 'fno_opportunities' in dir() else []
            
            if _d_scores:
                _top5 = sorted(_d_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                _passed = sum(1 for s in _d_scores.values() if s >= 52)
                _fno_count = len(_d_fno)
                _scan_mode = "FULL F&O" if _full_scan_mode else "CURATED+WC"
                print(f"📊 Scored: {len(_d_scores)} stocks [{_scan_mode}] | Passed(≥52): {_passed} | F&O Ready: {_fno_count}")
                _top5_str = " | ".join(f"{s.replace('NSE:','')}={v:.0f}" for s, v in _top5)
                print(f"🏆 Top 5: {_top5_str}")
            
            # F&O opportunities that went to GPT
            if _d_fno:
                print(f"\n🎯 F&O SIGNALS SENT TO GPT:")
                for opp in _d_fno[:8]:
                    print(f"   {opp.strip()}")
            
            # What got rejected/blocked
            if self._rejected_this_cycle:
                print(f"\n🚫 REJECTED: {', '.join(self._rejected_this_cycle)}")
            
            # Current portfolio state
            _total_open = len(_active_now)
            _sniper_positions = [t for t in _active_now if t.get('is_sniper') or t.get('setup_type') == 'GMM_SNIPER']
            _ml_override_positions = [t for t in _active_now if t.get('setup_type') == 'ML_OVERRIDE_WGMM']
            _model_tracker_positions = [t for t in _active_now if t.get('setup_type') in ('MODEL_TRACKER', 'ALL_AGREE')]
            _gpt_positions = [t for t in _active_now if t.get('setup_type') in ('', 'MANUAL', 'GPT') or not t.get('setup_type')]
            if _total_open > 0:
                _breakdown_parts = []
                if _sniper_positions:
                    _breakdown_parts.append(f"🎯{len(_sniper_positions)} sniper")
                if _ml_override_positions:
                    _breakdown_parts.append(f"🔄{len(_ml_override_positions)} ML-override-wGMM")
                if _model_tracker_positions:
                    _breakdown_parts.append(f"🧠{len(_model_tracker_positions)} score-based")
                if _gpt_positions:
                    _breakdown_parts.append(f"🤖{len(_gpt_positions)} GPT")
                _breakdown = " | ".join(_breakdown_parts) if _breakdown_parts else f"{len(_options_now)} options"
                print(f"\n💼 PORTFOLIO: {_total_open} positions ({_breakdown} | {len(_spreads_now)} spreads, {len(_ics_now)} ICs)")
                
                # Show sniper trades detail
                if _sniper_positions:
                    print(f"   🎯 SNIPER TRADES:")
                    for st in _sniper_positions:
                        _sym = st.get('underlying', st.get('symbol', '?')).replace('NSE:', '')
                        _dir = st.get('direction', '?')
                        _lots = st.get('lots', '?')
                        _mult = st.get('lot_multiplier', 1.0)
                        _entry = st.get('avg_price', 0)
                        print(f"      {_sym} ({_dir}) | {_lots} lots ({_mult}x) | Entry: ₹{_entry:.2f}")
            else:
                print(f"\n💼 PORTFOLIO: Empty — no open positions")
            
            # Hot watchlist
            try:
                from options_trader import get_hot_watchlist
                _wl = get_hot_watchlist()
                if _wl:
                    _wl_str = ", ".join(f"{k.replace('NSE:','')}({v.get('score',0):.0f})" for k, v in sorted(_wl.items(), key=lambda x: x[1].get('score',0), reverse=True)[:5])
                    print(f"🔥 WATCHLIST: {_wl_str}")
            except Exception:
                pass
            
            # P&L and timing
            _rs = self.risk_governor.state
            print(f"\n💰 P&L: ₹{_rs.daily_pnl:+,.0f} ({_rs.daily_pnl_pct:+.2f}%) | Trades: {_rs.trades_today} | W:{_rs.wins_today} L:{_rs.losses_today}")
            
            # Ticker stats
            if self.tools.ticker:
                _ts = self.tools.ticker.stats
                _ws_status = "🟢 LIVE" if _ts['connected'] else "🔴 REST"
                _fut_count = len(getattr(self.tools.ticker, '_futures_map', {}))
                print(f"🔌 Ticker: {_ws_status} | Sub:{_ts['subscribed']}(+{_fut_count} futures) | Hits:{_ts['cache_hits']} | Fallbacks:{_ts['fallbacks']} | Ticks:{_ts['ticks']}")
            
            print(f"⏱️ Cycle: {_cycle_elapsed:.0f}s | Next scan in ~{getattr(self, '_normal_interval', 5)}min")
            
            # === ADAPTIVE SCAN INTERVAL: Adjust next scan based on signal quality ===
            self._adapt_scan_interval(_pre_scores)
            
            # === DECISION LOG: Record all scored stocks this cycle ===
            self._log_cycle_decisions(
                cycle_time=_cycle_time,
                pre_scores=_pre_scores,
                fno_opportunities=fno_opportunities if 'fno_opportunities' in dir() else [],
                auto_fired=_auto_fired_syms,
                sorted_data=sorted_data,
                market_data=market_data
            )
            
            # Auto-fire stats in summary
            if _auto_fired_syms:
                print(f"🎯 Auto-Fired: {', '.join(s.replace('NSE:','') for s in _auto_fired_syms)}")
            if _dynamic_max != 3:
                print(f"📊 Dynamic Picks: {_dynamic_max} (was 3)")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._scanning = False  # Re-enable real-time dashboard
    
    def _record_trade(self, trade: dict):
        """Record a trade (paper or real) with risk checks"""
        # Check with risk governor first
        active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        position_value = trade.get('entry_price', 0) * trade.get('quantity', 0)
        
        permission = self.risk_governor.can_trade(
            symbol=trade['symbol'],
            position_value=position_value,
            active_positions=active_positions
        )
        
        if not permission.allowed:
            print(f"\n⛔ TRADE BLOCKED by Risk Governor:")
            print(f"   Symbol: {trade['symbol']}")
            print(f"   Reason: {permission.reason}")
            for w in permission.warnings:
                print(f"   ⚠️ {w}")
            return  # Don't record blocked trades
        
        # Apply size adjustment if suggested
        if permission.suggested_size_multiplier < 1.0:
            original_qty = trade.get('quantity', 0)
            trade['quantity'] = int(original_qty * permission.suggested_size_multiplier)
            print(f"   📉 Position size reduced: {original_qty} → {trade['quantity']}")
        
        trade['timestamp'] = datetime.now().isoformat()
        trade['paper'] = self.paper_mode
        
        if self.paper_mode:
            print(f"\n📝 PAPER TRADE: {trade['side']} {trade.get('quantity', 0)} {trade['symbol']}")
            print(f"   Entry: ₹{trade.get('entry_price', 0)}")
            print(f"   Stop: ₹{trade.get('stop_loss', 0)}")
            print(f"   Target: ₹{trade.get('target', 0)}")
        
        # Show warnings if any
        if permission.warnings:
            print(f"   ⚠️ Warnings: {', '.join(permission.warnings)}")
        
        # Register with Exit Manager for smart exits
        self.exit_manager.register_trade(
            symbol=trade['symbol'],
            side=trade['side'],
            entry_price=trade.get('entry_price', 0),
            stop_loss=trade.get('stop_loss', 0),
            target=trade.get('target', 0),
            quantity=trade.get('quantity', 0)
        )
        
        self.trades_today.append(trade)
        
        # Save to file
        with open('trade_log.json', 'a') as f:
            f.write(json.dumps(trade) + '\n')
    
    def monitor_positions(self):
        """Monitor open positions for exit signals"""
        if not self.positions:
            return
        
        print(f"👀 Monitoring {len(self.positions)} positions...")
        
        for pos in self.positions[:]:
            try:
                # Get current price
                data = self.tools.get_market_data([pos['symbol']])
                if pos['symbol'] in data:
                    ltp = data[pos['symbol']]['ltp']
                    
                    # Check stop loss
                    if ltp <= pos['stop_loss']:
                        self._exit_position(pos, ltp, "Stop Loss Hit")
                    
                    # Check target
                    elif ltp >= pos['target']:
                        self._exit_position(pos, ltp, "Target Hit")
            
            except Exception as e:
                print(f"❌ Error monitoring {pos['symbol']}: {e}")
    
    def _exit_position(self, pos: dict, exit_price: float, reason: str):
        """Exit a position"""
        pnl = (exit_price - pos['entry_price']) * pos['quantity']
        if pos['side'] == 'SELL':
            pnl = -pnl
        pnl -= calc_brokerage(pos['entry_price'], exit_price, pos['quantity'])
        
        with self._pnl_lock:
            self.daily_pnl += pnl
            self.capital += pnl
        
        print(f"\n🚪 EXIT: {pos['symbol']} @ ₹{exit_price:.2f}")
        print(f"   Reason: {reason}")
        print(f"   P&L: ₹{pnl:,.0f}")
        print(f"   Daily P&L: ₹{self.daily_pnl:,.0f}")
        
        self.positions.remove(pos)
    
    def _sync_broker_positions(self):
        """LIVE MODE ONLY: Check if any SL-M orders triggered at Zerodha that Titan missed.
        
        Compares Kite's actual open positions against Titan's tracked positions.
        If a position was closed by the broker SL order, update Titan's state.
        Also handles spread/condor positions by checking individual legs.
        """
        if self.paper_mode:
            return
        
        try:
            # Get actual positions from Zerodha
            broker_positions = self.tools.kite.positions()
            day_positions = broker_positions.get('day', [])
            
            # Get broker's open positions (non-zero quantity)
            broker_open = {}
            for bp in day_positions:
                symbol = f"{bp['exchange']}:{bp['tradingsymbol']}"
                if bp['quantity'] != 0:
                    broker_open[symbol] = bp
            
            # Get Titan's tracked open positions
            with self.tools._positions_lock:
                titan_open = [t for t in self.tools.paper_positions 
                            if t.get('status', 'OPEN') == 'OPEN']
            
            # Check each Titan position — if broker has ZERO qty, SL triggered
            for trade in titan_open:
                symbol = trade.get('symbol', '')
                
                # === SPREAD/CONDOR: Check individual legs ===
                if '|' in symbol:
                    # Composite symbol — check each leg individually
                    leg_symbols = symbol.split('|')
                    all_legs_closed = True
                    any_leg_closed = False
                    
                    for leg_sym in leg_symbols:
                        if leg_sym in broker_open:
                            all_legs_closed = False
                        else:
                            any_leg_closed = True
                    
                    if all_legs_closed:
                        # All legs closed at broker — mark trade as closed
                        entry = trade.get('avg_price', 0)
                        qty = trade.get('quantity', 0)
                        
                        # Get total P&L from broker positions for these legs
                        total_pnl = 0
                        for leg_sym in leg_symbols:
                            for bp in day_positions:
                                bp_sym = f"{bp['exchange']}:{bp['tradingsymbol']}"
                                if bp_sym == leg_sym:
                                    total_pnl += bp.get('realised', 0) + bp.get('unrealised', 0)
                                    break
                        
                        print(f"\n🔄 BROKER SYNC: Spread/Condor {trade.get('underlying', symbol)} closed by broker")
                        print(f"   P&L from broker: ₹{total_pnl:+,.2f}")
                        
                        self.tools.update_trade_status(symbol, 'BROKER_CLOSED', 0, total_pnl,
                            exit_detail={'exit_reason': 'BROKER_CLOSED', 'exit_type': 'SPREAD_CLOSE'})
                        with self._pnl_lock:
                            self.daily_pnl += total_pnl
                            self.capital += total_pnl
                    
                    elif any_leg_closed:
                        # Partial leg close — log warning (spreads should close together)
                        closed_legs = [s for s in leg_symbols if s not in broker_open]
                        print(f"   ⚠️ BROKER SYNC: Partial leg close on {trade.get('underlying', symbol)}: {closed_legs}")
                    
                    continue  # Skip single-symbol logic
                
                # === SINGLE-LEG POSITIONS ===
                if symbol not in broker_open:
                    # Position closed at broker (SL triggered or manual close)
                    entry = trade.get('avg_price', 0)
                    qty = trade.get('quantity', 0)
                    side = trade.get('side', 'BUY')
                    
                    # Get execution price from broker's filled SL order
                    exit_price = 0
                    try:
                        orders = self.tools.kite.orders()
                        sl_order_id = trade.get('sl_order_id', '')
                        for o in orders:
                            if str(o.get('order_id')) == str(sl_order_id) and o.get('status') == 'COMPLETE':
                                exit_price = o.get('average_price', 0)
                                break
                        # If SL order not found, check for any SELL order for this symbol
                        if not exit_price:
                            for o in orders:
                                if (o.get('tradingsymbol') in symbol and 
                                    o.get('status') == 'COMPLETE' and 
                                    o.get('transaction_type') in ('SELL', 'BUY')):
                                    exit_price = o.get('average_price', 0)
                    except Exception:
                        pass
                    
                    if not exit_price:
                        # Fallback: use LTP
                        try:
                            q = self.tools.kite.ltp([symbol])
                            exit_price = q[symbol]['last_price']
                        except Exception:
                            exit_price = entry  # Last resort
                    
                    # Calculate P&L
                    from config import calc_brokerage
                    if side == 'BUY':
                        pnl = (exit_price - entry) * qty
                    else:
                        pnl = (entry - exit_price) * qty
                    pnl -= calc_brokerage(entry, exit_price, qty)
                    
                    print(f"\n🔄 BROKER SYNC: {symbol} position closed by broker")
                    print(f"   Entry: ₹{entry:.2f} → Exit: ₹{exit_price:.2f} | P&L: ₹{pnl:+,.2f}")
                    
                    # Update Titan's state
                    self.tools.update_trade_status(symbol, 'STOPLOSS_HIT', exit_price, pnl, 
                        exit_detail={'exit_reason': 'BROKER_SL_TRIGGERED', 'exit_type': 'SL_HIT'})
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    
        except Exception as e:
            print(f"   ⚠️ Broker position sync failed: {e}")

    def run(self, scan_interval_minutes: int = 5):
        """Run the autonomous trader with dynamic scan intervals.
        
        Early session (before EARLY_SESSION.end_time): scans every 3 minutes
        using 3-minute candles for faster indicator maturation.
        After early session: switches to standard 5-minute scans.
        """
        from config import EARLY_SESSION
        
        # Determine initial scan interval based on time of day
        _now = datetime.now()
        _early_end_parts = EARLY_SESSION['end_time'].split(':')
        _early_end = _now.replace(hour=int(_early_end_parts[0]), minute=int(_early_end_parts[1]), second=0, microsecond=0)
        _is_early = EARLY_SESSION.get('enabled', True) and _now < _early_end
        
        if _is_early:
            current_interval = EARLY_SESSION.get('scan_interval_minutes', 3)
            print(f"\n🚀 Starting autonomous trading (EARLY SESSION MODE)...")
            print(f"   📊 Using {EARLY_SESSION.get('candle_interval', '3minute')} candles until {EARLY_SESSION['end_time']}")
            print(f"   Scanning every {current_interval} minutes (switches to {scan_interval_minutes}min after {EARLY_SESSION['end_time']})")
        else:
            current_interval = scan_interval_minutes
            print(f"\n🚀 Starting autonomous trading...")
            print(f"   Scanning every {current_interval} minutes")
        
        print(f"   Real-time monitoring every {self.monitor_interval} seconds")
        print(f"   Press Ctrl+C to stop\n")
        
        # Start real-time position monitor
        self.start_realtime_monitor()
        
        # Track whether we've switched from early to normal mode
        self._switched_to_normal = not _is_early
        # After 9:45 always use 5-min interval regardless of CLI arg
        self._normal_interval = 5
        
        # === DEBUG LOG FILE ===
        import traceback as _tb
        _debug_log = os.path.join(os.path.dirname(__file__), 'bot_debug.log')
        def _dbg(msg):
            """Write timestamped debug message to file + stdout"""
            _ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            _line = f"[{_ts}] {msg}"
            print(_line)
            try:
                with open(_debug_log, 'a', encoding='utf-8') as _f:
                    _f.write(_line + '\n')
            except Exception:
                pass

        _dbg("DEBUG: run() entered, scheduling tasks...")
        
        # Schedule tasks with current interval
        schedule.every(current_interval).minutes.do(self.scan_and_trade)
        _dbg(f"DEBUG: scheduled scan_and_trade every {current_interval} min")
        
        # Initial scan
        _dbg("DEBUG: calling initial scan_and_trade()...")
        try:
            self.scan_and_trade()
            _dbg("DEBUG: initial scan_and_trade() completed OK")
        except Exception as _e:
            _dbg(f"ERROR: initial scan_and_trade() CRASHED: {_e}")
            _dbg(f"TRACEBACK:\n{_tb.format_exc()}")
        
        _dbg("DEBUG: entering main while-loop...")
        # Run loop
        try:
            _loop_count = 0
            while True:
                try:
                    schedule.run_pending()
                except Exception as _e:
                    _dbg(f"ERROR: schedule.run_pending() failed: {_e}")
                    _dbg(f"TRACEBACK:\n{_tb.format_exc()}")
                
                # Check if we need to switch from early session to normal interval
                if not self._switched_to_normal:
                    _check_now = datetime.now()
                    if _check_now >= _early_end:
                        self._switched_to_normal = True
                        schedule.clear()
                        schedule.every(self._normal_interval).minutes.do(self.scan_and_trade)
                        _dbg(f"DEBUG: Early session ended — switching to {self._normal_interval}-min interval")
                
                _loop_count += 1
                if _loop_count % 60 == 0:  # Log heartbeat every 60s
                    _dbg(f"HEARTBEAT: loop iteration {_loop_count}, system_state={getattr(self.risk_governor.state, 'system_state', '?')}")
                
                time.sleep(1)
        except KeyboardInterrupt:
            _dbg("DEBUG: KeyboardInterrupt received, shutting down...")
            print("\n\n👋 Shutting down...")
            self.stop_realtime_monitor()
            self._print_summary()
        except Exception as _e:
            _dbg(f"FATAL: main loop crashed: {_e}")
            _dbg(f"TRACEBACK:\n{_tb.format_exc()}")
            self.stop_realtime_monitor()
            self._print_summary()
    
    def _print_summary(self):
        """Print trading summary using centralized Trade Ledger"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # === Collect data from Trade Ledger ===
        history = []
        try:
            from trade_ledger import get_trade_ledger
            trades = get_trade_ledger().get_trades_with_pnl(today)
            # Convert TradeLedger format to legacy format for compatibility
            for t in trades:
                history.append({
                    'symbol': t.get('symbol', ''),
                    'underlying': t.get('underlying', ''),
                    'side': t.get('direction', ''),
                    'pnl': t.get('total_pnl', 0),
                    'result': t.get('final_exit_type', ''),
                    'entry_score': t.get('smart_score', 0),
                    'score_tier': t.get('score_tier', ''),
                    'strategy_type': t.get('strategy_type', ''),
                    'is_sniper': t.get('is_sniper', False),
                    'avg_price': t.get('entry_price', 0),
                    'exit_price': t.get('exit_price', 0),
                    'timestamp': t.get('entry_time', ''),
                    'exit_detail': {
                        'candles_held': t.get('candles_held', 0),
                        'r_multiple_achieved': t.get('r_multiple', 0),
                        'max_favorable_excursion': t.get('max_favorable', 0),
                        'exit_reason': t.get('exit_reason', ''),
                    },
                    'entry_metadata': {
                        'entry_score': t.get('smart_score', 0),
                        'smart_score': t.get('smart_score', 0),
                        'score_tier': t.get('score_tier', ''),
                    },
                })
        except Exception:
            pass
        
        total_trades = len(history)
        winners = [t for t in history if t.get('pnl', 0) > 0]
        losers = [t for t in history if t.get('pnl', 0) < 0]
        breakevens = [t for t in history if t.get('pnl', 0) == 0]
        
        total_pnl = sum(t.get('pnl', 0) for t in history)
        avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
        avg_loss = sum(t['pnl'] for t in losers) / len(losers) if losers else 0
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        
        # By strategy
        by_strategy = {}
        for t in history:
            strat = t.get('strategy_type', t.get('spread_type', 'NAKED_OPTION'))
            if t.get('is_iron_condor'):
                strat = 'IRON_CONDOR'
            elif t.get('is_credit_spread'):
                strat = 'CREDIT_SPREAD'
            elif t.get('is_debit_spread'):
                strat = 'DEBIT_SPREAD'
            if strat not in by_strategy:
                by_strategy[strat] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('pnl', 0) > 0:
                by_strategy[strat]['wins'] += 1
            elif t.get('pnl', 0) < 0:
                by_strategy[strat]['losses'] += 1
            by_strategy[strat]['pnl'] += t.get('pnl', 0)
        
        # By exit type
        by_exit = {}
        for t in history:
            exit_type = t.get('result', 'UNKNOWN')
            by_exit[exit_type] = by_exit.get(exit_type, 0) + 1
        
        # By score tier
        by_tier = {}
        for t in history:
            tier = t.get('score_tier', t.get('entry_metadata', {}).get('score_tier', 'unknown'))
            if tier not in by_tier:
                by_tier[tier] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('pnl', 0) > 0:
                by_tier[tier]['wins'] += 1
            elif t.get('pnl', 0) < 0:
                by_tier[tier]['losses'] += 1
            by_tier[tier]['pnl'] += t.get('pnl', 0)
        
        # Candle gate analysis — did entry characteristics predict outcome?
        gate_analysis = {'ft_winners_avg': 0, 'ft_losers_avg': 0, 'adx_winners_avg': 0, 'adx_losers_avg': 0}
        for t in history:
            meta = t.get('entry_metadata', {})
            ft = meta.get('follow_through_candles', 0)
            adx = meta.get('adx', 0)
            if t.get('pnl', 0) > 0:
                gate_analysis['ft_winners_avg'] = (gate_analysis['ft_winners_avg'] + ft) / 2 if gate_analysis['ft_winners_avg'] else ft
                gate_analysis['adx_winners_avg'] = (gate_analysis['adx_winners_avg'] + adx) / 2 if gate_analysis['adx_winners_avg'] else adx
            elif t.get('pnl', 0) < 0:
                gate_analysis['ft_losers_avg'] = (gate_analysis['ft_losers_avg'] + ft) / 2 if gate_analysis['ft_losers_avg'] else ft
                gate_analysis['adx_losers_avg'] = (gate_analysis['adx_losers_avg'] + adx) / 2 if gate_analysis['adx_losers_avg'] else adx
        
        # Exit manager quality — candles held, R achieved
        avg_candles_held = 0
        avg_r_achieved = 0
        max_r_left_on_table = 0
        r_trades = 0
        for t in history:
            ed = t.get('exit_detail', {})
            candles = ed.get('candles_held', 0)
            r_mult = ed.get('r_multiple_achieved', 0)
            if candles > 0:
                avg_candles_held += candles
                r_trades += 1
            if r_mult:
                avg_r_achieved += r_mult
        if r_trades > 0:
            avg_candles_held /= r_trades
            avg_r_achieved /= r_trades
        
        # === PRINT TO TERMINAL ===
        print("\n" + "="*60)
        print(f"📊 DAILY TRADING SUMMARY — {today}")
        print("="*60)
        print(f"  Capital: ₹{self.start_capital:,.0f} → ₹{self.capital:,.0f}")
        print(f"  Daily P&L: ₹{self.daily_pnl:,.0f} ({(self.capital/self.start_capital - 1)*100:+.2f}%)")
        print(f"  Trades: {total_trades} | W: {len(winners)} | L: {len(losers)} | BE: {len(breakevens)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Avg Winner: ₹{avg_win:+,.0f} | Avg Loser: ₹{avg_loss:+,.0f}")
        if avg_loss != 0:
            print(f"  Payoff Ratio: {abs(avg_win/avg_loss):.2f}:1")
        
        if by_strategy:
            print(f"\n  📈 BY STRATEGY:")
            for strat, data in by_strategy.items():
                print(f"    {strat}: W{data['wins']}/L{data['losses']} P&L: ₹{data['pnl']:+,.0f}")
        
        if by_exit:
            print(f"\n  🚪 BY EXIT TYPE:")
            for exit_type, count in sorted(by_exit.items(), key=lambda x: x[1], reverse=True):
                print(f"    {exit_type}: {count}")
        
        if by_tier:
            print(f"\n  🏆 BY SCORE TIER:")
            for tier, data in by_tier.items():
                print(f"    {tier}: W{data['wins']}/L{data['losses']} P&L: ₹{data['pnl']:+,.0f}")
        
        if r_trades > 0:
            print(f"\n  ⏱️ EXIT QUALITY:")
            print(f"    Avg Candles Held: {avg_candles_held:.1f}")
            print(f"    Avg R-Multiple: {avg_r_achieved:.2f}")
        
        print("="*60)
        
        # Flush OI collector buffers to parquet (EOD)
        try:
            if hasattr(self, '_oi_collector'):
                self._oi_collector.flush_all()
                print(f"  💾 OI snapshots flushed to parquet for ML training")
        except Exception:
            pass
        
        print(f"\n  💾 All trade data logged to Trade Ledger (trade_ledger/)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Trading Bot')
    parser.add_argument('--capital', type=float, default=500000, help='Starting capital')
    parser.add_argument('--live', action='store_true', default=None, help='Enable live trading (overrides .env TRADING_MODE)')
    parser.add_argument('--paper', action='store_true', default=None, help='Force paper trading (overrides .env TRADING_MODE)')
    parser.add_argument('--interval', type=int, default=5, help='Scan interval in minutes')
    
    args = parser.parse_args()
    
    # Priority: CLI flags > .env TRADING_MODE > default (paper)
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        # Read from .env / config
        import config as _cfg_main
        paper_mode = _cfg_main.PAPER_MODE  # Already parsed from TRADING_MODE env var
        _source = os.environ.get('TRADING_MODE', 'PAPER').strip().upper()
        print(f"  📋 Mode from .env TRADING_MODE={_source} → {'PAPER' if paper_mode else 'LIVE'}")
    
    import traceback as _main_tb
    _debug_log = os.path.join(os.path.dirname(__file__), 'bot_debug.log')
    
    def _main_dbg(msg):
        _ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        _line = f"[{_ts}] {msg}"
        print(_line)
        try:
            with open(_debug_log, 'a', encoding='utf-8') as _f:
                _f.write(_line + '\n')
        except Exception:
            pass
    
    _main_dbg("DEBUG main(): creating AutonomousTrader...")
    try:
        bot = AutonomousTrader(
            capital=args.capital,
            paper_mode=paper_mode
        )
        _main_dbg("DEBUG main(): AutonomousTrader created OK, calling run()...")
        bot.run(scan_interval_minutes=args.interval)
    except Exception as _e:
        _main_dbg(f"FATAL main(): {_e}")
        _main_dbg(f"TRACEBACK:\n{_main_tb.format_exc()}")
        raise


if __name__ == "__main__":
    main()
