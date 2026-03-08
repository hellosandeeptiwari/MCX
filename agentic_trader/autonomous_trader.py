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

from config import HARD_RULES, APPROVED_UNIVERSE, TRADING_HOURS, FNO_CONFIG, TIER_1_OPTIONS, TIER_2_OPTIONS, FULL_FNO_SCAN, BREAKOUT_WATCHER, calc_brokerage
from llm_agent import TradingAgent
from zerodha_tools import get_tools, reset_tools
from market_scanner import get_market_scanner
from options_trader import update_fno_lot_sizes
from exit_manager import get_exit_manager, calculate_structure_sl, check_underlying_adverse
from execution_guard import get_execution_guard
from risk_governor import get_risk_governor, SystemState
from correlation_guard import get_correlation_guard
from regime_score import get_regime_scorer
from position_reconciliation import get_position_reconciliation
from data_health_gate import get_data_health_gate
from trade_ledger import get_trade_ledger
from state_db import get_state_db


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
        self._profit_target_hit = False  # Kill-all profit switch
        self._eod_ledger_reconciled = False  # EOD orphan reconciliation (once per day)
        
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
            # Headless: auto-confirm (mode is controlled by .env TRADING_MODE)
            if sys.stdin.isatty():
                confirm = input("  Type 'CONFIRM' to proceed: ")
                if confirm != "CONFIRM":
                    print("  Aborted.")
                    sys.exit(0)
            else:
                print("  Headless mode — auto-confirmed via .env TRADING_MODE")
        
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
        
        # Wire manual-exit reconciliation callback (Kite app exits)
        self.position_recon.on_manual_exit_callback = self._on_recon_manual_exit
        
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
        
        # print("\n  ✅ Bot initialized!")
        # print("  🟢 Auto-execution: ON")
        # print("  ⚡ Real-time monitoring: ENABLED (every 3 sec)")
        # print("  📊 Exit Manager: ACTIVE")
        # print("  🛡️ Execution Guard: ACTIVE")
        # print("  ⚖️ Risk Governor: ACTIVE")
        # print("  🔗 Correlation Guard: ACTIVE")
        # print("  📈 Regime Scorer: ACTIVE")
        # print("  🔄 Position Reconciliation: ACTIVE (every 10s)")
        # print("  🛡️ Data Health Gate: ACTIVE")
        # print("  📊 Options Trading: ACTIVE (F&O stocks)")
        # print("  🛡️ GTT Safety Net: ACTIVE (server-side SL+target)")
        # print("  📦 Autoslice: ACTIVE (freeze qty protection)")
        # print("  ⚡ IOC Validity: ACTIVE (spread legs)")
        print(f"  ✅ All subsystems initialized")
        
        # === NEW: Decision Log + Elite Auto-Fire + Adaptive Scan ===
        from config import DECISION_LOG, ELITE_AUTO_FIRE, ADAPTIVE_SCAN, DYNAMIC_MAX_PICKS, DOWN_RISK_GATING
        self._decision_log_cfg = DECISION_LOG
        self._elite_auto_fire_cfg = ELITE_AUTO_FIRE
        self._adaptive_scan_cfg = ADAPTIVE_SCAN
        self._dynamic_max_picks_cfg = DYNAMIC_MAX_PICKS
        self._auto_fired_this_session = set()   # Symbols auto-fired today (no re-fire)
        self._watcher_fired_this_session = set()    # All symbols traded via breakout watcher today
        self._watcher_momentum_tracker = {}          # opt_symbol → {underlying, direction, entry_ul_price, entry_time, peak_price, trough_price}
        self._wme_session_start = time.time()        # WME: skip positions that pre-existed this session
        self._last_watcher_scan_time = 0             # Cooldown between watcher focused scans (epoch)
        self._watcher_drain_count = 0                # Total drain calls today
        self._watcher_total_drained = 0              # Total triggers drained today
        self._watcher_total_actionable = 0           # Total triggers that passed filter
        self._watcher_total_pipeline_sent = 0        # Total triggers sent to focused scan
        self._watcher_total_gate_blocked = 0         # Total blocked by pipeline gates
        self._watcher_total_placed = 0               # Total trades successfully placed
        self._watcher_total_pos_exhausted = 0        # Times blocked by position limit

        # === SETTINGS HOT-RELOAD (titan_settings.json) ===
        self._settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titan_settings.json')
        self._settings_mtime = 0  # Tracks file modification time
        self._apply_settings_overrides()  # Apply on startup
        self._last_market_breadth = 'MIXED'          # Cached market breadth from last scan_and_trade
        self._last_signal_quality = 'normal'     # Track signal quality for adaptive scan
        
        # if ELITE_AUTO_FIRE.get('enabled'): print("  🎯 Elite Auto-Fire: ACTIVE (score ≥78 → instant execution)")
        # if DYNAMIC_MAX_PICKS.get('enabled'): print(f"  📊 Dynamic Max Picks: ACTIVE (3→5 when signals are hot)")
        # if ADAPTIVE_SCAN.get('enabled'): print(f"  ⏱️ Adaptive Scan: ACTIVE ({ADAPTIVE_SCAN['fast_interval_minutes']}min/{ADAPTIVE_SCAN['normal_interval_minutes']}min/{ADAPTIVE_SCAN['slow_interval_minutes']}min)")
        # if DECISION_LOG.get('enabled'): print(f"  📝 Decision Log: ACTIVE ({DECISION_LOG['file']})")
        
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
        # TEST_GMM: Pure DR model play (bypass ALL gates)
        from config import TEST_GMM
        self._test_gmm_cfg = TEST_GMM
        self._test_gmm_trades_today = 0
        self._test_gmm_date = datetime.now().date()
        self._test_gmm_symbols = set()
        # TEST_XGB: Pure XGBoost model play (bypass GMM/smart_score)
        from config import TEST_XGB
        self._test_xgb_cfg = TEST_XGB
        self._test_xgb_trades_today = 0
        self._test_xgb_date = datetime.now().date()
        self._test_xgb_symbols = set()
        # ARBTR: Sector Arbitrage — laggard convergence play
        try:
            from config import ARBTR_CONFIG, ARBTR_SECTOR_MAP
            self._arbtr_cfg = ARBTR_CONFIG
            self._arbtr_sector_map = ARBTR_SECTOR_MAP
        except ImportError:
            self._arbtr_cfg = {'enabled': False}
            self._arbtr_sector_map = {}
        self._arbtr_trades_today = 0
        self._arbtr_date = datetime.now().date()
        self._arbtr_symbols = set()
        self._arbtr_sector_cooldowns = {}  # sector_name → last_entry_time
        # GCR: GMM Conviction Recheck — re-query GMM on losing positions
        try:
            from config import GCR_CONFIG
            self._gcr_cfg = GCR_CONFIG
        except ImportError:
            self._gcr_cfg = {'enabled': False}
        self._gcr_oppose_counts = {}   # symbol → consecutive opposing count
        self._gcr_exits_today = 0
        self._gcr_date = datetime.now().date()
        self._gcr_exit_times = {}      # underlying → last exit time (for cooldown)
        # VIX REGIME: India VIX-based entry/SL/trailing adjustment
        try:
            from config import VIX_REGIME_CONFIG
            self._vix_cfg = VIX_REGIME_CONFIG
        except ImportError:
            self._vix_cfg = {'enabled': False}
        self._current_vix = self._vix_cfg.get('fallback_vix', 14.0)
        self._vix_regime = 'NORMAL'       # LOW / NORMAL / HIGH / EXTREME
        self._vix_last_fetch = 0.0        # timestamp of last VIX fetch
        # GMM Contrarian (DR_FLIP) state
        self._dr_flip_symbols = set()
        self._dr_flip_trades_today = 0
        # ML_OVERRIDE gates: loaded from config.ML_OVERRIDE_GATES
        try:
            from config import ML_OVERRIDE_GATES
            self._ml_ovr_cfg = ML_OVERRIDE_GATES
        except ImportError:
            self._ml_ovr_cfg = {'min_move_prob': 0.55, 'max_updr_score': 0.12,
                                'max_downdr_score': 0.09,
                                'min_directional_prob': 0.30, 'max_concurrent_open': 3}
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
        # if DOWN_RISK_GATING.get('enabled'):
        #     print(f"  🛡️ Down-Risk Graduated Scoring: ACTIVE (boost +{DOWN_RISK_GATING.get('clean_boost', 8)} / caution −{DOWN_RISK_GATING.get('mid_risk_penalty', 8)} / block −{DOWN_RISK_GATING.get('high_risk_penalty', 15)})")
        #     print(f"  📊 Model Tracker: ACTIVE ({DOWN_RISK_GATING.get('model_tracker_trades', 7)} exclusive trades/day for model evaluation)")
        # if GMM_SNIPER.get('enabled'):
        #     print(f"  🎯 GMM Sniper: ACTIVE ({GMM_SNIPER.get('max_sniper_trades_per_day', 5)}/day, {GMM_SNIPER.get('lot_multiplier', 2.0)}x lots, dr<{GMM_SNIPER.get('max_dr_score', 0.10)})")
        # if TEST_GMM.get('enabled'):
        #     print(f"  🧪 TEST_GMM: ACTIVE ({TEST_GMM.get('max_trades_per_day', 4)}/day, pure GMM play, dr<{TEST_GMM.get('max_dr_score', 0.06)})")
        # if getattr(self, '_sniper_engine', None):
        #     _se_status = self._sniper_engine.get_status()
        #     if _se_status['oi_unwinding']['enabled']:
        #         print(f"  🔫 Sniper-OIUnwinding: ACTIVE ({_se_status['oi_unwinding']['max_per_day']}/day, OI reversal at S/R)")
        #     if _se_status['pcr_extreme']['enabled']:
        #         print(f"  🔫 Sniper-PCRExtreme: ACTIVE ({_se_status['pcr_extreme']['max_per_day']}/day, PCR contrarian fade)")
        
        # === ML MOVE PREDICTOR ===
        self._ml_predictor = None
        try:
            from ml_models.predictor import MovePredictor
            self._ml_predictor = MovePredictor()
            if self._ml_predictor.ready:
                # print("  🧠 ML Move Predictor: ACTIVE (XGBoost score booster)")
                pass
            else:
                # print("  🧠 ML Move Predictor: STANDBY (no trained model yet)")
                pass
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
                # print(f"  📊 Historical 5-min candles: {len(self._hist_5min_cache)} stocks + {'✓' if self._nifty_5min_df is not None else '✗'} NIFTY50")
            # Load NIFTY50 daily candles
            _nifty_daily_path = os.path.join(_daily_dir, 'NIFTY50.parquet')
            if os.path.exists(_nifty_daily_path):
                self._nifty_daily_df = _pd_hist.read_parquet(_nifty_daily_path)
                self._nifty_daily_df['date'] = _pd_hist.to_datetime(self._nifty_daily_df['date'])
                # print(f"  📊 NIFTY50 daily context: ✓ ({len(self._nifty_daily_df)} days)")
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
                # CRITICAL: Use _gap > 1 (not 3) so that after-holiday gaps
                # still trigger refresh. March 3 was a holiday, so March 2→4
                # is only 2 days gap but the 5-min data is stale and shows
                # March 2's bullish intraday recovery instead of today's crash.
                if _gap > 1:
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
                            # print(f"  ✅ 5-min parquets refreshed: {_refreshed}/{len(_all_syms)} stocks (now up to {str(_new_last)[:10]})")
                        else:
                            print(f"  ⚠️ 5-min parquet refresh: 0 stocks updated (Kite API may be unavailable)")
                        
                    except ImportError:
                        print("  ⚠️ data_fetcher not available — 5-min refresh skipped")
                    except Exception as _5m_err:
                        print(f"  ⚠️ 5-min parquet refresh error: {_5m_err}")
                else:
                    # print(f"  ✅ 5-min parquets: fresh (last={_last_date.date()}, {_gap}d ago)")
                    pass
        except Exception as _5m_chk_e:
            print(f"  ⚠ 5-min freshness check: {_5m_chk_e}")

        # === NIFTY DAILY BACKFILL (independent of 5-min staleness) ===
        # CRITICAL: This MUST run every startup, not just when 5-min is stale.
        # The direction model needs ≥60 daily rows for EMA-50 feature warmup.
        # If only 45 rows exist, nifty_daily_trend=0 and nifty_daily_rsi=0
        # which blinds the model to the overall market regime, causing
        # uniform UP predictions even in a crash.
        try:
            import pandas as _pd_nd
            from ml_models.data_fetcher import fetch_candles
            _nifty_daily_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'candles_daily', 'NIFTY50.parquet')
            _kite_nd = self.tools.kite
            _now_nd = _pd_nd.Timestamp.now()

            _need_full_backfill = (
                self._nifty_daily_df is None or len(self._nifty_daily_df) < 80
            )

            if _need_full_backfill:
                _nd_count = len(self._nifty_daily_df) if self._nifty_daily_df is not None else 0
                print(f"  ⚠️ NIFTY daily: only {_nd_count} rows (need ≥80 for EMA-50) — backfilling 120 days...")
                _full_daily = fetch_candles(_kite_nd, 'NIFTY 50', days=120, interval='day')
                if len(_full_daily) > 0:
                    _full_daily['date'] = _pd_nd.to_datetime(_full_daily['date'])
                    if _full_daily['date'].dt.tz is not None:
                        _full_daily['date'] = _full_daily['date'].dt.tz_localize(None)
                    # Merge with existing (if any) to keep oldest data
                    if self._nifty_daily_df is not None and len(self._nifty_daily_df) > 0:
                        _nd_copy = self._nifty_daily_df.copy()
                        if _nd_copy['date'].dt.tz is not None:
                            _nd_copy['date'] = _nd_copy['date'].dt.tz_localize(None)
                        _full_daily = _pd_nd.concat([_nd_copy, _full_daily], ignore_index=True)
                    _full_daily = _full_daily.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                    os.makedirs(os.path.dirname(_nifty_daily_path), exist_ok=True)
                    _full_daily.to_parquet(_nifty_daily_path, index=False)
                    self._nifty_daily_df = _full_daily
                    print(f"  ✅ NIFTY daily backfilled: {len(_full_daily)} days (needed ≥60 for EMA-50)")
                else:
                    print(f"  ⚠️ NIFTY daily backfill: Kite returned 0 rows")
            elif self._nifty_daily_df is not None:
                # Incremental refresh: append missing days
                _nd_last = self._nifty_daily_df['date'].max()
                if hasattr(_nd_last, 'tz') and _nd_last.tz is not None:
                    _nd_last = _nd_last.tz_localize(None)
                _nd_gap = (_now_nd - _nd_last).days
                if _nd_gap > 1:
                    _new_daily = fetch_candles(_kite_nd, 'NIFTY 50', days=min(_nd_gap + 1, 60), interval='day')
                    if len(_new_daily) > 0:
                        _new_daily['date'] = _pd_nd.to_datetime(_new_daily['date'])
                        _nd_copy = self._nifty_daily_df.copy()
                        if _nd_copy['date'].dt.tz is not None:
                            _nd_copy['date'] = _nd_copy['date'].dt.tz_localize(None)
                        if _new_daily['date'].dt.tz is not None:
                            _new_daily['date'] = _new_daily['date'].dt.tz_localize(None)
                        _nd_combined = _pd_nd.concat([_nd_copy, _new_daily], ignore_index=True)
                        _nd_combined = _nd_combined.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
                        _nd_combined.to_parquet(_nifty_daily_path, index=False)
                        self._nifty_daily_df = _nd_combined
                        # print(f"  ✅ NIFTY daily refreshed: {len(_nd_combined)} days")
            if self._nifty_daily_df is not None:
                print(f"  📊 NIFTY daily: {len(self._nifty_daily_df)} rows ({self._nifty_daily_df['date'].min().date()} to {self._nifty_daily_df['date'].max().date()})")
        except Exception as _nd_err:
            print(f"  ⚠️ NIFTY daily backfill: {_nd_err}")
        
        # === AUTO-RENEW DHAN TOKEN (before OI backfill) ===
        try:
            from dhan_token_manager import ensure_token_fresh
            if ensure_token_fresh():
                print("  ✅ Dhan token: fresh")
            else:
                print("  ⚠️ Dhan token: expired or renewal failed — OI features may be degraded")
        except Exception as _tok_e:
            print(f"  ⚠️ Dhan token check: {_tok_e}")

        # === AUTO-REFRESH STALE OI DATA (runs ONCE per calendar day) ===
        try:
            import pandas as _pd_oi_chk
            from datetime import timedelta as _td_oi
            _oi_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'data', 'futures_oi')
            _oi_marker = os.path.join(_oi_dir, f'.oi_backfill_done_{_pd_oi_chk.Timestamp.now().strftime("%Y-%m-%d")}')

            if os.path.exists(_oi_marker):
                # Already ran backfill today — skip entirely
                print(f"  ✅ Futures OI data: backfill already done today (marker exists)")
            else:
                _sample_files = [f for f in os.listdir(_oi_dir) if f.endswith('_futures_oi.parquet')][:3]
                _oi_stale = False

                # Compute last trading day (skip weekends + 1-day holiday tolerance)
                # DhanHQ daily data appears EOD, so today's data isn't available
                # until tonight. Allow up to 2 business days old to handle
                # single-day holidays (e.g., Holi, Republic Day, etc.).
                _today_oi = _pd_oi_chk.Timestamp.now().normalize()
                _last_trading_day = _today_oi
                if _last_trading_day.weekday() == 0:    # Monday → Friday
                    _last_trading_day -= _td_oi(days=3)
                elif _last_trading_day.weekday() == 6:  # Sunday → Friday
                    _last_trading_day -= _td_oi(days=2)
                elif _last_trading_day.weekday() == 5:  # Saturday → Friday
                    _last_trading_day -= _td_oi(days=1)
                # If before 10 AM on a weekday, use previous trading day
                if _pd_oi_chk.Timestamp.now().hour < 10 and _today_oi.weekday() < 5:
                    _last_trading_day -= _td_oi(days=1)
                    if _last_trading_day.weekday() >= 5:
                        _last_trading_day -= _td_oi(days=(_last_trading_day.weekday() - 4))
                # Holiday tolerance: allow 1 extra business day gap
                # (handles single-day holidays like Holi, Diwali, etc.)
                _holiday_tolerance = _last_trading_day - _td_oi(days=1)
                if _holiday_tolerance.weekday() >= 5:
                    _holiday_tolerance -= _td_oi(days=(_holiday_tolerance.weekday() - 4))

                if not _sample_files:
                    _oi_stale = True
                else:
                    for _sf in _sample_files:
                        _sdf = _pd_oi_chk.read_parquet(os.path.join(_oi_dir, _sf))
                        _last = _pd_oi_chk.Timestamp(_sdf['date'].max()).normalize()
                        if _last < _holiday_tolerance:
                            _oi_stale = True
                            break
                if _oi_stale:
                    print(f"  ⚠️ Futures OI data stale (last={_last.date() if '_last' in dir() else '?'}, need≥{_holiday_tolerance.date()}) — refreshing (once)...")
                    from dhan_futures_oi import FuturesOIFetcher
                    _oi_fetcher = FuturesOIFetcher()
                    if _oi_fetcher.ready:
                        _oi_fetcher.backfill_all(months_back=1)
                        # Verify data actually got fresher before writing marker
                        _verify_fresh = False
                        _sample_after = [f for f in os.listdir(_oi_dir) if f.endswith('_futures_oi.parquet')][:3]
                        for _vf in _sample_after:
                            _vdf = _pd_oi_chk.read_parquet(os.path.join(_oi_dir, _vf))
                            _vlast = _pd_oi_chk.Timestamp(_vdf['date'].max()).normalize()
                            if _vlast >= _holiday_tolerance:
                                _verify_fresh = True
                                break
                        if _verify_fresh:
                            with open(_oi_marker, 'w') as _mf:
                                _mf.write(f"done at {_pd_oi_chk.Timestamp.now().isoformat()}\n")
                            # Clean up old markers (keep only today's)
                            for _old in os.listdir(_oi_dir):
                                if _old.startswith('.oi_backfill_done_') and _old != os.path.basename(_oi_marker):
                                    try: os.remove(os.path.join(_oi_dir, _old))
                                    except: pass
                            print(f"  ✅ OI backfill complete — data fresh (last={_vlast.date()}) — marker written")
                        else:
                            print(f"  ⚠️ OI backfill ran but data still stale (last={_vlast.date() if '_vlast' in dir() else '?'}) — NO marker (will retry next restart)")
                    else:
                        print("  ⚠️ DhanHQ not ready — OI refresh skipped (no marker, will retry)")
                else:
                    print(f"  ✅ Futures OI data: fresh (last={_last.date() if '_last' in dir() else '?'}, need≥{_last_trading_day.date()})")
                    # Data is fresh, write marker so we skip check on restart
                    with open(_oi_marker, 'w') as _mf:
                        _mf.write(f"fresh at {_pd_oi_chk.Timestamp.now().isoformat()}\n")
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
                # print("  📊 OI Flow Analyzer: ACTIVE (real-time PCR/IV/MaxPain overlay)")
                pass
            else:
                # print("  📊 OI Flow Analyzer: STANDBY (no chain fetcher)")
                self._oi_analyzer = None
        except Exception as e:
            print(f"  📊 OI Flow Analyzer: DISABLED ({e})")
        
        print("="*60)
    
    # ========== HELPERS ==========
    @staticmethod
    def _dr_tag(regime) -> str:
        """Return 'Down_Flag' or 'UP_Flag' based on GMM regime used."""
        if isinstance(regime, dict):
            regime = regime.get('ml_gmm_regime_used', 'UP')
        if regime == 'BOTH':
            return 'UP+Down'
        return 'Down_Flag' if regime == 'DOWN' else 'UP_Flag'
    
    # ========== ML_OVERRIDE_WGMM GATE CHECK ==========
    def _ml_override_allowed(self, sym_clean: str, ml: dict, dr_score: float,
                             path: str = 'MODEL_TRACKER', p_score: float = 0.0) -> tuple:
        """Check tightened ML_OVERRIDE_WGMM gates. Returns (allowed: bool, reason: str).
        Call BEFORE flipping direction.
        
        DR SCORE INTERPRETATION (correct — anomaly-based, dual-regime):
          UP_Flag=True (high up_score) = hidden DOWN risk (crash) → confirms SELL, opposes BUY
          Down_Flag=True (high down_score) = hidden UP risk (bounce) → confirms BUY, opposes SELL
          gmm_confirms_direction = True means NO anomaly = clean from BOTH models
          → ML_OVERRIDE fires when XGB opposes AND GMM confirms XGB via gmm_confirms
        
        DIRECTIONAL DR SCORE:
          Gate 3 uses the score that CONFIRMS the XGB flip direction, not max(both):
            XGB=UP (flip to BUY)  → use down_score (bounce confirms BUY)
            XGB=DOWN (flip to SELL) → use up_score (crash confirms SELL)
          max(both) would be misleading — the opposing regime's score doesn't help.
        
        Gates:
          1. Concurrent open position limit
          2. ml_move_prob minimum (higher bar than general gate)
          3. Directional dr_score MINIMUM — confirming regime must show signal
          4. XGB directional probability minimum
          5. Smart score floor — blocks weak-conviction overrides
        """
        cfg = self._ml_ovr_cfg
        
        # Resolve directional DR score: pick the regime that CONFIRMS the flip
        # XGB=UP → flip to BUY → down_score (bounce) confirms BUY
        # XGB=DOWN → flip to SELL → up_score (crash) confirms SELL
        _xgb_signal = ml.get('ml_signal', 'UNKNOWN')
        _up_score = ml.get('ml_up_score', dr_score)
        _down_score = ml.get('ml_down_score', dr_score)
        if _xgb_signal == 'UP':
            _confirm_dr = _down_score   # bounce signal confirms BUY
            _confirm_tag = 'down_score'
        elif _xgb_signal == 'DOWN':
            _confirm_dr = _up_score     # crash signal confirms SELL
            _confirm_tag = 'up_score'
        else:
            _confirm_dr = dr_score      # fallback to max(both)
            _confirm_tag = 'dr_score'
        
        # Gate 1: Max concurrent open ML_OVERRIDE_WGMM positions
        _max_concurrent = cfg.get('max_concurrent_open', 3)
        try:
            _active = getattr(self.tools, 'paper_positions', []) or []
            _open_ovr = sum(1 for t in _active if t.get('setup_type') == 'ML_OVERRIDE_WGMM' and not t.get('closed'))
            if _open_ovr >= _max_concurrent:
                return False, f"concurrent open {_open_ovr}/{_max_concurrent}"
        except Exception:
            pass
        
        # Gate 2: ml_move_prob minimum (round to 4dp to avoid float display confusion)
        _min_move = cfg.get('min_move_prob', 0.58)
        _move_prob = round(ml.get('ml_move_prob', ml.get('ml_p_move', 0.0)), 4)
        if _move_prob < _min_move - 0.001:  # >= with tiny epsilon for float safety
            return False, f"move_prob {_move_prob:.4f} < {_min_move}"
        
        # Gate 3: Directional dr_score MINIMUM — only for non-SNIPER paths
        # Uses the CONFIRMING regime score, not max(both).
        # Sniper already enforces BOTH GMM regimes clean — skip there.
        if path != 'SNIPER':
            _min_dr = cfg.get('min_dr_score', 0.15)
            if _confirm_dr < _min_dr:
                return False, f"{_confirm_tag} {_confirm_dr:.3f} < {_min_dr} (need confirming regime signal)"
        
        # Gate 4: XGB directional probability
        _min_dir_prob = cfg.get('min_directional_prob', 0.30)
        if _xgb_signal == 'UP':
            _dir_prob = ml.get('ml_prob_up', 0.0)
        elif _xgb_signal == 'DOWN':
            _dir_prob = ml.get('ml_prob_down', 0.0)
        else:
            _dir_prob = 0.0
        if _dir_prob < _min_dir_prob - 0.001:  # >= with epsilon for float safety
            return False, f"dir_prob({_xgb_signal}) {_dir_prob:.2f} < {_min_dir_prob}"
        
        # Gate 5: Smart score floor — estimate smart_score from available data
        # Uses confirming dr_score for safety (higher confirm = more edge = higher safety)
        _min_smart = cfg.get('min_smart_score', 58)
        if _min_smart > 0 and p_score > 0:
            _ml_conf = ml.get('ml_confidence', 0.0)
            _est_conviction = _move_prob * max(_ml_conf, 0.01) * 40.0
            _est_safety = _confirm_dr * 20.0 + 5.0   # higher confirming anomaly = more edge
            _est_technical = min(p_score, 100) * 0.20
            _est_move_bonus = _move_prob * 15.0
            _est_smart = _est_conviction + _est_safety + _est_technical + _est_move_bonus
            if _est_smart < _min_smart:
                return False, f"smart_score {_est_smart:.1f} < {_min_smart} floor"
        
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
            if hasattr(decision, 'recommended_direction') and decision.recommended_direction not in ('HOLD', None, ''):
                direction = decision.recommended_direction
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
            # FT measures candles since ORB breakout — only relevant for ORB-based entries.
            # ELITE auto-fire comes from scan_and_trade (score-based), not from watcher
            # ticker triggers. The FT gate still applies here because ELITE entries
            # are fundamentally about ORB/technical breakouts, not live momentum events.
            if isinstance(data, dict):
                ft_candles = data.get('follow_through_candles', 0)
                adx_val = data.get('adx', 20)
                oi_signal = data.get('oi_signal', 'NEUTRAL')
                
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
            
            # === ML CONFIDENCE FLOOR (prevent coin-flip ML from auto-firing) ===
            # VBL had ml_confidence=0.48, ml_move_prob=0.477 → coin flip, lost -₹9,915
            _min_ml_conf = cfg.get('min_ml_confidence', 0.55)
            _min_ml_move = cfg.get('min_ml_move_prob', 0.52)
            try:
                _cached_ml = cycle_decisions.get(sym, {}).get('ml_prediction', {})
                _ml_conf = _cached_ml.get('ml_confidence', 0)
                _ml_move = _cached_ml.get('ml_move_prob', 0)
                if _ml_conf > 0 and _ml_conf < _min_ml_conf:
                    self._log_decision(cycle_time, sym, score, 'ELITE_LOW_ML_CONF',
                                      reason=f'ML confidence {_ml_conf:.3f} < {_min_ml_conf} — too weak for auto-fire',
                                      direction=direction)
                    continue
                if _ml_move > 0 and _ml_move < _min_ml_move:
                    self._log_decision(cycle_time, sym, score, 'ELITE_LOW_ML_MOVE',
                                      reason=f'ML move_prob {_ml_move:.3f} < {_min_ml_move} — directional signal too weak',
                                      direction=direction)
                    continue
            except Exception:
                pass  # ML data unavailable — proceed with auto-fire
            
            # === MOVE EXHAUSTION GATE (prevent late entries after big moves) ===
            # VBL entered after 3.11% drop → only 0.34% remaining, IV crushed premium
            _max_move = cfg.get('max_existing_move_pct', 2.5)
            _min_accel = cfg.get('exhaustion_min_acceleration', 0.3)
            if isinstance(data, dict):
                _chg_pct = abs(data.get('change_pct', 0))
                if _chg_pct > _max_move:
                    # Allow override if recent acceleration shows fresh momentum
                    _last_candle_chg = abs(data.get('last_candle_change_pct', 0))
                    if _last_candle_chg >= _min_accel:
                        print(f"   ⚡ ELITE MOVE EXHAUSTION OVERRIDE: {sym} moved {_chg_pct:.1f}% but last candle {_last_candle_chg:.2f}% — fresh momentum")
                    else:
                        self._log_decision(cycle_time, sym, score, 'ELITE_MOVE_EXHAUSTED',
                                          reason=f'Stock already moved {_chg_pct:.1f}% > {_max_move}% — late entry risk, last candle {_last_candle_chg:.2f}%',
                                          direction=direction)
                        continue
            
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
                        'up_flag': _elite_ml.get('ml_up_flag', False),
                        'down_flag': _elite_ml.get('ml_down_flag', False),
                        'up_score': _elite_ml.get('ml_up_score', 0),
                        'down_score': _elite_ml.get('ml_down_score', 0),
                        'down_risk_bucket': _elite_ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': _elite_ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': _elite_ml.get('ml_gmm_regime_used', 'BOTH'),
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
                    setup_type="ELITE",
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
    
    # ========== WATCHER LOG — writes to bot_debug.log + watcher_debug.log ==========
    def _wlog(self, msg: str):
        """Write timestamped debug message to bot_debug.log AND watcher_debug.log."""
        _ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        _line = f"[{_ts}] WATCHER: {msg}"
        print(_line)
        for _fname in ('bot_debug.log', 'watcher_debug.log'):
            try:
                _path = os.path.join(os.path.dirname(__file__), _fname)
                with open(_path, 'a', encoding='utf-8') as _f:
                    _f.write(_line + '\n')
            except Exception:
                pass

    # ========== BREAKOUT WATCHER QUEUE DRAIN ==========
    def _process_breakout_triggers(self):
        """
        Drain the breakout watcher queue and feed triggered stocks through
        the FULL scan_and_trade pipeline.  The watcher is DETECTION ONLY —
        all safety gates (scoring, ML, OI, sector, DR, setup validation,
        follow-through, ADX, OI-conflict, ML-flat veto) run identically to
        a normal 5-min scan cycle.  No shortcuts.  No bypasses.
        
        Called from the main loop every ~1 second.
        """
        if not BREAKOUT_WATCHER.get('enabled', False):
            return
        
        ticker = getattr(self.tools, 'ticker', None)
        if not ticker or not hasattr(ticker, 'breakout_watcher') or not ticker.breakout_watcher:
            return
        
        # Cooldown check BEFORE draining — keeps triggers in queue until we can process
        import time as _wt
        _cooldown_elapsed = _wt.time() - self._last_watcher_scan_time
        if _cooldown_elapsed < 5:   # 5-sec cooldown (was 20s) — fast reaction to breakouts
            return
        
        watcher = ticker.breakout_watcher
        triggers = watcher.drain_queue()
        if not triggers:
            return
        
        # === WATCHER ACTIVITY BANNER ===
        _sym_list = ', '.join(t['symbol'].replace('NSE:', '') for t in triggers[:5])
        self._wlog(f"🔔 TRIGGER — {len(triggers)} breakout(s) detected: {_sym_list}")
        self._wlog(f"   ⚡ Processing immediately (cooldown={_cooldown_elapsed:.0f}s)...")
        
        self._watcher_drain_count += 1
        self._watcher_total_drained += len(triggers)
        
        # --- Position exhaustion pre-check ---
        _breadth = getattr(self, '_last_market_breadth', 'MIXED')
        active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        if _breadth == 'MIXED':
            _max_pos = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)
        elif _breadth in ('BULLISH', 'BEARISH'):
            _max_pos = HARD_RULES.get('MAX_POSITIONS_TRENDING', 12)
        else:
            _max_pos = HARD_RULES['MAX_POSITIONS']
        _pos_slots_free = max(0, _max_pos - len(active_positions))
        
        if _pos_slots_free == 0:
            self._watcher_total_pos_exhausted += 1
            _syms_brief = ', '.join(t['symbol'].replace('NSE:', '') for t in triggers[:5])
            self._wlog(f"❌ BLOCKED: Position slots full ({len(active_positions)}/{_max_pos} {_breadth}) — {_syms_brief}")
            return
        
        # Filter: skip already-held, already-watcher-fired
        # NOTE: We do NOT filter on _auto_fired_this_session — the watcher is an
        # INDEPENDENT entry path.  A stock the main scan auto-fired should still be
        # eligible for a watcher breakout trade if it's a new trigger event.
        actionable = []
        _skip_reasons = []
        for t in triggers:
            sym = t['symbol']
            if self.tools.is_symbol_in_active_trades(sym):
                _skip_reasons.append(f"{sym.replace('NSE:', '')}=held")
                continue
            if sym in self._watcher_fired_this_session:
                _skip_reasons.append(f"{sym.replace('NSE:', '')}=watcher-done")
                continue
            if not self.risk_governor.is_trading_allowed():
                _skip_reasons.append(f"{sym.replace('NSE:', '')}=risk-gov")
                break
            actionable.append(t)
        
        # Log EVERY drain — critical for debugging
        _trigger_summary = ', '.join(
            f"{t['symbol'].replace('NSE:', '')}({t.get('trigger_type', '?')[:8]} {t.get('move_pct', 0):+.1f}%)"
            for t in triggers
        )
        
        if not actionable:
            self._wlog(f"⚠️ ALL {len(triggers)} triggers filtered: {', '.join(_skip_reasons)} | triggers=[{_trigger_summary}]")
            return
        
        self._watcher_total_actionable += len(actionable)
        
        # Dynamic batch: use configurable max, capped by free position slots
        _max_batch = BREAKOUT_WATCHER.get('max_triggers_per_batch', 6)
        _batch = actionable[:min(_pos_slots_free, _max_batch)]
        self._watcher_total_pipeline_sent += len(_batch)
        
        _actionable_summary = ', '.join(
            f"{t['symbol'].replace('NSE:', '')}({t.get('trigger_type', '?')[:8]} {t.get('move_pct', 0):+.1f}%)"
            for t in _batch
        )
        _filter_note = f" | filtered=[{', '.join(_skip_reasons)}]" if _skip_reasons else ''
        self._wlog(f"🚀 PIPELINE: Sending {len(_batch)} stocks → [{_actionable_summary}] (slots={_pos_slots_free}){_filter_note}")
        
        # Run the FULL pipeline for these symbols (identical gates as scan_and_trade)
        _pipe_start = _wt.time()
        self._watcher_focused_scan(_batch)
        _pipe_dur = _wt.time() - _pipe_start
        self._wlog(f"✅ PIPELINE DONE in {_pipe_dur:.1f}s | placed={self._watcher_total_placed} total today")
        self._last_watcher_scan_time = _wt.time()
    
    # ========== WATCHER FOCUSED SCAN (FULL PIPELINE) ==========
    def _watcher_focused_scan(self, triggers: list):
        """
        Run the IDENTICAL pipeline as scan_and_trade for watcher-detected stocks.
        
        ALL gates present in scan_and_trade are replicated here:
          ✅ Market data fetch (candles, indicators)
          ✅ Intraday scoring (IntradaySignal + scorer)
          ✅ Market breadth (from last scan, not hardcoded MIXED)
          ✅ ML predictions (XGB + GMM via get_titan_signals)
          ✅ OI flow overlay (adjusts ML with live options chain)
          ✅ OI cross-validation (penalises direction conflict)
          ✅ Sector index cross-validation (penalises against-sector trades)
          ✅ Down-risk soft scoring (±5 nudge from GMM DR model)
          ✅ Setup validation (ORB / VWAP / EMA / RSI)
          ✅ Follow-through candle gate (no stale breakouts)
          ✅ ADX trend strength gate (≥25)
          ✅ OI conflict veto (institutions vs direction)
          ✅ ML flat veto (high P(flat) blocks auto-fire)
          ✅ XGB direction conflict veto (strong opposing signal blocks)
          ✅ XGB move probability floor (P(move) >= 0.55)
          ✅ GMM down-risk veto (extreme anomaly opposes direction → block)
          ✅ Position limit (regime-aware)
        
        The ONLY difference from a 5-min scan:
          - Universe is limited to watcher-triggered symbols (speed)
          - Score threshold is min_score (default 66) instead of 78 (ELITE)
          - GPT analysis is skipped (watcher trigger + full gates = sufficient)
        
        No shortcuts.  No bypasses.  Full pipeline parity.
        """
        _ts = datetime.now().strftime('%H:%M:%S')
        _trigger_map = {t['symbol']: t for t in triggers}
        _symbols = [t['symbol'] for t in triggers]
        
        try:
            # ================================================================
            # 0) F&O ELIGIBILITY — skip non-F&O stocks BEFORE the expensive pipeline
            # ================================================================
            _fo_set = getattr(self, '_fno_universe', None)
            if not _fo_set:
                try:
                    _nfo_inst = self.tools.get_nfo_instruments()
                    _fo_set = {f"NSE:{i['name']}" for i in _nfo_inst if i.get('segment') == 'NFO-OPT' and i.get('instrument_type') == 'CE'} if _nfo_inst else set()
                    if not _fo_set:
                        _fo_set = {f"NSE:{s}" for s in self.market_scanner.fo_stocks} if hasattr(self.market_scanner, 'fo_stocks') else set()
                    self._fno_universe = _fo_set
                except Exception:
                    _fo_set = set()
            
            if _fo_set:
                _before = len(_symbols)
                _fno_triggers = [t for t in triggers if t['symbol'] in _fo_set]
                _non_fno = [t['symbol'] for t in triggers if t['symbol'] not in _fo_set]
                if _non_fno:
                    self._wlog(f"PIPELINE: skipping non-F&O: {_non_fno}")
                if not _fno_triggers:
                    self._wlog(f"PIPELINE: all {_before} triggers are non-F&O — skipping pipeline")
                    return
                triggers = _fno_triggers
                _trigger_map = {t['symbol']: t for t in triggers}
                _symbols = [t['symbol'] for t in triggers]

            # ================================================================
            # 1) MARKET DATA — fresh candles + indicators for triggered stocks
            #    force_fresh=True bypasses 10-min indicator cache so RSI/VWAP/ADX
            #    are computed from live candles (watcher only processes 1-3 stocks)
            # ================================================================
            market_data = self.tools.get_market_data(_symbols, force_fresh=True)
            _sorted_data = [(s, d) for s, d in market_data.items()
                            if isinstance(d, dict) and 'ltp' in d]
            if not _sorted_data:
                self._wlog(f"PIPELINE: no market data for {[s for s in _symbols]} — skipping")
                return
            
            # ================================================================
            # 2) SCORING — identical IntradaySignal + scorer as scan_and_trade
            # ================================================================
            from options_trader import get_intraday_scorer, IntradaySignal
            _scorer = get_intraday_scorer()
            
            # Use cached market breadth from last full scan (not hardcoded)
            _breadth = getattr(self, '_last_market_breadth', 'MIXED')
            
            _pre_scores = {}
            _cycle_decisions = {}
            
            for _sym, _d in _sorted_data:
                _d['market_breadth'] = _breadth
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
                    _dec = _scorer.score_intraday_signal(_sig, market_data=_d, caller_direction=None, source='watcher')
                    _pre_scores[_sym] = _dec.confidence_score
                    _cycle_decisions[_sym] = {
                        'decision': _dec,
                        'direction': None,
                        'score': _dec.confidence_score,
                        'raw_score': _dec.confidence_score,
                    }
                except Exception:
                    pass
            
            if not _pre_scores:
                self._wlog(f"PIPELINE: scoring failed for all symbols — skipping")
                return
            
            # ================================================================
            # 3) ML PREDICTIONS — same get_titan_signals as scan_and_trade
            # ================================================================
            _ml_results = {}
            try:
                if self._ml_predictor:
                    import pandas as _pd_ml
                    _candle_cache = getattr(self.tools, '_candle_cache', {})
                    _daily_cache = getattr(self.tools, '_daily_cache', {})
                    _futures_oi_cache = getattr(self, '_futures_oi_data', {}) or {}
                    _sector_5min_cache = getattr(self, '_sector_5min_cache', {})
                    _sector_daily_cache = getattr(self, '_sector_daily_cache', {})
                    _nifty_5min = getattr(self, '_nifty_5min_df', None)
                    _nifty_daily = getattr(self, '_nifty_daily_df', None)
                    _hist_5min_cache = getattr(self, '_hist_5min_cache', {})
                    
                    try:
                        from ml_models.feature_engineering import get_sector_for_symbol as _get_sector
                    except ImportError:
                        _get_sector = lambda s: ''
                    
                    for _sym in list(_pre_scores.keys()):
                        try:
                            _sym_clean = _sym.replace('NSE:', '')
                            _live_intraday = _candle_cache.get(_sym)
                            _daily_df = _daily_cache.get(_sym)
                            _hist_5min = _hist_5min_cache.get(_sym_clean)
                            
                            # Build best possible 5-min candle series (same logic as scan_and_trade)
                            _ml_candles = None
                            if _hist_5min is not None:
                                if _live_intraday is not None and len(_live_intraday) >= 2:
                                    try:
                                        _live_copy = _live_intraday.copy()
                                        _live_copy['date'] = _pd_ml.to_datetime(_live_copy['date'])
                                        if _live_copy['date'].dt.tz is not None:
                                            _live_copy['date'] = _live_copy['date'].dt.tz_localize(None)
                                        _hist_copy = _hist_5min.copy()
                                        if _hist_copy['date'].dt.tz is not None:
                                            _hist_copy['date'] = _hist_copy['date'].dt.tz_localize(None)
                                        _gap_days = (_live_copy['date'].min() - _hist_copy['date'].max()).days
                                        if _gap_days > 3:
                                            _ml_candles = _hist_5min.tail(500)
                                        else:
                                            _hist_tail = _hist_copy.tail(500)
                                            _live_start = _live_copy['date'].min()
                                            _hist_tail = _hist_tail[_hist_tail['date'] < _live_start]
                                            _common_cols = [c for c in ['date','open','high','low','close','volume']
                                                           if c in _hist_tail.columns and c in _live_copy.columns]
                                            _ml_candles = _pd_ml.concat(
                                                [_hist_tail[_common_cols], _live_copy[_common_cols]],
                                                ignore_index=True)
                                    except Exception:
                                        _ml_candles = _hist_5min.tail(500)
                                else:
                                    _ml_candles = _hist_5min.tail(500)
                            elif _live_intraday is not None and len(_live_intraday) >= 50:
                                _ml_candles = _live_intraday
                            
                            if _ml_candles is None or len(_ml_candles) < 50:
                                continue
                            
                            _fut_oi = _futures_oi_cache.get(_sym_clean)
                            _sec_name = _get_sector(_sym_clean)
                            _sec_5m = _sector_5min_cache.get(_sec_name) if _sec_name else None
                            _sec_dl = _sector_daily_cache.get(_sec_name) if _sec_name else None
                            
                            _pred = self._ml_predictor.get_titan_signals(
                                _ml_candles,
                                daily_df=_daily_df,
                                futures_oi_df=_fut_oi,
                                nifty_5min_df=_nifty_5min,
                                nifty_daily_df=_nifty_daily,
                                sector_5min_df=_sec_5m,
                                sector_daily_df=_sec_dl
                            )
                            if _pred:
                                if _pred.get('ml_score_boost', 0) != 0:
                                    _pre_scores[_sym] += _pred['ml_score_boost']
                                if _pred.get('ml_signal') != 'UNKNOWN':
                                    _ml_results[_sym] = _pred
                                    if _sym in _cycle_decisions:
                                        _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                                        _cycle_decisions[_sym]['ml_prediction'] = _pred
                        except Exception:
                            pass
            except Exception:
                pass  # ML failed — continue without it (fail-safe)
            
            # ================================================================
            # 4) OI FLOW OVERLAY — adjusts ML with live options chain data
            # ================================================================
            _oi_results = {}
            try:
                if self._oi_analyzer:
                    for _sym in list(_pre_scores.keys()):
                        try:
                            _oi_data = self._oi_analyzer.analyze(_sym)
                            if _oi_data:
                                _oi_results[_sym] = _oi_data
                                if self._ml_predictor and _sym in _ml_results:
                                    self._ml_predictor.apply_oi_overlay(_ml_results[_sym], _oi_data)
                                    if _sym in _cycle_decisions:
                                        _cycle_decisions[_sym]['ml_prediction'] = _ml_results[_sym]
                        except Exception:
                            pass
            except Exception:
                pass
            
            # ================================================================
            # 5) OI CROSS-VALIDATION — penalise direction conflicts
            # ================================================================
            for _oi_sym, _oi_data in _oi_results.items():
                try:
                    _oi_bias = _oi_data.get('flow_bias', 'NEUTRAL')
                    _oi_conf = _oi_data.get('flow_confidence', 0.0)
                    if _oi_bias == 'NEUTRAL' or _oi_conf < 0.55:
                        continue
                    _cd = _cycle_decisions.get(_oi_sym)
                    if not _cd or not _cd.get('decision'):
                        continue
                    _scored_dir = _cd['decision'].recommended_direction
                    if _scored_dir == 'HOLD':
                        continue
                    _oi_dir = 'BUY' if _oi_bias == 'BULLISH' else 'SELL'
                    if _scored_dir != _oi_dir:
                        _oi_penalty = -5 if _oi_conf >= 0.70 else -3
                        _pre_scores[_oi_sym] += _oi_penalty
                        if _oi_sym in _cycle_decisions:
                            _cycle_decisions[_oi_sym]['score'] = _pre_scores[_oi_sym]
                        self._wlog(f"OI cross-val: {_oi_sym.replace('NSE:', '')} "
                              f"penalised {_oi_penalty} (OI {_oi_bias} vs scored {_scored_dir})")
                except Exception:
                    pass
            
            # ================================================================
            # 6) SECTOR INDEX CROSS-VALIDATION — penalise against-sector trades
            # ================================================================
            _sec_changes = getattr(self, '_sector_index_changes_cache', {})
            _stock_to_sector = getattr(self, '_stock_to_sector', {})
            
            for _sym, _score in list(_pre_scores.items()):
                try:
                    _stock_name = _sym.replace('NSE:', '')
                    _sec_match = _stock_to_sector.get(_stock_name)
                    if not _sec_match:
                        continue
                    _sec_name, _sec_index = _sec_match
                    _sec_chg = _sec_changes.get(_sec_index)
                    if _sec_chg is None:
                        continue
                    _cd = _cycle_decisions.get(_sym)
                    if not _cd or not _cd.get('decision'):
                        continue
                    _scored_dir = _cd['decision'].recommended_direction
                    if _scored_dir == 'HOLD':
                        continue
                    _stk_data = market_data.get(_sym, {})
                    _stk_chg = _stk_data.get('change_pct', 0) if isinstance(_stk_data, dict) else 0
                    
                    if _sec_chg <= -1.0 and _scored_dir == 'BUY':
                        if _stk_chg > 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                            continue
                        _sec_penalty = -5 if _sec_chg <= -2.0 else -3
                        _pre_scores[_sym] += _sec_penalty
                        if _sym in _cycle_decisions:
                            _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                        self._wlog(f"Sector cross-val: {_stock_name} penalised {_sec_penalty} "
                              f"(sector {_sec_chg:+.1f}% vs BUY)")
                    elif _sec_chg >= 1.0 and _scored_dir == 'SELL':
                        if _stk_chg < 0 and abs(_stk_chg) >= abs(_sec_chg) * 2:
                            continue
                        _sec_penalty = -5 if _sec_chg >= 2.0 else -3
                        _pre_scores[_sym] += _sec_penalty
                        if _sym in _cycle_decisions:
                            _cycle_decisions[_sym]['score'] = _pre_scores[_sym]
                        self._wlog(f"Sector cross-val: {_stock_name} penalised {_sec_penalty} "
                              f"(sector {_sec_chg:+.1f}% vs SELL)")
                except Exception:
                    pass
            
            # ================================================================
            # 7) DOWN-RISK SOFT SCORING — ±5 nudge from GMM DR model
            # ================================================================
            try:
                self._apply_down_risk_soft_scores(_ml_results, _pre_scores)
            except Exception:
                pass
            
            # ================================================================
            # 8) GATE-CHECKED EXECUTION — same gates as ELITE auto-fire
            #    Setup validation, FT, ADX, OI conflict, ML flat veto
            #    Score threshold: min_score (default 66) from BREAKOUT_WATCHER config
            # ================================================================
            _min_score = BREAKOUT_WATCHER.get('min_score', 66)
            _max_per_scan = BREAKOUT_WATCHER.get('max_trades_per_scan', 2)
            _fired_count = 0
            
            # ── VIX-based score penalty: elevated VIX = expensive options ──
            _vix = getattr(self, '_current_vix', 14.0)
            _vix_penalty_above = BREAKOUT_WATCHER.get('vix_penalty_above', 18.0)
            _vix_penalty_per_pt = BREAKOUT_WATCHER.get('vix_penalty_per_point', 3)
            _vix_hard_block = BREAKOUT_WATCHER.get('vix_hard_block_above', 28.0)
            _vix_penalty = 0
            if _vix > _vix_hard_block:
                self._wlog(f"🚫 VIX BLOCK: India VIX={_vix:.1f} > {_vix_hard_block} — blocking ALL watcher entries")
                return
            if _vix > _vix_penalty_above:
                _vix_penalty = round((_vix - _vix_penalty_above) * _vix_penalty_per_pt)
                self._wlog(f"⚠️ VIX PENALTY: India VIX={_vix:.1f} → -{_vix_penalty} score penalty on all candidates")
            
            # ════════════════════════════════════════════════════════════════
            # TWO-PASS PIPELINE: Run all gates first, collect candidates,
            # then RANK by P(move) and place BEST trades only.
            # ════════════════════════════════════════════════════════════════
            _candidates = []  # [{sym, direction, score, ml_move_prob, ...}]
            
            for _sym, _score in sorted(_pre_scores.items(), key=lambda x: x[1], reverse=True):
                _trig = _trigger_map.get(_sym, {})
                _trigger_type = _trig.get('trigger_type', '?')
                _move_pct = _trig.get('move_pct', 0)
                _data = market_data.get(_sym, {})
                if not isinstance(_data, dict):
                    continue
                
                _cached = _cycle_decisions.get(_sym, {})
                _decision = _cached.get('decision')
                if not _decision:
                    continue
                
                # Score from decision (includes ML boost, OI penalty, sector penalty, DR nudge)
                _final_score = _pre_scores.get(_sym, 0)
                # Apply VIX penalty
                _final_score -= _vix_penalty
                
                # --- Direction from scorer (authoritative, not from trigger) ---
                direction = None
                if hasattr(_decision, 'recommended_direction') and _decision.recommended_direction not in ('HOLD', None, ''):
                    direction = _decision.recommended_direction
                if not direction:
                    # Fallback: infer from trigger
                    if _trigger_type in ('PRICE_SPIKE_UP', 'NEW_DAY_HIGH'):
                        direction = 'BUY'
                    elif _trigger_type in ('PRICE_SPIKE_DOWN', 'NEW_DAY_LOW'):
                        direction = 'SELL'
                    else:
                        direction = 'BUY' if _move_pct > 0 else 'SELL'
                
                # Score audit from scorer (shows component breakdown)
                _score_audit = getattr(_scorer, '_last_score_audit', '')
                _stock_name = _sym.replace('NSE:', '')
                _ml_pred = _ml_results.get(_sym, {})
                _ml_signal = _ml_pred.get('ml_signal', 'N/A')
                _ml_conf = _ml_pred.get('ml_confidence', 0)
                _ml_move_prob = _ml_pred.get('ml_move_prob', 0)
                _dr_score = _ml_pred.get('ml_down_risk_score', 0)
                _up_score = _ml_pred.get('ml_up_score', 0)
                _down_score = _ml_pred.get('ml_down_score', 0)
                _prob_up = _ml_pred.get('ml_prob_up', 0)
                _prob_down = _ml_pred.get('ml_prob_down', 0)
                self._wlog(f"GATE CHECK: {_stock_name} | score={_final_score:.0f} dir={direction} "
                           f"trigger={_trigger_type}({_move_pct:+.1f}%) | "
                           f"XGB={_ml_signal} conf={_ml_conf:.2f} P(move)={_ml_move_prob:.2f} P(up)={_prob_up:.2f} P(dn)={_prob_down:.2f} | "
                           f"GMM DR={_dr_score:.3f} up={_up_score:.3f} dn={_down_score:.3f} | "
                           f"ORB={_data.get('orb_signal', '?')} VWAP={_data.get('price_vs_vwap', '?')} "
                           f"VOL={_data.get('volume_regime', '?')} ADX={_data.get('adx', 0):.0f} "
                           f"FT={_data.get('follow_through_candles', 0)} RSI={_data.get('rsi_14', 50):.0f}")
                if _score_audit:
                    self._wlog(f"  AUDIT: {_score_audit}")
                
                # --- GATE A: Score threshold ---
                if _final_score < _min_score:
                    self._wlog(f"  BLOCKED(A-SCORE): {_stock_name} score={_final_score:.0f} < {_min_score}")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_LOW_SCORE',
                                      reason=f'Breakout {_trigger_type} but score {_final_score:.0f} < {_min_score}',
                                      direction=direction)
                    continue
                
                # --- GATE B: Chop zone filter ---
                if _data.get('chop_zone', False):
                    self._wlog(f"  BLOCKED(B-CHOP): {_stock_name} in chop zone: {_data.get('chop_reason', '')}")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_CHOP_ZONE',
                                      reason=f'Breakout {_trigger_type} in chop zone: {_data.get("chop_reason", "")}',
                                      direction=direction)
                    continue
                
                # --- GATE C: Setup validation (relaxed for watcher) ---
                # The watcher already confirmed a real-time trigger (volume surge,
                # new day extreme, or price spike). These ARE setups — the ticker
                # detected them from raw ticks, so we grant implicit setup credit
                # for strong trigger types or VWAP alignment, not just classic ORB.
                orb = _data.get('orb_signal', 'INSIDE_ORB')
                vwap = _data.get('price_vs_vwap', 'AT_VWAP')
                vol = _data.get('volume_regime', 'NORMAL')
                ema = _data.get('ema_regime', 'NORMAL')
                _rsi = _data.get('rsi_14', 50)
                
                # Classic setups (same as ELITE)
                _classic_setup = (
                    orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') or
                    (vwap in ('ABOVE_VWAP', 'BELOW_VWAP') and vol in ('HIGH', 'EXPLOSIVE')) or
                    ema == 'COMPRESSED' or
                    _rsi < 30 or _rsi > 70
                )
                
                # Watcher-implicit setups: the ticker's trigger IS the setup evidence
                # VOLUME_SURGE + VWAP alignment = institutional activity + trend
                # NEW_DAY_LOW/HIGH = structural break of day's range
                # PRICE_SPIKE = momentum event
                # SLOW_GRIND = persistent multi-minute move detected
                _watcher_setup = (
                    _trigger_type in ('NEW_DAY_LOW', 'NEW_DAY_HIGH', 'PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN', 'SLOW_GRIND_UP', 'SLOW_GRIND_DOWN') or
                    (_trigger_type == 'VOLUME_SURGE' and vwap in ('ABOVE_VWAP', 'BELOW_VWAP')) or
                    (_trigger_type == 'VOLUME_SURGE' and _data.get('adx', 0) >= 30)
                )
                
                has_setup = _classic_setup or _watcher_setup
                if not has_setup:
                    self._wlog(f"  BLOCKED(C-SETUP): {_stock_name} no setup | ORB={orb} VWAP={vwap} VOL={vol} EMA={ema} RSI={_rsi:.0f} trigger={_trigger_type}")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_NO_SETUP',
                                      reason=f'Breakout {_trigger_type} but no setup (classic or watcher-implicit)',
                                      direction=direction)
                    continue
                
                # Log which setup pathway qualified
                _setup_path = 'classic' if _classic_setup else f'watcher-implicit({_trigger_type})'
                self._wlog(f"  PASSED(C-SETUP): {_stock_name} via {_setup_path}")
                
                # --- GATE D: Follow-through candle gate ---
                # FT measures candles since ORB breakout. This is ONLY relevant
                # when the trade thesis is ORB-based. For VOLUME_SURGE, SLOW_GRIND,
                # NEW_DAY_HIGH triggers, the ORB age is irrelevant — the trigger
                # is fresh momentum, not an old breakout.
                # [FIX Mar 6] UNITDSPR scored 68 with EXPLOSIVE vol, triggered by
                # VOLUME_SURGE, but was blocked because ORB breakout was from morning.
                # FT from the morning ORB has nothing to do with an afternoon volume surge.
                ft_candles = _data.get('follow_through_candles', 0)
                orb_hold = _data.get('orb_hold_candles', 0)
                _is_orb_trigger = orb in ('BREAKOUT_UP', 'BREAKOUT_DOWN') and _trigger_type not in (
                    'VOLUME_SURGE', 'SLOW_GRIND_UP', 'SLOW_GRIND_DOWN',
                    'NEW_DAY_HIGH', 'NEW_DAY_LOW', 'PRICE_SPIKE_UP', 'PRICE_SPIKE_DOWN'
                )
                if ft_candles == 0 and orb_hold > 2 and _is_orb_trigger:
                    self._wlog(f"  BLOCKED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} — stale ORB breakout")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_NO_FOLLOWTHROUGH',
                                      reason=f'FT=0, ORB hold={orb_hold} candles — stale ORB breakout',
                                      direction=direction)
                    continue
                if ft_candles == 0 and orb_hold > 2 and not _is_orb_trigger:
                    self._wlog(f"  PASSED(D-FT): {_stock_name} FT=0 ORB_hold={orb_hold} but trigger={_trigger_type} — FT gate N/A for non-ORB triggers")
                
                # --- GATE E: ADX trend strength gate (same as ELITE: ≥25) ---
                adx_val = _data.get('adx', 20)
                if adx_val < 25:
                    self._wlog(f"  BLOCKED(E-ADX): {_stock_name} ADX={adx_val:.0f} < 25")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_WEAK_ADX',
                                      reason=f'ADX={adx_val:.0f} < 25',
                                      direction=direction)
                    continue
                
                # --- GATE F: OI conflict veto (same as ELITE) ---
                oi_signal = _data.get('oi_signal', 'NEUTRAL')
                if direction == 'BUY' and oi_signal == 'SHORT_BUILDUP':
                    self._wlog(f"  BLOCKED(F-OI): {_stock_name} BUY vs SHORT_BUILDUP")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_OI_CONFLICT',
                                      reason=f'BUY direction but OI={oi_signal}',
                                      direction=direction)
                    continue
                if direction == 'SELL' and oi_signal == 'LONG_BUILDUP':
                    self._wlog(f"  BLOCKED(F-OI): {_stock_name} SELL vs LONG_BUILDUP")
                    self._watcher_total_gate_blocked += 1
                    self._log_decision(_ts, _sym, _final_score, 'WATCHER_OI_CONFLICT',
                                      reason=f'SELL direction but OI={oi_signal}',
                                      direction=direction)
                    continue
                
                # --- GATE G: ML flat veto (same as ELITE) ---
                try:
                    _cached_ml = _cycle_decisions.get(_sym, {}).get('ml_prediction', {})
                    if _cached_ml.get('ml_elite_ok') is False:
                        _ml_flat_p = _cached_ml.get('ml_prob_flat', 0)
                        self._wlog(f"  BLOCKED(G-ML): {_stock_name} P(flat)={_ml_flat_p:.0%}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_ML_FLAT',
                                          reason=f'ML FLAT ({_ml_flat_p:.0%} flat prob)',
                                          direction=direction)
                        continue
                except Exception:
                    pass
                
                # --- GATE G2: XGB Direction Conflict (watcher-specific) ---
                # If XGB strongly predicts the OPPOSITE direction, block.
                # Relaxed vs main scan (0.55→0.50 move_prob, needs strong opposing prob).
                try:
                    _xgb_ml = _ml_results.get(_sym, {})
                    _xgb_signal = _xgb_ml.get('ml_signal', 'UNKNOWN')
                    _xgb_move_prob = _xgb_ml.get('ml_move_prob', 0)
                    _xgb_prob_up = _xgb_ml.get('ml_prob_up', 0)
                    _xgb_prob_down = _xgb_ml.get('ml_prob_down', 0)
                    _xgb_conf = _xgb_ml.get('ml_confidence', 0)
                    
                    # Hard block: XGB says opposite with strong conviction
                    _xgb_opposes = False
                    if direction == 'BUY' and _xgb_signal == 'DOWN' and _xgb_prob_down >= 0.40 and _xgb_conf >= 0.50:
                        _xgb_opposes = True
                    elif direction == 'SELL' and _xgb_signal == 'UP' and _xgb_prob_up >= 0.40 and _xgb_conf >= 0.50:
                        _xgb_opposes = True
                    
                    if _xgb_opposes:
                        self._wlog(f"  BLOCKED(G2-XGB): {_stock_name} XGB={_xgb_signal} opposes {direction} "
                                   f"(P_up={_xgb_prob_up:.2f} P_down={_xgb_prob_down:.2f} conf={_xgb_conf:.2f})")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_XGB_CONFLICT',
                                          reason=f'XGB {_xgb_signal} opposes {direction} '
                                                 f'(P_up={_xgb_prob_up:.2f} P_down={_xgb_prob_down:.2f} conf={_xgb_conf:.2f})',
                                          direction=direction)
                        continue
                except Exception:
                    pass
                
                # --- GATE G3: XGB Move Probability Floor (watcher-specific) ---
                # XGB must show minimum conviction that a move (up OR down) will happen.
                # Watcher threshold: 0.55 (tightened from 0.40 — Mar-5 data shows
                # all winners had P(move)>=0.568, all IV-crush losers had <=0.52).
                _G3_MIN_MOVE_PROB = 0.55
                try:
                    _xgb_ml = _ml_results.get(_sym, {})
                    _xgb_move_prob = _xgb_ml.get('ml_move_prob', 0)
                    # Only gate if ML data is available (don't block when ML fails)
                    if _xgb_ml and _xgb_move_prob > 0 and _xgb_move_prob < _G3_MIN_MOVE_PROB:
                        self._wlog(f"  BLOCKED(G3-XGB_PROB): {_stock_name} P(move)={_xgb_move_prob:.2f} < {_G3_MIN_MOVE_PROB}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_XGB_LOW_PROB',
                                          reason=f'XGB P(move)={_xgb_move_prob:.2f} < {_G3_MIN_MOVE_PROB}',
                                          direction=direction)
                        continue
                except Exception:
                    pass
                
                # --- GATE G4: GMM Down-Risk Veto (watcher-specific) ---
                # If GMM DR score opposes trade direction AND is very high → block.
                # UP_Flag=True (UP regime, high dr) = hidden crash risk → opposes BUY
                # Down_Flag=True (DOWN regime, high dr) = hidden bounce risk → opposes SELL
                # Threshold: 0.30 (generous — only blocks extreme anomaly, main sniper uses 0.08-0.13)
                try:
                    _gmm_ml = _ml_results.get(_sym, {})
                    _dr_score = _gmm_ml.get('ml_down_risk_score', 0)
                    _up_flag = _gmm_ml.get('ml_up_flag', False)
                    _down_flag = _gmm_ml.get('ml_down_flag', False)
                    _up_score = _gmm_ml.get('ml_up_score', 0)
                    _down_score = _gmm_ml.get('ml_down_score', 0)
                    
                    _gmm_blocks = False
                    _gmm_reason = ''
                    
                    if direction == 'BUY' and _up_flag and _up_score >= 0.30:
                        # UP regime flagged = hidden crash risk → opposes BUY
                        _gmm_blocks = True
                        _gmm_reason = f'UP_flag=True up_score={_up_score:.3f}>=0.30 (crash risk opposes BUY)'
                    elif direction == 'SELL' and _down_flag and _down_score >= 0.30:
                        # DOWN regime flagged = hidden bounce risk → opposes SELL
                        _gmm_blocks = True
                        _gmm_reason = f'DOWN_flag=True down_score={_down_score:.3f}>=0.30 (bounce risk opposes SELL)'
                    
                    if _gmm_blocks:
                        self._wlog(f"  BLOCKED(G4-GMM): {_stock_name} {_gmm_reason}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_GMM_DR_VETO',
                                          reason=f'GMM DR veto: {_gmm_reason}',
                                          direction=direction)
                        continue
                except Exception:
                    pass
                
                # --- GATE H: Position limit (regime-aware, same as ELITE) ---
                active_positions = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
                if _breadth == 'MIXED':
                    _max_pos = HARD_RULES.get('MAX_POSITIONS_MIXED', 6)
                elif _breadth in ('BULLISH', 'BEARISH'):
                    _max_pos = HARD_RULES.get('MAX_POSITIONS_TRENDING', 12)
                else:
                    _max_pos = HARD_RULES['MAX_POSITIONS']
                if len(active_positions) >= _max_pos:
                    self._wlog(f"  BLOCKED(H-POS): {_stock_name} positions={len(active_positions)}/{_max_pos} ({_breadth}) — EXHAUSTED")
                    self._watcher_total_pos_exhausted += 1
                    break
                
                # === ALL GATES PASSED — Execute trade ===
                self._wlog(f"  ✅ ALL GATES PASSED: {_stock_name} score={_final_score:.0f} dir={direction} "
                           f"trigger={_trigger_type} pos={len(active_positions)}/{_max_pos} — EXECUTING")
                
                # Build ML data payload (same format as ELITE auto-fire)
                _elite_ml = _ml_results.get(_sym, {})
                _ml_data = {
                    'smart_score': _final_score,
                    'p_score': _final_score,
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
                        'up_flag': _elite_ml.get('ml_up_flag', False),
                        'down_flag': _elite_ml.get('ml_down_flag', False),
                        'up_score': _elite_ml.get('ml_up_score', 0),
                        'down_score': _elite_ml.get('ml_down_score', 0),
                        'down_risk_bucket': _elite_ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': _elite_ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': _elite_ml.get('ml_gmm_regime_used', 'BOTH'),
                        'gmm_action': 'WATCHER_BREAKOUT',
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': False,
                } if _elite_ml else {}
                
                _setup_type = 'ORB_BREAKOUT' if 'DAY' in _trigger_type or 'SPIKE' in _trigger_type else 'WATCHER'
                
                # --- GATE I: ORB-specific tightening (higher bar for ORB_BREAKOUT) ---
                if _setup_type == 'ORB_BREAKOUT':
                    _orb_min_score = BREAKOUT_WATCHER.get('orb_min_score', 45)
                    _orb_min_move = BREAKOUT_WATCHER.get('orb_min_move_prob', 0.65)
                    _xgb_mp = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                    if _final_score < _orb_min_score:
                        self._wlog(f"  BLOCKED(I-ORB_SCORE): {_stock_name} ORB score={_final_score:.0f} < {_orb_min_score}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_ORB_LOW_SCORE',
                                          reason=f'ORB_BREAKOUT score {_final_score:.0f} < {_orb_min_score}',
                                          direction=direction)
                        continue
                    if _xgb_mp > 0 and _xgb_mp < _orb_min_move:
                        self._wlog(f"  BLOCKED(I-ORB_MOVE): {_stock_name} ORB P(move)={_xgb_mp:.2f} < {_orb_min_move}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_ORB_LOW_MOVE',
                                          reason=f'ORB_BREAKOUT P(move)={_xgb_mp:.2f} < {_orb_min_move}',
                                          direction=direction)
                        continue

                # --- GATE I-W: WATCHER P(move) floor ---
                if _setup_type == 'WATCHER':
                    _w_min_move = BREAKOUT_WATCHER.get('watcher_min_move_prob', 0.57)
                    _w_mp = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                    if _w_mp > 0 and _w_mp < _w_min_move:
                        self._wlog(f"  BLOCKED(I-W_MOVE): {_stock_name} WATCHER P(move)={_w_mp:.2f} < {_w_min_move}")
                        self._watcher_total_gate_blocked += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_LOW_MOVE',
                                          reason=f'WATCHER P(move)={_w_mp:.2f} < {_w_min_move}',
                                          direction=direction)
                        continue
                
                # All gates passed — add to candidates for P(move) ranking
                _cand_pmove = _ml_results.get(_sym, {}).get('ml_move_prob', 0)
                _candidates.append({
                    'sym': _sym,
                    'direction': direction,
                    'score': _final_score,
                    'ml_move_prob': _cand_pmove,
                    'trigger_type': _trigger_type,
                    'move_pct': _move_pct,
                    'setup_type': _setup_type,
                    'ml_data': _ml_data,
                })
                self._wlog(f"  ✅ PASSED ALL GATES: {_sym.replace('NSE:', '')} "
                           f"score={_final_score:.0f} P(move)={_cand_pmove:.2f} "
                           f"trigger={_trigger_type}({_move_pct:+.1f}%)")
            
            # ════════════════════════════════════════════════════════════════
            # RANK CANDIDATES BY P(move) DESCENDING — place best trades first
            # ════════════════════════════════════════════════════════════════
            if not _candidates:
                self._wlog(f"  ⚠️ No candidates passed all gates")
            else:
                _candidates.sort(key=lambda c: c['ml_move_prob'], reverse=True)
                _rank_summary = ' > '.join(
                    f"{c['sym'].replace('NSE:', '')}(P={c['ml_move_prob']:.2f},S={c['score']:.0f})"
                    for c in _candidates
                )
                self._wlog(f"  📊 P(move) RANKING: {_rank_summary}")
                
                for _cand in _candidates:
                    if _fired_count >= _max_per_scan:
                        self._wlog(f"  ⏸ Max trades per scan ({_max_per_scan}) reached — skipping remaining")
                        break
                    
                    _sym = _cand['sym']
                    _direction = _cand['direction']
                    _trigger_type = _cand['trigger_type']
                    _move_pct = _cand['move_pct']
                    _setup_type = _cand['setup_type']
                    _final_score = _cand['score']
                    _ml_data = _cand['ml_data']
                    
                    result = self.tools.place_option_order(
                        underlying=_sym,
                        direction=_direction,
                        strike_selection="ATM",
                        rationale=(f"WATCHER→FULL_PIPELINE: {_trigger_type} ({_move_pct:+.1f}%) — "
                                  f"Score {_final_score:.0f} P(move)={_cand['ml_move_prob']:.2f}, "
                                  f"ranked #{_candidates.index(_cand)+1}/{len(_candidates)} by P(move)"),
                        setup_type=_setup_type,
                        ml_data=_ml_data
                    )
                    
                    if result and result.get('success'):
                        self._wlog(f"  🎯 TRADE PLACED: {_sym.replace('NSE:', '')} ({_direction}) "
                                   f"score={_final_score:.0f} P(move)={_cand['ml_move_prob']:.2f} "
                                   f"trigger={_trigger_type}({_move_pct:+.1f}%) "
                                   f"rank=#{_candidates.index(_cand)+1}/{len(_candidates)} "
                                   f"order={result.get('order_id', '?')} setup={_setup_type}")
                        self._watcher_fired_this_session.add(_sym)
                        self._auto_fired_this_session.add(_sym)  # Prevent ELITE re-fire
                        _fired_count += 1
                        self._watcher_total_placed += 1
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_FIRED',
                                          reason=(f'Breakout {_trigger_type} ({_move_pct:+.1f}%) — '
                                                 f'FULL PIPELINE: score+ML+OI+sector+DR+setup+FT+ADX all passed | '
                                                 f'P(move)={_cand["ml_move_prob"]:.2f} rank #{_candidates.index(_cand)+1}/{len(_candidates)}'),
                                          direction=_direction, setup=_setup_type)
                    else:
                        _err = result.get('error', 'unknown') if result else 'no result'
                        self._wlog(f"  ⚠️ TRADE FAILED: {_sym.replace('NSE:', '')} — {_err}")
                        self._log_decision(_ts, _sym, _final_score, 'WATCHER_TRADE_FAILED',
                                          reason=f'{_trigger_type}: {str(_err)[:80]}',
                                          direction=_direction)
        
        except Exception as _e:
            self._wlog(f"PIPELINE ERROR: {_e}")
            import traceback
            traceback.print_exc()
    
    # ========== WATCHER MOMENTUM EXIT (spike peaked / crater bottomed) ==========
    def _check_watcher_momentum_exits(self):
        """Check open WATCHER positions for momentum reversal and exit.
        
        Detects when a spike has peaked (for CE/BUY trades) or a crater has
        bottomed (for PE/SELL trades) by tracking the underlying's extreme since
        entry and triggering exit on reversal.
        
        After exit, cooldown is BYPASSED — the watcher can immediately re-enter
        if another spike/crater forms on the same symbol.
        """
        from config import BREAKOUT_WATCHER
        _cfg = BREAKOUT_WATCHER.get('momentum_exit', {})
        if not _cfg.get('enabled', False):
            return
        
        import time as _wme_time
        
        # --- Auto-discover new WATCHER trades not yet tracked ---
        with self.tools._positions_lock:
            _active = [t.copy() for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        _watcher_trades = [t for t in _active
                           if t.get('setup_type') in ('WATCHER', 'ORB_BREAKOUT')
                           and t.get('is_option', False)
                           and not t.get('is_debit_spread', False)
                           and not t.get('is_credit_spread', False)]
        
        # Register any new watcher trades
        for _t in _watcher_trades:
            _opt_sym = _t['symbol']
            if _opt_sym in self._watcher_momentum_tracker:
                continue
            _ul = _t.get('underlying', '')
            if not _ul:
                continue
            # Parse entry time
            try:
                _entry_epoch = datetime.fromisoformat(_t.get('timestamp', '')).timestamp()
            except Exception:
                _entry_epoch = _wme_time.time()
            # Skip positions that existed before this bot session (stale UL price on restart)
            if _entry_epoch < self._wme_session_start:
                continue
            # Get underlying LTP from WebSocket cache
            _ul_ltp = 0
            if self.tools.ticker and self.tools.ticker.connected:
                _ul_prices = self.tools.ticker.get_ltp_batch([_ul])
                _ul_ltp = _ul_prices.get(_ul, 0)
            if _ul_ltp <= 0:
                continue
            
            _direction = _t.get('direction', 'BUY')
            self._watcher_momentum_tracker[_opt_sym] = {
                'underlying': _ul,
                'direction': _direction,
                'entry_ul_price': _ul_ltp,
                'entry_time': _entry_epoch,
                'peak_price': _ul_ltp,
                'trough_price': _ul_ltp,
                # Multi-signal tracking
                'price_samples': [(_wme_time.time(), _ul_ltp)],
                'vol_samples': [],       # (timestamp, cumulative_volume)
                'peak_momentum': 0.0,    # Max price velocity seen (%/s, direction-adjusted)
            }
            _ul_short = _ul.replace('NSE:', '')
            print(f"   🌊 WME: Tracking {_ul_short} ({_direction}) UL=₹{_ul_ltp:.2f} opt={_opt_sym.split(':')[-1]}")
        
        # Clean tracker: remove entries whose positions are no longer open
        _active_opt_syms = {t['symbol'] for t in _watcher_trades}
        _stale = [k for k in self._watcher_momentum_tracker if k not in _active_opt_syms]
        for _s in _stale:
            del self._watcher_momentum_tracker[_s]
        
        if not self._watcher_momentum_tracker:
            return
        
        # --- Get underlying QUOTES via WebSocket cache (zero API calls) ---
        # Quotes include volume, buy_quantity, sell_quantity — needed for multi-signal detection
        _all_ul = list({v['underlying'] for v in self._watcher_momentum_tracker.values()})
        if not self.tools.ticker or not self.tools.ticker.connected:
            return
        _ul_quotes = self.tools.ticker.get_quote_batch(_all_ul)
        _ul_prices = {sym: q.get('last_price', 0) for sym, q in _ul_quotes.items()}
        # Fallback: symbols with quote but no LTP (shouldn't happen, but defensive)
        for _sym in _all_ul:
            if _sym not in _ul_prices or _ul_prices[_sym] <= 0:
                _ltp_batch = self.tools.ticker.get_ltp_batch([_sym])
                _ul_prices[_sym] = _ltp_batch.get(_sym, 0)
        
        _reversal_pct = _cfg.get('reversal_pct', 0.5) / 100
        _confirmed_rev_pct = _cfg.get('confirmed_reversal_pct', 0.25) / 100
        _partial_rev_pct = _cfg.get('partial_confirm_reversal_pct', 0.35) / 100
        _min_hold_s = _cfg.get('min_hold_seconds', 60)
        _min_move_pct = _cfg.get('min_favorable_move_pct', 0.5) / 100
        _bypass_cooldown = _cfg.get('bypass_cooldown', True)
        _only_profit = _cfg.get('only_in_profit', True)
        _skip_trailing = _cfg.get('skip_trailing_active', True)
        _vol_dryup_ratio = _cfg.get('volume_dryup_ratio', 0.30)
        _mom_decay_thresh = _cfg.get('momentum_decay_threshold', 0.70)
        _pressure_enabled = _cfg.get('pressure_shift_enabled', True)
        _pressure_ratio = _cfg.get('pressure_shift_ratio', 1.5)
        _sample_window = _cfg.get('sample_window_seconds', 90)
        _now = _wme_time.time()
        _exits_to_process = []
        
        # --- Check each tracked position ---
        for _opt_sym, _track in list(self._watcher_momentum_tracker.items()):
            _ul = _track['underlying']
            _ul_ltp = _ul_prices.get(_ul, 0)
            if _ul_ltp <= 0:
                continue
            
            _direction = _track['direction']
            _entry_ul = _track['entry_ul_price']
            _ul_quote = _ul_quotes.get(_ul, {})
            
            # ── ALWAYS collect samples (even before min_hold) to build baselines ──
            _track['price_samples'].append((_now, _ul_ltp))
            _cum_vol = _ul_quote.get('volume', 0)
            if _cum_vol > 0:
                _track['vol_samples'].append((_now, _cum_vol))
            
            # Trim samples to rolling window
            _cutoff = _now - _sample_window
            _track['price_samples'] = [(t, p) for t, p in _track['price_samples'] if t >= _cutoff]
            _track['vol_samples'] = [(t, v) for t, v in _track['vol_samples'] if t >= _cutoff]
            
            # ── Update peak/trough price ──
            if _direction == 'BUY':
                if _ul_ltp > _track['peak_price']:
                    _track['peak_price'] = _ul_ltp
            elif _direction == 'SELL':
                if _ul_ltp < _track['trough_price']:
                    _track['trough_price'] = _ul_ltp
            
            # ── Update peak momentum (max velocity seen over any ~15s window) ──
            _ps = _track['price_samples']
            if len(_ps) >= 4:
                _p_now = _ps[-1]
                _p_prev_idx = max(0, len(_ps) - 4)  # ~15-20s ago
                _p_prev = _ps[_p_prev_idx]
                _pdt = _p_now[0] - _p_prev[0]
                if _pdt > 0 and _p_prev[1] > 0:
                    _vel = (_p_now[1] - _p_prev[1]) / _p_prev[1] / _pdt  # %/s
                    # Track peak velocity in favorable direction
                    if _direction == 'BUY' and _vel > _track['peak_momentum']:
                        _track['peak_momentum'] = _vel
                    elif _direction == 'SELL' and _vel < _track['peak_momentum']:
                        _track['peak_momentum'] = _vel  # Negative for SELL
            
            # ── Min hold time: let the trade breathe (but data is already collected above) ──
            if (_now - _track['entry_time']) < _min_hold_s:
                continue
            
            # Skip trades already managed by trailing stop (let winners run)
            if _skip_trailing:
                _em_state = self.exit_manager.get_trade_state(_opt_sym)
                if _em_state and _em_state.trailing_active:
                    continue
            
            # Check minimum favorable move from entry
            if _direction == 'BUY':
                _favorable = (_track['peak_price'] - _entry_ul) / _entry_ul if _entry_ul > 0 else 0
            else:
                _favorable = (_entry_ul - _track['trough_price']) / _entry_ul if _entry_ul > 0 else 0
            if _favorable < _min_move_pct:
                continue
            
            # ── Compute current reversal from peak/trough ──
            if _direction == 'BUY':
                _rev = (_track['peak_price'] - _ul_ltp) / _track['peak_price'] if _track['peak_price'] > 0 else 0
            else:
                _rev = (_ul_ltp - _track['trough_price']) / _track['trough_price'] if _track['trough_price'] > 0 else 0
            
            # ════════════════════════════════════════════════
            # MULTI-SIGNAL REVERSAL CONFIRMATION
            # Mirrors watcher entry: spike, volume, grind → now: price reversal, volume dry-up, momentum decay, pressure shift
            # ════════════════════════════════════════════════
            _confirmations = 0
            _signal_reasons = []
            
            # ── Signal 1: VOLUME DRY-UP ──
            # Compare recent volume rate vs average rate over tracking period.
            # Spikes have explosive volume; when volume drops to < 30% of avg, the spike is over.
            _vs = _track['vol_samples']
            if len(_vs) >= 4:
                # Average volume rate over all tracked data
                _total_dt = _vs[-1][0] - _vs[0][0]
                if _total_dt > 10:  # Need at least 10s of data
                    _avg_vol_rate = (_vs[-1][1] - _vs[0][1]) / _total_dt
                    # Recent volume rate (last ~15s)
                    _rv_idx = max(0, len(_vs) - 4)
                    _recent_dt = _vs[-1][0] - _vs[_rv_idx][0]
                    if _recent_dt > 0 and _avg_vol_rate > 0:
                        _recent_vol_rate = (_vs[-1][1] - _vs[_rv_idx][1]) / _recent_dt
                        if _recent_vol_rate < _avg_vol_rate * _vol_dryup_ratio:
                            _confirmations += 1
                            _signal_reasons.append('VOL_DRYUP')
            
            # ── Signal 2: MOMENTUM DECAY ──
            # If price velocity has decayed to < 30% of peak, or reversed direction entirely.
            if len(_ps) >= 4 and abs(_track['peak_momentum']) > 1e-8:
                _p_now = _ps[-1]
                _p_prev_idx = max(0, len(_ps) - 4)
                _p_prev = _ps[_p_prev_idx]
                _pdt = _p_now[0] - _p_prev[0]
                if _pdt > 0 and _p_prev[1] > 0:
                    _cur_vel = (_p_now[1] - _p_prev[1]) / _p_prev[1] / _pdt
                    _peak_vel = _track['peak_momentum']
                    if _direction == 'BUY':
                        # Momentum reversed (velocity turned negative) = full decay
                        if _cur_vel <= 0:
                            _confirmations += 1
                            _signal_reasons.append('MOM_REVERSED')
                        # Momentum decayed to < 30% of peak
                        elif _peak_vel > 0 and _cur_vel < _peak_vel * (1 - _mom_decay_thresh):
                            _confirmations += 1
                            _signal_reasons.append('MOM_DECAY')
                    elif _direction == 'SELL':
                        if _cur_vel >= 0:
                            _confirmations += 1
                            _signal_reasons.append('MOM_REVERSED')
                        elif _peak_vel < 0 and abs(_cur_vel) < abs(_peak_vel) * (1 - _mom_decay_thresh):
                            _confirmations += 1
                            _signal_reasons.append('MOM_DECAY')
            
            # ── Signal 3: PRESSURE SHIFT (buy/sell quantity imbalance) ──
            # If opposing side's total order quantity dominates → participants are exiting.
            if _pressure_enabled:
                _buy_q = _ul_quote.get('buy_quantity', 0)
                _sell_q = _ul_quote.get('sell_quantity', 0)
                if _buy_q > 0 and _sell_q > 0:
                    if _direction == 'BUY' and _sell_q > _buy_q * _pressure_ratio:
                        _confirmations += 1
                        _signal_reasons.append('SELL_PRESSURE')
                    elif _direction == 'SELL' and _buy_q > _sell_q * _pressure_ratio:
                        _confirmations += 1
                        _signal_reasons.append('BUY_PRESSURE')
            
            # ── Adaptive reversal threshold ──
            if _confirmations >= 2:
                _effective_rev = _confirmed_rev_pct   # 0.25% — high confidence reversal
            elif _confirmations == 1:
                _effective_rev = _partial_rev_pct     # 0.35% — medium confidence
            else:
                _effective_rev = _reversal_pct        # 0.50% — price-only (original)
            
            # ── Check if reversal exceeds adaptive threshold ──
            if _rev >= _effective_rev:
                _exits_to_process.append((_opt_sym, _track, _ul_ltp, _rev * 100, _confirmations, _signal_reasons, _effective_rev * 100))
        
        # --- Process momentum exits ---
        for _opt_sym, _track, _ul_ltp, _rev_pct, _n_confirms, _sig_reasons, _eff_rev in _exits_to_process:
            _direction = _track['direction']
            _ul = _track['underlying']
            _ul_short = _ul.replace('NSE:', '')
            
            # Find the active trade
            with self.tools._positions_lock:
                _trade = next((t for t in self.tools.paper_positions
                               if t['symbol'] == _opt_sym and t.get('status', 'OPEN') == 'OPEN'), None)
            
            if not _trade:
                self._watcher_momentum_tracker.pop(_opt_sym, None)
                continue
            
            # Get current option premium
            try:
                if self.tools.ticker and self.tools.ticker.connected:
                    _opt_prices = self.tools.ticker.get_ltp_batch([_opt_sym])
                    _opt_ltp = _opt_prices.get(_opt_sym, 0)
                else:
                    _q = self.tools.kite.ltp([_opt_sym])
                    _opt_ltp = _q.get(_opt_sym, {}).get('last_price', 0)
            except Exception:
                continue
            
            if _opt_ltp <= 0:
                continue
            
            # Calculate P&L
            _entry_price = _trade['avg_price']
            _qty = _trade['quantity']
            _pnl = (_opt_ltp - _entry_price) * _qty
            from execution_guard import calc_brokerage
            _pnl -= calc_brokerage(_entry_price, _opt_ltp, _qty)
            
            # Only exit if option is in profit (don't cut losers — let exit manager handle)
            if _only_profit and _pnl <= 0:
                continue
            
            _extreme_label = 'peak' if _direction == 'BUY' else 'trough'
            _extreme_val = _track['peak_price'] if _direction == 'BUY' else _track['trough_price']
            _trigger_label = 'spike peaked' if _direction == 'BUY' else 'crater bottomed'
            _signals_str = '+'.join(_sig_reasons) if _sig_reasons else 'PRICE_ONLY'
            
            print(f"\n🌊 WATCHER MOMENTUM EXIT: {_ul_short}")
            print(f"   Direction: {_direction} | Trigger: {_trigger_label}")
            print(f"   Signals: {_signals_str} ({_n_confirms}/3 confirmed) → threshold {_eff_rev:.2f}%")
            print(f"   Underlying: entry ₹{_track['entry_ul_price']:.2f} → {_extreme_label} ₹{_extreme_val:.2f} → now ₹{_ul_ltp:.2f} ({_rev_pct:.2f}% reversal)")
            print(f"   Option: {_opt_sym.split(':')[-1]} — entry ₹{_entry_price:.2f} → exit ₹{_opt_ltp:.2f}")
            print(f"   P&L: ₹{_pnl:+,.0f}")
            
            # Execute exit
            self.tools.update_trade_status(_opt_sym, 'WATCHER_MOMENTUM_EXIT', _opt_ltp, _pnl)
            with self._pnl_lock:
                self.daily_pnl += _pnl
                self.capital += _pnl
            
            # Record with Risk Governor
            _remaining = [t for t in self.tools.paper_positions
                          if t.get('status') == 'OPEN' and t['symbol'] != _opt_sym]
            _unrealized = self.risk_governor._calc_unrealized_pnl(_remaining)
            self.risk_governor.record_trade_result(_opt_sym, _pnl, _pnl > 0, unrealized_pnl=_unrealized)
            self.risk_governor.update_capital(self.capital)
            
            # Remove from exit manager
            self.exit_manager.remove_trade(_opt_sym)
            
            # Remove from tracker
            self._watcher_momentum_tracker.pop(_opt_sym, None)
            
            # === COOLDOWN BYPASS: Allow watcher to re-enter this symbol ===
            if _bypass_cooldown:
                _nse_sym = f"NSE:{_ul_short}"
                self._watcher_fired_this_session.discard(_nse_sym)
                # Also clear the ticker-level cooldown for this symbol
                if self.tools.ticker and hasattr(self.tools.ticker, 'breakout_watcher'):
                    _bw = self.tools.ticker.breakout_watcher
                    if _bw and hasattr(_bw, '_cooldowns'):
                        _bw._cooldowns.pop(_nse_sym, None)
                print(f"   🔄 COOLDOWN BYPASSED: {_ul_short} — watcher can re-enter on next spike/crater")
            
            # Notify scorer
            try:
                from options_trader import get_intraday_scorer
                _scorer = get_intraday_scorer()
                if _pnl > 0:
                    _scorer.record_symbol_win(_opt_sym)
                else:
                    _scorer.record_symbol_loss(_opt_sym)
            except Exception:
                pass
    
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
            # print(f"📅 New trading day — model-tracker counter reset")
    
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
            dr_score = ml.get('ml_down_risk_score', None)
            
            if dr_score is None:
                continue  # No down-risk prediction for this symbol
            
            # Dual-regime: read independent UP and DOWN scores + flags
            _up_score = ml.get('ml_up_score', dr_score)
            _down_score = ml.get('ml_down_score', dr_score)
            _up_flag = ml.get('ml_up_flag', False)
            _down_flag = ml.get('ml_down_flag', False)
            
            # Get trade direction from IntradayScorer (if available)
            _cd = _cycle_decs.get(sym, {})
            _decision = _cd.get('decision')
            direction = None
            if _decision and hasattr(_decision, 'recommended_direction'):
                direction = _decision.recommended_direction
            
            _sym_diag = sym.replace('NSE:', '')
            _dir_tag = direction or '?'
            
            # Dual-regime directional scoring:
            #   UP_Flag (crash risk)   → opposes BUY, confirms SELL
            #   Down_Flag (bounce risk) → opposes SELL, confirms BUY
            #   Both clean → genuine pattern → boost
            #   Both flagged → conflicting → penalize
            if direction and direction != 'HOLD':
                if direction == 'BUY':
                    # BUY: Down_Flag confirms (bounce=UP), UP_Flag opposes (crash)
                    _confirm_flag = _down_flag
                    _oppose_flag = _up_flag
                    _confirm_score = _down_score
                    _oppose_score = _up_score
                    _confirm_lbl = 'Down_Flag(bounce)'
                    _oppose_lbl = 'UP_Flag(crash)'
                else:  # SELL
                    # SELL: UP_Flag confirms (crash=DOWN), Down_Flag opposes (bounce)
                    _confirm_flag = _up_flag
                    _oppose_flag = _down_flag
                    _confirm_score = _up_score
                    _oppose_score = _down_score
                    _confirm_lbl = 'UP_Flag(crash)'
                    _oppose_lbl = 'Down_Flag(bounce)'
                
                if _confirm_flag and not _oppose_flag:
                    # Anomaly CONFIRMS trade → BOOST (this is the 2x edge)
                    pre_scores[sym] += clean_boost
                    adjustments.append(f"🟣 {_sym_diag} +{clean_boost} ({_confirm_lbl} CONFIRMS {_dir_tag} score={_confirm_score:.3f})")
                elif _oppose_flag and not _confirm_flag:
                    # Anomaly OPPOSES trade → strong penalty
                    pre_scores[sym] -= high_penalty
                    adjustments.append(f"🔴 {_sym_diag} −{high_penalty} ({_oppose_lbl} OPPOSES {_dir_tag} score={_oppose_score:.3f})")
                elif _confirm_flag and _oppose_flag:
                    # BOTH flagged (conflicting/extreme vol) → penalize
                    pre_scores[sym] -= mid_penalty
                    adjustments.append(f"🟠 {_sym_diag} −{mid_penalty} (BOTH_FLAGS {_dir_tag} UP={_up_score:.3f} DN={_down_score:.3f})")
                elif _confirm_score > high_thresh:
                    # High confirming score (not flagged yet) → small boost
                    _small_boost = clean_boost // 2
                    pre_scores[sym] += _small_boost
                    adjustments.append(f"🔵 {_sym_diag} +{_small_boost} (HIGH_{_confirm_lbl} {_dir_tag} score={_confirm_score:.3f})")
                elif _oppose_score > high_thresh:
                    # High opposing score (not flagged yet) → caution penalty
                    pre_scores[sym] -= mid_penalty
                    adjustments.append(f"🟠 {_sym_diag} −{mid_penalty} (HIGH_{_oppose_lbl} vs {_dir_tag} score={_oppose_score:.3f})")
                elif max(_up_score, _down_score) < clean_thresh:
                    # Both very low → genuine clean pattern → boost
                    pre_scores[sym] += clean_boost
                    adjustments.append(f"🟢 {_sym_diag} +{clean_boost} (CLEAN {_dir_tag} UP={_up_score:.3f} DN={_down_score:.3f})")
                # else: neutral zone — no adjustment
            else:
                # No direction / HOLD — skip scoring (can't determine confirm/oppose)
                pass
        
        if adjustments:
            # print(f"   🛡️ GMM GRADUATED SCORE: {len(adjustments)} adjusted — {' | '.join(adjustments[:8])}")
            # if len(adjustments) > 8:
            #     print(f"      ... and {len(adjustments) - 8} more")
            pass
    
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
            
            # ══════════════════════════════════════════════════════════════
            # 18-CASE DECISION MATRIX (3 XGB × 2 Titan × 3 GMM states)
            # ══════════════════════════════════════════════════════════════
            # Now predictor.py runs BOTH regime models → independent UP_Flag + Down_Flag.
            #
            # GMM states:
            #   Clean       = both False  → no anomaly edge → BLOCK
            #   UP_Flag     = crash risk  → predicts DOWN   → confirms SELL, opposes BUY
            #   Down_Flag   = bounce risk → predicts UP     → confirms BUY, opposes SELL
            #   Both True   = conflicting → extreme vol     → BLOCK
            #
            # RULE: Only trade when anomaly CONFIRMS trade direction.
            #   BUY needs Down_Flag (bounce=UP). SELL needs UP_Flag (crash=DOWN).
            #   If XGB also aligns → ALL_AGREE. Else → GMM_CONTRARIAN.
            #   Clean / opposing anomaly / conflicting → BLOCK.
            # ══════════════════════════════════════════════════════════════
            trade_type = 'MODEL_TRACKER'
            gmm_action = 'ALLOW'  # default (will be overridden or continue'd)
            _xgb_signal = ml.get('ml_signal', 'UNKNOWN')
            gmm_regime = ml.get('ml_gmm_regime_used', 'BOTH')
            _up_flag = ml.get('ml_up_flag', False)
            _down_flag = ml.get('ml_down_flag', False)
            _up_score = ml.get('ml_up_score', ml.get('ml_down_risk_score', 0.0))
            _down_score = ml.get('ml_down_score', ml.get('ml_down_risk_score', 0.0))

            # ── Step A: Determine if anomaly confirms trade direction ──
            # BUY needs Down_Flag (bounce=UP to confirm), SELL needs UP_Flag (crash=DOWN to confirm)
            if direction == 'BUY':
                _anomaly_confirms = _down_flag and not _up_flag   # bounce confirms BUY
                _anomaly_opposes  = _up_flag                       # crash opposes BUY
                _confirm_score = _down_score
                _confirm_tag = 'Down_Flag(bounce)'
            else:  # SELL
                _anomaly_confirms = _up_flag and not _down_flag    # crash confirms SELL
                _anomaly_opposes  = _down_flag                      # bounce opposes SELL
                _confirm_score = _up_score
                _confirm_tag = 'UP_Flag(crash)'

            _is_clean = not _up_flag and not _down_flag
            _is_conflicting = _up_flag and _down_flag

            # ── Step B: Decision ──
            # BLOCK cases: Clean (1,4,7,10,13,16), opposing anomaly (2,6,8,12,14,18),
            #              conflicting (both flags True)
            if _is_clean or _anomaly_opposes or _is_conflicting:
                # No anomaly edge, or anomaly opposes trade, or conflicting → BLOCK
                continue

            # At this point: anomaly CONFIRMS trade direction (6 surviving cases: 3,5,9,11,15,17)
            # ── Step C: XGB alignment decides trade type ──
            _xgb_aligns = ((_xgb_signal == 'UP' and direction == 'BUY') or
                           (_xgb_signal == 'DOWN' and direction == 'SELL'))

            # ── Confirm score floor: anomaly must be strong enough ──
            _min_confirm = self._down_risk_cfg.get('min_confirm_score', 0.0)
            if _confirm_score < _min_confirm:
                continue  # Anomaly too weak — borderline flag

            if _xgb_aligns:
                # Cases 3, 11: Anomaly + XGB + Titan TRIPLE ALIGN → ALL_AGREE
                # [TIGHTENED Mar 6] Direction-specific GMM confirm score floors
                # BUY needs strong Down_Flag (bounce ≥ 0.30), SELL needs strong UP_Flag (crash ≥ 0.30)
                if direction == 'BUY':
                    _aa_min_confirm = self._down_risk_cfg.get('all_agree_min_down_score', 0.30)
                else:
                    _aa_min_confirm = self._down_risk_cfg.get('all_agree_min_up_score', 0.30)
                if _confirm_score < _aa_min_confirm:
                    sym_clean_tmp = sym.replace('NSE:', '')
                    print(f"      ⛔ ALL_AGREE BLOCKED: {sym_clean_tmp} — {_confirm_tag}={_confirm_score:.3f} < {_aa_min_confirm} (weak GMM signal)")
                    self._log_decision(cycle_time, sym, pre_scores.get(sym, 0), 'ALL_AGREE_WEAK_GMM',
                                      reason=f"{_confirm_tag}={_confirm_score:.3f} < {_aa_min_confirm} floor",
                                      direction=direction)
                    continue
                # [TIGHTENED Mar 2] Tech floor: pre_score must be ≥ 45 to qualify as ALL_AGREE
                _aa_p_score = pre_scores.get(sym, 0)
                if _aa_p_score < 45:
                    sym_clean_tmp = sym.replace('NSE:', '')
                    print(f"      ⛔ ALL_AGREE BLOCKED: {sym_clean_tmp} — tech={_aa_p_score:.0f} < 45 (weak technical setup)")
                    self._log_decision(cycle_time, sym, _aa_p_score, 'ALL_AGREE_WEAK_TECH',
                                      reason=f"tech={_aa_p_score:.0f} < 45 floor despite XGB+GMM+Titan align",
                                      direction=direction)
                    continue
                trade_type = 'ALL_AGREE'
                gmm_action = 'BOOST'
                sym_clean_tmp = sym.replace('NSE:', '')
                print(f"      ✅ ALL_AGREE: {sym_clean_tmp} — XGB={_xgb_signal} + Titan={direction} + "
                      f"{_confirm_tag}={_confirm_score:.3f} → BOOST (tech={_aa_p_score:.0f})")
                self._log_decision(cycle_time, sym, _aa_p_score, 'ALL_AGREE',
                                  reason=f"XGB={_xgb_signal} + Titan={direction} + {_confirm_tag}={_confirm_score:.3f}",
                                  direction=direction)
            else:
                # Cases 5, 9, 15, 17: Anomaly confirms Titan, XGB opposes/flat → GMM_CONTRARIAN
                from config import GMM_CONTRARIAN as _contr_cfg
                _contr_ok = False
                _contr_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
                if _contr_cfg.get('enabled', False) and _confirm_score >= _contr_cfg.get('min_dr_score', 0.15):
                    _contr_today = getattr(self, '_dr_flip_trades_today', 0)
                    _contr_max_day = _contr_cfg.get('max_trades_per_day', 4)
                    _contr_open = sum(1 for t in (getattr(self.tools, 'paper_positions', []) or [])
                                     if t.get('setup_type') in ('DR_FLIP', 'GMM_CONTRARIAN') and t.get('status', 'OPEN') == 'OPEN')
                    _contr_max_concurrent = _contr_cfg.get('max_concurrent_open', 3)
                    _contr_min_gate = _contr_cfg.get('min_gate_prob', 0.50)
                    _contr_confidence = ml.get('ml_confidence', 0.0)

                    if (_contr_today < _contr_max_day
                        and _contr_open < _contr_max_concurrent
                        and _contr_move_prob >= _contr_min_gate
                        and _contr_confidence >= 0.45):  # [TIGHTENED Mar 2] Confidence floor — weak ML conviction blocks contrarian
                        _contr_ok = True
                        trade_type = 'GMM_CONTRARIAN'
                        gmm_action = 'BOOST'
                        sym_clean_tmp = sym.replace('NSE:', '')
                        print(f"      🎯 GMM_CONTRARIAN: {sym_clean_tmp} — XGB={_xgb_signal} but "
                              f"{_confirm_tag}={_confirm_score:.3f} confirms {direction} | P(MOVE)={_contr_move_prob:.2f}")
                        self._log_decision(cycle_time, sym, pre_scores.get(sym, 0), 'GMM_CONTRARIAN',
                                          reason=f"XGB={_xgb_signal} but {_confirm_tag}={_confirm_score:.3f} confirms {direction}",
                                          direction=direction)

                if not _contr_ok:
                    continue  # GMM_CONTRARIAN gates failed → BLOCK
            
            # Gather available ML metrics (Gate model still runs)
            ml_confidence = ml.get('ml_confidence', 0.0)  # Gate confidence
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))  # P(MOVE) from Gate
            p_score = pre_scores.get(sym, 0)
            sym_clean = sym.replace('NSE:', '')
            sector = _get_sector_mt(sym_clean)

            # ── Gate floor: P(MOVE) must be meaningful ──
            # [TIGHTENED Mar 2] ALL_AGREE needs higher gate (0.55) since it bypasses scanner
            _gate_floor = 0.55 if trade_type == 'ALL_AGREE' else 0.40
            if ml_move_prob < _gate_floor:
                continue  # Gate says stock unlikely to move — skip
            
            # ── Composite smart score (dual-regime GMM-aware) ──
            # Weight 1: Conviction = Gate P(MOVE) × ML confidence (0-40 pts)
            # NOTE: Uses ml_confidence (not p_score) to avoid double-counting
            # p_score already represented in Weight 3 (technical)
            conviction = ml_move_prob * max(ml_confidence, 0.01) * 40.0
            
            # Weight 2: Anomaly conviction (0-25 pts)
            # Higher confirming anomaly score = stronger edge = higher score
            safety = min(_confirm_score, 1.0) * 20.0 + 5.0  # up to 25 pts
            
            # Weight 3: Technical strength = pre_score normalized (0-100 → 0-20 pts)
            technical = min(p_score, 100) * 0.20
            
            # Weight 4: Move probability bonus (0-1 → 0-15 pts)
            # Increased weight since we no longer have direction model agreement signal
            move_bonus = ml_move_prob * 15.0
            
            smart_score = conviction + safety + technical + move_bonus
            
            # ── ALIGNMENT BONUS: triple/dual-agree gets extra conviction ──
            # ALL_AGREE (XGB + Titan + GMM anomaly) = strongest possible signal → +15 pts
            # GMM_CONTRARIAN (Titan + GMM anomaly, XGB flat/opposed) = strong GMM edge → +10 pts
            if trade_type == 'ALL_AGREE':
                smart_score += 15.0
            elif trade_type == 'GMM_CONTRARIAN':
                smart_score += 10.0
            
            # ── SECTOR BREADTH PENALTY: penalize trades against sector direction ──
            # Uses sector index % changes cached from prior cycle.
            # If NIFTY METAL is bullish (+1%+) and trade is SELL → penalize.
            # If NIFTY IT is bearish (-1%+) and trade is BUY → penalize.
            _sector_penalty_applied = 0
            _sec_idx_chg_cache = getattr(self, '_sector_index_changes_cache', {})
            if _sec_idx_chg_cache and sector:
                # Map sector label (from STOCK_SECTOR_MAP) → NIFTY index symbol
                _sector_to_nifty = {
                    'METAL': 'NSE:NIFTY METAL', 'IT': 'NSE:NIFTY IT',
                    'BANK': 'NSE:NIFTY BANK', 'AUTO': 'NSE:NIFTY AUTO',
                    'PHARMA': 'NSE:NIFTY PHARMA', 'ENERGY': 'NSE:NIFTY ENERGY',
                    'FMCG': 'NSE:NIFTY FMCG', 'REALTY': 'NSE:NIFTY REALTY',
                    'INFRA': 'NSE:NIFTY INFRA',
                }
                _nifty_idx = _sector_to_nifty.get(sector)
                if _nifty_idx:
                    _sec_pct = _sec_idx_chg_cache.get(_nifty_idx)
                    if _sec_pct is not None:
                        from config import SECTOR_BREADTH_PENALTY
                        _sbp_cfg = SECTOR_BREADTH_PENALTY
                        _sbp_threshold = _sbp_cfg.get('threshold_pct', 1.0)
                        _sbp_penalty = _sbp_cfg.get('penalty', 10)
                        # Sector BULLISH but trade is SELL → counter-sector
                        if _sec_pct >= _sbp_threshold and direction == 'SELL':
                            smart_score -= _sbp_penalty
                            _sector_penalty_applied = _sbp_penalty
                            # print(f"      🏭 SECTOR BREADTH: {sym_clean} SELL vs {sector} +{_sec_pct:.1f}% → −{_sbp_penalty} penalty")
                        # Sector BEARISH but trade is BUY → counter-sector
                        elif _sec_pct <= -_sbp_threshold and direction == 'BUY':
                            smart_score -= _sbp_penalty
                            _sector_penalty_applied = _sbp_penalty
                            # print(f"      🏭 SECTOR BREADTH: {sym_clean} BUY vs {sector} {_sec_pct:+.1f}% → −{_sbp_penalty} penalty")
            
            # ── HIGH P(MOVE) BONUS: strong directional conviction → +25 ──
            _pmove_bonus_threshold = self._down_risk_cfg.get('pmove_bonus_threshold', 0.80)
            _pmove_bonus_pts = self._down_risk_cfg.get('pmove_bonus_points', 25)
            if ml_move_prob >= _pmove_bonus_threshold:
                smart_score += _pmove_bonus_pts
                print(f"      🚀 P(MOVE) BONUS: {sym_clean} P(move)={ml_move_prob:.2f} ≥ {_pmove_bonus_threshold} → +{_pmove_bonus_pts} (score={smart_score:.0f})")

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
                'up_flag': ml.get('ml_up_flag', False),
                'down_flag': ml.get('ml_down_flag', False),
                'up_score': _up_score,
                'down_score': _down_score,
                'confirm_score': _confirm_score,
                'confirm_tag': _confirm_tag,
                'gmm_regime': gmm_regime,
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
                        'up_flag': ml.get('ml_up_flag', False),
                        'down_flag': ml.get('ml_down_flag', False),
                        'up_score': _up_score,
                        'down_score': _down_score,
                        'confirm_score': _confirm_score,
                        'confirm_tag': _confirm_tag,
                        'down_risk_score': dr_score,
                        'down_risk_flag': dr_flag,
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', 'BOTH'),
                        'gmm_action': gmm_action,
                    },
                    'scored_direction': direction,
                    'xgb_signal': _xgb_signal,
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
        
        _mt_min_smart = self._down_risk_cfg.get('min_smart_score', 55)
        _mt_min_smart_contr = 58  # [TIGHTENED Mar 2] GMM_CONTRARIAN needs higher smart floor — contrarian = extra conviction required
        _mt_smart_rejected = []
        for cand in raw_candidates:
            if len(selected) >= budget:
                break
            _smart_floor = _mt_min_smart_contr if cand.get('trade_type') == 'GMM_CONTRARIAN' else _mt_min_smart
            if cand['smart_score'] < _smart_floor:
                _mt_smart_rejected.append(f"{cand['sym_clean']}({cand['smart_score']:.1f})")
                continue  # Below smart score floor
            sec = cand['sector']
            if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                continue  # Sector full — skip to next
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            selected.append(cand)
        if _mt_smart_rejected:
            print(f"      🚫 MT SMART FLOOR: {len(_mt_smart_rejected)} blocked (< {_mt_min_smart}) — {', '.join(_mt_smart_rejected[:5])}")

        
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
                _gmm_tag = f" {c.get('confirm_tag', '?')}={c.get('confirm_score', 0):.3f}"
                print(f"      #{i} {c['sym_clean']:<12s} {c['direction']:<4s} smart={c['smart_score']:5.1f} "
                      f"(gate={c['ml_move_prob']:.2f}, UP={c.get('up_score', 0):.3f}, DN={c.get('down_score', 0):.3f}, "
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
                
                # ALL_AGREE = strongest conviction (all 3 models) → amplified lot sizing
                _lot_mult = 1.0
                if trade_type == 'ALL_AGREE':
                    _lot_mult = self._down_risk_cfg.get('all_agree_lot_multiplier', 1.5)
                    print(f"   🚀 ALL_AGREE AMPLIFIED: {_lot_mult}x lots (strongest conviction)")
                
                result = self.tools.place_option_order(
                    underlying=sym,
                    direction=direction,
                    strike_selection="ATM",
                    use_intraday_scoring=False,  # Bypass scorer — model-tracker evaluates independently
                    lot_multiplier=_lot_mult,
                    rationale=(f"{trade_type} smart-pick #{selected.index(cand)+1}: "
                              f"smart={cand['smart_score']:.1f}, {cand.get('confirm_tag', '?')}={cand.get('confirm_score', 0):.3f}, "
                              f"gate={cand['ml_move_prob']:.2f}, gmm_action={cand['gmm_action']}, "
                              f"sector={cand['sector']}, lots={_lot_mult}x"),
                    setup_type=trade_type,
                    ml_data=cand.get('ml_data', {}),
                    sector=cand.get('sector', ''),
                )
                if result and result.get('success'):
                    print(f"   ✅ MODEL-TRACKER PLACED: {cand['sym_clean']} ({direction}) [smart={cand['smart_score']:.1f}]{_type_label}")
                    self._model_tracker_symbols.add(sym)
                    self._model_tracker_trades_today += 1
                    # Track GMM_CONTRARIAN / DR_FLIP trades for daily limit
                    if trade_type in ('DR_FLIP', 'GMM_CONTRARIAN'):
                        self._dr_flip_trades_today += 1
                    placed.append(cand['sym_clean'])
                    self._log_decision(cycle_time, sym, cand['p_score'], f'{trade_type}_PLACED',
                                      reason=(f"Smart pick: score={cand['smart_score']:.1f}, "
                                             f"{cand.get('confirm_tag', '?')}={cand.get('confirm_score', 0):.3f}, gate={cand['ml_move_prob']:.2f}, "
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
          1. GMM down_risk_score < max_updr/downdr_score (regime-proportional gate)
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
        max_dr_up = cfg.get('max_updr_score', 0.10)
        max_dr_down = cfg.get('max_downdr_score', 0.08)
        min_gate = cfg.get('min_gate_prob', 0.55)
        score_tier = cfg.get('score_tier', 'premium')
        
        print(f"\n   🎯 SNIPER SCAN: gates → UPDR<{max_dr_up}/DownDR<{max_dr_down}, smart>={min_smart}, gate>={min_gate} | {self._gmm_sniper_trades_today}/{max_per_day} used")
        
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
            
            # Strict GMM cleanliness gate — BOTH regimes must be clean
            # Predictor now returns independent up_score + down_score
            _snp_up_score = ml.get('ml_up_score', ml.get('ml_down_risk_score', 0.0))
            _snp_down_score = ml.get('ml_down_score', ml.get('ml_down_risk_score', 0.0))
            _snp_up_flag = ml.get('ml_up_flag', False)
            _snp_down_flag = ml.get('ml_down_flag', False)
            if _snp_up_score > max_dr_up or _snp_down_score > max_dr_down:
                _sniper_reject_counts['dr_high'] += 1
                _tag = f"UP={_snp_up_score:.3f}/{max_dr_up}, DN={_snp_down_score:.3f}/{max_dr_down}"
                if (_snp_up_score <= max_dr_up + 0.05) and (_snp_down_score <= max_dr_down + 0.05):  # Near miss
                    _sniper_near_misses.append(f"{sym_diag}({_tag})")
                continue
            if _snp_up_flag or _snp_down_flag:
                _sniper_reject_counts['dr_flagged'] += 1
                _sniper_near_misses.append(f"{sym_diag}(UP_Flag={_snp_up_flag} Down_Flag={_snp_down_flag} UP={_snp_up_score:.3f} DN={_snp_down_score:.3f})")
                continue  # BOTH must be clean (no flags)
            
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
                # FIXED: use gmm_confirms_direction (clean = confirms XGB), not raw dr_flag
                sym_clean_tmp = sym.replace('NSE:', '')
                _sniper_gmm_confirms = ml.get('ml_gmm_confirms_direction', False)
                if _sniper_gmm_confirms:
                    # GMM clean (no anomaly) → confirms XGB direction → check ML_OVERRIDE gates → FLIP
                    _ovr_ok, _ovr_reason = self._ml_override_allowed(sym_clean_tmp, ml, dr_score, path='SNIPER', p_score=pre_scores.get(sym, 0))
                    if not _ovr_ok:
                        print(f"      🚫 SNIPER ML_OVR BLOCKED: {sym_clean_tmp} — {_ovr_reason} ({self._dr_tag(ml)}={dr_score:.3f})")
                        _sniper_reject_counts['xgb_block'] += 1
                        continue
                    old_direction = direction
                    direction = 'BUY' if _xgb_signal == 'UP' else 'SELL'
                    _sniper_was_flipped = True
                    print(f"      🔄 SNIPER ML_OVERRIDE_WGMM: {sym_clean_tmp} — XGB={_xgb_signal} + GMM clean "
                          f"({self._dr_tag(ml)}={dr_score:.4f}) → FLIPPED {old_direction}→{direction}")
                else:
                    # GMM flagged anomaly → hidden risk in XGB's direction too → unreliable
                    # Sniper needs full alignment — block on unconfirmed XGB opposition
                    print(f"      🚫 SNIPER XGB_OPPOSE: {sym_clean_tmp} — XGB={_xgb_signal} vs {direction}, "
                          f"GMM anomaly ({self._dr_tag(ml)}={dr_score:.3f}) → BLOCK")
                    _sniper_reject_counts['xgb_block'] += 1
                    continue
            
            # XGB gate probability floor
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
            if ml_move_prob < min_gate:
                _sniper_reject_counts['gate_low'] += 1
                _sniper_near_misses.append(f"{sym_diag}(gate={ml_move_prob:.2f}<{min_gate}, {self._dr_tag(ml)}={dr_score:.3f})")
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
                              f"GMM {self._dr_tag(ml)}={dr_score:.3f} → BOTH disagree")
                        _sniper_reject_counts['dir_block'] += 1
                        continue
                    else:
                        print(f"      ⚠️ SNIPER XGB CONFLICT: {sym_clean_chk} — XGB={_ml_signal} vs scored={direction}, "
                              f"GMM clean ({self._dr_tag(ml)}={dr_score:.3f}) → −{_dir_conflict_cfg.get('xgb_penalty', 15)} penalty")
            
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
                _sniper_near_misses.append(f"{sym_diag}(smart={smart_score:.1f}<{min_smart}, {self._dr_tag(ml)}={dr_score:.3f}, gate={ml_move_prob:.2f})")
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
                'up_score': _snp_up_score,
                'down_score': _snp_down_score,
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
                        'up_flag': _snp_up_flag,
                        'down_flag': _snp_down_flag,
                        'up_score': _snp_up_score,
                        'down_score': _snp_down_score,
                        'down_risk_score': dr_score,
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', False),
                        'gmm_regime_used': 'BOTH',
                        'gmm_action': 'BOOST',
                    },
                    'scored_direction': direction,
                    'xgb_signal': _xgb_signal,
                },
            })
        
        # --- SNIPER DIAGNOSTIC SUMMARY ---
        _total_syms = len(pre_scores)
        _passed = len(candidates)
        _reject_str = ', '.join(f"{k}={v}" for k, v in _sniper_reject_counts.items() if v > 0)
        print(f"   🎯 SNIPER RESULT: {_passed} candidates from {_total_syms} symbols | Rejected: {_reject_str or 'none'}")
        if candidates:
            for i, c in enumerate(candidates[:5]):
                print(f"      #{i+1} {c['sym_clean']} {c['direction']} | UP={c.get('up_score',0):.4f} DN={c.get('down_score',0):.4f} smart={c['smart_score']:.1f} gate={c['ml_move_prob']:.2f} | {c['sector']}")
        if _sniper_near_misses and not candidates:
            print(f"      Near misses: {' | '.join(_sniper_near_misses[:6])}")
        
        if not candidates:
            return None
        
        # Pick the ONE with lowest max(up_score, down_score) = cleanest from BOTH models
        # Tiebreak by highest smart_score
        candidates.sort(key=lambda c: (c['dr_score'], -c['smart_score']))
        pick = candidates[0]
        
        print(f"\n   🎯 GMM SNIPER: {pick['sym_clean']} ({pick['direction']}) "
              f"| UP={pick.get('up_score',0):.4f} DN={pick.get('down_score',0):.4f} | smart={pick['smart_score']:.1f} "
              f"| gate={pick['ml_move_prob']:.2f} | {pick['sector']} "
              f"| {lot_multiplier}x lots")
        if len(candidates) > 1:
            runner_up = candidates[1]
            print(f"      Runner-up: {runner_up['sym_clean']} (UP={runner_up.get('up_score',0):.4f} DN={runner_up.get('down_score',0):.4f}, smart={runner_up['smart_score']:.1f})")
        
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
                rationale=(f"{_sniper_setup}: UP={pick.get('up_score',0):.4f} DN={pick.get('down_score',0):.4f}, smart={pick['smart_score']:.1f}, "
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
                      f"[{lot_multiplier}x lots] UP={pick.get('up_score',0):.4f} DN={pick.get('down_score',0):.4f}")
                
                self._log_decision(cycle_time, pick['sym'], pick['p_score'], 'GMM_SNIPER_PLACED',
                                  reason=(f"Sniper: UP={pick.get('up_score',0):.4f} DN={pick.get('down_score',0):.4f}, smart={pick['smart_score']:.1f}, "
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

    # ========== TEST_GMM: PURE DR MODEL PLAY ==========
    def _place_test_gmm_trades(self, ml_results: dict, pre_scores: dict, market_data: dict, cycle_time: str):
        """Place trades based on GMM Flag-Confirmed Regime Divergence.
        
        Direction follows the FLAGGING model (not contrarian):
        
          down_flag + HIGH down_score + LOW up_score → confirmed downside → BUY PUT
          up_flag   + HIGH up_score + LOW down_score → confirmed upside   → BUY CALL
        
        The flag IS the directional confirmation. No contrarian flip.
        DOWN model (AUROC=0.62) is more reliable → PUT side gets easier thresholds.
        UP model (AUROC=0.56) is weaker → CALL side needs higher anomaly bar.
        
        Tagged as 'TEST_GMM' for P&L tracking.
        """
        cfg = self._test_gmm_cfg
        if not cfg.get('enabled', False):
            return []
        
        # Reset on new day
        today = datetime.now().date()
        if today != self._test_gmm_date:
            self._test_gmm_trades_today = 0
            self._test_gmm_date = today
            self._test_gmm_symbols = set()
        
        max_per_day = cfg.get('max_trades_per_day', 3)
        budget = max_per_day - self._test_gmm_trades_today
        if budget <= 0:
            return []
        
        # Flag-confirmed divergence thresholds
        dn_min = cfg.get('down_min_score', 0.25)        # DOWN model must be this high to signal PUT
        dn_max_opp = cfg.get('down_max_opposite', 0.10)  # UP model must be this low (clean)
        up_min = cfg.get('up_min_score', 0.22)            # UP model must be this high to signal CALL
        up_max_opp = cfg.get('up_max_opposite', 0.10)    # DOWN model must be this low (clean)
        min_gap = cfg.get('min_divergence_gap', 0.20)     # Strict: signaling vs clean gap ≥ 0.20
        
        # Flag-based conviction gates
        require_signaling_flag = cfg.get('require_signaling_flag', True)
        require_clean_no_flag = cfg.get('require_clean_no_flag', True)
        
        # Quality gates
        require_xgb = cfg.get('require_xgb_agree', True)
        min_gate = cfg.get('min_gate_prob', 0.50)
        max_gate = cfg.get('max_gate_prob', 1.0)          # Cap: high gate = confirmed momentum, bad for contrarian
        max_confidence = cfg.get('max_ml_confidence', 1.0) # Cap: high XGB confidence = fighting real trend
        min_smart = cfg.get('min_smart_score', 45)
        lot_mult = cfg.get('lot_multiplier', 1.5)
        
        # Build candidate pool — regime divergence plays
        candidates = []
        active_syms = {p.get('underlying', '') for p in getattr(self.tools, 'paper_positions', [])
                      if p.get('status', 'OPEN') == 'OPEN'}
        
        for sym in ml_results:
            ml = ml_results[sym]
            dr_score = ml.get('ml_down_risk_score', None)
            if dr_score is None:
                continue
            
            up_score = ml.get('ml_up_score', dr_score)
            down_score = ml.get('ml_down_score', dr_score)
            up_flag = ml.get('ml_up_flag', False)
            down_flag = ml.get('ml_down_flag', False)
            
            # FLAG-CONFIRMED DIVERGENCE CHECK:
            # BUY PUT:  DOWN model HIGH + down_flag confirms + UP clean → go WITH the down signal
            put_ok = (down_score >= dn_min) and (up_score <= dn_max_opp) and ((down_score - up_score) >= min_gap)
            # BUY CALL: UP model HIGH + up_flag confirms + DOWN clean → go WITH the up signal
            call_ok = (up_score >= up_min) and (down_score <= up_max_opp) and ((up_score - down_score) >= min_gap)
            
            if not call_ok and not put_ok:
                continue
            
            # FLAG CONVICTION: signaling model must fire its own calibrated flag
            if require_signaling_flag:
                if put_ok and not down_flag:
                    put_ok = False   # DOWN model score high but didn't breach its own threshold
                if call_ok and not up_flag:
                    call_ok = False  # UP model score high but didn't breach its own threshold
                if not call_ok and not put_ok:
                    continue
            
            # CLEAN SIDE: opposite model must NOT flag (no conflicting signal)
            if require_clean_no_flag:
                if put_ok and up_flag:
                    put_ok = False   # UP model also firing = confused signal, skip
                if call_ok and down_flag:
                    call_ok = False  # DOWN model also firing = confused signal, skip
                if not call_ok and not put_ok:
                    continue
            
            # Determine direction — flag-confirmed (not contrarian)
            if call_ok and not put_ok:
                direction = 'BUY'
                side_tag = 'CALL'
                divergence_score = up_score - down_score
            elif put_ok and not call_ok:
                direction = 'SELL'
                side_tag = 'PUT'
                divergence_score = down_score - up_score
            else:
                # Both pass (rare) — pick stronger divergence
                call_div = up_score - down_score
                put_div = down_score - up_score
                if call_div >= put_div:
                    direction, side_tag, divergence_score = 'BUY', 'CALL', call_div
                else:
                    direction, side_tag, divergence_score = 'SELL', 'PUT', put_div
            
            # Gate: XGB P(MOVE) floor & ceiling
            ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
            if ml_move_prob < min_gate:
                continue
            if max_gate < 1.0 and ml_move_prob > max_gate:
                continue  # Too-high gate = confirmed momentum, contrarian divergence play fails
            
            # Gate: XGB confidence ceiling — high confidence = fighting real trend
            ml_conf = ml.get('ml_confidence', 0.0)
            if max_confidence < 1.0 and ml_conf > max_confidence:
                continue
            
            # Gate: XGB direction agreement (or neutral)
            if require_xgb:
                xgb_hint = ml.get('ml_direction_hint', 'NEUTRAL')
                prob_up = ml.get('ml_prob_up', 0.33)
                prob_down = ml.get('ml_prob_down', 0.33)
                # Hard block: XGB strongly opposes our divergence direction
                if direction == 'BUY' and xgb_hint == 'DOWN':
                    continue
                if direction == 'SELL' and xgb_hint == 'UP':
                    continue
                # Soft block: XGB leans opposite by 30%+ probability margin
                if direction == 'BUY' and prob_down > prob_up * 1.3:
                    continue
                if direction == 'SELL' and prob_up > prob_down * 1.3:
                    continue
            
            # Gate: minimum smart score (low bar — divergence IS the signal)
            p_score = pre_scores.get(sym, 0)
            if min_smart > 0 and p_score < min_smart:
                continue
            
            sym_clean = sym.replace('NSE:', '')
            
            # Skip if already traded today (any path)
            if sym in self._test_gmm_symbols:
                continue
            if sym in self._model_tracker_symbols or sym in self._gmm_sniper_symbols:
                continue
            if sym in active_syms:
                continue
            
            candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'side_tag': side_tag,
                'up_score': up_score,
                'down_score': down_score,
                'divergence_score': divergence_score,
                'dr_score': dr_score,
                'gmm_regime': ml.get('ml_gmm_regime_used', 'UP'),
                'ml_move_prob': ml_move_prob,
                'p_score': p_score,
                'ml_data': {
                    'smart_score': p_score,
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
                        'up_flag': ml.get('ml_up_flag', False),
                        'down_flag': ml.get('ml_down_flag', False),
                        'up_score': up_score,
                        'down_score': down_score,
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', True),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', 'BOTH'),
                        'gmm_action': f'TEST_GMM_DIVERGE_{side_tag}',
                        'divergence_score': round(divergence_score, 4),
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': False,
                },
            })
        
        # ── Diagnostic logging: always show status ──
        _total_syms = len(ml_results) if ml_results else 0
        if not candidates:
            print(f"   🧪 TEST_GMM: 0 candidates from {_total_syms} symbols "
                  f"| thresholds: dn≥{dn_min}/up≥{up_min} opp≤{dn_max_opp} gap≥{min_gap} "
                  f"flag={require_signaling_flag} budget={budget}")
            return []
        
        print(f"   🧪 TEST_GMM: {len(candidates)} candidates from {_total_syms} symbols | budget={budget}")
        
        # Sort by divergence strength (largest gap → highest priority)
        candidates.sort(key=lambda c: c['divergence_score'], reverse=True)
        
        # Take up to budget
        selected = candidates[:budget]
        
        # Place trades
        placed = []
        for cand in selected:
            _dir_label = 'BUY(CALL)' if cand['direction'] == 'BUY' else 'SELL(PUT)'
            try:
                result = self.tools.place_option_order(
                    underlying=cand['sym'],
                    direction=cand['direction'],
                    strike_selection="ATM",
                    use_intraday_scoring=False,
                    lot_multiplier=lot_mult,
                    rationale=(f"TEST_GMM_DIVERGE_{cand['side_tag']}: UP={cand['up_score']:.3f} "
                              f"DN={cand['down_score']:.3f} gap={cand['divergence_score']:.3f} "
                              f"gate={cand['ml_move_prob']:.2f} — flag-confirmed direction"),
                    setup_type='TEST_GMM',
                    ml_data=cand.get('ml_data', {}),
                    sector='',
                )
                if result and result.get('success'):
                    self._test_gmm_trades_today += 1
                    self._test_gmm_symbols.add(cand['sym'])
                    placed.append(cand['sym_clean'])
                    self._log_decision(cycle_time, cand['sym'], cand['p_score'], 'TEST_GMM_PLACED',
                                      reason=(f"Diverge {cand['side_tag']}: UP={cand['up_score']:.3f} "
                                              f"DN={cand['down_score']:.3f} gap={cand['divergence_score']:.3f}"),
                                      direction=cand['direction'], setup='TEST_GMM')
                else:
                    error = result.get('error', 'unknown') if result else 'no result'
                    print(f"   ⚠️ TEST_GMM failed: {cand['sym_clean']}: {error}")
            except Exception as e:
                print(f"   ❌ TEST_GMM error: {cand['sym_clean']}: {e}")
        
        if placed:
            remaining = max_per_day - self._test_gmm_trades_today
        
        return placed

    def _place_test_xgb_trades(self, ml_results: dict, pre_scores: dict, market_data: dict, cycle_time: str):
        """Place trades purely based on XGBoost model — bypass GMM, smart_score, all other gates.
        
        Direction from XGB directional probabilities (prob_up vs prob_down).
        Conviction from gate model P(MOVE).
        No GMM/DR gating, no smart_score, no IntradayScorer direction.
        Pure XGB conviction play for testing XGB accuracy in isolation.
        
        Tagged as 'TEST_XGB' for P&L tracking.
        """
        cfg = self._test_xgb_cfg
        if not cfg.get('enabled', False):
            return []
        
        # Reset on new day
        today = datetime.now().date()
        if today != self._test_xgb_date:
            self._test_xgb_trades_today = 0
            self._test_xgb_date = today
            self._test_xgb_symbols = set()
        
        max_per_day = cfg.get('max_trades_per_day', 3)
        budget = max_per_day - self._test_xgb_trades_today
        if budget <= 0:
            return []
        
        # XGB conviction thresholds
        min_move = cfg.get('min_move_prob', 0.65)
        min_dir_prob = cfg.get('min_directional_prob', 0.48)
        min_dir_margin = cfg.get('min_directional_margin', 0.15)
        min_conf = cfg.get('min_ml_confidence', 0.60)
        lot_mult = cfg.get('lot_multiplier', 1.0)
        
        # ── SAFETY: Respect herd detection + market/sector breadth ──
        # TEST_XGB is "pure XGB" but must NOT fight the market OR sector.
        # Priority: sector trend > market breadth (sector is more specific).
        # If stock's sector is bullish but market is bearish, follow sector.
        # If stock has no sector mapping, fall back to market breadth.
        _breadth = getattr(self, '_last_market_breadth', 'MIXED')
        _sector_changes = getattr(self, '_sector_index_changes_cache', {})
        _stock_to_sector = getattr(self, '_stock_to_sector', {})
        
        # Build candidate pool — pure XGB plays
        candidates = []
        active_syms = {p.get('underlying', '') for p in getattr(self.tools, 'paper_positions', [])
                      if p.get('status', 'OPEN') == 'OPEN'}
        _xgb_skipped_herd = 0
        _xgb_skipped_breadth = 0
        _xgb_sector_override = 0
        
        for sym in ml_results:
            ml = ml_results[sym]
            
            # Gate: HERD — skip stocks explicitly flattened (Tier 3 low-conf during conflict)
            # ml_herd_survived=True → sector-aligned or high-confidence → LET THROUGH
            # ml_herd_caution=True → low-conf, boost reduced → LET THROUGH (sector gate below filters)
            # ml_herd_flattened=True → legacy hard-kill (no longer set, kept for safety)
            if ml.get('ml_herd_flattened', False):
                _xgb_skipped_herd += 1
                continue
            
            # Gate: P(MOVE) must be strong
            ml_move_prob = round(ml.get('ml_move_prob', ml.get('ml_p_move', 0.0)), 4)
            if ml_move_prob < min_move:
                continue
            
            # Gate: XGB confidence floor — model must be sure, not just sensing volatility
            ml_conf = ml.get('ml_confidence', 0.0)
            if ml_conf < min_conf:
                continue
            
            # Direction: from prob_up vs prob_down
            prob_up = ml.get('ml_prob_up', 0.33)
            prob_down = ml.get('ml_prob_down', 0.33)
            margin = abs(prob_up - prob_down)
            
            # Must have clear directional lean
            if margin < min_dir_margin:
                continue
            
            if prob_up > prob_down and prob_up >= min_dir_prob:
                direction = 'BUY'
                side_tag = 'CALL'
                conviction = prob_up
            elif prob_down > prob_up and prob_down >= min_dir_prob:
                direction = 'SELL'
                side_tag = 'PUT'
                conviction = prob_down
            else:
                continue
            
            # ── Gate: SECTOR-AWARE BREADTH CONFLICT ──
            # Step 1: Check if stock belongs to a known sector
            # Step 2: If sector has a clear trend (>= +1% or <= -1%), use sector trend
            # Step 3: If no sector data, fall back to overall market breadth
            # Logic: Don't buy PEs in a bullish environment, don't buy CEs in bearish
            #        "environment" = sector if available, else market breadth
            sym_clean = sym.replace('NSE:', '')
            _sec_match = _stock_to_sector.get(sym_clean)  # (sector_name, index_symbol)
            _effective_trend = None  # 'BULLISH', 'BEARISH', or None (no opinion)
            _trend_source = 'MARKET'
            _sec_chg_val = None
            
            if _sec_match and _sector_changes:
                _sec_name, _sec_index = _sec_match
                _sec_chg_val = _sector_changes.get(_sec_index)
                if _sec_chg_val is not None:
                    if _sec_chg_val >= 1.0:
                        _effective_trend = 'BULLISH'
                        _trend_source = f'SECTOR({_sec_name}:{_sec_chg_val:+.1f}%)'
                    elif _sec_chg_val <= -1.0:
                        _effective_trend = 'BEARISH'
                        _trend_source = f'SECTOR({_sec_name}:{_sec_chg_val:+.1f}%)'
                    # else: sector is flat (-1% to +1%), fall through to market breadth
            
            # Fall back to market breadth if sector gave no clear signal
            if _effective_trend is None and _breadth in ('BULLISH', 'BEARISH'):
                _effective_trend = _breadth
                _trend_source = f'MARKET({_breadth})'
            
            # Block if direction contradicts the effective trend
            # SELL (PE) in BULLISH trend = fighting the trend → skip
            # BUY (CE) in BEARISH trend = fighting the trend → skip
            if _effective_trend:
                _conflicts = (
                    (direction == 'SELL' and _effective_trend == 'BULLISH') or
                    (direction == 'BUY' and _effective_trend == 'BEARISH')
                )
                if _conflicts:
                    _xgb_skipped_breadth += 1
                    continue
                # Track when sector OVERRODE market breadth (allowed a trade market would block)
                if _sec_match and _trend_source.startswith('SECTOR') and _breadth in ('BULLISH', 'BEARISH'):
                    _market_would_block = (
                        (direction == 'SELL' and _breadth == 'BULLISH') or
                        (direction == 'BUY' and _breadth == 'BEARISH')
                    )
                    if _market_would_block:
                        _xgb_sector_override += 1
            
            # Skip if already traded today (any path)
            if sym in self._test_xgb_symbols:
                continue
            if sym in self._model_tracker_symbols or sym in self._gmm_sniper_symbols:
                continue
            if sym in self._test_gmm_symbols:
                continue
            if sym in active_syms:
                continue
            
            dr_score = ml.get('ml_down_risk_score', 0)
            p_score = pre_scores.get(sym, 0)
            
            # Sector info for this candidate (for rationale + logging)
            _cand_sector = _sec_match[0] if _sec_match else ''
            _cand_sec_chg = _sec_chg_val if _sec_chg_val is not None else 0
            _cand_trend_src = _trend_source
            
            candidates.append({
                'sym': sym,
                'sym_clean': sym_clean,
                'direction': direction,
                'side_tag': side_tag,
                'ml_move_prob': ml_move_prob,
                'prob_up': prob_up,
                'prob_down': prob_down,
                'conviction': conviction,
                'margin': margin,
                'dr_score': dr_score,
                'p_score': p_score,
                'sector': _cand_sector,
                'sector_chg': _cand_sec_chg,
                'trend_source': _cand_trend_src,
                'ml_data': {
                    'smart_score': 0,  # Not used — pure XGB play
                    'p_score': p_score,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'ml_confidence': ml.get('ml_confidence', 0),
                    'xgb_model': {
                        'signal': ml.get('ml_signal', 'UNKNOWN'),
                        'move_prob': ml_move_prob,
                        'prob_up': prob_up,
                        'prob_down': prob_down,
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
                        'up_flag': ml.get('ml_up_flag', False),
                        'down_flag': ml.get('ml_down_flag', False),
                        'up_score': ml.get('ml_up_score', dr_score),
                        'down_score': ml.get('ml_down_score', dr_score),
                        'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                        'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', True),
                        'gmm_regime_used': ml.get('ml_gmm_regime_used', 'BOTH'),
                        'gmm_action': f'TEST_XGB_{side_tag}',
                    },
                    'scored_direction': direction,
                    'xgb_disagrees': False,
                },
            })
        
        # ── Log gate summary ──
        _total_xgb_syms = len(ml_results) if ml_results else 0
        if _xgb_skipped_herd or _xgb_skipped_breadth or _xgb_sector_override:
            _gate_parts = []
            if _xgb_skipped_herd:
                _gate_parts.append(f'herd_blocked={_xgb_skipped_herd}')
            if _xgb_skipped_breadth:
                _gate_parts.append(f'breadth_blocked={_xgb_skipped_breadth}(mkt={_breadth})')
            if _xgb_sector_override:
                _gate_parts.append(f'sector_overrides={_xgb_sector_override}')
            print(f"   \U0001f6e1 TEST_XGB GATES: {' | '.join(_gate_parts)} → {len(candidates)} candidates remaining")
        
        if not candidates:
            print(f"   🧪 TEST_XGB: 0 candidates from {_total_xgb_syms} symbols "
                  f"| thresholds: P(move)≥{min_move} dir≥{min_dir_prob} margin≥{min_dir_margin} conf≥{min_conf} "
                  f"breadth={_breadth} budget={budget}")
            return []
        
        print(f"   🧪 TEST_XGB: {len(candidates)} candidates from {_total_xgb_syms} symbols | budget={budget}")
        
        # Sort by XGB conviction: highest P(MOVE) * directional_margin first
        candidates.sort(key=lambda c: c['ml_move_prob'] * c['margin'], reverse=True)
        
        # Take up to budget
        selected = candidates[:budget]
        
        # Place trades
        placed = []
        for cand in selected:
            _dir_label = 'BUY(CALL)' if cand['direction'] == 'BUY' else 'SELL(PUT)'
            try:
                # Build rationale string with sector info
                _sec_part = ''
                if cand.get('sector'):
                    _sc = cand['sector_chg']
                    _sec_part = f" | {cand['sector']}:{_sc:+.1f}%"
                _xgb_rationale = (
                    f"TEST_XGB_{cand['side_tag']}: P(MOVE)={cand['ml_move_prob']:.2f} "
                    f"P(UP)={cand['prob_up']:.2f} P(DN)={cand['prob_down']:.2f} "
                    f"margin={cand['margin']:.2f}{_sec_part} [{cand.get('trend_source', 'MARKET')}]"
                )
                # Pass pre-fetched market_data so IV crush gate can compute RV (ATR-based)
                _sym_md = market_data.get(cand['sym'], {})
                if not _sym_md:
                    _sym_md = market_data.get(cand['sym_clean'], {})
                result = self.tools.place_option_order(
                    underlying=cand['sym'],
                    direction=cand['direction'],
                    strike_selection="ATM",
                    use_intraday_scoring=False,
                    lot_multiplier=lot_mult,
                    rationale=_xgb_rationale,
                    setup_type='TEST_XGB',
                    ml_data=cand.get('ml_data', {}),
                    sector=cand.get('sector', ''),
                    pre_fetched_market_data=_sym_md if _sym_md else None,
                )
                if result and result.get('success'):
                    self._test_xgb_trades_today += 1
                    self._test_xgb_symbols.add(cand['sym'])
                    placed.append(cand['sym_clean'])
                    self._log_decision(cycle_time, cand['sym'], cand['p_score'], 'TEST_XGB_PLACED',
                                      reason=(f"XGB {cand['side_tag']}: P(MOVE)={cand['ml_move_prob']:.2f} "
                                              f"margin={cand['margin']:.2f}"),
                                      direction=cand['direction'], setup='TEST_XGB')
                else:
                    error = result.get('error', 'unknown') if result else 'no result'
                    print(f"   \u26a0\ufe0f TEST_XGB failed: {cand['sym_clean']}: {error}")
            except Exception as e:
                print(f"   \u274c TEST_XGB error: {cand['sym_clean']}: {e}")
        
        if placed:
            remaining = max_per_day - self._test_xgb_trades_today
        
        return placed

    # ========== ARBTR: SECTOR ARBITRAGE — LAGGARD CONVERGENCE ==========
    def _place_arbtr_trades(self, ml_results: dict, pre_scores: dict,
                            market_data: dict, sector_index_changes: dict,
                            cycle_time: str):
        """Sector Arbitrage strategy — trade laggard stocks within a sector.

        Logic:
        1. For each sector, check if the sector index moved >= min threshold.
        2. Count how many sector stocks "aligned" (moved ≥50% of index move, same direction).
        3. If enough alignment (breadth), find the laggards: stocks that barely moved.
        4. Trade laggards in the sector's direction, expecting convergence.
        5. All trades gated by ML, volume, chop, HTF, GMM, and smart_score.
        
        Tagged as 'ARBTR' for P&L tracking.
        """
        cfg = self._arbtr_cfg
        if not cfg.get('enabled', False):
            return []

        # Daily reset
        today = datetime.now().date()
        if today != self._arbtr_date:
            self._arbtr_trades_today = 0
            self._arbtr_date = today
            self._arbtr_symbols = set()
            self._arbtr_sector_cooldowns = {}

        max_per_day = cfg.get('max_trades_per_day', 6)
        budget = max_per_day - self._arbtr_trades_today
        if budget <= 0:
            return []

        # Timing window
        now = datetime.now()
        now_str = now.strftime('%H:%M')
        earliest = cfg.get('earliest_entry', '09:45')
        latest = cfg.get('no_entry_after', '14:00')
        if now_str < earliest or now_str > latest:
            return []

        # ── ARBTR diagnostic: log every 10th cycle (reduce noise) ──
        _arbtr_diag_counter = getattr(self, '_arbtr_diag_counter', 0) + 1
        self._arbtr_diag_counter = _arbtr_diag_counter
        _arbtr_diag = (_arbtr_diag_counter % 10 == 1)  # First call + every 10th

        # Diagnostic: show sector index data availability
        if _arbtr_diag:
            if sector_index_changes:
                _sec_strs = [f"{k.replace('NSE:NIFTY ', '')}:{v:+.1f}%" for k, v in sector_index_changes.items()]
                print(f"   \U0001F504 ARBTR scan: sectors=[{', '.join(_sec_strs)}] budget={budget}")
            else:
                print(f"   \U0001F504 ARBTR scan: NO sector index data (empty dict) — check get_quote_batch for index symbols")

        # Config thresholds
        min_sector_move = cfg.get('min_sector_move_pct', 1.5)
        min_aligned_ratio = cfg.get('min_sector_stocks_aligned', 0.60)
        max_laggard_move = cfg.get('max_laggard_move_pct', 0.50)
        min_divergence = cfg.get('min_divergence_pct', 1.0)
        max_divergence = cfg.get('max_divergence_pct', 5.0)
        min_volume_ratio = cfg.get('min_volume_ratio', 0.8)
        min_ml_move = cfg.get('min_ml_move_prob', 0.40)
        min_ml_conf = cfg.get('min_ml_confidence', 0.45)
        max_ml_flat = cfg.get('max_ml_flat_prob', 0.55)
        require_no_chop = cfg.get('require_no_chop_zone', True)
        require_htf_not_opposed = cfg.get('require_htf_not_opposed', True)
        min_smart = cfg.get('min_smart_score', 35)
        use_gmm_veto = cfg.get('use_gmm_veto', True)
        max_dr = cfg.get('max_dr_score', 0.25)
        lot_mult = cfg.get('lot_multiplier', 1.0)
        cooldown_mins = cfg.get('cooldown_per_sector_minutes', 15)
        max_simul = cfg.get('max_simultaneous_arbtr', 3)

        # Count current ARBTR open positions
        _open_arbtr = sum(1 for p in getattr(self.tools, 'paper_positions', [])
                         if p.get('status', 'OPEN') == 'OPEN' and p.get('setup_type') == 'ARBTR')
        if _open_arbtr >= max_simul:
            return []

        # Active symbols (any strategy)
        active_syms = {p.get('underlying', '') for p in getattr(self.tools, 'paper_positions', [])
                      if p.get('status', 'OPEN') == 'OPEN'}

        # ── ARBTR data enrichment: use ticker as fallback for sector stocks ──
        # market_data only has ~40 stocks; sector maps need ~87.
        # Use WebSocket quote cache to fill gaps (zero API cost).
        _arbtr_market = dict(market_data)  # shallow copy, don't pollute original
        _missing_syms = []
        for _sec_info in self._arbtr_sector_map.values():
            for _stk in _sec_info.get('stocks', []):
                _sk = f'NSE:{_stk}'
                if _sk not in _arbtr_market and _stk not in _arbtr_market:
                    _missing_syms.append(_sk)
        if _missing_syms and self.tools.ticker:
            try:
                _ws_quotes = self.tools.ticker.get_quote_batch(_missing_syms)
                _filled = 0
                for _mk_sym, _mk_q in _ws_quotes.items():
                    if _mk_q:
                        _ohlc = _mk_q.get('ohlc', {})
                        _prev = _ohlc.get('close', 0)
                        _ltp = _mk_q.get('last_price', 0)
                        if _prev > 0 and _ltp > 0:
                            _chg_pct = ((_ltp - _prev) / _prev) * 100
                            _arbtr_market[_mk_sym] = {
                                'last_price': _ltp, 'change_pct': _chg_pct,
                                'ohlc': _ohlc, 'volume': _mk_q.get('volume', 0),
                                '_arbtr_ws_fill': True,  # marker for diagnostic
                            }
                            _filled += 1
                if _arbtr_diag:
                    print(f"   \U0001F504 ARBTR data: {_filled}/{len(_missing_syms)} missing stocks filled from ticker cache")
            except Exception as _e:
                if _arbtr_diag:
                    print(f"   \U0001F504 ARBTR data fill error: {_e}")

        # Reverse map: index symbol → sector name
        _idx_to_sector = {}
        for _sec_name, _sec_info in self._arbtr_sector_map.items():
            _idx_to_sector[_sec_info['index']] = _sec_name

        candidates = []

        for sector_name, sec_info in self._arbtr_sector_map.items():
            idx_symbol = sec_info['index']
            stocks = sec_info.get('stocks', [])
            if not stocks:
                continue

            # 1. Check if sector index has moved enough
            sec_change = sector_index_changes.get(idx_symbol, 0)
            if abs(sec_change) < min_sector_move:
                if _arbtr_diag and abs(sec_change) >= 0.5:
                    print(f"      ARBTR {sector_name}: {sec_change:+.1f}% < {min_sector_move}% threshold (skip)")
                continue

            if _arbtr_diag:
                print(f"      ARBTR {sector_name}: {sec_change:+.1f}% \u2265 {min_sector_move}% \u2713 checking laggards...")

            sector_direction = 'BUY' if sec_change > 0 else 'SELL'  # Trade laggard in same direction

            # Sector cooldown check
            last_entry = self._arbtr_sector_cooldowns.get(sector_name)
            if last_entry and (now - last_entry).total_seconds() < cooldown_mins * 60:
                if _arbtr_diag:
                    print(f"      ARBTR {sector_name}: on cooldown (skip)")
                continue

            # 2. Count aligned stocks in this sector
            aligned_count = 0
            total_checked = 0
            for stk in stocks:
                stk_sym = f'NSE:{stk}'
                stk_data = _arbtr_market.get(stk_sym, _arbtr_market.get(stk, {}))
                if not stk_data:
                    continue
                total_checked += 1
                stk_change = stk_data.get('change_pct', 0) if isinstance(stk_data, dict) else 0
                # Stock is "aligned" if it moved in same direction as sector, at least 50% of sector move
                if sec_change > 0 and stk_change >= sec_change * 0.5:
                    aligned_count += 1
                elif sec_change < 0 and stk_change <= sec_change * 0.5:
                    aligned_count += 1

            if total_checked < 3:
                if _arbtr_diag:
                    print(f"      ARBTR {sector_name}: only {total_checked} stocks with data (need 3+, skip)")
                continue  # Not enough data to judge breadth
            alignment_ratio = aligned_count / total_checked
            if alignment_ratio < min_aligned_ratio:
                if _arbtr_diag:
                    print(f"      ARBTR {sector_name}: alignment {aligned_count}/{total_checked}={alignment_ratio:.0%} < {min_aligned_ratio:.0%} (skip)")
                continue  # Sector not broadly aligned — don't trade laggards

            # 3. Find laggards — stocks that barely moved despite sector moving
            _laggard_rejects = {}   # reason → count (for diagnostics)
            for stk in stocks:
                stk_sym = f'NSE:{stk}'
                stk_data = _arbtr_market.get(stk_sym, _arbtr_market.get(stk, {}))
                if not stk_data or not isinstance(stk_data, dict):
                    _laggard_rejects['no_data'] = _laggard_rejects.get('no_data', 0) + 1
                    continue

                stk_change = stk_data.get('change_pct', 0)

                # Laggard: moved less than threshold, and within max divergence from sector
                if abs(stk_change) > max_laggard_move:
                    _laggard_rejects['moved_too_much'] = _laggard_rejects.get('moved_too_much', 0) + 1
                    continue  # Stock already moved significantly — not a laggard
                # Check for opposing movers: laggard should not be moving AGAINST sector
                if sec_change > 0 and stk_change < -max_laggard_move:
                    _laggard_rejects['counter_move'] = _laggard_rejects.get('counter_move', 0) + 1
                    continue  # Counter-moving — might have stock-specific bad news, skip
                if sec_change < 0 and stk_change > max_laggard_move:
                    _laggard_rejects['counter_move'] = _laggard_rejects.get('counter_move', 0) + 1
                    continue  # Counter-moving upward

                divergence = abs(sec_change) - abs(stk_change)
                if divergence < min_divergence:
                    _laggard_rejects['low_divergence'] = _laggard_rejects.get('low_divergence', 0) + 1
                    continue  # Not enough gap
                if divergence > max_divergence:
                    _laggard_rejects['high_divergence'] = _laggard_rejects.get('high_divergence', 0) + 1
                    continue  # Too large — might be stock-specific reason

                # === QUALITY GATES ===

                # Gate: Skip already traded symbols
                if stk_sym in self._arbtr_symbols:
                    _laggard_rejects['already_traded'] = _laggard_rejects.get('already_traded', 0) + 1
                    continue
                if stk_sym in active_syms:
                    _laggard_rejects['active_pos'] = _laggard_rejects.get('active_pos', 0) + 1
                    continue
                if stk_sym in self._model_tracker_symbols:
                    _laggard_rejects['model_tracker'] = _laggard_rejects.get('model_tracker', 0) + 1
                    continue
                if stk_sym in self._test_gmm_symbols:
                    _laggard_rejects['test_gmm'] = _laggard_rejects.get('test_gmm', 0) + 1
                    continue
                if stk_sym in self._test_xgb_symbols:
                    _laggard_rejects['test_xgb'] = _laggard_rejects.get('test_xgb', 0) + 1
                    continue
                if stk_sym in self._gmm_sniper_symbols:
                    _laggard_rejects['gmm_sniper'] = _laggard_rejects.get('gmm_sniper', 0) + 1
                    continue

                # Gate: Volume regime
                vol_regime = stk_data.get('volume_regime', 'NORMAL')
                if min_volume_ratio > 0 and vol_regime in ('LOW', 'DEAD'):
                    _laggard_rejects['low_volume'] = _laggard_rejects.get('low_volume', 0) + 1
                    continue  # Low volume = no conviction to converge

                # Gate: Chop zone
                if require_no_chop and stk_data.get('chop_zone', False):
                    _laggard_rejects['chop_zone'] = _laggard_rejects.get('chop_zone', 0) + 1
                    continue

                # Gate: HTF not opposed
                if require_htf_not_opposed:
                    htf = stk_data.get('htf_trend', 'NEUTRAL')
                    if sector_direction == 'BUY' and htf == 'BEARISH':
                        _laggard_rejects['htf_opposed'] = _laggard_rejects.get('htf_opposed', 0) + 1
                        continue
                    if sector_direction == 'SELL' and htf == 'BULLISH':
                        _laggard_rejects['htf_opposed'] = _laggard_rejects.get('htf_opposed', 0) + 1
                        continue

                # Gate: ML results (skipped when require_ml_move_signal=False)
                _require_ml = cfg.get('require_ml_move_signal', True)
                ml = ml_results.get(stk_sym, {})
                if _require_ml:
                    if not ml:
                        _laggard_rejects['no_ml'] = _laggard_rejects.get('no_ml', 0) + 1
                        continue

                    ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0))
                    if ml_move_prob < min_ml_move:
                        _laggard_rejects['ml_move_low'] = _laggard_rejects.get('ml_move_low', 0) + 1
                        continue

                    ml_conf = ml.get('ml_confidence', 0.0)
                    if ml_conf < min_ml_conf:
                        _laggard_rejects['ml_conf_low'] = _laggard_rejects.get('ml_conf_low', 0) + 1
                        continue

                    ml_flat = ml.get('ml_prob_flat', 0.5)
                    if ml_flat > max_ml_flat:
                        _laggard_rejects['ml_flat_high'] = _laggard_rejects.get('ml_flat_high', 0) + 1
                        continue

                # Gate: GMM down-risk veto (only when ML data available)
                dr_score = ml.get('ml_down_risk_score', ml.get('dr_score', 0.15)) if ml else 0.15
                if use_gmm_veto and ml and dr_score > max_dr:
                    _laggard_rejects['gmm_dr_high'] = _laggard_rejects.get('gmm_dr_high', 0) + 1
                    continue

                # Gate: Smart score floor (only when pre_scores available for this stock)
                p_score = pre_scores.get(stk_sym, 0)
                if p_score > 0 and p_score < min_smart:
                    _laggard_rejects['score_low'] = _laggard_rejects.get('score_low', 0) + 1
                    continue

                # Gate: ML direction should not strongly oppose sector direction (only with ML)
                if _require_ml and ml:
                    xgb_hint = ml.get('ml_direction_hint', 'NEUTRAL')
                    if sector_direction == 'BUY' and xgb_hint == 'DOWN':
                        prob_down = ml.get('ml_prob_down', 0.33)
                        prob_up = ml.get('ml_prob_up', 0.33)
                        if prob_down > prob_up * 1.3:
                            _laggard_rejects['xgb_opposed'] = _laggard_rejects.get('xgb_opposed', 0) + 1
                            continue  # XGB strongly bearish — don't fight it
                    if sector_direction == 'SELL' and xgb_hint == 'UP':
                        prob_up = ml.get('ml_prob_up', 0.33)
                        prob_down = ml.get('ml_prob_down', 0.33)
                        if prob_up > prob_down * 1.3:
                            _laggard_rejects['xgb_opposed'] = _laggard_rejects.get('xgb_opposed', 0) + 1
                            continue  # XGB strongly bullish — don't fight it

                # Build candidate
                side_tag = 'CALL' if sector_direction == 'BUY' else 'PUT'
                ml_move_prob = ml.get('ml_move_prob', ml.get('ml_p_move', 0.0)) if ml else 0.0
                ml_conf = ml.get('ml_confidence', 0.0) if ml else 0.0
                candidates.append({
                    'sym': stk_sym,
                    'sym_clean': stk,
                    'sector': sector_name,
                    'direction': sector_direction,
                    'side_tag': side_tag,
                    'sec_change': sec_change,
                    'stk_change': stk_change,
                    'divergence': divergence,
                    'alignment_ratio': alignment_ratio,
                    'dr_score': dr_score,
                    'ml_move_prob': ml_move_prob,
                    'p_score': p_score,
                    'ml_data': {
                        'smart_score': p_score,
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
                            'up_flag': ml.get('ml_up_flag', False),
                            'down_flag': ml.get('ml_down_flag', False),
                            'up_score': ml.get('ml_up_score', 0),
                            'down_score': ml.get('ml_down_score', 0),
                            'down_risk_bucket': ml.get('ml_down_risk_bucket', 'LOW'),
                            'gmm_confirms_direction': ml.get('ml_gmm_confirms_direction', True),
                            'gmm_regime_used': ml.get('ml_gmm_regime_used', 'BOTH'),
                            'gmm_action': f'ARBTR_{side_tag}',
                            'divergence_score': round(divergence, 4),
                        },
                        'scored_direction': sector_direction,
                        'xgb_disagrees': False,
                        'arbtr_meta': {
                            'sector': sector_name,
                            'sector_index': idx_symbol,
                            'sector_change_pct': round(sec_change, 2),
                            'stock_change_pct': round(stk_change, 2),
                            'divergence_pct': round(divergence, 2),
                            'alignment_ratio': round(alignment_ratio, 2),
                        },
                    },
                })

            # Diagnostic: per-sector laggard rejection summary
            if _arbtr_diag and _laggard_rejects:
                _rej_str = ', '.join(f"{k}={v}" for k, v in sorted(_laggard_rejects.items(), key=lambda x: -x[1]))
                print(f"      ARBTR {sector_name} laggard rejects: {_rej_str}")

        if not candidates:
            return []

        # Sort by divergence (biggest gap = strongest convergence opportunity)
        candidates.sort(key=lambda c: c['divergence'], reverse=True)

        # Take up to remaining budget
        selected = candidates[:budget]

        # Place trades
        placed = []
        for cand in selected:
            # Respect simultaneous limit mid-loop
            _cur_open = _open_arbtr + len(placed)
            if _cur_open >= max_simul:
                break

            _dir_label = f"{cand['direction']}({cand['side_tag']})"
            try:
                _sym_md = _arbtr_market.get(cand['sym'], _arbtr_market.get(cand['sym_clean'], {}))
                result = self.tools.place_option_order(
                    underlying=cand['sym'],
                    direction=cand['direction'],
                    strike_selection="ATM",
                    use_intraday_scoring=False,
                    lot_multiplier=lot_mult,
                    rationale=(f"ARBTR_{cand['sector']}_{cand['side_tag']}: "
                              f"Sector {cand['sec_change']:+.1f}% vs Stock {cand['stk_change']:+.1f}% "
                              f"gap={cand['divergence']:.1f}% align={cand['alignment_ratio']:.0%} "
                              f"move={cand['ml_move_prob']:.2f} score={cand['p_score']:.0f}"),
                    setup_type='ARBTR',
                    ml_data=cand.get('ml_data', {}),
                    sector=cand['sector'],
                    pre_fetched_market_data=_sym_md if _sym_md else None,
                )
                if result and result.get('success'):
                    self._arbtr_trades_today += 1
                    self._arbtr_symbols.add(cand['sym'])
                    self._arbtr_sector_cooldowns[cand['sector']] = now
                    placed.append(cand['sym_clean'])
                    self._log_decision(cycle_time, cand['sym'], cand['p_score'], 'ARBTR_PLACED',
                                      reason=(f"Sector={cand['sector']} idx={cand['sec_change']:+.1f}% "
                                              f"stk={cand['stk_change']:+.1f}% gap={cand['divergence']:.1f}%"),
                                      direction=cand['direction'], setup='ARBTR')
                    print(f"\n   🔄 ARBTR: {cand['sym_clean']} {_dir_label} | "
                          f"Sector {cand['sector']} {cand['sec_change']:+.1f}% → Stock {cand['stk_change']:+.1f}% "
                          f"| Gap {cand['divergence']:.1f}% | Align {cand['alignment_ratio']:.0%} "
                          f"| ML={cand['ml_move_prob']:.2f} Score={cand['p_score']:.0f}")
                else:
                    error = result.get('error', 'unknown') if result else 'no result'
                    print(f"   ⚠️ ARBTR failed: {cand['sym_clean']}: {error}")
            except Exception as e:
                print(f"   ❌ ARBTR error: {cand['sym_clean']}: {e}")

        if placed:
            remaining = max_per_day - self._arbtr_trades_today
            print(f"   📊 ARBTR placed {len(placed)}: {', '.join(placed)} | {remaining}/{max_per_day} remaining today")

        return placed

    # ═══════════════════════════════════════════════════════════════════════════
    # VIX REGIME — Fetch India VIX & determine current regime
    # Cached for `cache_seconds` to avoid API spam. Falls back to default on error.
    # ═══════════════════════════════════════════════════════════════════════════
    def _fetch_india_vix(self):
        """Fetch India VIX and update self._current_vix / self._vix_regime.
        Uses Kite LTP with a cache to avoid hitting API every 3 seconds.
        """
        if not self._vix_cfg.get('enabled', False):
            return self._current_vix

        import time as _t
        cache_s = self._vix_cfg.get('cache_seconds', 120)
        now_ts = _t.time()
        if now_ts - self._vix_last_fetch < cache_s:
            return self._current_vix  # Use cached value

        try:
            vix_key = self._vix_cfg.get('vix_instrument', 'NSE:INDIA VIX')
            raw = self.tools.kite.ltp([vix_key])
            if raw and vix_key in raw:
                vix_val = raw[vix_key].get('last_price', 0)
                if vix_val and vix_val > 0:
                    self._current_vix = round(vix_val, 2)
                    self._vix_last_fetch = now_ts
        except Exception as e:
            # Silently use fallback — don't let VIX fetch failure block trading
            pass

        # Determine regime
        low = self._vix_cfg.get('low_vix_upper', 13.0)
        normal = self._vix_cfg.get('normal_vix_upper', 18.0)
        high = self._vix_cfg.get('high_vix_upper', 25.0)
        if self._current_vix < low:
            self._vix_regime = 'LOW'
        elif self._current_vix < normal:
            self._vix_regime = 'NORMAL'
        elif self._current_vix < high:
            self._vix_regime = 'HIGH'
        else:
            self._vix_regime = 'EXTREME'

        return self._current_vix

    def _get_vix_multipliers(self):
        """Return (score_mult, lot_mult, sl_widen, trail_reduce) for current VIX regime."""
        regime = self._vix_regime.lower()
        cfg = self._vix_cfg
        return {
            'score_multiplier': cfg.get(f'score_multiplier_{regime}', 1.0),
            'lot_multiplier': cfg.get(f'lot_multiplier_{regime}', 1.0),
            'sl_widen': cfg.get(f'sl_widen_{regime}', 1.0),
            'trail_retain_reduce': cfg.get(f'trail_retain_reduce_{regime}', 0.0),
            'regime': self._vix_regime,
            'vix': self._current_vix,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # GCR — GMM CONVICTION RECHECK
    # Re-queries GMM DR scores on open LOSING positions every scan cycle.
    # If the model now OPPOSES trade direction for N consecutive checks → exit.
    # Anti-whipsaw: must see N consecutive opposing checks, not just 1 flash.
    # ═══════════════════════════════════════════════════════════════════════════
    def _gmm_conviction_recheck(self, ml_results: dict, cycle_time: str):
        """Check if GMM now opposes any open losing positions.
        
        For CE positions (direction=BUY): if ml_down_risk_score > threshold → opposing
        For PE positions (direction=SELL): if ml_up_score > threshold → opposing
        
        Uses consecutive-check anti-whipsaw: must oppose N scans in a row
        before triggering exit.
        """
        cfg = self._gcr_cfg
        if not cfg.get('enabled', False):
            return []

        # Day reset
        today = datetime.now().date()
        if self._gcr_date != today:
            self._gcr_date = today
            self._gcr_exits_today = 0
            self._gcr_oppose_counts.clear()
            self._gcr_exit_times.clear()

        # Daily cap
        max_exits = cfg.get('max_exits_per_day', 4)
        if self._gcr_exits_today >= max_exits:
            return []

        # Time window check
        now_str = datetime.now().strftime('%H:%M')
        t_start, t_end = cfg.get('time_window', ('09:30', '15:10'))
        if now_str < t_start or now_str > t_end:
            return []

        min_loss_pct = cfg.get('min_loss_pct', 7)
        dr_threshold = cfg.get('dr_oppose_threshold', 0.25)
        consec_required = cfg.get('consecutive_checks_required', 3)
        skip_setups = set(cfg.get('skip_setup_types', []))
        cooldown_min = cfg.get('cooldown_after_exit_min', 10)

        # Get open option positions (naked only — no spreads, no hedges)
        with self.tools._positions_lock:
            open_trades = [t.copy() for t in self.tools.paper_positions
                          if t.get('status', 'OPEN') == 'OPEN'
                          and t.get('is_option', False)
                          and not t.get('is_credit_spread', False)
                          and not t.get('is_debit_spread', False)
                          and not t.get('is_iron_condor', False)]

        if not open_trades:
            # No open positions → clear any stale counters
            if self._gcr_oppose_counts:
                self._gcr_oppose_counts.clear()
            return []

        # Track which symbols are still open (to clean stale counters)
        current_open_syms = set()
        exited = []

        for trade in open_trades:
            symbol = trade.get('symbol', '')
            underlying = trade.get('underlying', '')
            direction = trade.get('direction', '')  # BUY = bullish (CE), SELL = bearish (PE)
            setup_type = trade.get('setup_type', '')
            entry_price = trade.get('avg_price', 0)

            if not symbol or not underlying or not direction:
                continue

            current_open_syms.add(symbol)

            # Skip excluded setup types
            if setup_type in skip_setups:
                continue

            # Skip if already hedged
            if trade.get('hedged_from_tie', False) or trade.get('is_hedge', False):
                continue

            # Cooldown check: don't re-exit same underlying too quickly
            last_exit = self._gcr_exit_times.get(underlying)
            if last_exit:
                minutes_since = (datetime.now() - last_exit).total_seconds() / 60
                if minutes_since < cooldown_min:
                    continue

            # Get current LTP for P&L calculation
            ltp = 0
            try:
                if self.tools.ticker and self.tools.ticker.connected:
                    cached = self.tools.ticker.get_ltp_batch([symbol])
                    ltp = cached.get(symbol, 0)
                if not ltp:
                    from state_db import get_state_db
                    live = get_state_db().load_live_pnl() or {}
                    lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
                    ltp = lp['ltp'] if lp else 0
            except Exception:
                pass

            if ltp <= 0 or entry_price <= 0:
                continue

            # Calculate current loss %
            if direction in ('BUY', 'LONG'):
                pnl_pct = ((ltp - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - ltp) / entry_price) * 100

            # Only check positions that are losing ≥ min_loss_pct
            if pnl_pct > -min_loss_pct:
                # Not losing enough — reset counter (position recovered)
                if symbol in self._gcr_oppose_counts:
                    del self._gcr_oppose_counts[symbol]
                continue

            # Look up underlying's current GMM scores from this cycle's ML results
            ul_key = underlying if underlying in ml_results else f'NSE:{underlying.replace("NSE:", "")}'
            ml = ml_results.get(ul_key, {})
            if not ml:
                # No ML data this cycle — don't count as opposing or confirming
                continue

            dr_down = ml.get('ml_down_risk_score', 0) or ml.get('ml_down_score', 0)
            dr_up = ml.get('ml_up_score', 0)
            gmm_regime = ml.get('ml_gmm_regime_used', 'UNKNOWN')

            # Determine if GMM now opposes the trade direction
            gmm_opposes = False
            oppose_reason = ''

            if direction in ('BUY', 'LONG'):
                # Holding CE (bullish) — GMM opposes if down-risk is high
                if dr_down > dr_threshold:
                    gmm_opposes = True
                    oppose_reason = f'DR_DOWN={dr_down:.2%}>{dr_threshold:.0%} (regime={gmm_regime})'
            elif direction in ('SELL', 'SHORT'):
                # Holding PE (bearish) — GMM opposes if up-move score is high
                if dr_up > dr_threshold:
                    gmm_opposes = True
                    oppose_reason = f'DR_UP={dr_up:.2%}>{dr_threshold:.0%} (regime={gmm_regime})'

            sym_clean = underlying.replace('NSE:', '')

            if gmm_opposes:
                # Increment consecutive opposing counter
                prev = self._gcr_oppose_counts.get(symbol, 0)
                self._gcr_oppose_counts[symbol] = prev + 1
                count = self._gcr_oppose_counts[symbol]
                print(f"   🔄 GCR: {sym_clean} [{setup_type}] opposing {count}/{consec_required} — {oppose_reason} | loss={pnl_pct:+.1f}%")

                if count >= consec_required:
                    # ═══ TRIGGER EXIT ═══
                    qty = abs(trade.get('quantity', 0))
                    if direction in ('BUY', 'LONG'):
                        pnl = (ltp - entry_price) * qty
                    else:
                        pnl = (entry_price - ltp) * qty

                    print(f"\n   🚨 GCR EXIT: {sym_clean} [{setup_type}] — GMM opposed {count}x consecutive")
                    print(f"      Direction: {direction} | Entry: ₹{entry_price:.2f} → Exit: ₹{ltp:.2f} | P&L: ₹{pnl:+,.0f}")
                    print(f"      Reason: {oppose_reason}")

                    try:
                        exit_detail = {
                            'exit_type': 'GCR_GMM_OPPOSE',
                            'gcr_consecutive_count': count,
                            'gcr_dr_down': round(dr_down, 4),
                            'gcr_dr_up': round(dr_up, 4),
                            'gcr_gmm_regime': gmm_regime,
                            'gcr_loss_pct': round(pnl_pct, 2),
                            'gcr_oppose_reason': oppose_reason,
                            'direction': direction,
                            'setup_type': setup_type,
                        }
                        self.tools.update_trade_status(symbol, 'GCR_GMM_OPPOSE', ltp, pnl, exit_detail=exit_detail)
                        with self._pnl_lock:
                            self.daily_pnl += pnl
                            self.capital += pnl

                        # Record with Risk Governor
                        open_pos = [t for t in open_trades if t.get('symbol') != symbol]
                        unrealized = self.risk_governor._calc_unrealized_pnl(open_pos)
                        self.risk_governor.record_trade_result(symbol, pnl, pnl > 0, unrealized_pnl=unrealized)
                        self.risk_governor.update_capital(self.capital)

                        # Notify scorer
                        try:
                            from options_trader import get_intraday_scorer
                            scorer = get_intraday_scorer()
                            if pnl > 0:
                                scorer.record_symbol_win(symbol)
                            else:
                                scorer.record_symbol_loss(symbol)
                        except Exception:
                            pass

                        # Update tracking
                        self._gcr_exits_today += 1
                        self._gcr_exit_times[underlying] = datetime.now()
                        del self._gcr_oppose_counts[symbol]
                        exited.append(sym_clean)

                        self._log_decision(cycle_time, underlying, 0, 'GCR_GMM_OPPOSE_EXIT',
                                          reason=oppose_reason, direction=direction,
                                          setup=setup_type)

                        print(f"   ✅ GCR exit done: {sym_clean} | {self._gcr_exits_today}/{max_exits} exits today")

                    except Exception as e:
                        print(f"   ❌ GCR exit error for {sym_clean}: {e}")
            else:
                # GMM no longer opposes — reset counter (streak broken)
                if symbol in self._gcr_oppose_counts:
                    old_count = self._gcr_oppose_counts[symbol]
                    del self._gcr_oppose_counts[symbol]
                    if old_count > 0:
                        print(f"   🔄 GCR: {sym_clean} streak reset (was {old_count}/{consec_required}) — GMM no longer opposes")

        # Clean stale counters for closed positions
        stale = [s for s in self._gcr_oppose_counts if s not in current_open_syms]
        for s in stale:
            del self._gcr_oppose_counts[s]

        return exited

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
        """Persist ORB trade tracking to SQLite for restart safety"""
        try:
            get_state_db().save_orb_trades(self.orb_trades_today)
        except Exception as e:
            print(f"⚠️ Failed to save ORB state: {e}")

    def _load_orb_state(self):
        """Load persisted ORB state from SQLite; returns empty if missing or stale"""
        today = datetime.now().date()
        try:
            trades = get_state_db().load_orb_trades(str(today))
            if trades is not None:
                count = sum(1 for s in trades.values() for d, v in s.items() if v)
                return trades, today
        except Exception as e:
            print(f"⚠️ Failed to load ORB state from SQLite: {e}")
        # Fallback: legacy JSON
        try:
            if os.path.exists(self._orb_state_file):
                with open(self._orb_state_file, 'r') as f:
                    state = json.load(f)
                saved_date = state.get('date', '')
                if saved_date == str(today):
                    trades = state.get('trades', {})
                    return trades, today
        except Exception as e:
            print(f"⚠️ Failed to load ORB state from JSON: {e}")
        return {}, today

    def _reset_orb_tracker_if_new_day(self):
        """Reset ORB tracker at start of new trading day"""
        today = datetime.now().date()
        if today != self.orb_tracking_date:
            self.orb_trades_today = {}
            self.orb_tracking_date = today
            self._save_orb_state()
            # print(f"📅 New trading day - ORB tracker reset")
    
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
            # print(f"📊 Exit Manager: Synced {registered} existing positions for exit management")
            pass
    
    def _mark_orb_trade_taken(self, symbol: str, direction: str):
        """Mark ORB direction as used for symbol today"""
        self._reset_orb_tracker_if_new_day()
        if symbol not in self.orb_trades_today:
            self.orb_trades_today[symbol] = {"UP": False, "DOWN": False}
        self.orb_trades_today[symbol][direction] = True
        self._save_orb_state()
        # print(f"📊 ORB {direction} marked as taken for {symbol} today")
    
    def start_realtime_monitor(self):
        """Start the real-time position monitor in a separate thread"""
        if self.monitor_running:
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self._realtime_monitor_loop, daemon=True)
        self.monitor_thread.start()
        # print("⚡ Real-time monitor started")
    
    def stop_realtime_monitor(self):
        """Stop the real-time monitor"""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        # print("⚡ Real-time monitor stopped")
    
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
                # print(f"   [PLH] {symbol} loss {loss_pct:.1f}% (trigger at {loss_trigger}%)")
                pass

        if hedged_this_cycle:
            print(f"\n🛡️ PROACTIVE HEDGE CYCLE: converted {len(hedged_this_cycle)} positions")

    def _realtime_monitor_loop(self):
        """Continuous loop that checks positions every few seconds"""
        candle_timer = 0  # Track time for candle increment
        _proactive_hedge_timer = 0  # Track time for proactive hedge check
        _profit_target_timer = 0  # Track time for profit target check (every 60s)
        _tie_timer = 0  # Track time for TIE thesis check (every 60s instead of every 3s)
        _gcr_timer = 0  # Track time for GCR conviction recheck (every 180s / 3 min)
        _wme_timer = 0  # Track time for watcher momentum exit check
        while self.monitor_running:
            try:
                if self.is_trading_hours():
                    # TIE runs every 60s, not every 3s — throttle via timer
                    _tie_timer += self.monitor_interval
                    _run_tie = False
                    if _tie_timer >= 60:
                        _tie_timer = 0
                        _run_tie = True
                    self._check_positions_realtime(skip_tie=not _run_tie)
                    self._check_eod_exit()  # Check if need to exit before close

                    # === PORTFOLIO PROFIT TARGET (every 60s) ===
                    _profit_target_timer += self.monitor_interval
                    if _profit_target_timer >= 60:
                        _profit_target_timer = 0
                        try:
                            self._check_portfolio_profit_target()
                        except Exception as _pt_err:
                            print(f"   \u26a0\ufe0f Profit target check error: {_pt_err}")
                    
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
                    
                    # === GCR: GMM Conviction Recheck (every 3 min) ===
                    _gcr_timer += self.monitor_interval
                    if _gcr_timer >= 180:
                        _gcr_timer = 0
                        try:
                            _gcr_ml = getattr(self, '_cycle_ml_results', {})
                            if _gcr_ml:
                                self._gmm_conviction_recheck(
                                    _gcr_ml, datetime.now().strftime('%H:%M:%S')
                                )
                        except Exception as _gcr_err:
                            print(f"   ⚠️ GCR error (non-fatal): {_gcr_err}")

                    # === WME: Watcher Momentum Exit (configurable interval, default 5s) ===
                    _wme_timer += self.monitor_interval
                    from config import BREAKOUT_WATCHER as _wme_bw_cfg
                    _wme_interval = _wme_bw_cfg.get('momentum_exit', {}).get('check_interval_seconds', 5)
                    if _wme_timer >= _wme_interval:
                        _wme_timer = 0
                        try:
                            self._check_watcher_momentum_exits()
                        except Exception as _wme_err:
                            print(f"   ⚠️ WME error (non-fatal): {_wme_err}")

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
        """Exit all positions before market close (3:30 PM).
        
        Two-phase approach:
        1. Close all positions in paper_positions (in-memory)
        2. Reconcile with trade ledger: find ENTRY records with no EXIT and force-close them
           This catches trades lost from paper_positions during bot restarts.
        """
        now = datetime.now().time()
        eod_exit_time = datetime.strptime("15:22", "%H:%M").time()  # Exit 8 mins before 3:30
        
        if now >= eod_exit_time:
            # === PHASE 1: Close all in-memory positions ===
            with self.tools._positions_lock:
                active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            
            if active_trades:
                print(f"\n⏰ END OF DAY - Closing all positions ({len(active_trades)} active)...")
                
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
                    # Fall through to Phase 2 reconciliation
                    pass
                else:
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
                    except Exception as _eod_q_err:
                        print(f"   ⚠️ EOD EXIT: quote fetch failed: {_eod_q_err} — will retry next cycle")
                        # Don't return — fall through to Phase 2 reconciliation
                        quotes = {}
                
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

            # === PHASE 2: Ledger reconciliation — catch orphaned trades ===
            # Trades can be lost from paper_positions during bot restarts.
            # Scan the trade ledger for ENTRY records with no matching EXIT.
            if not getattr(self, '_eod_ledger_reconciled', False):
                try:
                    self._reconcile_orphaned_trades()
                except Exception as _recon_err:
                    print(f"   ⚠️ EOD ledger reconciliation error: {_recon_err}")

            # ── EOD DB maintenance (runs once after all positions closed) ──
            self._eod_db_maintenance()

    def _reconcile_orphaned_trades(self):
        """Find trades with ENTRY in ledger but no EXIT, and force-close them.
        
        This catches trades that were lost from paper_positions during bot restarts.
        Uses trade ledger as the source of truth. Runs once per day at EOD.
        """
        if getattr(self, '_eod_ledger_reconciled', False):
            return
        self._eod_ledger_reconciled = True
        
        try:
            from trade_ledger import get_trade_ledger
            today = datetime.now().strftime('%Y-%m-%d')
            tl = get_trade_ledger()
            entries = tl.get_entries(today)
            exits = tl.get_exits(today)
        except Exception as _tl_err:
            print(f"   ⚠️ Orphan reconciliation: ledger error: {_tl_err}")
            return
        
        # Build set of symbols that have EXIT records (including spread symbols)
        exited_symbols = set()
        for ex in exits:
            sym = ex.get('symbol', '')
            exited_symbols.add(sym)
            # Spread exits use combined symbol "NFO:X|NFO:Y" — extract components
            if '|' in sym:
                for part in sym.split('|'):
                    exited_symbols.add(part.strip())
        
        # Also check in-memory positions (some may have been closed but not yet in ledger)
        with self.tools._positions_lock:
            for t in self.tools.paper_positions:
                if t.get('status', 'OPEN') != 'OPEN':
                    exited_symbols.add(t.get('symbol', ''))
        
        # Safety net: also check symbols exited from paper_positions this session
        # (covers trades where ledger write failed but exit DID happen in memory)
        exited_symbols.update(getattr(self.tools, '_exited_symbols_today', set()))
        
        # Find orphaned entries
        orphaned = []
        for entry in entries:
            sym = entry.get('symbol', '')
            if sym and sym not in exited_symbols:
                orphaned.append(entry)
        
        if not orphaned:
            return
        
        print(f"\n🔍 EOD RECONCILIATION: Found {len(orphaned)} orphaned trade(s) — closing...")
        
        # Fetch quotes for orphaned symbols
        orphan_syms = [e['symbol'] for e in orphaned]
        try:
            if self.tools.ticker and self.tools.ticker.connected:
                ltp_data = self.tools.ticker.get_ltp_batch(orphan_syms)
                quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                missing = [s for s in orphan_syms if s not in quotes]
                if missing:
                    quotes.update(self.tools.kite.ltp(missing))
            else:
                quotes = self.tools.kite.ltp(orphan_syms)
        except Exception as _q_err:
            print(f"   ⚠️ Orphan quote fetch failed: {_q_err} — writing EXIT with entry price (zero P&L)")
            quotes = {}
        
        for entry in orphaned:
            sym = entry.get('symbol', '')
            entry_price = entry.get('entry_price', 0)
            qty = entry.get('quantity', 0)
            direction = entry.get('direction', 'BUY')
            source = entry.get('source', 'UNKNOWN')
            order_id = entry.get('order_id', '')
            underlying = entry.get('underlying', '')
            
            ltp = quotes.get(sym, {}).get('last_price', 0)
            if ltp and ltp > 0:
                if direction == 'BUY':
                    pnl = (ltp - entry_price) * qty
                else:
                    pnl = (entry_price - ltp) * qty
                pnl -= calc_brokerage(entry_price, ltp, qty)
            else:
                # No quote available — record zero P&L (better than orphaning)
                ltp = entry_price
                pnl = 0
            
            print(f"   🔄 ORPHAN EXIT: {sym} (src={source}) entry=₹{entry_price} exit=₹{ltp:.2f} P&L=₹{pnl:+,.0f}")
            
            # Write EXIT event to trade ledger directly
            try:
                tl.log_exit(
                    symbol=sym,
                    underlying=underlying,
                    direction=direction,
                    source=source,
                    exit_type='EOD_ORPHAN_RECONCILE',
                    entry_price=entry_price,
                    exit_price=ltp,
                    quantity=qty,
                    pnl=round(pnl, 2),
                    exit_reason=f'EOD orphan reconciliation: trade lost from paper_positions during bot restart',
                    order_id=order_id,
                    hold_minutes=0,
                    candles_held=0,
                    r_multiple=0,
                    smart_score=entry.get('smart_score', 0),
                    final_score=entry.get('final_score', 0),
                    dr_score=entry.get('dr_score', 0),
                    score_tier=entry.get('score_tier', ''),
                    strategy_type=entry.get('strategy_type', ''),
                    entry_time=entry.get('ts', ''),
                )
                with self._pnl_lock:
                    self.daily_pnl += pnl
                    self.capital += pnl
            except Exception as _ex_err:
                print(f"   ⚠️ Failed to record orphan exit for {sym}: {_ex_err}")

    def _eod_db_maintenance(self):
        """Run after EOD close: WAL checkpoint + purge stale data (>30 days)."""
        if getattr(self, '_eod_maintenance_done', False):
            return
        self._eod_maintenance_done = True
        try:
            from state_db import get_state_db
            db = get_state_db()
            # Checkpoint WAL → merge into main DB file
            db.checkpoint()
            # Purge data older than 30 days
            cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            with db._lock:
                for tbl in ['active_trades', 'exit_states', 'order_idempotency',
                            'data_health', 'reconciliation_state', 'orb_trades', 'risk_state']:
                    db._conn.execute(f"DELETE FROM [{tbl}] WHERE date < ?", (cutoff,))
                # slippage_log: keep last 500
                db._conn.execute(
                    "DELETE FROM slippage_log WHERE id NOT IN (SELECT id FROM slippage_log ORDER BY id DESC LIMIT 500)"
                )
                # daily_state: keep last 30 days
                db._conn.execute("DELETE FROM daily_state WHERE date < ?", (cutoff,))
                db._conn.commit()
            print(f"🗄️ EOD DB maintenance: checkpoint + purge (cutoff {cutoff})")
        except Exception as e:
            print(f"⚠️ EOD DB maintenance error: {e}")

    def _check_portfolio_profit_target(self):
        """KILL-ALL PROFIT SWITCH: Close ALL positions when cumulative unrealized P&L >= 15% of capital.
        
        Uses _compute_live_unrealized_pnl() for real-time unrealized P&L.
        Runs every 60s in background. After booking profit, resets so next scan can continue.
        """
        if self._profit_target_hit:
            return  # Close-all already in progress this cycle
        
        from config import HARD_RULES
        target_pct = HARD_RULES.get('PORTFOLIO_PROFIT_TARGET', 0.15)
        target_amount = self.start_capital * target_pct
        
        unrealized = self._compute_live_unrealized_pnl()
        if unrealized < target_amount:
            return  # Not yet at target
        
        # ═══ PROFIT TARGET HIT — CLOSE EVERYTHING ═══
        self._profit_target_hit = True
        print(f"\n{'='*70}")
        print(f"💰💰💰 PORTFOLIO PROFIT TARGET HIT! Unrealized: ₹{unrealized:+,.0f} >= {target_pct*100:.0f}% of ₹{self.start_capital:,.0f} (₹{target_amount:,.0f})")
        print(f"💰💰💰 CLOSING ALL POSITIONS TO BOOK PROFIT")
        print(f"{'='*70}\n")
        
        with self.tools._positions_lock:
            active_trades = [t for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
        
        if not active_trades:
            return
        
        # Separate trade types (same pattern as _check_eod_exit)
        regular_trades = [t for t in active_trades if not t.get('is_credit_spread', False) and not t.get('is_debit_spread', False) and not t.get('is_iron_condor', False)]
        spread_trades_pt = [t for t in active_trades if t.get('is_credit_spread', False)]
        debit_spread_trades_pt = [t for t in active_trades if t.get('is_debit_spread', False)]
        ic_trades_pt = [t for t in active_trades if t.get('is_iron_condor', False)]
        
        # Collect all symbols for quotes
        all_symbols = [t['symbol'] for t in regular_trades]
        for st in spread_trades_pt:
            sold_sym = st.get('sold_symbol', ''); hedge_sym = st.get('hedge_symbol', '')
            if sold_sym: all_symbols.append(sold_sym)
            if hedge_sym: all_symbols.append(hedge_sym)
        for dt in debit_spread_trades_pt:
            buy_sym = dt.get('buy_symbol', ''); sell_sym = dt.get('sell_symbol', '')
            if buy_sym: all_symbols.append(buy_sym)
            if sell_sym: all_symbols.append(sell_sym)
        for ict in ic_trades_pt:
            for leg in ['sold_ce_symbol', 'hedge_ce_symbol', 'sold_pe_symbol', 'hedge_pe_symbol']:
                sym = ict.get(leg, '')
                if sym: all_symbols.append(sym)
        
        if not all_symbols:
            return
        
        try:
            if self.tools.ticker and self.tools.ticker.connected:
                ltp_data = self.tools.ticker.get_ltp_batch(all_symbols)
                quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
                missing = [s for s in all_symbols if s not in quotes]
                if missing:
                    quotes.update(self.tools.kite.ltp(missing))
            else:
                quotes = self.tools.kite.ltp(all_symbols)
        except Exception as _pt_err:
            print(f"   ⚠️ Profit target quotes failed: {_pt_err}")
            self._profit_target_hit = False  # Allow retry next cycle
            return
        
        total_booked = 0
        
        # --- Close regular trades ---
        for trade in regular_trades:
            symbol = trade['symbol']
            if symbol not in quotes:
                continue
            ltp = quotes[symbol]['last_price']
            entry = trade['avg_price']; qty = trade['quantity']; side = trade['side']
            pnl = (ltp - entry) * qty if side == 'BUY' else (entry - ltp) * qty
            pnl -= calc_brokerage(entry, ltp, qty)
            print(f"   💰 PROFIT EXIT: {symbol} @ ₹{ltp:.2f} | P&L: ₹{pnl:+,.2f}")
            self.tools.update_trade_status(symbol, 'PROFIT_TARGET_EXIT', ltp, pnl,
                                           exit_detail={'exit_reason': 'PORTFOLIO_PROFIT_TARGET', 'exit_type': 'PROFIT_TARGET_KILL_ALL'})
            with self._pnl_lock:
                self.daily_pnl += pnl; self.capital += pnl
            total_booked += pnl
        
        # --- Close credit spreads ---
        for trade in spread_trades_pt:
            sold_sym = trade.get('sold_symbol', ''); hedge_sym = trade.get('hedge_symbol', '')
            qty = trade['quantity']; net_credit = trade.get('net_credit', 0)
            sold_ltp = quotes.get(sold_sym, {}).get('last_price', 0) if sold_sym else 0
            hedge_ltp = quotes.get(hedge_sym, {}).get('last_price', 0) if hedge_sym else 0
            current_debit = sold_ltp - hedge_ltp
            pnl = (net_credit - current_debit) * qty
            pnl -= calc_brokerage(net_credit, current_debit, qty)
            print(f"   💰 PROFIT EXIT SPREAD: {trade['symbol']} | P&L: ₹{pnl:+,.2f}")
            self.tools.update_trade_status(trade['symbol'], 'PROFIT_TARGET_EXIT', current_debit, pnl,
                                           exit_detail={'exit_reason': 'PORTFOLIO_PROFIT_TARGET', 'exit_type': 'PROFIT_TARGET_KILL_ALL'})
            with self._pnl_lock:
                self.daily_pnl += pnl; self.capital += pnl
            total_booked += pnl
        
        # --- Close debit spreads ---
        for trade in debit_spread_trades_pt:
            buy_sym = trade.get('buy_symbol', ''); sell_sym = trade.get('sell_symbol', '')
            qty = trade['quantity']; net_debit = trade.get('net_debit', 0)
            buy_ltp = quotes.get(buy_sym, {}).get('last_price', 0) if buy_sym else 0
            sell_ltp = quotes.get(sell_sym, {}).get('last_price', 0) if sell_sym else 0
            current_value = buy_ltp - sell_ltp
            pnl = (current_value - net_debit) * qty
            pnl -= calc_brokerage(net_debit, current_value, qty)
            print(f"   💰 PROFIT EXIT DEBIT SPREAD: {trade['symbol']} | P&L: ₹{pnl:+,.2f}")
            self.tools.update_trade_status(trade['symbol'], 'PROFIT_TARGET_EXIT', current_value, pnl,
                                           exit_detail={'exit_reason': 'PORTFOLIO_PROFIT_TARGET', 'exit_type': 'PROFIT_TARGET_KILL_ALL'})
            with self._pnl_lock:
                self.daily_pnl += pnl; self.capital += pnl
            total_booked += pnl
        
        # --- Close iron condors ---
        for trade in ic_trades_pt:
            qty = trade['quantity']; total_credit = trade.get('total_credit', 0)
            s_ce = quotes.get(trade.get('sold_ce_symbol', ''), {}).get('last_price', 0)
            h_ce = quotes.get(trade.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
            s_pe = quotes.get(trade.get('sold_pe_symbol', ''), {}).get('last_price', 0)
            h_pe = quotes.get(trade.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
            current_debit = (s_ce - h_ce) + (s_pe - h_pe)
            pnl = (total_credit - current_debit) * qty
            pnl -= calc_brokerage(total_credit, current_debit, qty)
            print(f"   💰 PROFIT EXIT IC: {trade['symbol']} | P&L: ₹{pnl:+,.2f}")
            self.tools.update_trade_status(trade['symbol'], 'PROFIT_TARGET_EXIT', current_debit, pnl,
                                           exit_detail={'exit_reason': 'PORTFOLIO_PROFIT_TARGET', 'exit_type': 'PROFIT_TARGET_KILL_ALL', 'strategy': 'IRON_CONDOR'})
            with self._pnl_lock:
                self.daily_pnl += pnl; self.capital += pnl
            total_booked += pnl
        
        print(f"\n{'='*70}")
        print(f"💰 PORTFOLIO PROFIT TARGET COMPLETE — {len(active_trades)} positions closed")
        print(f"💰 Total Booked: ₹{total_booked:+,.0f} | Daily P&L: ₹{self.daily_pnl:+,.0f} | Capital: ₹{self.capital:,.0f}")
        print(f"💰 Profit booked. Next scan cycle will continue as normal.")
        print(f"{'='*70}\n")

        # Log summary event to centralized trade ledger
        try:
            from trade_ledger import get_trade_ledger
            get_trade_ledger().log_scan_decision(
                symbol='PORTFOLIO',
                action='PROFIT_TARGET_KILL_ALL',
                reason=f"Unrealized ₹{unrealized:+,.0f} >= {target_pct*100:.0f}% of ₹{self.start_capital:,.0f}. "
                       f"Closed {len(active_trades)} positions. Booked ₹{total_booked:+,.0f}. "
                       f"Daily P&L: ₹{self.daily_pnl:+,.0f}",
                source='PROFIT_TARGET',
                direction='KILL_ALL',
                smart_score=0,
                dr_score=0,
            )
        except Exception as _ledger_err:
            print(f"   ⚠️ Ledger log error: {_ledger_err}")

        # Reset flag so the next scan cycle can take fresh entries
        self._profit_target_hit = False

    def _compute_live_unrealized_pnl(self) -> float:
        """Compute total unrealized P&L from all open positions using live ticker quotes.
        
        Uses the same WebSocket cache that the real-time monitor uses.
        Returns 0.0 if no positions or no ticker available.
        """
        try:
            with self.tools._positions_lock:
                active = [t.copy() for t in self.tools.paper_positions if t.get('status', 'OPEN') == 'OPEN']
            if not active:
                # print(f"   [RG-UNREAL] no active positions → 0")
                return 0.0
            
            # Collect all symbols needed for LTP
            all_symbols = set()
            for t in active:
                if t.get('is_iron_condor'):
                    for k in ('sold_ce_symbol', 'hedge_ce_symbol', 'sold_pe_symbol', 'hedge_pe_symbol'):
                        s = t.get(k, '')
                        if s: all_symbols.add(s)
                elif t.get('is_credit_spread'):
                    for k in ('sold_symbol', 'hedge_symbol'):
                        s = t.get(k, '')
                        if s: all_symbols.add(s)
                elif t.get('is_debit_spread'):
                    for k in ('buy_symbol', 'sell_symbol'):
                        s = t.get(k, '')
                        if s: all_symbols.add(s)
                else:
                    all_symbols.add(t['symbol'])
            
            all_symbols.discard('')
            if not all_symbols:
                # print(f"   [RG-UNREAL] no symbols from {len(active)} positions → 0")
                return 0.0
            
            # Fetch LTPs from WebSocket cache
            quotes = {}
            _ticker_ok = self.tools.ticker and self.tools.ticker.connected
            if _ticker_ok:
                ltp_data = self.tools.ticker.get_ltp_batch(list(all_symbols))
                quotes = {sym: {'last_price': ltp} for sym, ltp in ltp_data.items()}
            else:
                try:
                    quotes = self.tools.kite.ltp(list(all_symbols))
                except Exception:
                    # print(f"   [RG-UNREAL] ticker not connected, kite.ltp failed → 0")
                    return 0.0
            
            if not quotes:
                # print(f"   [RG-UNREAL] no quotes for {all_symbols} (ticker={'Y' if _ticker_ok else 'N'}) → 0")
                return 0.0
            
            # Compute total unrealized P&L (mirrors _check_positions_realtime logic)
            total = 0.0
            for t in active:
                if t.get('is_credit_spread'):
                    sold_ltp = quotes.get(t.get('sold_symbol', ''), {}).get('last_price', 0)
                    hedge_ltp = quotes.get(t.get('hedge_symbol', ''), {}).get('last_price', 0)
                    if sold_ltp and hedge_ltp:
                        current_debit = sold_ltp - hedge_ltp
                        total += (t.get('net_credit', 0) - current_debit) * t['quantity']
                elif t.get('is_debit_spread'):
                    buy_ltp = quotes.get(t.get('buy_symbol', ''), {}).get('last_price', 0)
                    sell_ltp = quotes.get(t.get('sell_symbol', ''), {}).get('last_price', 0)
                    if buy_ltp and sell_ltp:
                        current_value = buy_ltp - sell_ltp
                        total += (current_value - t.get('net_debit', 0)) * t['quantity']
                elif t.get('is_iron_condor'):
                    s_ce = quotes.get(t.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                    h_ce = quotes.get(t.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                    s_pe = quotes.get(t.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                    h_pe = quotes.get(t.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                    if all([s_ce, h_ce, s_pe, h_pe]):
                        ic_debit = (s_ce - h_ce) + (s_pe - h_pe)
                        total += (t.get('total_credit', 0) - ic_debit) * t['quantity']
                else:
                    sym = t['symbol']
                    ltp = quotes.get(sym, {}).get('last_price', 0)
                    if ltp > 0:
                        entry = t.get('avg_price', 0)
                        qty = t.get('quantity', 0)
                        if t.get('side', 'BUY') == 'BUY':
                            total += (ltp - entry) * qty
                        else:
                            total += (entry - ltp) * qty
            return total
        except Exception as _unreal_err:
            # print(f"   [RG-UNREAL] exception: {_unreal_err}")
            return 0.0

    # ── Manual exit processing (dashboard → bot in-memory sync) ──────
    MANUAL_EXIT_FILE = os.path.join(os.path.dirname(__file__), 'manual_exit_requests.json')

    def _process_manual_exit_requests(self):
        """Process manual exit requests written by the dashboard.
        
        The dashboard writes a signal file with exits to process.
        This method:
        1. Reads the signal file
        2. Removes matching positions from in-memory paper_positions
        3. Places real exit orders (LIVE mode)
        4. Clears the processed signal file
        
        Called at the start of every _check_positions_realtime() cycle.
        """
        if not os.path.exists(self.MANUAL_EXIT_FILE):
            return
        
        try:
            with open(self.MANUAL_EXIT_FILE, 'r') as f:
                requests = json.load(f)
            
            if not requests:
                os.remove(self.MANUAL_EXIT_FILE)
                return
            
            print(f"\n   🖐️ MANUAL EXIT: Processing {len(requests)} dashboard exit request(s)...")
            
            processed = []
            for req in requests:
                symbol = req.get('symbol', '')
                exit_price = req.get('exit_price', 0)
                pnl = req.get('pnl', 0)
                
                with self.tools._positions_lock:
                    found = False
                    for i, trade in enumerate(self.tools.paper_positions):
                        if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                            found = True
                            
                            # === LIVE MODE: Place real exit order (only if dashboard didn't already) ===
                            if not self.tools.paper_mode and not req.get('live_exit_placed', False):
                                try:
                                    self.tools._execute_live_exit(trade)
                                    print(f"   🖐️ LIVE exit order placed for {symbol}")
                                except Exception as e:
                                    print(f"   🚨 LIVE exit order FAILED for {symbol}: {e}")
                            elif req.get('live_exit_placed'):
                                print(f"   🖐️ LIVE exit already placed by dashboard for {symbol}")
                            
                            # Cancel GTT if exists
                            gtt_id = trade.get('gtt_trigger_id')
                            if gtt_id:
                                try:
                                    self.tools._cancel_gtt(gtt_id, symbol)
                                except Exception:
                                    pass
                            
                            # Remove from in-memory positions
                            self.tools.paper_positions.pop(i)
                            self.tools.paper_pnl += pnl
                            # Update in-memory daily P&L and capital (was missing → dashboard exits not reflected)
                            with self._pnl_lock:
                                self.daily_pnl += pnl
                                self.capital += pnl
                            print(f"   🖐️ {symbol} removed from memory | P&L: ₹{pnl:+,.2f} | exit@{exit_price} | Daily P&L: ₹{self.daily_pnl:+,.0f}")
                            processed.append(symbol)
                            break
                    
                    if not found:
                        # Already removed by state_db update (dashboard did this)
                        print(f"   🖐️ {symbol} already removed from memory (OK)")
                        processed.append(symbol)
                
                # Save updated state
                self.tools._save_active_trades()
            
            # Clear signal file
            try:
                os.remove(self.MANUAL_EXIT_FILE)
            except Exception:
                pass
            
            print(f"   🖐️ Manual exit complete: {len(processed)} processed ✅")
            
        except json.JSONDecodeError:
            # Corrupt file — remove it
            try:
                os.remove(self.MANUAL_EXIT_FILE)
            except Exception:
                pass
        except Exception as e:
            print(f"   ⚠️ Manual exit processing error: {e}")

    # ── Reconciliation callback: Kite app manual exit detected ───
    def _on_recon_manual_exit(self, symbol: str, local_pos: dict, reason: str):
        """Called by PositionReconciliation when a position exists locally
        but is gone from broker (= exited on Kite app/web).
        
        Cleans up in-memory state, logs to trade_ledger, updates state_db.
        No LIVE order needed — broker already has no position.
        """
        if not symbol or not local_pos:
            return
        
        print(f"\n   🔄 KITE EXIT DETECTED: {symbol} (reason: {reason})")
        
        entry_price = local_pos.get('avg_price') or local_pos.get('entry_price') or 0
        qty = abs(local_pos.get('quantity', 0))
        # Use 'side' (actual transaction side: BUY/SELL) NOT 'direction' (market view: BUY=bullish, SELL=bearish)
        # For PE options: direction=SELL but side=BUY → must use side for correct P&L
        direction = local_pos.get('side') or local_pos.get('direction', 'BUY')
        
        # Try to get last traded price for P&L calculation
        ltp = 0
        try:
            if self.tools.ticker and self.tools.ticker.connected:
                cached = self.tools.ticker.get_ltp_batch([symbol])
                ltp = cached.get(symbol, 0)
            if not ltp:
                live = get_state_db().load_live_pnl() or {}
                lp = live.get(symbol) or live.get(symbol.replace('NFO:', ''))
                ltp = lp['ltp'] if lp else 0
        except Exception:
            pass
        
        if ltp > 0 and entry_price > 0:
            pnl = (ltp - entry_price) * qty if direction in ('BUY', 'LONG') else (entry_price - ltp) * qty
        else:
            pnl = local_pos.get('unrealized_pnl', 0)
        
        exit_price = ltp if ltp > 0 else entry_price
        
        # Remove from in-memory positions
        with self.tools._positions_lock:
            for i, trade in enumerate(self.tools.paper_positions):
                if trade.get('symbol') == symbol and trade.get('status', 'OPEN') == 'OPEN':
                    # Cancel GTT if exists
                    gtt_id = trade.get('gtt_trigger_id')
                    if gtt_id:
                        try:
                            self.tools._cancel_gtt(gtt_id, symbol)
                        except Exception:
                            pass
                    self.tools.paper_positions.pop(i)
                    self.tools.paper_pnl += pnl
                    # Update in-memory daily P&L and capital (was missing → recon exits not reflected)
                    with self._pnl_lock:
                        self.daily_pnl += pnl
                        self.capital += pnl
                    break
        
        self.tools._save_active_trades()
        
        # Log to trade_ledger
        try:
            from trade_ledger import get_trade_ledger
            import re
            _underlying = local_pos.get('underlying', '')
            if not _underlying:
                m = re.match(r'(?:NFO:)?([A-Z]+)\d', symbol.replace('NFO:', ''))
                _underlying = f"NSE:{m.group(1)}" if m else symbol
            _hold_mins = 0
            try:
                _et = local_pos.get('timestamp', '')
                if _et:
                    _hold_mins = int((datetime.now() - datetime.fromisoformat(_et)).total_seconds() / 60)
            except Exception:
                pass
            _pnl_pct = (pnl / (entry_price * qty) * 100) if entry_price > 0 and qty > 0 else 0
            get_trade_ledger().log_exit(
                symbol=symbol,
                underlying=_underlying,
                direction=direction,
                source=local_pos.get('setup_type', local_pos.get('strategy_type', '')),
                sector=local_pos.get('sector', ''),
                exit_type=reason,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl,
                pnl_pct=_pnl_pct,
                smart_score=local_pos.get('smart_score', 0),
                final_score=local_pos.get('entry_score', 0),
                dr_score=local_pos.get('dr_score', 0),
                exit_reason=f'Position closed outside Titan ({reason})',
                hold_minutes=_hold_mins,
                entry_time=local_pos.get('timestamp', ''),
            )
        except Exception as e:
            print(f"   ⚠️ Trade ledger log failed for recon exit {symbol}: {e}")
        
        print(f"   🔄 {symbol}: Cleaned up | P&L: ₹{pnl:+,.2f} | exit@{exit_price:.2f} ✅")
    
    def _check_positions_realtime(self, skip_tie: bool = False):
        """Check all positions for target/stoploss hits using Exit Manager
        
        skip_tie: if True, skip TIE thesis checks this cycle (throttled to 60s).
        """
        # === PROCESS MANUAL EXITS FROM DASHBOARD ===
        self._process_manual_exit_requests()
        
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
                                            em_state.candles_since_entry = 0  # Fresh start after unwind
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
        exit_signals = self.exit_manager.check_all_exits(price_dict, underlying_prices=underlying_prices, skip_tie=skip_tie)
        
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
                            not trade.get('hedged_from_tie', False) and
                            not trade.get('hedge_unwound', False)  # Don't re-hedge unwound positions
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
                            # Underlying confirmation gate — don't hedge if underlying moving against us
                            _thp_can_hedge = True
                            _thp_ul_reason = ''
                            _thp_em = self.exit_manager.get_trade_state(symbol)
                            if _thp_em:
                                _thp_ul_adverse, _thp_ul_reason = check_underlying_adverse(_thp_em, underlying_prices.get(symbol, 0))
                                if _thp_ul_adverse:
                                    _thp_can_hedge = False
                            if _tie_loss_pct > _max_hedge_loss:
                                print(f"\n📉 THP: {symbol} — loss {_tie_loss_pct:.1f}% > {_max_hedge_loss}% cap — too deep to hedge, exiting")
                            elif not _thp_can_hedge:
                                print(f"\n   🔴 THP: {symbol} — {_thp_ul_reason} — exiting instead of hedging")
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
                            not trade.get('hedged_from_tie', False) and
                            not trade.get('hedge_unwound', False)  # Don't re-hedge unwound positions
                        )
                        _thp_enabled = _thp_cfg.get('enabled', False) and _thp_cfg.get('hedge_time_stop', False)
                        if _is_naked_opt and _thp_enabled:
                            # Check current loss is moderate enough to justify hedging
                            _ts_ltp = price_dict.get(symbol, 0) or signal.exit_price
                            _ts_entry = trade.get('avg_price', 0)
                            _ts_loss_pct = ((_ts_entry - _ts_ltp) / _ts_entry * 100) if _ts_entry > 0 and _ts_ltp > 0 else 999
                            _max_loss_for_hedge = _thp_cfg.get('max_hedge_loss_pct', 12)
                            # Underlying confirmation gate
                            _ts_can_hedge = True
                            _ts_em = self.exit_manager.get_trade_state(symbol)
                            if _ts_em:
                                _ts_adverse, _ts_ul_rsn = check_underlying_adverse(_ts_em, underlying_prices.get(symbol, 0))
                                if _ts_adverse:
                                    _ts_can_hedge = False
                                    print(f"   🔴 THP: {symbol} — {_ts_ul_rsn} — exiting instead of hedging")
                            if _ts_loss_pct <= _max_loss_for_hedge and _ts_can_hedge:
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
                            not trade.get('hedged_from_tie', False) and
                            not trade.get('hedge_unwound', False)  # Don't re-hedge unwound positions
                        )
                        _thp_sl_enabled = _thp_sl_cfg.get('enabled', False)
                        if _is_naked_opt_sl and _thp_sl_enabled:
                            _sl_ltp = price_dict.get(symbol, 0) or signal.exit_price
                            _sl_entry = trade.get('avg_price', 0)
                            _sl_loss_pct = ((_sl_entry - _sl_ltp) / _sl_entry * 100) if _sl_entry > 0 and _sl_ltp > 0 else 999
                            _max_sl_hedge_loss = _thp_sl_cfg.get('max_hedge_loss_pct', 20)
                            # Underlying confirmation gate
                            _sl_can_hedge = True
                            _sl_em = self.exit_manager.get_trade_state(symbol)
                            if _sl_em:
                                _sl_adverse, _sl_ul_rsn = check_underlying_adverse(_sl_em, underlying_prices.get(symbol, 0))
                                if _sl_adverse:
                                    _sl_can_hedge = False
                                    print(f"   🔴 THP: {symbol} — {_sl_ul_rsn} — exiting at SL instead of hedging")
                            if _sl_loss_pct <= _max_sl_hedge_loss and _sl_can_hedge:
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
            elif _setup == 'ORB_BREAKOUT':
                _type_tag = '📊ORB'
            elif _setup.startswith('VWAP'):
                _type_tag = '📈VWAP'
            elif _setup == 'WATCHER':
                _type_tag = '⚡WATCH'
            elif _setup == 'CONTRARIAN':
                _type_tag = '🔄CONTR'
            elif _setup == 'TEST_GMM':
                _type_tag = '🧪GMM'
            elif _setup == 'TEST_XGB':
                _type_tag = '🧪XGB'
            elif _setup == 'ARBTR':
                _type_tag = '🔄ARBTR'
            elif _setup in ('', 'MANUAL', 'GPT') or not _setup:
                _type_tag = '🤖GPT'
            else:
                _type_tag = _setup[:10]
            
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

        # ── Always persist live P&L snapshot for dashboard (not gated by show_status) ──
        if active_trades and quotes:
            try:
                live_snaps = []
                _total_upnl = 0.0
                for t in active_trades:
                    if t.get('status', 'OPEN') != 'OPEN':
                        continue
                    sym = t['symbol']
                    ltp = 0.0
                    upnl = 0.0
                    if t.get('is_credit_spread'):
                        sold_sym = t.get('sold_symbol', '')
                        hedge_sym = t.get('hedge_symbol', '')
                        s_ltp = quotes.get(sold_sym, {}).get('last_price', 0)
                        h_ltp = quotes.get(hedge_sym, {}).get('last_price', 0)
                        ltp = s_ltp - h_ltp
                        upnl = (t.get('net_credit', 0) - ltp) * t['quantity']
                    elif t.get('is_debit_spread'):
                        b_ltp = quotes.get(t.get('buy_symbol', ''), {}).get('last_price', 0)
                        sl_ltp = quotes.get(t.get('sell_symbol', ''), {}).get('last_price', 0)
                        ltp = b_ltp - sl_ltp
                        upnl = (ltp - t.get('net_debit', 0)) * t['quantity']
                    elif t.get('is_iron_condor'):
                        s_ce = quotes.get(t.get('sold_ce_symbol', ''), {}).get('last_price', 0)
                        h_ce = quotes.get(t.get('hedge_ce_symbol', ''), {}).get('last_price', 0)
                        s_pe = quotes.get(t.get('sold_pe_symbol', ''), {}).get('last_price', 0)
                        h_pe = quotes.get(t.get('hedge_pe_symbol', ''), {}).get('last_price', 0)
                        ltp = (s_ce - h_ce) + (s_pe - h_pe)
                        upnl = (t.get('total_credit', 0) - ltp) * t['quantity']
                    elif sym in quotes:
                        ltp = quotes[sym]['last_price']
                        if t['side'] == 'BUY':
                            upnl = (ltp - t['avg_price']) * t['quantity']
                        else:
                            upnl = (t['avg_price'] - ltp) * t['quantity']
                    _total_upnl += upnl
                    live_snaps.append({'symbol': sym, 'ltp': round(ltp, 2), 'unrealized_pnl': round(upnl, 2)})
                get_state_db().save_live_pnl(live_snaps, round(_total_upnl, 2))
            except Exception as e:
                pass  # Silent — don't spam logs with dashboard bridge errors
    
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
        """Check if within trading hours (also blocks weekends)."""
        today = datetime.now()
        if today.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        now = today.time()
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
        
        # (Profit target check runs in background every 60s — no scan gate needed)
        
        # === RISK GOVERNOR CHECK ===
        # Compute live unrealized P&L from ticker quotes and pass to risk governor.
        # Positions don't have LTP data — only the ticker WebSocket has live prices.
        _rg_unrealized = self._compute_live_unrealized_pnl()
        _rg_allowed = self.risk_governor.is_trading_allowed(unrealized_pnl=_rg_unrealized)
        _scan_dbg(f"SCAN: risk_governor.is_trading_allowed() = {_rg_allowed} (unrealized={_rg_unrealized:+,.0f})")
        if not _rg_allowed:
            _scan_dbg(f"SCAN: EXIT - risk governor blocked")
            print(self.risk_governor.get_status(unrealized_pnl=_rg_unrealized))
            return
        
        _scan_dbg("SCAN: all pre-checks passed, proceeding to scan...")
        
        # === INDIA VIX REGIME CHECK ===
        # Fetch India VIX (cached for 2 min) and determine regime.
        # Printed once per scan cycle for visibility.
        try:
            _vix_val = self._fetch_india_vix()
            _vix_m = self._get_vix_multipliers()
            _vix_emoji = {'LOW': '🟢', 'NORMAL': '🔵', 'HIGH': '🟠', 'EXTREME': '🔴'}.get(self._vix_regime, '⚪')
            _scan_dbg(f"SCAN: India VIX = {_vix_val:.1f} | Regime: {_vix_emoji} {self._vix_regime} "
                       f"| Score×{_vix_m['score_multiplier']:.2f} Lots×{_vix_m['lot_multiplier']:.2f} "
                       f"SL widen×{_vix_m['sl_widen']:.2f} Trail-{_vix_m['trail_retain_reduce']:.0%}")
            # Push VIX multipliers to tools so place_option_order can access them
            self.tools._vix_multipliers = _vix_m
            # Push VIX trailing adjustment to exit_manager
            self.exit_manager.vix_trail_retain_reduce = _vix_m.get('trail_retain_reduce', 0.0)
        except Exception as _vix_err:
            _scan_dbg(f"SCAN: VIX fetch error (non-fatal): {_vix_err}")
            self.tools._vix_multipliers = {'score_multiplier': 1.0, 'lot_multiplier': 1.0,
                                            'sl_widen': 1.0, 'trail_retain_reduce': 0.0,
                                            'regime': 'NORMAL', 'vix': 14.0}
        
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
        
        print(self.risk_governor.get_status(self.tools.paper_positions if hasattr(self, 'tools') and self.tools else []))
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
                    _max_indicator_stocks = FULL_FNO_SCAN.get('max_indicator_stocks', 40)
                    
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
                        
                        # print(f"   🔍 Quality gate input: {len(_all_raw)} raw stocks, min_change={_min_change}%, vol_p75={_vol_p75:,.0f}")
                        
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
                        # print(f"   🔍 Quality gate: {_after_min} above min_change → {len(_quality_candidates)} passed, {_dropped} dropped")
                        # print(f"      Gate A (≥1.0% chg): {_gate_a_count} | Gate B (day extreme): {_gate_b_count} | Gate C (vol top25): {_gate_c_count}")
                        
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
                            # print(f"      Top 5 selected:")
                            # for _i, _r in enumerate(_selected[:5]):
                            #     _dr2 = _r.day_high - _r.day_low
                            #     _pos2 = ((_r.ltp - _r.day_low) / _dr2 * 100) if _dr2 > 0 else 50
                            #     _gates = []
                            #     if abs(_r.change_pct) >= 1.0: _gates.append("A:chg")
                            #     if _dr2 > 0 and (_pos2 >= 85 or _pos2 <= 15): _gates.append("B:ext")
                            #     if _r.volume >= _vol_p75 and _vol_p75 > 0: _gates.append("C:vol")
                            #     print(f"        {_i+1}. {_r.symbol} chg={_r.change_pct:+.1f}% dayPos={_pos2:.0f}% vol={_r.volume:,.0f} gates=[{','.join(_gates)}]")
                    
                    # Always include stocks we already hold (need exit monitoring)
                    for _t in self.tools.paper_positions:
                        if _t.get('status', 'OPEN') == 'OPEN':
                            _held_sym = _t.get('symbol', '')
                            if _held_sym:
                                _full_universe.add(_held_sym)
                    
                    scan_universe = list(_full_universe)
                    _skipped = len(_all_fo) - len(scan_universe)
                    _qc_count = len(_quality_candidates) if _quality_candidates else 0
                    print(f"   📡 Pre-score: {_qc_count} quality → top {_max_indicator_stocks} selected → {len(scan_universe)} universe (skipped {_skipped} flat/illiquid from {len(_all_fo)} F&O)")
                    
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
                        # print(f"   📡 Scan universe: {len(APPROVED_UNIVERSE)} curated + {len(self._wildcard_symbols)} wildcards = {len(scan_universe)} total (from {len(_all_fo)} F&O)")
                        pass
                except Exception as _e:
                    _all_fo_syms = set()
                    print(f"   ⚠️ Could not get F&O universe: {_e}")
            
            # Get fresh market data for fixed universe + wild-cards
            # Split into 2 batches with watcher drain between them so breakout
            # triggers don't starve during the ~60-120s indicator calculation.
            # Each batch uses its own kite.quote() + ThreadPoolExecutor (finishes
            # cleanly before the drain fires — no nested API calls).
            if BREAKOUT_WATCHER.get('enabled', False) and len(scan_universe) > 10:
                _mid = len(scan_universe) // 2
                print(f"   📡 Market data: batch 1/{2} ({_mid} stocks)...")
                market_data = self.tools.get_market_data(scan_universe[:_mid])
                # === WATCHER MID-SCAN DRAIN (between market data batches) ===
                try:
                    self._process_breakout_triggers()
                except Exception:
                    pass
                print(f"   📡 Market data: batch 2/{2} ({len(scan_universe) - _mid} stocks)...")
                market_data.update(self.tools.get_market_data(scan_universe[_mid:]))
            else:
                market_data = self.tools.get_market_data(scan_universe)
            
            print(f"   📡 Market data: {len(market_data)} stocks fetched")
            # === WATCHER MID-SCAN DRAIN #1 (after all market data) ===
            try:
                if BREAKOUT_WATCHER.get('enabled', False):
                    self._process_breakout_triggers()
            except Exception:
                pass
            
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
            self._last_market_breadth = _pre_breadth  # Cache for watcher focused scans
            
            try:
                from options_trader import get_intraday_scorer, IntradaySignal
                _scorer = get_intraday_scorer()
                for _sym, _d in sorted_data:
                    if not isinstance(_d, dict) or 'ltp' not in _d:
                        continue
                    # Skip symbols already traded by breakout watcher this cycle
                    if _sym in self._watcher_fired_this_session:
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
                _scored_above_52 = sum(1 for s in _pre_scores.values() if s >= 52)
                _scored_above_45 = sum(1 for s in _pre_scores.values() if s >= 45)
                print(f"   📊 Scored {len(_pre_scores)} stocks: {_scored_above_52} ≥52 (tradeable), {_scored_above_45} ≥45")
                _top5 = sorted(_pre_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                if _top5:
                    _top5_str = " | ".join(f"{s.replace('NSE:', '')}={v:.0f}" for s, v in _top5)
                    print(f"   🏆 Top 5: {_top5_str}")
                
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
                                    # print(f"   📊 Sector indices loaded: {', '.join(sorted(self._sector_5min_cache.keys()))}")
                                    pass
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
                        # === PERF: Run ML on ALL stocks for GMM DR scores ===
                        # Sniper/TEST_GMM need ml_down_risk_score on every stock.
                        # ML runs in parallel (8 threads, ~50ms/stock) so 50 stocks ≈ 300ms.
                        # Previously threshold=40 caused 46/50 stocks to have no DR score,
                        # blocking sniper entirely (no_ml=46).
                        _ML_SCORE_THRESHOLD = 0
                        _ml_eligible = [s for s in list(_pre_scores.keys()) if _pre_scores.get(s, 0) >= _ML_SCORE_THRESHOLD]
                        _ml_skipped = len(_pre_scores) - len(_ml_eligible)
                        with ThreadPoolExecutor(max_workers=8) as _ml_executor:
                            _ml_futures = {_ml_executor.submit(_predict_one, s): s for s in _ml_eligible}
                            for _fut in as_completed(_ml_futures):
                                _sym, _pred = _fut.result()
                                if _pred:
                                    _ml_predictions[_sym] = _pred
                        
                        # ── HERD DETECTION: Detect when ML gives same signal to all stocks ──
                        # When >80% of DIRECTIONAL stocks get the same signal, it MIGHT mean the
                        # model is responding to a market-wide factor (nifty crash, OI stale,
                        # extreme day returns) rather than stock-specific signals.
                        # FIX Mar-05: TIERED response instead of nuclear flatten-all.
                        #   - Sector-aligned stocks KEEP their signal (genuine sector play)
                        #   - High-confidence stocks KEEP signal (stock-specific alpha)
                        #   - Low-confidence stocks get boost reduced (not zeroed)
                        #   - Signals are NEVER blanket-forced to FLAT (kills all trading)
                        # FIX Mar-04: Only count directional signals (UP/DOWN) for herd ratio.
                        _herd_mode = False
                        _herd_signal = None
                        _directional_signals = [p.get('ml_signal') for p in _ml_predictions.values() if p.get('ml_signal') in ('UP', 'DOWN')]
                        _all_valid = [p.get('ml_signal') for p in _ml_predictions.values() if p.get('ml_signal') not in ('UNKNOWN', None)]
                        if len(_directional_signals) >= 8:
                            from collections import Counter as _Counter
                            _signal_counts = _Counter(_directional_signals)
                            _dominant_signal, _dominant_count = _signal_counts.most_common(1)[0]
                            _herd_ratio = _dominant_count / len(_directional_signals)
                            if _herd_ratio >= 0.80:
                                _herd_mode = True
                                _herd_signal = _dominant_signal
                                
                                _cur_breadth = getattr(self, '_last_market_breadth', 'MIXED')
                                _herd_vs_breadth_conflict = (
                                    (_dominant_signal == 'UP' and _cur_breadth == 'BEARISH') or
                                    (_dominant_signal == 'DOWN' and _cur_breadth == 'BULLISH')
                                )
                                
                                # Sector + confidence caches for tiered response
                                _h_stock_to_sector = getattr(self, '_stock_to_sector', {})
                                _h_sector_changes = getattr(self, '_sector_index_changes_cache', {})
                                _h_survived = 0
                                _h_reduced = 0
                                _h_sector_kept = 0
                                _h_conf_kept = 0
                                
                                for _hp_sym, _hp in _ml_predictions.items():
                                    _hp['ml_herd_signal'] = True
                                    _hp_conf = _hp.get('ml_confidence', 0)
                                    _hp_sig = _hp.get('ml_signal')
                                    _hp_sym_clean = _hp_sym.replace('NSE:', '')
                                    
                                    # --- Tier 1: Sector-aligned → KEEP signal ---
                                    # If stock's sector is moving in the SAME direction as herd,
                                    # this is a genuine sector play, not noise.
                                    _hp_sector_aligned = False
                                    _hp_sec_match = _h_stock_to_sector.get(_hp_sym_clean)
                                    if _hp_sec_match and _h_sector_changes:
                                        _hp_sec_name, _hp_sec_idx = _hp_sec_match
                                        _hp_sec_chg = _h_sector_changes.get(_hp_sec_idx)
                                        if _hp_sec_chg is not None:
                                            if _dominant_signal == 'DOWN' and _hp_sec_chg <= -0.8:
                                                _hp_sector_aligned = True
                                            elif _dominant_signal == 'UP' and _hp_sec_chg >= 0.8:
                                                _hp_sector_aligned = True
                                    
                                    if _hp_sector_aligned and _hp_sig == _dominant_signal:
                                        # Sector confirms direction → keep signal, halve boost
                                        _hp['ml_score_boost'] = round(_hp.get('ml_score_boost', 0) * 0.5, 2)
                                        _hp['ml_herd_survived'] = True
                                        _hp['ml_herd_reason'] = 'sector_aligned'
                                        _h_sector_kept += 1
                                        _h_survived += 1
                                        continue
                                    
                                    # --- Tier 2: High confidence → KEEP signal ---
                                    # Stock with ml_confidence >= 0.72 may have genuine
                                    # stock-specific features, not just market-wide noise.
                                    if _hp_conf >= 0.72 and _hp_sig == _dominant_signal:
                                        _hp['ml_score_boost'] = round(_hp.get('ml_score_boost', 0) * 0.5, 2)
                                        _hp['ml_herd_survived'] = True
                                        _hp['ml_herd_reason'] = 'high_confidence'
                                        _h_conf_kept += 1
                                        _h_survived += 1
                                        continue
                                    
                                    # --- Tier 3: Everyone else → reduce boost, keep signal direction ---
                                    # Don't flatten to FLAT — that kills all trading.
                                    # Instead: reduce boost by 70%, mark herd_caution.
                                    # Downstream strategies (TEST_XGB, etc.) have their own
                                    # sector/breadth gates to filter bad trades.
                                    _hp['ml_score_boost'] = round(_hp.get('ml_score_boost', 0) * 0.3, 2)
                                    _hp['ml_herd_caution'] = True
                                    _h_reduced += 1
                                
                                if _herd_vs_breadth_conflict:
                                    _surv_detail = f"sector_kept={_h_sector_kept}, conf_kept={_h_conf_kept}"
                                    print(f"   ⚠️ ML HERD + BREADTH CONFLICT: {_dominant_count}/{len(_directional_signals)} predict {_dominant_signal} but breadth={_cur_breadth} → {_h_survived} survived ({_surv_detail}), {_h_reduced} boost-reduced")
                                else:
                                    print(f"   ⚠️ ML HERD DETECTED: {_dominant_count}/{len(_directional_signals)} directional ({_herd_ratio:.0%}) predict {_dominant_signal} → boosts halved/reduced ({_h_survived} survived, {_h_reduced} reduced)")
                        
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
                            _herd_survived = sum(1 for r in _ml_results.values() if r.get('ml_herd_survived'))
                            _herd_cautioned = sum(1 for r in _ml_results.values() if r.get('ml_herd_caution'))
                            _herd_note = ""
                            if _herd_mode:
                                _herd_note = f" | HERD({_herd_signal}: {_herd_survived} survived, {_herd_cautioned} cautioned)"
                            print(f"   🧠 ML: {len(_ml_results)} analyzed | {_ml_boosted} score-adjusted | {_ml_up} UP | {_ml_down} DOWN | {_ml_flat} FLAT | {_ml_caution} CAUTION{_skip_note}{_herd_note}")
                        else:
                            _eligible = sum(1 for s in _pre_scores if (_candle_cache.get(s) is not None and len(_candle_cache.get(s, [])) >= 50) or (_daily_cache.get(s) is not None and len(_daily_cache.get(s, [])) >= 50))
                            # print(f"   🧠 ML: NO predictions (candle_cache={_candle_count}, daily_cache={_daily_count}, eligible={_eligible}/{len(_pre_scores)})")
                            pass
                except Exception as _ml_err:
                    print(f"   ⚠️ ML predictor error (non-fatal, continuing without ML): {_ml_err}")
                
                # Store ML results for downstream use (GPT prompt, sizing, etc.)
                self._cycle_ml_results = _ml_results
                
                # === WATCHER MID-SCAN DRAIN #2 (after ML, ~120-180s into scan) ===
                try:
                    if BREAKOUT_WATCHER.get('enabled', False):
                        print(f"   ⚡ Watcher mid-scan drain (post-ML)...")
                        self._process_breakout_triggers()
                except Exception:
                    pass
                
                # === OI FLOW OVERLAY: Adjust ML predictions with live options chain data ===
                # Only analyze top-scoring F&O stocks to minimize API calls (max 15)
                # FAIL-SAFE: If OI analyzer crashes, _ml_results stays unchanged
                _oi_results = {}
                try:
                    if self._oi_analyzer and _ml_results:
                        # Pick top stocks worth analyzing OI for (scored >=30 + have ML data)
                        # Widened from 15→30 and pre_score 45→30 so sniper strategies
                        # (OI Unwinding, PCR Extreme) see more of the universe.
                        _oi_candidates = [
                            sym for sym in list(_pre_scores.keys())
                            if _pre_scores.get(sym, 0) >= 30 and sym in _ml_results
                        ]
                        # Sort by score descending, limit to 30 to balance API cost vs sniper coverage
                        _oi_candidates.sort(key=lambda s: _pre_scores.get(s, 0), reverse=True)
                        _oi_candidates = _oi_candidates[:30]
                        
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
                                # Include ALL OI data in _oi_results so sniper strategies
                                # (OIUnwinding, PCRExtreme) can see stocks with NEUTRAL flow_bias
                                # that still have valid PCR/unwinding signals.
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
                            _directional = sum(1 for d in _oi_results.values() if d.get('flow_bias') != 'NEUTRAL')
                            _has_unwinding = sum(1 for d in _oi_results.values()
                                                 if d.get('nse_oi_buildup') in ('LONG_UNWINDING', 'SHORT_COVERING'))
                            _has_pcr_extreme = sum(1 for d in _oi_results.values()
                                                   if d.get('pcr_oi') and (d['pcr_oi'] >= 1.2 or d['pcr_oi'] <= 0.7))
                            print(f"   📊 OI: {len(_oi_candidates)} analyzed | {len(_oi_results)} with data | "
                                  f"{_directional} directional | {_oi_adjusted_count} ML-adjusted"
                                  f"{f' | {_has_unwinding} unwinding' if _has_unwinding else ''}"
                                  f"{f' | {_has_pcr_extreme} PCR-extreme' if _has_pcr_extreme else ''}")
                except Exception as _oi_err:
                    print(f"   ⚠️ OI analyzer error (non-fatal): {_oi_err}")
                
                # === WATCHER MID-SCAN DRAIN #3 (after OI, ~180-240s into scan) ===
                try:
                    if BREAKOUT_WATCHER.get('enabled', False):
                        print(f"   ⚡ Watcher mid-scan drain (post-OI)...")
                        self._process_breakout_triggers()
                except Exception:
                    pass
                
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
                    # print(f"   📊 OI CROSS-VAL: {_oi_score_adjusted} stocks penalized for direction conflict")
                    pass

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
                self._stock_to_sector = _stock_to_sector  # Cache for watcher focused scans

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
                    # print(f"   🏭 SECTOR CROSS-VAL: {_sector_score_adjusted} stocks penalized — {', '.join(_sector_penalized_details[:5])}")
                    pass
                elif _sector_index_changes:
                    _bearish_secs = [f"{k.replace('NSE:NIFTY ', '')}:{v:+.1f}%" for k, v in _sector_index_changes.items() if v <= -1.0]
                    if _bearish_secs:
                        # print(f"   🏭 SECTOR INDEX: Bearish sectors: {', '.join(_bearish_secs)} (no scored stocks conflicting)")
                        pass

                # Cache sector index changes for GPT prompt section
                self._sector_index_changes_cache = _sector_index_changes

                # Store OI results for GPT prompt
                self._cycle_oi_results = _oi_results
                
                # === DR SCORE LEADERBOARD: Show highest & lowest DR scores ===
                try:
                    _dr_entries = []
                    for _dr_sym, _dr_ml in _ml_results.items():
                        _dr_val = _dr_ml.get('ml_down_risk_score')
                        if _dr_val is not None:
                            _dr_regime = _dr_ml.get('ml_gmm_regime_used', 'UP')
                            _dr_flag_active = _dr_ml.get('ml_up_flag', False) if _dr_regime == 'UP' else _dr_ml.get('ml_down_flag', False)
                            _dr_entries.append((_dr_sym.replace('NSE:', ''), _dr_val, _dr_flag_active))
                    if _dr_entries:
                        _dr_entries.sort(key=lambda x: x[1])
                        _dr_lowest = _dr_entries[:2]
                        _dr_highest = _dr_entries[-2:][::-1]
                        _low_str = ' | '.join(f"{s} {d:.4f}{'⚑' if f else ''}" for s, d, f in _dr_lowest)
                        _high_str = ' | '.join(f"{s} {d:.4f}{'⚑' if f else ''}" for s, d, f in _dr_highest)
                        # print(f"\n   📊 DR SCORE LEADERBOARD ({len(_dr_entries)} stocks):")
                        # print(f"      🟢 LOWEST  (safest): {_low_str}")
                        # print(f"      🔴 HIGHEST (riskiest): {_high_str}")
                        # Count extreme zones
                        _ultra_low = [s for s, d, f in _dr_entries if d < 0.06]
                        _ultra_high = [s for s, d, f in _dr_entries if d > 0.75]
                        if _ultra_low:
                            # print(f"      🧪 TEST_GMM ZONE (dr<6%): {', '.join(_ultra_low)} ({len(_ultra_low)} stocks)")
                            pass
                        if _ultra_high:
                            # print(f"      ⚡ EXTREME HIGH (dr>75%): {', '.join(_ultra_high)} ({len(_ultra_high)} stocks)")
                            pass
                except Exception as _lb_err:
                    print(f"   ⚠️ DR leaderboard error (non-fatal): {_lb_err}")

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
                
                # === TEST_GMM: Pure DR model play (bypass ALL gates) ===
                # DR < 6% = model extremely confident no downside → BUY CALL
                # No risk gates, no scores, no XGB — pure GMM conviction play.
                try:
                    _test_gmm_placed = self._place_test_gmm_trades(
                        _ml_results, _pre_scores, market_data, datetime.now().strftime('%H:%M:%S')
                    )
                except Exception as _tg_err:
                    print(f"   ⚠️ TEST_GMM error (non-fatal): {_tg_err}")
                    _test_gmm_placed = []

                # === TEST_XGB: Pure XGBoost play (bypass GMM, smart_score) ===
                # High P(MOVE) + clear directional lean → pure XGB conviction trade.
                try:
                    _test_xgb_placed = self._place_test_xgb_trades(
                        _ml_results, _pre_scores, market_data, datetime.now().strftime('%H:%M:%S')
                    )
                except Exception as _tx_err:
                    print(f"   ⚠️ TEST_XGB error (non-fatal): {_tx_err}")
                    _test_xgb_placed = []

                # === ARBTR: Sector Arbitrage — laggard convergence play ===
                # When sector index moves but peer stock lags, trade the laggard.
                try:
                    _arbtr_placed = self._place_arbtr_trades(
                        _ml_results, _pre_scores, market_data,
                        _sector_index_changes, datetime.now().strftime('%H:%M:%S')
                    )
                except Exception as _arb_err:
                    print(f"   ⚠️ ARBTR error (non-fatal): {_arb_err}")
                    _arbtr_placed = []

                # === SNIPER STRATEGIES: OI Unwinding + PCR Extreme ===
                # Scans OI data for high-edge reversal / contrarian setups.
                # Independent from GMM Sniper — these use OI flow signals + GMM confirmation.
                try:
                    if getattr(self, '_sniper_engine', None) and _oi_results and _ml_results:
                        # --- Diagnostic: show OI data availability for snipers ---
                        _snp_unwind_syms = [s.replace('NSE:', '') for s, d in _oi_results.items()
                                            if d.get('nse_oi_buildup') in ('LONG_UNWINDING', 'SHORT_COVERING')]
                        _snp_pcr_ext = [(s.replace('NSE:', ''), d.get('pcr_oi', 0))
                                        for s, d in _oi_results.items()
                                        if d.get('pcr_oi') and (d['pcr_oi'] >= 1.2 or d['pcr_oi'] <= 0.7)]
                        if _snp_unwind_syms or _snp_pcr_ext:
                            _diag_parts = []
                            if _snp_unwind_syms:
                                _diag_parts.append(f"Unwinding: {', '.join(_snp_unwind_syms[:5])}")
                            if _snp_pcr_ext:
                                _diag_parts.append(f"PCR↗↘: {', '.join(f'{s}({p:.2f})' for s, p in _snp_pcr_ext[:5])}")
                            print(f"   🔎 Sniper data: {' | '.join(_diag_parts)}")
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
                                _oiu_dr_lbl = self._dr_tag(_oiu_cand.get('ml_data', {}).get('gmm_model', {}).get('gmm_regime_used', 'UP'))
                                print(f"\n   🔫 SNIPER-OIUnwinding: {_oiu_cand['sym_clean']} ({_oiu_cand['direction']}) "
                                      f"| {_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f} "
                                      f"| chg={_oiu_cand.get('change_pct', 0):+.1f}% RSI={_oiu_cand.get('rsi', 50):.0f} "
                                      f"| exhaust={_oiu_cand.get('exhaustion_score', 0):.2f} "
                                      f"| spot→SR {_oiu_cand['dist_from_sr_pct']:.1f}% "
                                      f"| {_oiu_dr_lbl}={_oiu_cand['dr_score']:.4f} | smart={_oiu_cand['smart_score']:.1f} "
                                      f"| {_oiu_lot_mult}x lots")
                                _oiu_result = self.tools.place_option_order(
                                    underlying=_oiu_cand['sym'], direction=_oiu_cand['direction'],
                                    strike_selection="ATM", use_intraday_scoring=False,
                                    lot_multiplier=_oiu_lot_mult,
                                    rationale=(f"SNIPER_OI_UNWINDING: {_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f}, "
                                              f"spot→SR={_oiu_cand['dist_from_sr_pct']:.1f}%, {_oiu_dr_lbl}={_oiu_cand['dr_score']:.4f}, "
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
                                                      reason=f"OI={_oiu_cand['oi_buildup']} str={_oiu_cand['oi_strength']:.2f}, {_oiu_dr_lbl}={_oiu_cand['dr_score']:.4f}",
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
                                _pcr_dr_lbl = self._dr_tag(_pcr_cand.get('ml_data', {}).get('gmm_model', {}).get('gmm_regime_used', 'UP'))
                                _pcr_thr_info = f"thr={_pcr_cand.get('adaptive_oversold_thr', 1.35):.2f}/{_pcr_cand.get('adaptive_overbought_thr', 0.65):.2f}"
                                _pcr_idx_tag = ' IDX✓' if _pcr_cand.get('index_agrees') else ''
                                print(f"\n   🔫 SNIPER-PCRExtreme: {_pcr_cand['sym_clean']} ({_pcr_cand['direction']}) "
                                      f"| PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}) "
                                      f"| edge={_pcr_cand['pcr_edge']:.3f} {_pcr_thr_info}{_pcr_idx_tag} "
                                      f"| {_pcr_dr_lbl}={_pcr_cand['dr_score']:.4f} | smart={_pcr_cand['smart_score']:.1f} "
                                      f"| {_pcr_lot_mult}x lots")
                                _pcr_result = self.tools.place_option_order(
                                    underlying=_pcr_cand['sym'], direction=_pcr_cand['direction'],
                                    strike_selection="ATM", use_intraday_scoring=False,
                                    lot_multiplier=_pcr_lot_mult,
                                    rationale=(f"SNIPER_PCR_EXTREME: PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}), "
                                              f"edge={_pcr_cand['pcr_edge']:.3f}, {_pcr_dr_lbl}={_pcr_cand['dr_score']:.4f}, "
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
                                                      reason=f"PCR={_pcr_cand['blended_pcr']:.3f} ({_pcr_cand['pcr_regime']}), {_pcr_dr_lbl}={_pcr_cand['dr_score']:.4f}",
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
                                # print(f"   📋 DhanHQ OI snapshots logged: {_dhan_logged} stocks")
                                pass
                        
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
                                # print(f"   📋 NSE OI snapshots logged: {_nse_logged} stocks")
                                pass
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
                            # print(f"   ⭐ WILDCARD CHOP-BLOCKED: {symbol} — {chop_reason}")
                            pass
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
                    # [TIGHTENED Mar 2] ORB now requires EXPLOSIVE volume (was HIGH+) + HTF must not be NEUTRAL
                    if orb_signal == "BREAKOUT_UP" and volume_regime == "EXPLOSIVE" and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "BUY"
                        # HTF Check: Block BUY if HTF is BEARISH or NEUTRAL (tightened — need trend confirmation)
                        if htf_trend in ["BEARISH", "NEUTRAL"] and htf_ema_slope != "RISING":
                            htf_blocked = True
                            setup = "⛔HTF-BLOCKS-ORB-BUY" if htf_trend == "BEARISH" else "⛔HTF-NEUTRAL-ORB-BUY"
                        elif self._is_orb_trade_allowed(symbol, "UP"):
                            setup = "🚀ORB-BREAKOUT-BUY"
                            regime_signals.append(f"  🚀 {symbol}: ORB↑ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BULL | HTF:{htf_trend}")
                        else:
                            setup = "⛔ORB-UP-ALREADY-TAKEN"
                    elif orb_signal == "BREAKOUT_DOWN" and volume_regime == "EXPLOSIVE" and ema_regime in ["EXPANDING", "NORMAL"]:
                        intended_direction = "SELL"
                        # HTF Check: Block SELL if HTF is BULLISH or NEUTRAL (tightened)
                        if htf_trend in ["BULLISH", "NEUTRAL"] and htf_ema_slope != "FALLING":
                            htf_blocked = True
                            setup = "⛔HTF-BLOCKS-ORB-SHORT" if htf_trend == "BULLISH" else "⛔HTF-NEUTRAL-ORB-SHORT"
                        elif self._is_orb_trade_allowed(symbol, "DOWN"):
                            setup = "🔻ORB-BREAKOUT-SHORT"
                            regime_signals.append(f"  🔻 {symbol}: ORB↓ +{orb_strength:.1f}% | Vol:{volume_regime} | EMA:BEAR")
                        else:
                            setup = "⛔ORB-DOWN-ALREADY-TAKEN"
                    elif ema_regime == "COMPRESSED" and volume_regime in ["HIGH", "EXPLOSIVE"]:
                        setup = "💥SQUEEZE-PENDING"
                        regime_signals.append(f"  💥 {symbol}: EMA SQUEEZE + High Volume - BREAKOUT IMMINENT | HTF:{htf_trend}")
                    elif price_vs_vwap == "ABOVE_VWAP" and vwap_slope == "RISING" and rsi < 60:
                        # [TIGHTENED Mar 2] VWAP needs HTF aligned + ADX > 20
                        _vwap_adx = data.get('adx', 20) if isinstance(data, dict) else 20
                        if htf_trend == "BEARISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-BUY-HTF-CONFLICT"
                        elif htf_trend == "NEUTRAL":
                            setup = "⚠️VWAP-BUY-HTF-NEUTRAL"  # No trend confirmation
                        elif _vwap_adx < 20:
                            setup = f"⚠️VWAP-BUY-WEAK-ADX({_vwap_adx:.0f})"
                        else:
                            setup = "📈VWAP-TREND-BUY"
                    elif price_vs_vwap == "BELOW_VWAP" and vwap_slope == "FALLING" and rsi > 40:
                        # [TIGHTENED Mar 2] VWAP needs HTF aligned + ADX > 20
                        _vwap_adx = data.get('adx', 20) if isinstance(data, dict) else 20
                        if htf_trend == "BULLISH" and volume_regime not in ["HIGH", "EXPLOSIVE"]:
                            setup = "⚠️VWAP-SHORT-HTF-CONFLICT"
                        elif htf_trend == "NEUTRAL":
                            setup = "⚠️VWAP-SHORT-HTF-NEUTRAL"  # No trend confirmation
                        elif _vwap_adx < 20:
                            setup = f"⚠️VWAP-SHORT-WEAK-ADX({_vwap_adx:.0f})"
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
            self._cycle_detected_setups = {}  # Store detected setup per symbol for GPT execution
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
                    
                    # [FIX Mar 4] ORB_BREAKOUT removed from GPT pipeline.
                    # ORB trades now fire ONLY through the watcher pipeline which has
                    # 8 quality gates (score, ADX, FT, OI, ML flat veto, etc.).
                    # Previously, GPT path bypassed ALL gates → entered with smart_score=0,
                    # ml_move_prob as low as 0.117 (SHRIRAMFIN), causing losses.
                    # if orb_sig == "BREAKOUT_UP" and vol_reg == "EXPLOSIVE":
                    #     setup_type = "ORB_BREAKOUT"
                    #     direction = "BUY"
                    # elif orb_sig == "BREAKOUT_DOWN" and vol_reg == "EXPLOSIVE":
                    #     setup_type = "ORB_BREAKOUT"
                    #     direction = "SELL"
                    # [DISABLED Mar 6] VWAP_TREND removed from GPT pipeline.
                    # Root cause: GPT path places at smart_score=0, bypasses all ML/OI gates,
                    # and orphaned trades lose SL management. ASTRAL lost ₹24,482 (49% against).
                    # VWAP signals should only fire through watcher pipeline (if re-enabled).
                    # if pvw == "ABOVE_VWAP" and vs == "RISING" and ema_reg == "EXPANDING":
                    #     _adx_val = data.get('adx', 20)
                    #     if _adx_val >= 25 and htf_a not in ["NEUTRAL", "BEARISH_ALIGNED"]:
                    #         setup_type = "VWAP_TREND"
                    #         direction = "BUY"
                    # elif pvw == "BELOW_VWAP" and vs == "FALLING" and ema_reg == "EXPANDING":
                    #     _adx_val = data.get('adx', 20)
                    #     if _adx_val >= 25 and htf_a not in ["NEUTRAL", "BULLISH_ALIGNED"]:
                    #         setup_type = "VWAP_TREND"
                    #         direction = "SELL"
                    # [DISABLED Mar 6] ALL GPT pipeline setup detection removed.
                    # GPT path bypasses all quality gates (smart_score=0, no ML veto, no OI check).
                    # Trades placed via GPT lost ₹27K+ on VWAP alone. Only watcher + model-tracker
                    # pipelines have proper gating. GPT now only provides analysis, not trade placement.
                    # if pvw == "ABOVE_VWAP" and vs == "RISING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                    #     setup_type = "MOMENTUM"
                    #     direction = "BUY"
                    # elif pvw == "BELOW_VWAP" and vs == "FALLING" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                    #     setup_type = "MOMENTUM"
                    #     direction = "SELL"
                    # elif ema_reg == "COMPRESSED" and vol_reg in ["HIGH", "EXPLOSIVE"]:
                    #     setup_type = "EMA_SQUEEZE"
                    #     direction = "BUY" if data.get('change_pct', 0) > 0 else "SELL"
                    # elif rsi < 30 and htf_a != "BEARISH_ALIGNED":
                    #     setup_type = "RSI_REVERSAL"
                    #     direction = "BUY"
                    # elif rsi > 70 and htf_a != "BULLISH_ALIGNED":
                    #     setup_type = "RSI_REVERSAL"
                    #     direction = "SELL"
                    #
                    # WILDCARD MOMENTUM SETUP: disabled with all GPT setups
                    # if not setup_type and is_wildcard:
                    #     wc_chg = self._wildcard_change.get(symbol, 0)
                    #     if abs(wc_chg) >= 2.0 and vol_reg in ['HIGH', 'EXPLOSIVE']:
                    #         setup_type = "WILDCARD_MOMENTUM"
                    #         direction = "BUY" if wc_chg > 0 else "SELL"
                    
                    # [DISABLED Mar 6] VWAP_GRIND removed from GPT pipeline.
                    # Same issue as VWAP_TREND: bypasses all quality gates, smart_score=0.
                    # BLUESTARCO won +₹7,170 but VEDL/ASTRAL lost ₹27,047 combined.
                    # if not setup_type and orb_sig == "INSIDE_ORB" and vol_reg == 'EXPLOSIVE':
                    #     _grind_ltp = data.get('ltp', 0)
                    #     _grind_open = data.get('open', _grind_ltp)
                    #     _grind_vwap = data.get('vwap', 0)
                    #     _grind_move = abs((_grind_ltp - _grind_open) / _grind_open * 100) if _grind_open > 0 else 0
                    #     if _grind_move >= 1.2 and _grind_ltp < _grind_vwap and _grind_ltp < _grind_open:
                    #         setup_type = "VWAP_GRIND"
                    #         direction = "SELL"
                    #     elif _grind_move >= 1.2 and _grind_ltp > _grind_vwap and _grind_ltp > _grind_open:
                    #         setup_type = "VWAP_GRIND"
                    #         direction = "BUY"
                    
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
                        # GPT-selected minimum: score + micro offset must reach 70
                        # (raw score ~58+ since micro adds ~12) [was 66, tightened Mar 2 +4pts]
                        if _fno_score > 0 and _fno_score + _micro_absent_offset < 70:
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
                        
                        # Store detected setup_type for downstream GPT execution routing
                        self._cycle_detected_setups[symbol] = setup_type
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
                                    # print(f"      🧠 ML BOOST: {symbol} {_ds_ml_signal} prob={_ds_max_dir:.0%} → +20 priority")
                                elif _ds_ml_signal in ('UP', 'DOWN') and _ds_max_dir >= 0.40:
                                    ds_priority += 10  # Moderate ML confirmation
                                elif _ds_ml_signal == 'FLAT' and _ds_prob_flat >= 0.70:
                                    # print(f"      🧠 ML SKIP: {symbol} FLAT prob={_ds_prob_flat:.0%} → skipping debit spread")
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
                                    # print(f"   ✅ Liquidity OK: {liq_reason}")
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
                                    # print(f"   ℹ️ Debit spread not viable for {symbol}: {result.get('error', 'unknown')}")
                                    pass
                            except Exception as e:
                                print(f"   ⚠️ Debit spread attempt failed for {symbol}: {e}")
                        
                        if debit_spread_placed:
                            print(f"\n   🚀 PROACTIVE DEBIT SPREADS PLACED: {', '.join(debit_spread_placed)}")
                        elif debit_candidates:
                            # print(f"\n   ℹ️ {len(debit_candidates)} debit spread candidates found, none viable this cycle")
                            pass
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
                    # Track index vs stock eligibility SEPARATELY so stock IC scan
                    # only runs on stock expiry days (DTE=0), not on index-only expiry days.
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    if not hasattr(self, '_ic_eligible_date') or self._ic_eligible_date != today_str:
                        self._ic_idx_eligible = False
                        self._ic_stk_eligible = False
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
                            idx_max_dte = idx_mode.get('max_dte', 0)
                            stk_max_dte = stk_mode.get('max_dte', 0)
                            today_date = datetime.now().date()
                            
                            # Check INDEX expiry
                            for idx_sym in IC_INDEX_SYMBOLS:
                                try:
                                    idx_expiry_sel = ExpirySelection[idx_mode.get('prefer_expiry', 'CURRENT_WEEK')]
                                    idx_expiry = _ot_check.chain_fetcher.get_nearest_expiry(idx_sym, idx_expiry_sel)
                                    if idx_expiry:
                                        from datetime import date as _date
                                        exp_d = idx_expiry if isinstance(idx_expiry, _date) and not isinstance(idx_expiry, datetime) else idx_expiry.date() if hasattr(idx_expiry, 'date') else idx_expiry
                                        idx_dte = (exp_d - today_date).days
                                        if idx_dte <= idx_max_dte:
                                            self._ic_idx_eligible = True
                                            print(f"   🦅 IC: Index {idx_sym} expiry {idx_expiry} DTE={idx_dte} ✅")
                                            break
                                except Exception:
                                    pass
                            
                            # Check STOCK expiry (separately — don't let index eligibility bypass this)
                            try:
                                stk_expiry_sel = ExpirySelection[stk_mode.get('prefer_expiry', 'CURRENT_MONTH')]
                                stk_expiry = _ot_check.chain_fetcher.get_nearest_expiry("NSE:RELIANCE", stk_expiry_sel)
                                if stk_expiry:
                                    from datetime import date as _date
                                    exp_d = stk_expiry if isinstance(stk_expiry, _date) and not isinstance(stk_expiry, datetime) else stk_expiry.date() if hasattr(stk_expiry, 'date') else stk_expiry
                                    stk_dte = (exp_d - today_date).days
                                    if stk_dte <= stk_max_dte:
                                        self._ic_stk_eligible = True
                                        print(f"   🦅 IC: Stock expiry {stk_expiry} DTE={stk_dte} ✅")
                            except Exception:
                                pass
                            
                            if not self._ic_idx_eligible and not self._ic_stk_eligible:
                                print(f"   🦅 IC SKIPPED: Not expiry day (index max_dte={idx_max_dte}, stock max_dte={stk_max_dte})")
                            elif self._ic_idx_eligible and not self._ic_stk_eligible:
                                print(f"   🦅 IC: Index expiry only — stock IC scan will be SKIPPED (saving bandwidth)")
                        except Exception as dte_err:
                            print(f"   ⚠️ IC DTE pre-check failed: {dte_err} — will skip IC scan")
                            self._ic_idx_eligible = False
                            self._ic_stk_eligible = False
                    
                    # Only run IC scan if today has eligible expiry AND within time window
                    now_time = datetime.now().time()
                    ic_earliest = datetime.strptime(IRON_CONDOR_CONFIG.get('earliest_entry', '10:30'), '%H:%M').time()
                    ic_cutoff = datetime.strptime(IRON_CONDOR_CONFIG.get('no_entry_after', '12:30'), '%H:%M').time()
                    _any_ic_eligible = self._ic_idx_eligible or self._ic_stk_eligible
                    
                    if _any_ic_eligible and ic_earliest <= now_time <= ic_cutoff:
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
                                    # print(f"   🦅 IC SKIP {idx_symbol}: moved {idx_change:.1f}% (>{max_move}%)")
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
                        # ONLY run if stocks have DTE=0 (expiry day), not on index-only expiry days.
                        if IRON_CONDOR_CONFIG.get('scan_rejected_stocks', True) and self._ic_stk_eligible:
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
                                    # print(f"\n   🦅 STOCK IC SCAN: {symbol} | Quality: {ic_quality}/100 | Chg: {abs(data.get('change_pct', 0)):.2f}% | RSI: {data.get('rsi_14', 50):.0f}")
                                    # print(f"      IC Factors: {reasons_str}")
                                    
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
                # print(f"   📊 DYNAMIC PICKS: GPT allowed up to {_dynamic_max} trades (signal quality {'HIGH' if _dynamic_max > 3 else 'LOW'})")
                pass
            
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
            
            # === BUILD FUTURES OI BUILDUP SUMMARY FOR GPT (fail-safe) ===
            # fut_oi_buildup is the #1 ML feature (46.5% of direction model importance).
            # It classifies institutional positioning from futures OI + price:
            #   +1.0 = LONG BUILDUP   (OI↑ + Price↑) — institutions adding longs, bullish
            #   -1.0 = SHORT BUILDUP  (OI↑ + Price↓) — institutions adding shorts, bearish
            #   +0.5 = SHORT COVERING (OI↓ + Price↑) — shorts exiting, mildly bullish
            #   -0.5 = LONG UNWINDING (OI↓ + Price↓) — longs exiting, mildly bearish
            #    0.0 = NEUTRAL        (no clear signal)
            _fut_buildup_section = 'Futures OI buildup data not available'
            try:
                _foc = getattr(self, '_futures_oi_data', {}) or {}
                if _foc:
                    _buildup_lines = []
                    _buildup_map = {
                        1.0: '🟢LONG_BUILDUP', -1.0: '🔴SHORT_BUILDUP',
                        0.5: '🟡SHORT_COVERING', -0.5: '🟠LONG_UNWINDING',
                    }
                    _buildup_counts = {'LONG_BUILDUP': 0, 'SHORT_BUILDUP': 0,
                                       'SHORT_COVERING': 0, 'LONG_UNWINDING': 0, 'NEUTRAL': 0}
                    _stock_buildups = {}
                    for _fsym, _fdf in _foc.items():
                        if _fdf is not None and len(_fdf) > 0:
                            _last = _fdf.iloc[-1]
                            _bval = float(_last.get('fut_oi_buildup', 0))
                            _basis = float(_last.get('fut_basis_pct', 0))
                            _oi_chg = float(_last.get('fut_oi_change_pct', 0))
                            _5d = float(_last.get('fut_oi_5d_trend', 0))
                            _label = _buildup_map.get(_bval, 'NEUTRAL')
                            if _bval == 0:
                                _label = 'NEUTRAL'
                            # Count
                            _key = _label.split('_', 1)[-1] if '🟢' in _label or '🔴' in _label or '🟡' in _label or '🟠' in _label else _label
                            _key_clean = _label.replace('🟢', '').replace('🔴', '').replace('🟡', '').replace('🟠', '')
                            _buildup_counts[_key_clean] = _buildup_counts.get(_key_clean, 0) + 1
                            if _bval != 0:
                                _stock_buildups[_fsym] = (_label, _basis, _oi_chg, _5d)
                    
                    # Market-wide summary
                    _lb = _buildup_counts.get('LONG_BUILDUP', 0)
                    _sb = _buildup_counts.get('SHORT_BUILDUP', 0)
                    _sc = _buildup_counts.get('SHORT_COVERING', 0)
                    _lu = _buildup_counts.get('LONG_UNWINDING', 0)
                    _neu = _buildup_counts.get('NEUTRAL', 0)
                    _bullish_total = _lb + _sc
                    _bearish_total = _sb + _lu
                    if _bullish_total > _bearish_total * 1.5:
                        _mkt_inst_bias = '🟢 INSTITUTIONAL BIAS: BULLISH'
                    elif _bearish_total > _bullish_total * 1.5:
                        _mkt_inst_bias = '🔴 INSTITUTIONAL BIAS: BEARISH'
                    else:
                        _mkt_inst_bias = '⚖️ INSTITUTIONAL BIAS: MIXED'
                    
                    _buildup_lines.append(f"  {_mkt_inst_bias} (Long Buildup:{_lb} | Short Buildup:{_sb} | Short Covering:{_sc} | Long Unwinding:{_lu} | Neutral:{_neu})")
                    
                    # Per-stock buildups (only for F&O candidates in the prompt)
                    _fno_syms_in_prompt = set()
                    for _opp in fno_opportunities:
                        import re as _re_bld
                        _m = _re_bld.search(r'🎯\s*(\S+):', _opp)
                        if _m:
                            _fno_syms_in_prompt.add(_m.group(1))
                    
                    for _fsym in sorted(_fno_syms_in_prompt):
                        if _fsym in _stock_buildups:
                            _lbl, _bas, _ochg, _5d = _stock_buildups[_fsym]
                            _buildup_lines.append(f"  {_fsym}: {_lbl} | Basis:{_bas:+.2f}% | OI_Chg:{_ochg:+.1f}% | 5d_Trend:{_5d:+.1f}%")
                    
                    if _buildup_lines:
                        _fut_buildup_section = chr(10).join(_buildup_lines)
            except Exception:
                pass  # Futures buildup summary failed — GPT continues without it
            
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

=== 🏗️ INSTITUTIONAL POSITIONING (Futures OI Buildup — #1 ML feature) ===
{_fut_buildup_section}

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
   - 🏗️ INSTITUTIONAL POSITIONING: LONG_BUILDUP + BUY = STRONGEST signal | SHORT_BUILDUP + SELL = STRONGEST signal
   - If Futures OI Buildup CONTRADICTS your direction → SKIP or reduce conviction
   - Institutional Bias (market-wide) helps gauge overall flow — align picks with institutional money
6. Check CONTRARIAN risks (chasing? extended? volume divergence? ML says FLAT?)
7. EXECUTE via tools — place_option_order(underlying, direction) for F&O, place_order() for cash
8. State your reasoning briefly: Setup | Score | ML Signal | OI Bias | Why

⚠️ CRITICAL RULES:
- Do NOT pick stocks outside the F&O READY SIGNALS list. They are NOT tradeable.
- Do NOT pick stocks scoring below 56. The scorer WILL block them.
- If ML says FLAT (🧠ML:FLAT) on a stock, do NOT pick it for directional trades — it will likely not move.
- If ML says MOVE and OI confirms direction — this is the STRONGEST signal. Prioritize these.
- LONG_BUILDUP + ML:UP + OI:BULLISH = TRIPLE CONFLUENCE — highest priority pick.
- SHORT_BUILDUP or LONG_UNWINDING on a BUY candidate — be very cautious, institutions are bearish.
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
                
                # [DISABLED Mar 6] GPT direct-placement completely disabled.
                # GPT path bypasses all quality gates — no smart_score, no ML veto,
                # no OI alignment, no GMM gating. Only watcher + model-tracker place trades.
                if False and unplaced and len(unplaced) <= 5:
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
                                    # print(f"   🔄 Using scored direction for {sym}: {direction} (GPT text unparseable)")
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
                            _gpt_dr_lbl = self._dr_tag(_gpt_ml)
                            _gpt_xgb_disagrees = False
                            if direction == 'BUY' and _gpt_ml_signal == 'DOWN' and _gpt_move_prob >= _gpt_dir_cfg.get('min_xgb_confidence', 0.55):
                                _gpt_xgb_disagrees = True
                            elif direction == 'SELL' and _gpt_ml_signal == 'UP' and _gpt_move_prob >= _gpt_dir_cfg.get('min_xgb_confidence', 0.55):
                                _gpt_xgb_disagrees = True
                            
                            if _gpt_xgb_disagrees:
                                # XGB actively opposes GPT trade direction.
                                # FIXED: use gmm_confirms (clean = confirms XGB), not raw dr_flag
                                _gpt_gmm_confirms = _gpt_ml.get('ml_gmm_confirms_direction', False)
                                if _gpt_gmm_confirms:
                                    # GMM clean (no anomaly) → confirms XGB's opposing direction
                                    # XGB + GMM both oppose → either FLIP or BLOCK
                                    _ovr_ok, _ovr_reason = self._ml_override_allowed(sym, _gpt_ml, _gpt_dr_score, path='GPT', p_score=_pre_scores.get(f'NSE:{sym}', 0))
                                    if _ovr_ok:
                                        # Gates passed → FLIP to XGB direction
                                        old_direction = direction
                                        direction = 'BUY' if _gpt_ml_signal == 'UP' else 'SELL'
                                        _gpt_was_flipped = True
                                        print(f"   🔄 GPT ML_OVERRIDE_WGMM: {sym} — XGB={_gpt_ml_signal} + GMM clean "
                                              f"({_gpt_dr_lbl}={_gpt_dr_score:.3f}) → FLIPPED {old_direction}→{direction}")
                                    else:
                                        # Gates failed → BLOCK (XGB+GMM both oppose but gates not met)
                                        print(f"   🚫 GPT ML_OVR BLOCKED: {sym} — {_ovr_reason}")
                                        continue
                                else:
                                    # GMM anomaly → check if anomaly actually CONFIRMS GPT direction
                                    _gpt_up_flag = _gpt_ml.get('ml_up_flag', False)
                                    _gpt_down_flag = _gpt_ml.get('ml_down_flag', False)
                                    _gpt_anomaly_confirms = False
                                    if direction == 'BUY' and _gpt_down_flag and not _gpt_up_flag:
                                        _gpt_anomaly_confirms = True  # bounce confirms BUY
                                    elif direction == 'SELL' and _gpt_up_flag and not _gpt_down_flag:
                                        _gpt_anomaly_confirms = True  # crash confirms SELL
                                    
                                    if _gpt_anomaly_confirms:
                                        print(f"   🎯 GPT GMM_CONTRARIAN: {sym} — XGB={_gpt_ml_signal} opposes but "
                                              f"anomaly CONFIRMS GPT={direction} (UP={_gpt_ml.get('ml_up_score',0):.3f} DN={_gpt_ml.get('ml_down_score',0):.3f})")
                                    else:
                                        # Anomaly opposes or conflicting → BLOCK
                                        print(f"   🚫 GPT ANOMALY_OPPOSE: {sym} — anomaly does NOT confirm {direction} "
                                              f"(UP_Flag={_gpt_up_flag} Down_Flag={_gpt_down_flag}) → BLOCK")
                                        continue
                        
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
                                    'up_flag': _gpt_ml_full.get('ml_up_flag', False),
                                    'down_flag': _gpt_ml_full.get('ml_down_flag', False),
                                    'up_score': _gpt_ml_full.get('ml_up_score', 0),
                                    'down_score': _gpt_ml_full.get('ml_down_score', 0),
                                    'down_risk_bucket': _gpt_ml_full.get('ml_down_risk_bucket', 'LOW'),
                                    'gmm_confirms_direction': _gpt_ml_full.get('ml_gmm_confirms_direction', False),
                                    'gmm_regime_used': _gpt_ml_full.get('ml_gmm_regime_used', 'BOTH'),
                                    'gmm_action': 'GPT_DIRECT',
                                },
                                'scored_direction': direction,
                                'xgb_disagrees': _gpt_xgb_disagrees if '_gpt_xgb_disagrees' in dir() else False,
                            } if _gpt_ml_full else {}
                            # Use detected setup_type from scanning (ORB_BREAKOUT, VWAP_TREND, etc.)
                            # so routing in place_option_order can reverse for ORB/ELITE
                            _detected_setup = getattr(self, '_cycle_detected_setups', {}).get(f'NSE:{sym}', '') or getattr(self, '_cycle_detected_setups', {}).get(sym, '')
                            _gpt_setup_type = 'ML_OVERRIDE_WGMM' if _gpt_was_flipped else (_detected_setup or 'GPT')
                            # ML_OVERRIDE gates already checked before flip above
                            
                            # [FIX Mar 4] GPT ML QUALITY GATE — safety net for all GPT-selected trades
                            # Prevents zero-score, low-ML-conviction entries from leaking through GPT pipeline.
                            # Watcher/ELITE/SNIPER/TEST paths have their own ML gates; GPT had NONE.
                            _gpt_pre_score = getattr(self, '_cycle_pre_scores', {}).get(f'NSE:{sym}', 0)
                            _gpt_ml_move = _gpt_ml_full.get('ml_move_prob', 0) if _gpt_ml_full else 0
                            _gpt_ml_conf = _gpt_ml_full.get('ml_confidence', 0) if _gpt_ml_full else 0
                            _gpt_ml_flat = _gpt_ml_full.get('ml_prob_flat', 0) if _gpt_ml_full else 0
                            
                            # Gate 1: ML flat veto — if ML says >60% chance flat, don't enter
                            if _gpt_ml_flat > 0.60:
                                print(f"   🚫 GPT ML_FLAT_VETO: {sym} — P(flat)={_gpt_ml_flat:.0%} > 60% → SKIP")
                                continue
                            
                            # Gate 2: ML move probability floor — need >35% move probability
                            if _gpt_ml_full and _gpt_ml_move < 0.35:
                                print(f"   🚫 GPT ML_LOW_MOVE: {sym} — move_prob={_gpt_ml_move:.0%} < 35% → SKIP")
                                continue
                            
                            # Gate 3: Pre-score sanity — if score exists and is very low, skip
                            # (score=0 means not scored yet, allow through; but if scored low, block)
                            if _gpt_pre_score > 0 and _gpt_pre_score < 45:
                                print(f"   🚫 GPT LOW_SCORE: {sym} — pre_score={_gpt_pre_score:.0f} < 45 → SKIP")
                                continue
                            
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
            
            # VIX Regime (safe access)
            _d_vix_emoji = {'LOW': '🟢', 'NORMAL': '🔵', 'HIGH': '🟠', 'EXTREME': '🔴'}.get(self._vix_regime, '⚪')
            print(f"📊 India VIX: {self._current_vix:.1f} | Regime: {_d_vix_emoji} {self._vix_regime}")
            
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
                # print(f"🔌 Ticker: {_ws_status} | Sub:{_ts['subscribed']}(+{_fut_count} futures) | Hits:{_ts['cache_hits']} | Fallbacks:{_ts['fallbacks']} | Ticks:{_ts['ticks']}")
            
            print(f"⏱️ Cycle: {_cycle_elapsed:.0f}s | Next scan in ~{getattr(self, '_normal_interval', 5)}min")
            
            # --- DR SCORE RECAP (end of cycle so it's always visible in terminal) ---
            try:
                _dr_recap = []
                for _rs2, _ml2 in _ml_results.items():
                    _dv = _ml2.get('ml_down_risk_score')
                    if _dv is not None:
                        _dr_recap.append((_rs2.replace('NSE:', ''), _dv))
                if _dr_recap:
                    _dr_recap.sort(key=lambda x: x[1])
                    _lo = _dr_recap[:2]
                    _hi = _dr_recap[-2:][::-1]
                    _lo_s = ' | '.join(f"{s} {d:.4f}" for s, d in _lo)
                    _hi_s = ' | '.join(f"{s} {d:.4f}" for s, d in _hi)
                    # print(f"📊 DR: LOW={_lo_s} | HIGH={_hi_s}")
            except Exception:
                pass
            
            print("=" * 95)
            
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

    # ══════════════════════════════════════════════════════════════
    #  SETTINGS HOT-RELOAD: Read titan_settings.json & apply overrides
    # ══════════════════════════════════════════════════════════════
    def _apply_settings_overrides(self):
        """Read titan_settings.json and apply overrides to config module.
        Called at startup and every 30s from the main loop.
        Only re-applies if file has been modified since last check."""
        import config as _cfg
        try:
            if not os.path.exists(self._settings_file):
                return
            mtime = os.path.getmtime(self._settings_file)
            if mtime == self._settings_mtime:
                return  # No change since last read
            self._settings_mtime = mtime

            with open(self._settings_file, 'r') as f:
                s = json.load(f)

            applied = []

            # --- Kill switch ---
            if s.get('kill_switch', False):
                print("🚨 KILL SWITCH ACTIVE — blocking all new trades")
                _cfg.HARD_RULES['MAX_POSITIONS'] = 0  # Zero positions = no new trades
                return  # Don't apply other overrides

            # --- Capital & Risk ---
            for key in ['CAPITAL', 'RISK_PER_TRADE', 'MAX_DAILY_LOSS', 'MAX_POSITIONS',
                        'REENTRY_COOLDOWN_MINUTES', 'MIN_OPTION_PREMIUM', 'PORTFOLIO_PROFIT_TARGET']:
                if key in s and s[key] != _cfg.HARD_RULES.get(key):
                    _cfg.HARD_RULES[key] = s[key]
                    applied.append(f"HARD_RULES.{key}={s[key]}")

            # --- Strategy toggles ---
            strat_map = {
                'BREAKOUT_WATCHER': 'BREAKOUT_WATCHER',
                'ELITE_AUTO_FIRE': 'ELITE_AUTO_FIRE',
                'DOWN_RISK_GATING': 'DOWN_RISK_GATING',
                'GMM_SNIPER': 'GMM_SNIPER',
                'GMM_CONTRARIAN': 'GMM_CONTRARIAN',
                'TEST_GMM': 'TEST_GMM',
                'TEST_XGB': 'TEST_XGB',
                'SNIPER_OI_UNWINDING': 'SNIPER_OI_UNWINDING',
                'SNIPER_PCR_EXTREME': 'SNIPER_PCR_EXTREME',
                'ARBTR_CONFIG': 'ARBTR_CONFIG',
                'IRON_CONDOR_CONFIG': 'IRON_CONDOR_CONFIG',
                'CREDIT_SPREAD_CONFIG': 'CREDIT_SPREAD_CONFIG',
                'DEBIT_SPREAD_CONFIG': 'DEBIT_SPREAD_CONFIG',
                'ML_DIRECTION_CONFLICT': 'ML_DIRECTION_CONFLICT',
                'GCR_CONFIG': 'GCR_CONFIG',
            }
            for key, attr_name in strat_map.items():
                skey = f'strategy_{key}'
                if skey in s:
                    cfg_dict = getattr(_cfg, attr_name, None)
                    if isinstance(cfg_dict, dict) and 'enabled' in cfg_dict:
                        new_val = bool(s[skey])
                        if cfg_dict['enabled'] != new_val:
                            cfg_dict['enabled'] = new_val
                            applied.append(f"{attr_name}.enabled={new_val}")

            # --- Watcher tunables ---
            bw = getattr(_cfg, 'BREAKOUT_WATCHER', {})
            for skey, cfgkey in [
                ('watcher_min_score', 'min_score'),
                ('watcher_max_trades_per_scan', 'max_trades_per_scan'),
                ('watcher_max_triggers_per_batch', 'max_triggers_per_batch'),
                ('watcher_sustain_seconds', 'sustain_seconds'),
                ('watcher_vix_hard_block', 'vix_hard_block_above'),
            ]:
                if skey in s and bw.get(cfgkey) != s[skey]:
                    bw[cfgkey] = s[skey]
                    applied.append(f"BREAKOUT_WATCHER.{cfgkey}={s[skey]}")
            if 'watcher_momentum_exit' in s:
                me = bw.get('momentum_exit', {})
                new_me = bool(s['watcher_momentum_exit'])
                if me.get('enabled') != new_me:
                    me['enabled'] = new_me
                    applied.append(f"momentum_exit.enabled={new_me}")

            # --- Lot multipliers ---
            for skey_prefix, attr_name in [
                ('lots_GMM_SNIPER', 'GMM_SNIPER'),
                ('lots_ARBTR_CONFIG', 'ARBTR_CONFIG'),
                ('lots_SNIPER_OI_UNWINDING', 'SNIPER_OI_UNWINDING'),
                ('lots_SNIPER_PCR_EXTREME', 'SNIPER_PCR_EXTREME'),
                ('lots_DOWN_RISK_GATING', 'DOWN_RISK_GATING'),
                ('lots_GMM_CONTRARIAN', 'GMM_CONTRARIAN'),
            ]:
                if skey_prefix in s:
                    cfg_dict = getattr(_cfg, attr_name, None)
                    if isinstance(cfg_dict, dict):
                        if attr_name == 'DOWN_RISK_GATING':
                            new_v = float(s[skey_prefix])
                            if cfg_dict.get('all_agree_lot_multiplier') != new_v:
                                cfg_dict['all_agree_lot_multiplier'] = new_v
                                applied.append(f"{attr_name}.all_agree_lot_multiplier={new_v}")
                        else:
                            new_v = float(s[skey_prefix])
                            if cfg_dict.get('lot_multiplier') != new_v:
                                cfg_dict['lot_multiplier'] = new_v
                                applied.append(f"{attr_name}.lot_multiplier={new_v}")

            # --- Global lot multiplier (applied to HARD_RULES for sizer) ---
            if 'global_lot_multiplier' in s:
                new_gl = float(s['global_lot_multiplier'])
                if _cfg.HARD_RULES.get('GLOBAL_LOT_MULTIPLIER') != new_gl:
                    _cfg.HARD_RULES['GLOBAL_LOT_MULTIPLIER'] = new_gl
                    applied.append(f"GLOBAL_LOT_MULTIPLIER={new_gl}")

            # --- Trading hours ---
            th = getattr(_cfg, 'TRADING_HOURS', {})
            for skey, cfgkey in [
                ('hours_start', 'start'),
                ('hours_end', 'end'),
                ('hours_no_new_after', 'no_new_after'),
            ]:
                if skey in s and th.get(cfgkey) != s[skey]:
                    th[cfgkey] = s[skey]
                    applied.append(f"TRADING_HOURS.{cfgkey}={s[skey]}")

            if applied:
                print(f"⚙️  Settings reloaded ({len(applied)} overrides): {', '.join(applied[:5])}"
                      f"{'...' if len(applied) > 5 else ''}")
        except Exception as e:
            print(f"⚠️ Settings reload failed: {e}")

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
                
                # === BREAKOUT WATCHER: Drain queue every 1 second ===
                # This runs between scan cycles — sub-second detection latency.
                # All trade logic stays single-threaded (no race conditions).
                try:
                    if self.is_trading_hours() and BREAKOUT_WATCHER.get('enabled', False):
                        self._process_breakout_triggers()
                except Exception as _bw_e:
                    if _loop_count % 300 == 0:  # Log watcher errors every 5 min (not spammy)
                        _dbg(f"WARN: breakout watcher drain error: {_bw_e}")
                
                # Check if we need to switch from early session to normal interval
                if not self._switched_to_normal:
                    _check_now = datetime.now()
                    if _check_now >= _early_end:
                        self._switched_to_normal = True
                        schedule.clear()
                        schedule.every(self._normal_interval).minutes.do(self.scan_and_trade)
                        _dbg(f"DEBUG: Early session ended — switching to {self._normal_interval}-min interval")
                
                # === SETTINGS HOT-RELOAD (every 30s — mtime check is very cheap) ===
                if _loop_count % 30 == 0:
                    try:
                        self._apply_settings_overrides()
                    except Exception:
                        pass
                
                _loop_count += 1

                # === WEEKEND SLEEP — skip tight loop on Sat/Sun ===
                if datetime.now().weekday() >= 5:
                    if _loop_count % 3600 == 1:  # Log once per hour
                        _dbg("💤 Weekend — sleeping (60s intervals)")
                    time.sleep(60)
                    continue

                # === AUTO-RENEW DHAN TOKEN (every ~2 hours) ===
                if _loop_count > 0 and _loop_count % 7200 == 0:  # every 7200s = 2 hours
                    try:
                        from dhan_token_manager import ensure_token_fresh
                        if ensure_token_fresh():
                            _dbg("DHAN_TOKEN: renewed/fresh ✅")
                        else:
                            _dbg("DHAN_TOKEN: expired ❌ — OI degraded")
                    except Exception as _tok_loop_e:
                        _dbg(f"DHAN_TOKEN: check error: {_tok_loop_e}")
                if _loop_count % 60 == 0:  # Log heartbeat every 60s
                    _watcher_stats = ''
                    _watcher_alive = ''
                    try:
                        _tw = getattr(self.tools, 'ticker', None)
                        if _tw and hasattr(_tw, 'breakout_watcher') and _tw.breakout_watcher:
                            _ws = _tw.breakout_watcher.stats
                            _wq = _ws.get('queued', 0)
                            _watcher_stats = (f" | watcher: q={_wq} "
                                             f"drains={self._watcher_drain_count} "
                                             f"drained={self._watcher_total_drained} "
                                             f"actionable={self._watcher_total_actionable} "
                                             f"piped={self._watcher_total_pipeline_sent} "
                                             f"gated={self._watcher_total_gate_blocked} "
                                             f"placed={self._watcher_total_placed} "
                                             f"pos_full={self._watcher_total_pos_exhausted} "
                                             f"fired={len(self._watcher_fired_this_session)}")
                            # Periodic watcher-alive banner every 5 min
                            if _loop_count % 300 == 0 and self.is_trading_hours():
                                _n_subs = _tw.stats.get('subscribed', 0) if hasattr(_tw, 'stats') else 0
                                _watcher_alive = (f"\n👁️  WATCHER ALIVE — monitoring {_n_subs} stocks | "
                                                 f"q={_wq} triggered={self._watcher_total_drained} "
                                                 f"placed={self._watcher_total_placed} | "
                                                 f"cooldown=5s, score≥35")
                    except Exception:
                        pass
                    _dbg(f"HEARTBEAT: loop iteration {_loop_count}, system_state={getattr(self.risk_governor.state, 'system_state', '?')}{_watcher_stats}")
                    if _watcher_alive:
                        _dbg(_watcher_alive)
                
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
