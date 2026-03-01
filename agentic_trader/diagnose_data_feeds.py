"""
TITAN DATA FEED DIAGNOSTICS
============================
Tests ALL live data sources (Kite WS, Kite REST, DhanHQ REST)
and verifies alignment with trading strategies.

Run: python diagnose_data_feeds.py
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"

results = []  # (section, test_name, status, detail)


def record(section, name, status, detail=""):
    results.append((section, name, status, detail))
    icon = PASS if status == "PASS" else FAIL if status == "FAIL" else WARN if status == "WARN" else INFO
    print(f"  {icon} {name}: {detail}")


def section_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# SECTION 1: CREDENTIALS & CONFIG
# ============================================================
def test_credentials():
    section_header("1. CREDENTIALS & CONFIG")

    # Kite
    try:
        from config import ZERODHA_API_KEY
        if ZERODHA_API_KEY and len(ZERODHA_API_KEY) > 5:
            record("CREDS", "Kite API Key", "PASS", f"{ZERODHA_API_KEY[:4]}...{ZERODHA_API_KEY[-4:]}")
        else:
            record("CREDS", "Kite API Key", "FAIL", "Missing or too short")
    except Exception as e:
        record("CREDS", "Kite API Key", "FAIL", str(e))

    # Kite access token (from .env only)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    env_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')
    if env_token and len(env_token) > 10:
        record("CREDS", "Kite Access Token (.env)", "PASS", f"{env_token[:6]}...")
    else:
        record("CREDS", "Kite Access Token (.env)", "FAIL", "ZERODHA_ACCESS_TOKEN not set or empty in .env")

    # DhanHQ
    try:
        dhan_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dhan_config.json')
        if os.path.exists(dhan_cfg_path):
            with open(dhan_cfg_path) as f:
                dcfg = json.load(f)
            cid = dcfg.get('client_id', '')
            dtoken = dcfg.get('access_token', '')
            if cid and dtoken:
                record("CREDS", "DhanHQ Config", "PASS", f"client_id={cid}, token={dtoken[:10]}...")
            else:
                record("CREDS", "DhanHQ Config", "FAIL", "client_id or access_token missing")
        else:
            record("CREDS", "DhanHQ Config", "FAIL", "dhan_config.json not found")
    except Exception as e:
        record("CREDS", "DhanHQ Config", "FAIL", str(e))


# ============================================================
# SECTION 2: KITE REST API CONNECTIVITY
# ============================================================
def test_kite_rest():
    section_header("2. KITE REST API")

    try:
        from kiteconnect import KiteConnect
        from config import ZERODHA_API_KEY
    except ImportError as e:
        record("KITE_REST", "Import", "FAIL", str(e))
        return None

    # Load token from .env
    kite = KiteConnect(api_key=ZERODHA_API_KEY)
    try:
        access_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')
        if not access_token:
            record("KITE_REST", "Token Load", "FAIL", "ZERODHA_ACCESS_TOKEN not set in .env")
            return None
        kite.set_access_token(access_token)
    except Exception as e:
        record("KITE_REST", "Token Load", "FAIL", str(e))
        return None

    # Test 1: Profile (auth check)
    try:
        profile = kite.profile()
        record("KITE_REST", "Authentication", "PASS", f"User: {profile.get('user_name', 'N/A')} ({profile.get('user_id', 'N/A')})")
    except Exception as e:
        record("KITE_REST", "Authentication", "FAIL", f"Token likely expired: {e}")
        return kite

    # Test 2: LTP (live data)
    test_symbols = ["NSE:SBIN", "NSE:RELIANCE", "NSE:NIFTY 50"]
    try:
        t0 = time.time()
        ltp_data = kite.ltp(test_symbols)
        elapsed = (time.time() - t0) * 1000
        prices = {s.split(':')[1]: ltp_data[s]['last_price'] for s in test_symbols if s in ltp_data}
        record("KITE_REST", "LTP Fetch", "PASS", f"{prices} ({elapsed:.0f}ms)")
    except Exception as e:
        record("KITE_REST", "LTP Fetch", "FAIL", str(e))

    # Test 3: Quote (OHLCV + OI)
    try:
        t0 = time.time()
        quotes = kite.quote(["NSE:SBIN"])
        elapsed = (time.time() - t0) * 1000
        q = quotes.get("NSE:SBIN", {})
        fields = ['last_price', 'volume', 'ohlc', 'last_trade_time', 'oi']
        found = [f for f in fields if f in q and q[f]]
        missing = [f for f in fields if f not in q or not q[f]]
        record("KITE_REST", "Quote Fields", "PASS" if len(found) >= 3 else "WARN",
               f"Found: {found}, Missing: {missing} ({elapsed:.0f}ms)")
    except Exception as e:
        record("KITE_REST", "Quote", "FAIL", str(e))

    # Test 4: Historical data (5-min candles for indicators)
    try:
        t0 = time.time()
        # Get SBIN instrument token
        instruments_nse = kite.instruments("NSE")
        sbin_token = None
        for inst in instruments_nse:
            if inst['tradingsymbol'] == 'SBIN':
                sbin_token = inst['instrument_token']
                break
        if sbin_token:
            from_dt = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
            to_dt = datetime.now()
            # Try today's candles; if market hasn't opened, use yesterday
            candles = kite.historical_data(sbin_token, from_dt, to_dt, "5minute")
            if not candles:
                # Try yesterday
                from_dt = from_dt - timedelta(days=1)
                to_dt = to_dt - timedelta(days=1)
                candles = kite.historical_data(sbin_token, from_dt, to_dt, "5minute")
            elapsed = (time.time() - t0) * 1000
            if candles and len(candles) > 0:
                record("KITE_REST", "Historical 5min", "PASS", f"{len(candles)} candles ({elapsed:.0f}ms)")
                # Check candle structure
                sample = candles[-1]
                expected_keys = ['date', 'open', 'high', 'low', 'close', 'volume']
                has_keys = all(k in sample for k in expected_keys)
                record("KITE_REST", "Candle Structure", "PASS" if has_keys else "FAIL",
                       f"Keys: {list(sample.keys())}")
            else:
                record("KITE_REST", "Historical 5min", "WARN", f"No candles (market may be closed) ({elapsed:.0f}ms)")
    except Exception as e:
        record("KITE_REST", "Historical 5min", "FAIL", str(e))

    # Test 5: NFO instruments (for options chain)
    try:
        t0 = time.time()
        nfo = kite.instruments("NFO")
        elapsed = (time.time() - t0) * 1000
        fut_count = sum(1 for i in nfo if i.get('instrument_type') == 'FUT')
        opt_count = sum(1 for i in nfo if i.get('instrument_type') in ('CE', 'PE'))
        record("KITE_REST", "NFO Instruments", "PASS", f"{len(nfo)} total ({fut_count} FUT, {opt_count} OPT) ({elapsed:.0f}ms)")
    except Exception as e:
        record("KITE_REST", "NFO Instruments", "FAIL", str(e))

    # Test 6: Positions & Orders
    try:
        positions = kite.positions()
        orders = kite.orders()
        net_pos = positions.get('net', [])
        record("KITE_REST", "Positions/Orders", "PASS",
               f"{len(net_pos)} positions, {len(orders)} orders today")
    except Exception as e:
        record("KITE_REST", "Positions/Orders", "FAIL", str(e))

    return kite


# ============================================================
# SECTION 3: KITE WEBSOCKET (TitanTicker)
# ============================================================
def test_kite_websocket(kite):
    section_header("3. KITE WEBSOCKET (TitanTicker)")

    if kite is None:
        record("KITE_WS", "Skip", "FAIL", "Kite REST failed — cannot test WS")
        return

    try:
        from kite_ticker import TitanTicker, HAS_KITE_TICKER
        if not HAS_KITE_TICKER:
            record("KITE_WS", "KiteTicker Import", "FAIL", "kiteconnect.KiteTicker not available")
            return
        record("KITE_WS", "KiteTicker Import", "PASS", "Module loaded")
    except ImportError as e:
        record("KITE_WS", "KiteTicker Import", "FAIL", str(e))
        return

    # Create ticker
    try:
        from config import ZERODHA_API_KEY
        access_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')
        if not access_token:
            record("KITE_WS", "Token Load", "FAIL", "ZERODHA_ACCESS_TOKEN not set in .env")
            return

        ticker = TitanTicker(ZERODHA_API_KEY, access_token, kite)
        ticker.start()
        record("KITE_WS", "Ticker Start", "PASS", "Thread launched")

        # Subscribe test symbols
        test_syms = ["NSE:SBIN", "NSE:RELIANCE", "NSE:TCS"]
        ticker.subscribe_symbols(test_syms, mode='quote')
        record("KITE_WS", "Subscribe", "PASS", f"Subscribed {test_syms}")

        # Wait for data
        print(f"  {INFO} Waiting 5 seconds for WebSocket data...")
        time.sleep(5)

        # Check connection
        connected = ticker._connected
        record("KITE_WS", "Connection Status", "PASS" if connected else "FAIL",
               f"Connected: {connected}")

        # Check stats
        stats = ticker._stats
        record("KITE_WS", "Ticks Received", "PASS" if stats['ticks_received'] > 0 else "WARN",
               f"ticks={stats['ticks_received']}, reconnects={stats['reconnects']}, errors={stats['errors']}")

        # Check LTP cache
        cache_size = len(ticker._ltp_cache)
        record("KITE_WS", "LTP Cache", "PASS" if cache_size > 0 else "WARN",
               f"{cache_size} instruments cached")

        # Test get_ltp via cache
        for sym in test_syms:
            ltp = ticker.get_ltp(sym)
            source = "WS cache" if stats['fallback_calls'] == 0 else "REST fallback"
            if ltp:
                record("KITE_WS", f"get_ltp({sym.split(':')[1]})", "PASS", f"₹{ltp:.2f} via {source}")
            else:
                record("KITE_WS", f"get_ltp({sym.split(':')[1]})", "WARN", "None (market may be closed)")

        # Check fallback ratio
        total = stats['cache_hits'] + stats['fallback_calls']
        if total > 0:
            cache_pct = stats['cache_hits'] / total * 100
            record("KITE_WS", "Cache Hit Rate", "PASS" if cache_pct > 50 else "WARN",
                   f"{cache_pct:.0f}% ({stats['cache_hits']} hits / {stats['fallback_calls']} fallbacks)")
        
        # Test quote (OHLCV) via WS
        quote_data = ticker.get_quote("NSE:SBIN")
        if quote_data:
            q_fields = list(quote_data.keys()) if isinstance(quote_data, dict) else ['raw']
            record("KITE_WS", "get_quote(SBIN)", "PASS", f"Fields: {q_fields[:8]}")
        else:
            record("KITE_WS", "get_quote(SBIN)", "WARN", "No quote data (market may be closed)")

        # Stop ticker
        ticker.stop()
        record("KITE_WS", "Ticker Stop", "PASS", "Cleaned up")
        
    except Exception as e:
        record("KITE_WS", "Ticker Test", "FAIL", f"{e}\n{traceback.format_exc()}")


# ============================================================
# SECTION 4: DHANHHQ REST API
# ============================================================
def test_dhan():
    section_header("4. DHANHHQ REST API")

    # Test DhanRiskTools
    try:
        from dhan_risk_tools import DhanRiskTools
        drt = DhanRiskTools()
        if drt.ready:
            record("DHAN", "DhanRiskTools Init", "PASS", f"client_id={drt.client_id}")
        else:
            record("DHAN", "DhanRiskTools Init", "FAIL", "Not ready (credentials missing)")
            return
    except Exception as e:
        record("DHAN", "DhanRiskTools Init", "FAIL", str(e))
        return

    # Test fund limit (basic auth check)
    try:
        status, resp = drt._request('GET', '/fundlimit')
        if status == 200:
            avail = resp.get('data', {}).get('availabelBalance', resp.get('availabelBalance', 'N/A'))
            record("DHAN", "Fund Limit (Auth)", "PASS", f"Available: ₹{avail}")
        else:
            record("DHAN", "Fund Limit (Auth)", "FAIL", f"HTTP {status}: {resp}")
    except Exception as e:
        record("DHAN", "Fund Limit (Auth)", "FAIL", str(e))

    # Test DhanOIFetcher
    try:
        from dhan_oi_fetcher import get_dhan_oi_fetcher
        fetcher = get_dhan_oi_fetcher()
        if fetcher:
            record("DHAN", "DhanOIFetcher Init", "PASS", "Fetcher ready")

            # Fetch option chain for SBIN
            try:
                t0 = time.time()
                oi_data = fetcher.fetch("SBIN")
                elapsed = (time.time() - t0) * 1000
                if oi_data and isinstance(oi_data, dict):
                    strikes = len(oi_data.get('strikes', []))
                    has_greeks = any(s.get('delta') for s in oi_data.get('strikes', [])[:5])
                    record("DHAN", "Option Chain (SBIN)", "PASS",
                           f"{strikes} strikes, Greeks: {has_greeks} ({elapsed:.0f}ms)")
                else:
                    record("DHAN", "Option Chain (SBIN)", "WARN", f"Empty/None response ({elapsed:.0f}ms)")
            except Exception as e:
                record("DHAN", "Option Chain (SBIN)", "FAIL", str(e))
        else:
            record("DHAN", "DhanOIFetcher Init", "WARN", "Fetcher not available")
    except ImportError:
        record("DHAN", "DhanOIFetcher", "WARN", "Module not importable")
    except Exception as e:
        record("DHAN", "DhanOIFetcher", "FAIL", str(e))

    # Test historical OI (for ML features)
    try:
        t0 = time.time()
        # Get SBIN daily candle with OI from DhanHQ
        status, resp = drt._request('POST', '/charts/historical', json_body={
            'securityId': '500112',  # SBIN security ID
            'exchangeSegment': 'NSE_EQ',
            'instrument': 'EQUITY',
            'fromDate': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
            'toDate': datetime.now().strftime('%Y-%m-%d'),
        })
        elapsed = (time.time() - t0) * 1000
        if status == 200 and resp:
            candle_count = len(resp.get('data', {}).get('close', resp.get('close', [])))
            record("DHAN", "Historical Candles", "PASS" if candle_count > 0 else "WARN",
                   f"{candle_count} candles ({elapsed:.0f}ms)")
        else:
            record("DHAN", "Historical Candles", "WARN", f"HTTP {status} ({elapsed:.0f}ms)")
    except Exception as e:
        record("DHAN", "Historical Candles", "FAIL", str(e))


# ============================================================
# SECTION 5: DATA FLOW → STRATEGY ALIGNMENT
# ============================================================
def test_strategy_alignment(kite):
    section_header("5. DATA FLOW → STRATEGY ALIGNMENT")

    if kite is None:
        record("ALIGN", "Skip", "FAIL", "Kite not available")
        return

    # Test: get_market_data produces all fields strategies need
    try:
        from zerodha_tools import get_tools
        tools = get_tools()
        
        t0 = time.time()
        market_data = tools.get_market_data(["NSE:SBIN", "NSE:RELIANCE"])
        elapsed = (time.time() - t0)
        
        if not market_data or "error" in market_data:
            record("ALIGN", "get_market_data()", "FAIL", f"Error: {market_data}")
            return
        
        record("ALIGN", "get_market_data()", "PASS", f"{len(market_data)} stocks in {elapsed:.1f}s")
        
        # Check required fields for strategy
        required_fields = [
            'ltp', 'open', 'high', 'low', 'close', 'volume', 'change_pct',
            'sma_20', 'rsi_14', 'atr_14', 'vwap', 'ema_9', 'ema_21',
            'orb_signal', 'volume_regime', 'trend', 'price_vs_vwap',
            'ema_regime'
        ]
        
        for sym, data in market_data.items():
            if isinstance(data, dict):
                present = [f for f in required_fields if f in data and data[f] is not None]
                missing = [f for f in required_fields if f not in data or data[f] is None]
                status = "PASS" if len(missing) == 0 else "WARN" if len(missing) <= 3 else "FAIL"
                record("ALIGN", f"Fields: {sym.split(':')[1]}", status,
                       f"{len(present)}/{len(required_fields)} present" +
                       (f", missing: {missing}" if missing else ""))
                
                # Check data freshness
                ts = data.get('timestamp', '')
                ltp = data.get('ltp', 0)
                vol = data.get('volume', 0)
                record("ALIGN", f"Values: {sym.split(':')[1]}", "PASS" if ltp > 0 else "WARN",
                       f"LTP=₹{ltp:.2f}, Vol={vol:,}, RSI={data.get('rsi_14', 'N/A')}, "
                       f"ORB={data.get('orb_signal', 'N/A')}, Vol_Regime={data.get('volume_regime', 'N/A')}")
                
                # === CRITICAL CHECK: ORB direction alignment ===
                orb = data.get('orb_signal', '')
                vwap_pos = data.get('price_vs_vwap', '')
                ema_regime = data.get('ema_regime', '')
                
                # Check if ORB + VWAP + EMA signals make sense together
                if orb == 'BREAKOUT_UP':
                    if vwap_pos == 'BELOW_VWAP':
                        record("ALIGN", f"Signal Coherence: {sym.split(':')[1]}", "WARN",
                               "ORB=BREAKOUT_UP but price BELOW_VWAP — potential misalignment")
                    else:
                        record("ALIGN", f"Signal Coherence: {sym.split(':')[1]}", "PASS",
                               f"ORB={orb}, VWAP={vwap_pos}, EMA={ema_regime} — coherent")
                elif orb == 'BREAKOUT_DOWN':
                    if vwap_pos == 'ABOVE_VWAP':
                        record("ALIGN", f"Signal Coherence: {sym.split(':')[1]}", "WARN",
                               "ORB=BREAKOUT_DOWN but price ABOVE_VWAP — potential misalignment")
                    else:
                        record("ALIGN", f"Signal Coherence: {sym.split(':')[1]}", "PASS",
                               f"ORB={orb}, VWAP={vwap_pos}, EMA={ema_regime} — coherent")
                else:
                    record("ALIGN", f"Signal Coherence: {sym.split(':')[1]}", INFO,
                           f"ORB={orb}, VWAP={vwap_pos}, EMA={ema_regime}")
            break  # Just test first stock in detail
        
    except Exception as e:
        record("ALIGN", "get_market_data()", "FAIL", f"{e}\n{traceback.format_exc()}")

    # Test: ML predictor can consume the data
    try:
        from ml_models.predictor import TitanPredictor
        predictor = TitanPredictor()
        if predictor.ready:
            record("ALIGN", "ML Predictor", "PASS", "Model loaded and ready")
            
            # Verify ML models exist
            import glob
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models', 'saved_models')
            gate_models = glob.glob(os.path.join(model_dir, 'meta_gate_*'))
            dir_models = glob.glob(os.path.join(model_dir, 'meta_dir_*'))
            record("ALIGN", "ML Model Files", "PASS" if gate_models and dir_models else "WARN",
                   f"Gate: {len(gate_models)} files, Direction: {len(dir_models)} files")
        else:
            record("ALIGN", "ML Predictor", "WARN", "Not ready (models not loaded)")
    except Exception as e:
        record("ALIGN", "ML Predictor", "WARN", str(e))

    # Test: Option chain fetcher alignment
    try:
        from options_trader import OptionChainFetcher
        # Just check it can be instantiated
        record("ALIGN", "OptionChainFetcher", "PASS", "Module importable")
    except ImportError as e:
        record("ALIGN", "OptionChainFetcher", "FAIL", str(e))

    # Test: Intraday scorer alignment
    try:
        from options_trader import IntradayOptionScorer
        record("ALIGN", "IntradayOptionScorer", "PASS",
               f"Thresholds: Premium≥{IntradayOptionScorer.PREMIUM_THRESHOLD}, "
               f"Standard≥{IntradayOptionScorer.STANDARD_THRESHOLD}, "
               f"Reject<{IntradayOptionScorer.REJECTION_THRESHOLD}")
    except Exception as e:
        record("ALIGN", "IntradayOptionScorer", "FAIL", str(e))

    # Test: Options flow analyzer (DhanHQ → strategy pipeline)
    try:
        from options_flow_analyzer import OptionsFlowAnalyzer
        analyzer = OptionsFlowAnalyzer(kite)
        record("ALIGN", "OptionsFlowAnalyzer", "PASS", "Initialized (Dhan OI → PCR/IV/MaxPain)")
    except Exception as e:
        record("ALIGN", "OptionsFlowAnalyzer", "WARN", str(e))


# ============================================================
# SECTION 6: CROSS-SOURCE DATA CONSISTENCY
# ============================================================
def test_cross_source(kite):
    section_header("6. CROSS-SOURCE CONSISTENCY (Kite vs Dhan)")

    if kite is None:
        record("XSRC", "Skip", "FAIL", "Kite not available")
        return

    # Compare SBIN LTP from Kite vs DhanHQ
    kite_ltp = None
    try:
        ltp_data = kite.ltp(["NSE:SBIN"])
        kite_ltp = ltp_data.get("NSE:SBIN", {}).get('last_price')
    except Exception:
        pass

    dhan_ltp = None
    try:
        from dhan_risk_tools import DhanRiskTools
        drt = DhanRiskTools()
        if drt.ready:
            # DhanHQ market quote for SBIN (security_id = 500112 for BSE, 11536 for NSE)
            # Note: DhanHQ uses different security IDs — we check if data flows at all
            status, resp = drt._request('POST', '/marketfeed/ltp', json_body={
                'NSE_EQ': [11536]  # SBIN NSE security ID
            })
            if status == 200 and resp:
                dhan_data = resp.get('data', resp)
                if isinstance(dhan_data, dict):
                    # DhanHQ response structure varies
                    for key, val in dhan_data.items():
                        if isinstance(val, dict) and 'last_price' in val:
                            dhan_ltp = val['last_price']
                            break
                        elif isinstance(val, (int, float)) and val > 0:
                            dhan_ltp = val
                            break
    except Exception:
        pass

    if kite_ltp and dhan_ltp:
        diff_pct = abs(kite_ltp - dhan_ltp) / kite_ltp * 100
        status = "PASS" if diff_pct < 1.0 else "WARN" if diff_pct < 5.0 else "FAIL"
        record("XSRC", "LTP Kite vs Dhan (SBIN)", status,
               f"Kite=₹{kite_ltp:.2f}, Dhan=₹{dhan_ltp:.2f}, diff={diff_pct:.2f}%")
    elif kite_ltp:
        record("XSRC", "LTP Comparison", "WARN", f"Kite=₹{kite_ltp:.2f}, Dhan=unavailable")
    else:
        record("XSRC", "LTP Comparison", "WARN", "Neither source returned LTP (market closed?)")


# ============================================================
# SECTION 7: TOKEN / SESSION HEALTH
# ============================================================
def test_session_health():
    section_header("7. TOKEN & SESSION HEALTH")

    # Check Kite token from .env
    try:
        env_token = os.getenv('ZERODHA_ACCESS_TOKEN', '')
        if env_token and len(env_token) > 10:
            record("SESSION", "Kite Token (.env)", "PASS", f"Token present: {env_token[:6]}...")
        else:
            record("SESSION", "Kite Token (.env)", "FAIL", "ZERODHA_ACCESS_TOKEN missing or too short")
    except Exception as e:
        record("SESSION", "Kite Token (.env)", "WARN", str(e))

    # Check DhanHQ token
    try:
        dhan_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dhan_config.json')
        if os.path.exists(dhan_cfg_path):
            with open(dhan_cfg_path) as f:
                dcfg = json.load(f)
            saved_at = dcfg.get('saved_at', dcfg.get('last_updated', ''))
            if saved_at:
                record("SESSION", "DhanHQ Token Saved", "PASS", f"Saved: {saved_at}")
            else:
                record("SESSION", "DhanHQ Token Saved", "WARN", "No timestamp — manually verify token validity")
    except Exception as e:
        record("SESSION", "DhanHQ Token Age", "WARN", str(e))


# ============================================================
# SUMMARY
# ============================================================
def print_summary():
    section_header("DIAGNOSTICS SUMMARY")

    pass_count = sum(1 for _, _, s, _ in results if s == "PASS")
    fail_count = sum(1 for _, _, s, _ in results if s == "FAIL")
    warn_count = sum(1 for _, _, s, _ in results if s == "WARN")
    total = len(results)

    print(f"\n  Total checks: {total}")
    print(f"  {PASS} PASS:  {pass_count}")
    print(f"  {FAIL} FAIL:  {fail_count}")
    print(f"  {WARN} WARN:  {warn_count}")

    if fail_count == 0:
        print(f"\n  🎉 All critical checks passed!")
        if warn_count > 0:
            print(f"  {WARN} Review {warn_count} warnings above")
    else:
        print(f"\n  {FAIL} {fail_count} critical failures need attention:")
        for section, name, status, detail in results:
            if status == "FAIL":
                print(f"    • [{section}] {name}: {detail}")

    # Data alignment summary
    print(f"\n  📊 DATA PIPELINE STATUS:")
    sections = {
        'KITE_REST': 'Kite REST API',
        'KITE_WS': 'Kite WebSocket',
        'DHAN': 'DhanHQ API',
        'ALIGN': 'Strategy Alignment',
        'XSRC': 'Cross-Source Consistency',
    }
    for sec_key, sec_name in sections.items():
        sec_results = [(s, n, st, d) for s, n, st, d in results if s == sec_key]
        if sec_results:
            sec_pass = sum(1 for _, _, st, _ in sec_results if st == "PASS")
            sec_total = len(sec_results)
            icon = PASS if sec_pass == sec_total else WARN if sec_pass > 0 else FAIL
            print(f"    {icon} {sec_name}: {sec_pass}/{sec_total} checks passed")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"  TITAN DATA FEED DIAGNOSTICS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    test_credentials()
    kite = test_kite_rest()
    test_kite_websocket(kite)
    test_dhan()
    test_strategy_alignment(kite)
    test_cross_source(kite)
    test_session_health()
    print_summary()
