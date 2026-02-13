"""
QUICK TEST — Full F&O Universe Scan
====================================
Tests the new scanning pipeline WITHOUT running the trading bot.
Safe to run anytime (after hours, weekends — uses cached/last-traded data).

What it validates:
1. Kite API connection + token load
2. WebSocket ticker init + futures subscription
3. OI cache reads (WebSocket vs REST fallback)
4. MarketScanner pre-screen of all ~200 F&O stocks
5. Indicator calculation on top movers (parallelized, 8 threads)
6. IntradayOptionScorer scoring (pure math)
7. Full cycle timing breakdown
"""

import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

# Fix encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

from datetime import datetime

def run_test():
    print(f"\n{'='*80}")
    print(f"🧪 FULL F&O SCAN TEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # =========================================================
    # STEP 1: Kite Connection
    # =========================================================
    t0 = time.time()
    print("1️⃣  Connecting to Kite API...")
    try:
        from zerodha_tools import get_tools
        tools = get_tools()
        kite = tools.kite
        print(f"   ✅ Kite connected ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"   ❌ Kite connection failed: {e}")
        return
    
    # =========================================================
    # STEP 2: Ticker Status
    # =========================================================
    print("\n2️⃣  Checking WebSocket Ticker...")
    ticker = tools.ticker
    if ticker:
        stats = ticker.stats
        print(f"   Connected: {'🟢 YES' if stats['connected'] else '🔴 NO (after-hours expected)'}")
        print(f"   Subscribed instruments: {stats['subscribed']}")
        print(f"   Futures map loaded: {len(getattr(ticker, '_futures_map', {}))}")
        print(f"   Cached LTPs: {stats['cached']}")
        print(f"   Ticks received: {stats['ticks']}")
    else:
        print("   ⚠️ Ticker not initialized — REST fallback mode")
    
    # =========================================================
    # STEP 3: F&O Universe Discovery
    # =========================================================
    print("\n3️⃣  Discovering F&O Universe...")
    t1 = time.time()
    from market_scanner import get_market_scanner
    scanner = get_market_scanner(kite)
    all_fo = scanner.get_all_fo_symbols()
    print(f"   Found {len(all_fo)} F&O stocks ({time.time()-t1:.1f}s)")
    
    # =========================================================
    # STEP 4: Market Scanner Pre-Screen
    # =========================================================
    print("\n4️⃣  Running Market Scanner (1 batch quote for all ~200 stocks)...")
    t2 = time.time()
    from config import APPROVED_UNIVERSE
    scan_result = scanner.scan(existing_universe=APPROVED_UNIVERSE)
    scan_time = time.time() - t2
    print(f"   Scanned: {scan_result.scanned}/{scan_result.total_fo_stocks} stocks in {scan_time:.1f}s")
    print(f"   Top Gainers: {len(scan_result.top_gainers)}")
    print(f"   Top Losers: {len(scan_result.top_losers)}")
    print(f"   Volume Surges: {len(scan_result.volume_surges)}")
    print(f"   Breakouts: {len(scan_result.breakouts)}")
    print(f"   Wildcards selected: {len(scan_result.wildcards)}")
    
    if scan_result.top_gainers:
        print(f"\n   🟢 Top 5 Gainers:")
        for g in scan_result.top_gainers[:5]:
            print(f"      {g.symbol:15} +{g.change_pct:.2f}%  ₹{g.ltp:.2f}")
    
    if scan_result.top_losers:
        print(f"\n   🔴 Top 5 Losers:")
        for l in scan_result.top_losers[:5]:
            print(f"      {l.symbol:15} {l.change_pct:.2f}%  ₹{l.ltp:.2f}")
    
    # =========================================================
    # STEP 5: OI Cache Test — WebSocket vs REST
    # =========================================================
    print(f"\n5️⃣  Testing OI Fetch (WebSocket cache vs REST)...")
    test_symbols = ["NSE:SBIN", "NSE:RELIANCE", "NSE:HDFCBANK", "NSE:TATASTEEL", "NSE:INFY"]
    
    # Test batch from ticker
    if ticker and hasattr(ticker, '_futures_map'):
        t3 = time.time()
        oi_batch = ticker.get_futures_oi_batch(test_symbols)
        ws_time = time.time() - t3
        print(f"\n   📡 WebSocket OI cache ({ws_time*1000:.1f}ms):")
        for sym, data in oi_batch.items():
            print(f"      {sym.replace('NSE:',''):15} OI: {data['oi']:>12,} | Futures: {data['futures_symbol']}")
        if not oi_batch:
            print(f"      (empty — WebSocket may need live market ticks to populate cache)")
    
    # Test via _get_futures_oi_quick (should use WS cache → REST fallback)
    t4 = time.time()
    rest_results = {}
    for sym in test_symbols:
        oi = tools._get_futures_oi_quick(sym)
        if oi.get('has_futures'):
            rest_results[sym] = oi
    rest_time = time.time() - t4
    print(f"\n   🔄 _get_futures_oi_quick ({rest_time:.1f}s for {len(test_symbols)} stocks):")
    for sym, data in rest_results.items():
        print(f"      {sym.replace('NSE:',''):15} OI: {data['oi']:>12,} | Signal: {data['oi_signal']} | {data['futures_symbol']}")
    if not rest_results:
        print(f"      (no OI data — this is expected on weekends/holidays)")
    
    # =========================================================
    # STEP 6: Build Full Scan Universe (with pre-filter)
    # =========================================================
    print(f"\n6️⃣  Building scan universe with pre-filter...")
    from config import FULL_FNO_SCAN
    
    _min_change = FULL_FNO_SCAN.get('min_change_pct_filter', 0.5)
    _max_stocks = FULL_FNO_SCAN.get('max_indicator_stocks', 80)
    
    full_universe = set(APPROVED_UNIVERSE)
    if hasattr(scanner, '_all_results'):
        moving = sorted(scanner._all_results, key=lambda r: abs(r.change_pct), reverse=True)
        for r in moving:
            if abs(r.change_pct) >= _min_change:
                full_universe.add(r.nse_symbol)
            if len(full_universe) >= _max_stocks:
                break
    
    scan_universe = list(full_universe)
    print(f"   Curated: {len(APPROVED_UNIVERSE)} | After pre-filter: {len(scan_universe)} | Cap: {_max_stocks}")
    
    # =========================================================
    # STEP 7: Indicator Calculation (parallel, 8 threads)
    # =========================================================
    print(f"\n7️⃣  Running indicator calculation on {len(scan_universe)} stocks (8 threads)...")
    t5 = time.time()
    market_data = tools.get_market_data(scan_universe)
    ind_time = time.time() - t5
    
    valid_count = sum(1 for v in market_data.values() if isinstance(v, dict) and 'ltp' in v)
    print(f"   ✅ Got data for {valid_count}/{len(scan_universe)} stocks in {ind_time:.1f}s")
    
    # =========================================================
    # STEP 8: Volume Analysis (OI signals)
    # =========================================================
    print(f"\n8️⃣  Running volume/OI analysis...")
    t6 = time.time()
    volume_analysis = tools.get_volume_analysis(scan_universe)
    vol_time = time.time() - t6
    
    oi_count = sum(1 for v in volume_analysis.values() if isinstance(v, dict) and v.get('has_futures_oi'))
    print(f"   ✅ Volume analysis for {len(volume_analysis)} stocks in {vol_time:.1f}s | {oi_count} with OI signals")
    
    # Show OI signals
    oi_signals = {}
    for sym, data in volume_analysis.items():
        if isinstance(data, dict) and data.get('oi_signal') and data.get('oi_signal') != 'N/A':
            sig = data.get('oi_signal', 'NEUTRAL')
            if sig != 'NEUTRAL':
                oi_signals[sym] = sig
    
    if oi_signals:
        print(f"\n   📊 Active OI Signals:")
        for sym, sig in sorted(oi_signals.items()):
            emoji = {"LONG_BUILDUP": "🟢", "SHORT_BUILDUP": "🔴", "SHORT_COVERING": "🟡", "LONG_UNWINDING": "🟡"}.get(sig, "⚪")
            print(f"      {emoji} {sym.replace('NSE:',''):15} → {sig}")
    
    # =========================================================
    # STEP 9: Score All Stocks
    # =========================================================
    print(f"\n9️⃣  Scoring all {valid_count} stocks...")
    t7 = time.time()
    scores = {}
    try:
        from options_trader import get_intraday_scorer, IntradaySignal
        scorer = get_intraday_scorer()
        
        for sym, data in market_data.items():
            if not isinstance(data, dict) or 'ltp' not in data:
                continue
            try:
                sig = IntradaySignal(
                    symbol=sym,
                    orb_signal=data.get('orb_signal', 'INSIDE_ORB'),
                    vwap_position=data.get('price_vs_vwap', data.get('vwap_position', 'AT_VWAP')),
                    vwap_trend=data.get('vwap_slope', data.get('vwap_trend', 'FLAT')),
                    ema_regime=data.get('ema_regime', 'NORMAL'),
                    volume_regime=data.get('volume_regime', 'NORMAL'),
                    rsi=data.get('rsi_14', 50.0),
                    price_momentum=data.get('momentum_15m', 0.0),
                    htf_alignment=data.get('htf_alignment', 'NEUTRAL'),
                    chop_zone=data.get('chop_zone', False),
                    follow_through_candles=data.get('follow_through_candles', 0),
                    range_expansion_ratio=data.get('range_expansion_ratio', 0.0),
                    vwap_slope_steepening=data.get('vwap_slope_steepening', False),
                    atr=data.get('atr_14', 0.0)
                )
                dec = scorer.score_intraday_signal(sig, market_data=data, caller_direction=None)
                scores[sym] = dec.confidence_score
            except Exception:
                pass
        
        score_time = time.time() - t7
        print(f"   ✅ Scored {len(scores)} stocks in {score_time:.2f}s (pure math)")
    except Exception as e:
        score_time = time.time() - t7
        print(f"   ⚠️ Scoring error: {e}")
    
    # =========================================================
    # RESULTS
    # =========================================================
    total_time = time.time() - t0
    
    print(f"\n{'='*80}")
    print(f"📊 FULL SCAN RESULTS")
    print(f"{'='*80}")
    
    if scores:
        passed_56 = sum(1 for s in scores.values() if s >= 56)
        passed_49 = sum(1 for s in scores.values() if s >= 49)
        passed_45 = sum(1 for s in scores.values() if s >= 45)
        
        print(f"\n   Total scored:    {len(scores)}")
        print(f"   Score ≥ 56:      {passed_56} (trade-ready)")
        print(f"   Score ≥ 49:      {passed_49} (watchlist)")
        print(f"   Score ≥ 45:      {passed_45} (warming up)")
        
        top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n   🏆 TOP 10 SCORES:")
        for i, (sym, score) in enumerate(top10, 1):
            chg = market_data.get(sym, {}).get('change_pct', 0)
            trend = market_data.get(sym, {}).get('trend', 'N/A')
            oi_sig = oi_signals.get(sym, '-')
            print(f"      {i:2}. {sym.replace('NSE:',''):15} Score: {score:5.1f} | {chg:+.2f}% | Trend: {trend} | OI: {oi_sig}")
    
    print(f"\n{'='*80}")
    print(f"⏱️  TIMING BREAKDOWN")
    print(f"{'='*80}")
    print(f"   Kite connect:       {t1-t0:6.1f}s")
    print(f"   F&O discovery:      {time.time()-t1 if t1 else 0:6.1f}s")
    print(f"   Market scanner:     {scan_time:6.1f}s")
    print(f"   OI test (5 stocks): {rest_time:6.1f}s")
    print(f"   Indicators ({len(scan_universe)} stk): {ind_time:6.1f}s  ← main bottleneck")
    print(f"   Volume/OI analysis: {vol_time:6.1f}s")
    print(f"   Scoring:            {score_time:6.1f}s")
    print(f"   {'─'*30}")
    print(f"   TOTAL:              {total_time:6.1f}s")
    print(f"{'='*80}\n")
    
    # Ticker final stats
    if ticker:
        ts = ticker.stats
        print(f"   🔌 Ticker: Subscribed={ts['subscribed']} | CacheHits={ts['cache_hits']} | Fallbacks={ts['fallbacks']} | Ticks={ts['ticks']}")
        fut_count = len(getattr(ticker, '_futures_map', {}))
        print(f"   📡 Futures subscribed: {fut_count}")
    
    print(f"\n✅ Test complete! This is what a full scan cycle would look like in the bot.")
    print(f"   Note: After-hours data is from last traded session. Live ticks populate WebSocket cache during market hours.")


if __name__ == "__main__":
    run_test()
