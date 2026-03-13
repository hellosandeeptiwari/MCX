"""Reconstruct OI signals for March 11 trades using futures_oi parquet data."""
import json, os, sys
from datetime import datetime

# March 11 trades (from ENTRY records)
trades = [
    {"sym": "DIXON", "dir": "BUY", "src": "WATCHER", "score": 54.5, "pnl": -7058, "exit": "DEBIT_SPREAD_TIME_EXIT", "time": "09:30"},
    {"sym": "CGPOWER", "dir": "BUY", "src": "ORB_BREAKOUT", "score": 56.4, "pnl": -4477, "exit": "DEBIT_SPREAD_TIME_EXIT", "time": "09:33"},
    {"sym": "POLYCAB", "dir": "SELL", "src": "WATCHER", "score": 49.5, "pnl": 14974, "exit": "SL_HIT+PARTIAL", "time": "09:34"},
    {"sym": "VEDL", "dir": "BUY", "src": "GMM_SNIPER", "score": 54.4, "pnl": -9483, "exit": "DEBIT_SPREAD_SL", "time": "09:50"},
    {"sym": "COLPAL", "dir": "SELL", "src": "GMM_SNIPER", "score": 63.0, "pnl": -984, "exit": "DEBIT_SPREAD_TIME_EXIT", "time": "09:58"},
    {"sym": "LAURUSLABS", "dir": "BUY", "src": "ORB_BREAKOUT", "score": 51.7, "pnl": 9055, "exit": "SL_HIT", "time": "09:59"},
    {"sym": "BLUESTARCO", "dir": "SELL", "src": "WATCHER", "score": 56.2, "pnl": -1563, "exit": "GREEKS_EXIT", "time": "10:00"},
    {"sym": "MOTHERSON", "dir": "SELL", "src": "WATCHER", "score": 45.9, "pnl": 5925, "exit": "SL_HIT", "time": "10:09"},
    {"sym": "INOXWIND", "dir": "BUY", "src": "WATCHER", "score": 42.0, "pnl": 760, "exit": "WATCHER_MOM_EXIT", "time": "10:42"},
    {"sym": "HEROMOTOCO", "dir": "SELL", "src": "ORB_BREAKOUT", "score": 44.2, "pnl": 12088, "exit": "SESSION_CUTOFF", "time": "10:44"},
    {"sym": "JIOFIN", "dir": "SELL", "src": "TEST_GMM", "score": 63.1, "pnl": 15133, "exit": "SESSION_CUTOFF", "time": "10:58"},
    {"sym": "TMPV", "dir": "SELL", "src": "TEST_GMM", "score": 7.9, "pnl": 19240, "exit": "SESSION_CUTOFF", "time": "11:17"},
    {"sym": "OIL", "dir": "BUY", "src": "ORB_BREAKOUT", "score": 56.8, "pnl": 9012, "exit": "SESSION_CUTOFF", "time": "11:42"},
    {"sym": "ASTRAL", "dir": "SELL", "src": "SNIPER_PCR", "score": 46.1, "pnl": -7860, "exit": "IV_CRUSH", "time": "11:55"},
    {"sym": "BEL", "dir": "SELL", "src": "ORB_BREAKOUT", "score": 55.4, "pnl": 1845, "exit": "DEBIT_SPREAD_TIME_EXIT", "time": "13:05"},
    {"sym": "APLAPOLLO", "dir": "SELL", "src": "WATCHER", "score": 51.5, "pnl": 21290, "exit": "PARTIAL+TARGET", "time": "13:28"},
]

# Try reading OI parquet data
try:
    import pandas as pd
    oi_dir = "/home/ubuntu/titan/agentic_trader/ml_models/data/futures_oi"
    
    results = []
    for trade in trades:
        sym = trade["sym"]
        parquet_path = os.path.join(oi_dir, f"{sym}_futures_oi.parquet")
        
        oi_signal = "NO_FUTURES"
        oi_change_pct = 0
        
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                # Look for March 11 data
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    mar11 = df[df['date'].dt.date == datetime(2026, 3, 11).date()]
                    mar10 = df[df['date'].dt.date == datetime(2026, 3, 10).date()]
                    
                    if not mar11.empty and not mar10.empty:
                        oi_11 = mar11['oi'].iloc[-1] if 'oi' in df.columns else 0
                        oi_10 = mar10['oi'].iloc[-1] if 'oi' in df.columns else 0
                        close_11 = mar11['close'].iloc[-1] if 'close' in df.columns else 0
                        close_10 = mar10['close'].iloc[-1] if 'close' in df.columns else 0
                        
                        if oi_10 > 0:
                            oi_change_pct = (oi_11 - oi_10) / oi_10 * 100
                        price_up = close_11 > close_10 if close_10 > 0 else None
                        
                        # Determine OI signal
                        if oi_change_pct > 2:  # OI increased
                            if price_up:
                                oi_signal = "LONG_BUILDUP"
                            else:
                                oi_signal = "SHORT_BUILDUP"
                        elif oi_change_pct < -2:  # OI decreased
                            if price_up:
                                oi_signal = "SHORT_COVERING"
                            else:
                                oi_signal = "LONG_UNWINDING"
                        else:
                            oi_signal = "NEUTRAL"
                    elif not mar11.empty:
                        # Only one day available
                        cols = list(df.columns)
                        oi_signal = "NO_PREV_DAY"
                    else:
                        oi_signal = "NO_MAR11_DATA"
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    mar11 = df[df['timestamp'].dt.date == datetime(2026, 3, 11).date()]
                    if not mar11.empty:
                        # Use intraday data 
                        first = mar11.iloc[0]
                        last = mar11.iloc[-1]
                        oi_first = first.get('oi', 0)
                        oi_last = last.get('oi', 0)
                        price_first = first.get('close', first.get('ltp', 0))
                        price_last = last.get('close', last.get('ltp', 0))
                        
                        if oi_first > 0:
                            oi_change_pct = (oi_last - oi_first) / oi_first * 100
                        price_up = price_last > price_first if price_first > 0 else None
                        
                        if abs(oi_change_pct) > 2:
                            if oi_change_pct > 0:
                                oi_signal = "LONG_BUILDUP" if price_up else "SHORT_BUILDUP"
                            else:
                                oi_signal = "SHORT_COVERING" if price_up else "LONG_UNWINDING"
                        else:
                            oi_signal = "NEUTRAL"
                    else:
                        oi_signal = "NO_MAR11_DATA"
                else:
                    oi_signal = f"COLS:{','.join(list(df.columns)[:5])}"
            except Exception as e:
                oi_signal = f"ERR:{str(e)[:30]}"
        
        # Determine conflict
        conflict = False
        if trade["dir"] == "BUY" and oi_signal in ("SHORT_BUILDUP", "LONG_UNWINDING"):
            conflict = True
        elif trade["dir"] == "SELL" and oi_signal in ("LONG_BUILDUP", "SHORT_COVERING"):
            conflict = True
        
        # OI direction
        if oi_signal in ("LONG_BUILDUP", "SHORT_COVERING"):
            oi_dir_label = "BULLISH"
        elif oi_signal in ("SHORT_BUILDUP", "LONG_UNWINDING"):
            oi_dir_label = "BEARISH"
        else:
            oi_dir_label = "NEUTRAL"
        
        results.append({
            **trade,
            "oi_signal": oi_signal,
            "oi_chg": oi_change_pct,
            "conflict": conflict,
            "oi_dir": oi_dir_label,
            "won": trade["pnl"] > 0
        })
    
    # Print results
    print("=" * 120)
    print("MARCH 11, 2026 — TRADES vs OI BUILDUP ANALYSIS")
    print("=" * 120)
    print(f"{'#':>2} {'Stock':<12} {'Dir':>4} {'Source':<14} {'Score':>5} {'PnL':>10} {'W/L':>3} {'OI Signal':<18} {'OI%':>6} {'Conflict':>8} {'OI Says':<8}")
    print("-" * 120)
    
    conflict_trades = []
    aligned_trades = []
    neutral_trades = []
    
    for i, r in enumerate(results, 1):
        w_l = "WIN" if r["won"] else "LOSS"
        conf = "YES ⚠️" if r["conflict"] else "no"
        pnl_str = f"Rs{r['pnl']:>+,}"
        oi_chg_str = f"{r['oi_chg']:>+.1f}%" if r['oi_chg'] != 0 else "N/A"
        
        print(f"{i:>2} {r['sym']:<12} {r['dir']:>4} {r['src']:<14} {r['score']:>5.0f} {pnl_str:>10} {w_l:>4} {r['oi_signal']:<18} {oi_chg_str:>6} {conf:>8} {r['oi_dir']:<8}")
        
        if r["conflict"]:
            conflict_trades.append(r)
        elif r["oi_dir"] != "NEUTRAL":
            aligned_trades.append(r)
        else:
            neutral_trades.append(r)
    
    print("-" * 120)
    
    # Summary
    total_pnl = sum(r["pnl"] for r in results)
    total_wins = len([r for r in results if r["won"]])
    print(f"\n   TOTAL: {len(results)} trades, {total_wins}W/{len(results)-total_wins}L, PnL = Rs{total_pnl:+,}")
    
    if conflict_trades:
        c_pnl = sum(r["pnl"] for r in conflict_trades)
        c_wins = len([r for r in conflict_trades if r["won"]])
        print(f"\n   ⚠️  OI CONFLICT ({len(conflict_trades)} trades): {c_wins}W/{len(conflict_trades)-c_wins}L, PnL = Rs{c_pnl:+,}")
        for r in conflict_trades:
            w = "WIN" if r["won"] else "LOSS"
            print(f"       {r['sym']:12s} {r['dir']} vs OI={r['oi_signal']} Rs{r['pnl']:>+,} ({w})")
    
    if aligned_trades:
        a_pnl = sum(r["pnl"] for r in aligned_trades)
        a_wins = len([r for r in aligned_trades if r["won"]])
        print(f"\n   ✅ OI ALIGNED ({len(aligned_trades)} trades): {a_wins}W/{len(aligned_trades)-a_wins}L, PnL = Rs{a_pnl:+,}")
    
    if neutral_trades:
        n_pnl = sum(r["pnl"] for r in neutral_trades)
        n_wins = len([r for r in neutral_trades if r["won"]])
        print(f"\n   ➖ OI NEUTRAL ({len(neutral_trades)} trades): {n_wins}W/{len(neutral_trades)-n_wins}L, PnL = Rs{n_pnl:+,}")
    
    # What-if: if we had flipped conflict trades
    if conflict_trades:
        print(f"\n   📊 WHAT-IF OI FLIP WAS ACTIVE ON MAR 11:")
        conflict_loss = sum(r["pnl"] for r in conflict_trades if r["pnl"] < 0)
        conflict_gain = sum(r["pnl"] for r in conflict_trades if r["pnl"] > 0)
        print(f"       Would have avoided Rs{abs(conflict_loss):,} in losses")
        print(f"       Would have flipped Rs{conflict_gain:,} in wins (may lose these)")
        print(f"       Net conflict PnL impact: Rs{sum(r['pnl'] for r in conflict_trades):+,}")
    
    print("=" * 120)
    
    # Debug: show parquet column names for first found file
    for trade in trades[:1]:
        pp = os.path.join(oi_dir, f"{trade['sym']}_futures_oi.parquet")
        if os.path.exists(pp):
            df = pd.read_parquet(pp)
            print(f"\nDEBUG: {trade['sym']} parquet columns: {list(df.columns)}")
            print(f"  Date range: {df.iloc[0][df.columns[0]]} to {df.iloc[-1][df.columns[0]]}")
            print(f"  Rows: {len(df)}")
            if 'date' in df.columns:
                dates = df['date'].unique()[-5:]
                print(f"  Last 5 dates: {dates}")

except ImportError:
    print("ERROR: pandas not installed on EC2. Cannot read parquet files.")
except Exception as e:
    import traceback
    traceback.print_exc()
