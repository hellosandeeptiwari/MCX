"""
GMM Calibration Data - Sniper Trade Profitability Analysis
Reads gmm_calibration_data.jsonl and analyzes which sniper signals are profitable.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime

DATA_FILE = "gmm_calibration_data.jsonl"

def load_data():
    records = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

def classify_outcome(rec):
    """Classify if the trade would have been profitable based on scorer direction."""
    direction = rec.get("scorer_direction", "HOLD")
    max_up = rec.get("max_up_pct", 0) or 0
    max_down = rec.get("max_down_pct", 0) or 0
    
    if direction == "BUY":
        # For BUY, profit if price went up
        return max_up, max_down
    elif direction == "SELL":
        # For SELL, profit if price went down (favorable = abs(down))
        return abs(max_down), -max_up if max_up else 0
    else:
        return 0, 0

def main():
    records = load_data()
    print(f"Total records: {len(records)}")
    
    completed = [r for r in records if r.get("completed")]
    print(f"Completed records: {len(completed)}")
    
    # ─── SECTION 1: Overall outcome distribution ───
    print("\n" + "="*80)
    print("SECTION 1: OVERALL OUTCOME DISTRIBUTION")
    print("="*80)
    outcomes = defaultdict(int)
    for r in completed:
        outcomes[r.get("outcome", "UNKNOWN")] += 1
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:4d}  ({v/len(completed)*100:.1f}%)")
    
    # ─── SECTION 2: By scorer direction ───
    print("\n" + "="*80)
    print("SECTION 2: BY SCORER DIRECTION (BUY/SELL/HOLD)")
    print("="*80)
    by_dir = defaultdict(list)
    for r in completed:
        by_dir[r.get("scorer_direction", "HOLD")].append(r)
    
    for direction in ["BUY", "SELL", "HOLD"]:
        recs = by_dir.get(direction, [])
        if not recs:
            continue
        wins = sum(1 for r in recs if 
                   (direction == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (direction == "SELL" and r.get("outcome") == "LOSS_DOWN") or
                   (direction == "HOLD" and r.get("outcome") in ("PROFIT_UP", "LOSS_DOWN")))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs) if recs else 0
        avg_adv = sum(abs(r.get("max_adverse", 0)) for r in recs) / len(recs) if recs else 0
        
        # For directional: calculate actual PnL
        pnls = []
        for r in recs:
            fav, adv = classify_outcome(r)
            pnls.append(fav if fav > 0 else -abs(adv))
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        
        print(f"\n  {direction}: {len(recs)} trades | Win rate: {wins}/{len(recs)} ({wins/len(recs)*100:.1f}%)")
        print(f"    Avg max favorable: {avg_fav:.2f}% | Avg max adverse: {avg_adv:.2f}%")
    
    # ─── SECTION 3: By symbol - Top profitable and losing ───
    print("\n" + "="*80)
    print("SECTION 3: BY SYMBOL - TOP PROFITABLE & LOSING STOCKS")
    print("="*80)
    
    by_symbol = defaultdict(list)
    for r in completed:
        sym = r.get("symbol", "?").replace("NSE:", "")
        by_symbol[sym].append(r)
    
    symbol_stats = {}
    for sym, recs in by_symbol.items():
        directional = [r for r in recs if r.get("scorer_direction") in ("BUY", "SELL")]
        if len(directional) < 2:
            continue
        
        wins = 0
        total_fav = 0
        total_adv = 0
        for r in directional:
            d = r.get("scorer_direction")
            if d == "BUY" and r.get("outcome") == "PROFIT_UP":
                wins += 1
            elif d == "SELL" and r.get("outcome") == "LOSS_DOWN":
                wins += 1
            fav, _ = classify_outcome(r)
            total_fav += fav
        
        win_rate = wins / len(directional) * 100 if directional else 0
        avg_fav = total_fav / len(directional) if directional else 0
        symbol_stats[sym] = {
            "trades": len(directional),
            "wins": wins,
            "win_rate": win_rate,
            "avg_favorable_move": avg_fav,
            "total_records": len(recs),
        }
    
    # Sort by win rate (min 3 trades)
    qualifying = {k: v for k, v in symbol_stats.items() if v["trades"] >= 3}
    
    print(f"\n  Symbols with ≥3 directional trades: {len(qualifying)}")
    
    print("\n  ── TOP 15 PROFITABLE (by win rate) ──")
    print(f"  {'Symbol':<16} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")
    for sym, st in sorted(qualifying.items(), key=lambda x: (-x[1]["win_rate"], -x[1]["avg_favorable_move"]))[:15]:
        print(f"  {sym:<16} {st['trades']:>6} {st['wins']:>5} {st['win_rate']:>7.1f}% {st['avg_favorable_move']:>7.2f}%")
    
    print("\n  ── BOTTOM 15 LOSING (by win rate) ──")
    print(f"  {'Symbol':<16} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")
    for sym, st in sorted(qualifying.items(), key=lambda x: (x[1]["win_rate"], x[1]["avg_favorable_move"]))[:15]:
        print(f"  {sym:<16} {st['trades']:>6} {st['wins']:>5} {st['win_rate']:>7.1f}% {st['avg_favorable_move']:>7.2f}%")
    
    # ─── SECTION 4: By smart_score bucket ───
    print("\n" + "="*80)
    print("SECTION 4: BY SMART SCORE BUCKET")
    print("="*80)
    
    directional_all = [r for r in completed if r.get("scorer_direction") in ("BUY", "SELL")]
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    
    print(f"\n  {'Bucket':<12} {'Count':>6} {'WinUp':>6} {'WinDn':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi in buckets:
        bucket_recs = [r for r in directional_all if lo <= r.get("smart_score", 0) < hi]
        if not bucket_recs:
            continue
        win_up = sum(1 for r in bucket_recs if r.get("outcome") == "PROFIT_UP" and r.get("scorer_direction") == "BUY")
        win_dn = sum(1 for r in bucket_recs if r.get("outcome") == "LOSS_DOWN" and r.get("scorer_direction") == "SELL")
        wins = win_up + win_dn
        avg_fav = sum(r.get("max_favorable", 0) for r in bucket_recs) / len(bucket_recs)
        print(f"  {lo:>2}-{hi:<8} {len(bucket_recs):>6} {win_up:>6} {win_dn:>6} {wins/len(bucket_recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 5: By scorer confidence bucket ───
    print("\n" + "="*80)
    print("SECTION 5: BY SCORER DIRECTION CONFIDENCE")
    print("="*80)
    
    conf_buckets = [(0, 30), (30, 50), (50, 70), (70, 85), (85, 100)]
    print(f"\n  {'Confidence':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi in conf_buckets:
        bucket_recs = [r for r in directional_all if lo <= r.get("scorer_dir_confidence", 0) < hi]
        if not bucket_recs:
            continue
        wins = sum(1 for r in bucket_recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in bucket_recs) / len(bucket_recs)
        print(f"  {lo:>2}-{hi:<8} {len(bucket_recs):>6} {wins:>6} {wins/len(bucket_recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 6: By GMM direction hint ───
    print("\n" + "="*80)
    print("SECTION 6: BY GMM DIRECTION HINT")
    print("="*80)
    
    by_hint = defaultdict(list)
    for r in directional_all:
        by_hint[r.get("direction_hint", "?")].append(r)
    
    print(f"\n  {'Hint':<20} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for hint, recs in sorted(by_hint.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {hint:<20} {len(recs):>6} {wins:>6} {wins/len(recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 7: GMM confirms direction vs not ───
    print("\n" + "="*80)
    print("SECTION 7: GMM CONFIRMS DIRECTION — YES vs NO")
    print("="*80)
    
    confirms = [r for r in directional_all if r.get("gmm_confirms_dir")]
    not_confirms = [r for r in directional_all if not r.get("gmm_confirms_dir")]
    
    for label, recs in [("GMM CONFIRMS", confirms), ("GMM REJECTS", not_confirms)]:
        if not recs:
            continue
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {label}: {len(recs)} trades | Win rate: {wins}/{len(recs)} ({wins/len(recs)*100:.1f}%) | Avg fav: {avg_fav:.2f}%")
    
    # ─── SECTION 8: XGB signal analysis ───
    print("\n" + "="*80)
    print("SECTION 8: XGB SIGNAL (UP/DOWN/FLAT)")
    print("="*80)
    
    by_xgb = defaultdict(list)
    for r in directional_all:
        by_xgb[r.get("xgb_signal", "?")].append(r)
    
    print(f"\n  {'XGB Signal':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8} {'AvgMoveProb':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*12}")
    for sig, recs in sorted(by_xgb.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        avg_mp = sum(r.get("move_prob", 0) for r in recs) / len(recs)
        print(f"  {sig:<12} {len(recs):>6} {wins:>6} {wins/len(recs)*100:>7.1f}% {avg_fav:>7.2f}% {avg_mp:>11.3f}")
    
    # ─── SECTION 9: ML elite OK ───
    print("\n" + "="*80)
    print("SECTION 9: ML ELITE OK — TRUE vs FALSE")
    print("="*80)
    
    elite_yes = [r for r in directional_all if r.get("ml_elite_ok")]
    elite_no = [r for r in directional_all if not r.get("ml_elite_ok")]
    
    for label, recs in [("ELITE OK=True", elite_yes), ("ELITE OK=False", elite_no)]:
        if not recs:
            continue
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {label}: {len(recs)} trades | Win rate: {wins}/{len(recs)} ({wins/len(recs)*100:.1f}%) | Avg fav: {avg_fav:.2f}%")
    
    # ─── SECTION 10: By date ───
    print("\n" + "="*80)
    print("SECTION 10: DAY-BY-DAY PERFORMANCE")
    print("="*80)
    
    by_date = defaultdict(list)
    for r in directional_all:
        by_date[r.get("date", "?")].append(r)
    
    print(f"\n  {'Date':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for dt, recs in sorted(by_date.items()):
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {dt:<12} {len(recs):>6} {wins:>6} {wins/len(recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 11: Move probability vs actual outcome ───
    print("\n" + "="*80)
    print("SECTION 11: MOVE PROBABILITY vs WIN RATE")
    print("="*80)
    
    mp_buckets = [(0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]
    print(f"\n  {'MoveProb':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi in mp_buckets:
        bucket_recs = [r for r in directional_all if lo <= r.get("move_prob", 0) < hi]
        if not bucket_recs:
            continue
        wins = sum(1 for r in bucket_recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in bucket_recs) / len(bucket_recs)
        print(f"  {lo:.2f}-{hi:<6.2f} {len(bucket_recs):>6} {wins:>6} {wins/len(bucket_recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 12: DR score analysis ───
    print("\n" + "="*80)
    print("SECTION 12: DOWN-RISK SCORE vs OUTCOME (BUY trades only)")
    print("="*80)
    
    buy_trades = [r for r in directional_all if r.get("scorer_direction") == "BUY"]
    sell_trades = [r for r in directional_all if r.get("scorer_direction") == "SELL"]
    
    dr_buckets = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    print(f"\n  BUY trades — downdr_score buckets:")
    print(f"  {'DR Score':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi in dr_buckets:
        bucket_recs = [r for r in buy_trades if lo <= r.get("downdr_score", 0) < hi]
        if not bucket_recs:
            continue
        wins = sum(1 for r in bucket_recs if r.get("outcome") == "PROFIT_UP")
        avg_fav = sum(r.get("max_favorable", 0) for r in bucket_recs) / len(bucket_recs)
        print(f"  {lo:.2f}-{hi:<6.2f} {len(bucket_recs):>6} {wins:>6} {wins/len(bucket_recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    print(f"\n  SELL trades — updr_score buckets:")
    print(f"  {'UpDR Score':<12} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for lo, hi in dr_buckets:
        bucket_recs = [r for r in sell_trades if lo <= r.get("updr_score", 0) < hi]
        if not bucket_recs:
            continue
        wins = sum(1 for r in bucket_recs if r.get("outcome") == "LOSS_DOWN")
        avg_fav = sum(r.get("max_favorable", 0) for r in bucket_recs) / len(bucket_recs)
        print(f"  {lo:.2f}-{hi:<6.2f} {len(bucket_recs):>6} {wins:>6} {wins/len(bucket_recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 13: Best combo — high confidence + GMM confirms + smart_score ───
    print("\n" + "="*80)
    print("SECTION 13: SWEET SPOT — HIGH CONFIDENCE + GMM CONFIRMS + SMART SCORE ≥40")
    print("="*80)
    
    sweet = [r for r in directional_all if 
             r.get("scorer_dir_confidence", 0) >= 70 and
             r.get("gmm_confirms_dir") and
             r.get("smart_score", 0) >= 40]
    
    if sweet:
        wins = sum(1 for r in sweet if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in sweet) / len(sweet)
        print(f"  Sweet spot trades: {len(sweet)} | Win rate: {wins}/{len(sweet)} ({wins/len(sweet)*100:.1f}%) | Avg fav: {avg_fav:.2f}%")
        
        # Show by symbol
        sweet_by_sym = defaultdict(list)
        for r in sweet:
            sweet_by_sym[r.get("symbol", "?").replace("NSE:", "")].append(r)
        
        print(f"\n  {'Symbol':<16} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'AvgFav%':>8}")
        print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*8} {'-'*8}")
        for sym, recs in sorted(sweet_by_sym.items(), key=lambda x: -len(x[1])):
            wins_s = sum(1 for r in recs if 
                       (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                       (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
            avg_f = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
            print(f"  {sym:<16} {len(recs):>6} {wins_s:>5} {wins_s/len(recs)*100:>7.1f}% {avg_f:>7.2f}%")
    
    # ─── SECTION 14: Time of day analysis ───
    print("\n" + "="*80)
    print("SECTION 14: TIME-OF-DAY ANALYSIS")
    print("="*80)
    
    by_hour = defaultdict(list)
    for r in directional_all:
        ct = r.get("cycle_time", "00:00:00")
        try:
            hr = int(ct.split(":")[0])
            by_hour[hr].append(r)
        except:
            pass
    
    print(f"\n  {'Hour':<8} {'Count':>6} {'Wins':>6} {'WinRate':>8} {'AvgFav%':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for hr in sorted(by_hour.keys()):
        recs = by_hour[hr]
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {hr:>2}:00   {len(recs):>6} {wins:>6} {wins/len(recs)*100:>7.1f}% {avg_fav:>7.2f}%")
    
    # ─── SECTION 15: Chop hint analysis ───
    print("\n" + "="*80)
    print("SECTION 15: ML CHOP HINT — TRUE vs FALSE")
    print("="*80)
    
    chop_yes = [r for r in directional_all if r.get("ml_chop_hint")]
    chop_no = [r for r in directional_all if not r.get("ml_chop_hint")]
    
    for label, recs in [("CHOP=True", chop_yes), ("CHOP=False", chop_no)]:
        if not recs:
            continue
        wins = sum(1 for r in recs if 
                   (r.get("scorer_direction") == "BUY" and r.get("outcome") == "PROFIT_UP") or
                   (r.get("scorer_direction") == "SELL" and r.get("outcome") == "LOSS_DOWN"))
        avg_fav = sum(r.get("max_favorable", 0) for r in recs) / len(recs)
        print(f"  {label}: {len(recs)} trades | Win rate: {wins}/{len(recs)} ({wins/len(recs)*100:.1f}%) | Avg fav: {avg_fav:.2f}%")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
