"""
GMM Calibration Deep Dive — Which sniper signals actually produce profitable moves?
Analyzes gmm_calibration_data.jsonl (updr/downdr, 50+ stocks, 8 candles forward)
"""
import json, sys
from collections import defaultdict

records = []
with open("gmm_calibration_data.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("completed"):
            records.append(r)

print(f"=== GMM CALIBRATION DEEP DIVE ===")
print(f"Total completed records: {len(records)}")
dates = sorted(set(r["date"] for r in records))
print(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
symbols = sorted(set(r["symbol"] for r in records))
print(f"Unique symbols: {len(symbols)}")

# Define profit: direction-aware
# BUY direction → max_up_pct ≥ threshold = PROFIT
# SELL direction → max_down_pct (abs) ≥ threshold = PROFIT
THRESHOLDS = [0.20, 0.30, 0.50]  # % move thresholds

def is_profitable(r, threshold=0.30):
    """Did price move ≥ threshold% in the predicted direction within 8 candles?"""
    direction = r.get("scorer_direction", "BUY")
    if direction == "BUY":
        return r.get("max_up_pct", 0) >= threshold
    else:
        return abs(r.get("max_down_pct", 0)) >= threshold

def is_any_move(r, threshold=0.30):
    """Did price move ≥ threshold% in ANY direction?"""
    return max(r.get("max_up_pct", 0), abs(r.get("max_down_pct", 0))) >= threshold

print(f"\n{'='*70}")
print("1. OVERALL WIN RATES BY THRESHOLD")
print(f"{'='*70}")
for t in THRESHOLDS:
    wins = sum(1 for r in records if is_profitable(r, t))
    moves = sum(1 for r in records if is_any_move(r, t))
    print(f"  ≥{t}% move in direction: {wins}/{len(records)} = {wins/len(records)*100:.1f}%")
    print(f"  ≥{t}% move ANY direction: {moves}/{len(records)} = {moves/len(records)*100:.1f}%")

# 2. UPDR score ranges
print(f"\n{'='*70}")
print("2. UPDR SCORE — Higher updr = stronger UP pressure from GMM")
print(f"{'='*70}")
updr_buckets = [(0, 0.08), (0.08, 0.10), (0.10, 0.12), (0.12, 0.15), (0.15, 0.20), (0.20, 1.0)]
for lo, hi in updr_buckets:
    subset = [r for r in records if lo <= r.get("updr_score", 0) < hi]
    if len(subset) < 5:
        continue
    buy_sub = [r for r in subset if r.get("scorer_direction") == "BUY"]
    sell_sub = [r for r in subset if r.get("scorer_direction") == "SELL"]
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100 if subset else 0
    buy_wins = sum(1 for r in buy_sub if is_profitable(r, 0.30))
    buy_wr = buy_wins/len(buy_sub)*100 if buy_sub else 0
    sell_wins = sum(1 for r in sell_sub if is_profitable(r, 0.30))
    sell_wr = sell_wins/len(sell_sub)*100 if sell_sub else 0
    avg_max_up = sum(r.get("max_up_pct", 0) for r in subset)/len(subset)
    avg_max_dn = sum(abs(r.get("max_down_pct", 0)) for r in subset)/len(subset)
    print(f"  updr [{lo:.2f}-{hi:.2f}): n={len(subset):4d}  WR={wr:5.1f}%  BUY_WR={buy_wr:5.1f}%({len(buy_sub)})  SELL_WR={sell_wr:5.1f}%({len(sell_sub)})  avg_up={avg_max_up:.3f}%  avg_dn={avg_max_dn:.3f}%")

# 3. DOWNDR score ranges
print(f"\n{'='*70}")
print("3. DOWNDR SCORE — Higher downdr = stronger DOWN pressure from GMM")
print(f"{'='*70}")
downdr_buckets = [(0, 0.12), (0.12, 0.15), (0.15, 0.18), (0.18, 0.20), (0.20, 0.22), (0.22, 1.0)]
for lo, hi in downdr_buckets:
    subset = [r for r in records if lo <= r.get("downdr_score", 0) < hi]
    if len(subset) < 5:
        continue
    buy_sub = [r for r in subset if r.get("scorer_direction") == "BUY"]
    sell_sub = [r for r in subset if r.get("scorer_direction") == "SELL"]
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100 if subset else 0
    buy_wins = sum(1 for r in buy_sub if is_profitable(r, 0.30))
    buy_wr = buy_wins/len(buy_sub)*100 if buy_sub else 0
    sell_wins = sum(1 for r in sell_sub if is_profitable(r, 0.30))
    sell_wr = sell_wins/len(sell_sub)*100 if sell_sub else 0
    avg_max_up = sum(r.get("max_up_pct", 0) for r in subset)/len(subset)
    avg_max_dn = sum(abs(r.get("max_down_pct", 0)) for r in subset)/len(subset)
    print(f"  downdr [{lo:.2f}-{hi:.2f}): n={len(subset):4d}  WR={wr:5.1f}%  BUY_WR={buy_wr:5.1f}%({len(buy_sub)})  SELL_WR={sell_wr:5.1f}%({len(sell_sub)})  avg_up={avg_max_up:.3f}%  avg_dn={avg_max_dn:.3f}%")

# 4. UPDR-DOWNDR SPREAD (directional regime detection)
print(f"\n{'='*70}")
print("4. UPDR-DOWNDR SPREAD (updr-downdr: +ve=bullish bias, -ve=bearish bias)")
print(f"{'='*70}")
spread_buckets = [(-1, -0.10), (-0.10, -0.05), (-0.05, 0), (0, 0.05), (0.05, 0.10), (0.10, 1)]
for lo, hi in spread_buckets:
    subset = [r for r in records if lo <= (r.get("updr_score", 0) - r.get("downdr_score", 0)) < hi]
    if len(subset) < 5:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    # check if aligned direction trades do better
    aligned = [r for r in subset if 
               (r.get("scorer_direction") == "BUY" and (r.get("updr_score", 0) - r.get("downdr_score", 0)) > 0) or
               (r.get("scorer_direction") == "SELL" and (r.get("updr_score", 0) - r.get("downdr_score", 0)) < 0)]
    aligned_wins = sum(1 for r in aligned if is_profitable(r, 0.30))
    aligned_wr = aligned_wins/len(aligned)*100 if aligned else 0
    print(f"  spread [{lo:+.2f},{hi:+.2f}): n={len(subset):4d}  WR={wr:5.1f}%  ALIGNED_WR={aligned_wr:5.1f}%({len(aligned)})")

# 5. MOVE PROBABILITY
print(f"\n{'='*70}")
print("5. MOVE PROBABILITY — GMM's predicted probability of significant move")
print(f"{'='*70}")
mp_buckets = [(0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]
for lo, hi in mp_buckets:
    subset = [r for r in records if lo <= r.get("move_prob", 0) < hi]
    if len(subset) < 5:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    big_wins = sum(1 for r in subset if is_profitable(r, 0.50))
    big_wr = big_wins/len(subset)*100
    avg_fav = sum(r.get("max_favorable", 0) for r in subset)/len(subset)
    print(f"  move_prob [{lo:.2f}-{hi:.2f}): n={len(subset):4d}  WR≥0.3%={wr:5.1f}%  WR≥0.5%={big_wr:5.1f}%  avg_favor={avg_fav:.3f}%")

# 6. SMART SCORE
print(f"\n{'='*70}")
print("6. SMART SCORE — Composite quality score")
print(f"{'='*70}")
ss_buckets = [(0, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 100)]
for lo, hi in ss_buckets:
    subset = [r for r in records if lo <= r.get("smart_score", 0) < hi]
    if len(subset) < 5:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    avg_fav = sum(r.get("max_favorable", 0) for r in subset)/len(subset)
    print(f"  smart [{lo:2d}-{hi:2d}): n={len(subset):4d}  WR={wr:5.1f}%  avg_favorable={avg_fav:.3f}%")

# 7. XGB SIGNAL
print(f"\n{'='*70}")
print("7. XGB SIGNAL — ML directional prediction")
print(f"{'='*70}")
for sig in ["UP", "DOWN", "FLAT"]:
    subset = [r for r in records if r.get("xgb_signal") == sig]
    if not subset:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    big_wins = sum(1 for r in subset if is_profitable(r, 0.50))
    big_wr = big_wins/len(subset)*100
    # aligned = xgb UP + scorer BUY, or xgb DOWN + scorer SELL
    aligned = [r for r in subset if 
               (sig == "UP" and r.get("scorer_direction") == "BUY") or
               (sig == "DOWN" and r.get("scorer_direction") == "SELL")]
    aligned_wins = sum(1 for r in aligned if is_profitable(r, 0.30))
    aligned_wr = aligned_wins/len(aligned)*100 if aligned else 0
    print(f"  XGB={sig:5s}: n={len(subset):4d}  WR={wr:5.1f}%  WR≥0.5%={big_wr:5.1f}%  ALIGNED_WR={aligned_wr:5.1f}%({len(aligned)})")

# 8. ML ELITE + CHOP
print(f"\n{'='*70}")
print("8. ML ELITE OK + CHOP HINT — Quality gates")
print(f"{'='*70}")
for elite in [True, False]:
    for chop in [True, False]:
        subset = [r for r in records if r.get("ml_elite_ok") == elite and r.get("ml_chop_hint") == chop]
        if len(subset) < 5:
            continue
        wins = sum(1 for r in subset if is_profitable(r, 0.30))
        wr = wins/len(subset)*100
        avg_fav = sum(r.get("max_favorable", 0) for r in subset)/len(subset)
        print(f"  elite={str(elite):5s} chop={str(chop):5s}: n={len(subset):4d}  WR={wr:5.1f}%  avg_favorable={avg_fav:.3f}%")

# 9. GMM DIRECTION HINT
print(f"\n{'='*70}")
print("9. GMM DIRECTION HINT — Does GMM direction alignment matter?")
print(f"{'='*70}")
for hint in ["BULLISH", "BEARISH", "BULLISH_LEAN", "BEARISH_LEAN", "NEUTRAL"]:
    subset = [r for r in records if r.get("direction_hint") == hint]
    if len(subset) < 5:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    # aligned = BULLISH*/BUY or BEARISH*/SELL
    if "BULLISH" in hint:
        aligned = [r for r in subset if r.get("scorer_direction") == "BUY"]
    elif "BEARISH" in hint:
        aligned = [r for r in subset if r.get("scorer_direction") == "SELL"]
    else:
        aligned = subset  # NEUTRAL, take all
    aligned_wins = sum(1 for r in aligned if is_profitable(r, 0.30))
    aligned_wr = aligned_wins/len(aligned)*100 if aligned else 0
    print(f"  {hint:15s}: n={len(subset):4d}  WR={wr:5.1f}%  ALIGNED_WR={aligned_wr:5.1f}%({len(aligned)})")

# 10. SCORER DIRECTION CONFIDENCE
print(f"\n{'='*70}")
print("10. SCORER DIRECTION CONFIDENCE — How confident is the scorer?")
print(f"{'='*70}")
conf_buckets = [(0, 50), (50, 65), (65, 75), (75, 85), (85, 95), (95, 100)]
for lo, hi in conf_buckets:
    subset = [r for r in records if lo <= r.get("scorer_dir_confidence", 0) < hi]
    if len(subset) < 5:
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    avg_fav = sum(r.get("max_favorable", 0) for r in subset)/len(subset)
    print(f"  conf [{lo:2d}-{hi:2d}): n={len(subset):4d}  WR={wr:5.1f}%  avg_favorable={avg_fav:.3f}%")

# 11. TOP PERFORMING STOCKS
print(f"\n{'='*70}")
print("11. TOP/BOTTOM STOCKS BY WIN RATE (min 15 records)")
print(f"{'='*70}")
stock_stats = {}
for r in records:
    sym = r["symbol"]
    if sym not in stock_stats:
        stock_stats[sym] = {"total": 0, "wins": 0, "big_wins": 0, "avg_fav": 0}
    stock_stats[sym]["total"] += 1
    if is_profitable(r, 0.30):
        stock_stats[sym]["wins"] += 1
    if is_profitable(r, 0.50):
        stock_stats[sym]["big_wins"] += 1
    stock_stats[sym]["avg_fav"] += r.get("max_favorable", 0)

top_stocks = []
for sym, s in stock_stats.items():
    if s["total"] >= 15:
        s["avg_fav"] /= s["total"]
        wr = s["wins"]/s["total"]*100
        big_wr = s["big_wins"]/s["total"]*100
        top_stocks.append((sym, s["total"], wr, big_wr, s["avg_fav"]))

top_stocks.sort(key=lambda x: x[2], reverse=True)
print("  TOP 10:")
for sym, n, wr, big_wr, avg_fav in top_stocks[:10]:
    print(f"    {sym:25s} n={n:3d}  WR≥0.3%={wr:5.1f}%  WR≥0.5%={big_wr:5.1f}%  avg_fav={avg_fav:.3f}%")
print("  BOTTOM 10:")
for sym, n, wr, big_wr, avg_fav in top_stocks[-10:]:
    print(f"    {sym:25s} n={n:3d}  WR≥0.3%={wr:5.1f}%  WR≥0.5%={big_wr:5.1f}%  avg_fav={avg_fav:.3f}%")

# 12. TIME OF DAY
print(f"\n{'='*70}")
print("12. TIME OF DAY — Which scan times produce best moves?")
print(f"{'='*70}")
time_stats = defaultdict(lambda: {"total": 0, "wins": 0})
for r in records:
    hour = r.get("cycle_time", "00:00:00")[:2]
    time_stats[hour]["total"] += 1
    if is_profitable(r, 0.30):
        time_stats[hour]["wins"] += 1
for hour in sorted(time_stats):
    s = time_stats[hour]
    wr = s["wins"]/s["total"]*100 if s["total"] else 0
    print(f"  {hour}:xx  n={s['total']:4d}  WR={wr:5.1f}%")

# 13. DAY BY DAY PERFORMANCE
print(f"\n{'='*70}")
print("13. DAY-BY-DAY PERFORMANCE")
print(f"{'='*70}")
for d in dates:
    day_recs = [r for r in records if r["date"] == d]
    wins = sum(1 for r in day_recs if is_profitable(r, 0.30))
    wr = wins/len(day_recs)*100 if day_recs else 0
    symbols_day = len(set(r["symbol"] for r in day_recs))
    print(f"  {d}: n={len(day_recs):4d}  symbols={symbols_day:2d}  WR={wr:5.1f}%")

# 14. THE GOLDEN COMBOS — multi-factor intersection
print(f"\n{'='*70}")
print("14. GOLDEN COMBOS — Multi-factor sweet spots")
print(f"{'='*70}")

combos = {
    "move_prob≥0.55 + elite + !chop": lambda r: r.get("move_prob",0)>=0.55 and r.get("ml_elite_ok") and not r.get("ml_chop_hint"),
    "move_prob≥0.55 + smart≥45 + elite": lambda r: r.get("move_prob",0)>=0.55 and r.get("smart_score",0)>=45 and r.get("ml_elite_ok"),
    "xgb!FLAT + gmm_confirms + smart≥40": lambda r: r.get("xgb_signal")!="FLAT" and r.get("gmm_confirms_dir") and r.get("smart_score",0)>=40,
    "xgb!FLAT + move_prob≥0.55 + elite": lambda r: r.get("xgb_signal")!="FLAT" and r.get("move_prob",0)>=0.55 and r.get("ml_elite_ok"),
    "xgb!FLAT + conf≥75 + elite + !chop": lambda r: r.get("xgb_signal")!="FLAT" and r.get("scorer_dir_confidence",0)>=75 and r.get("ml_elite_ok") and not r.get("ml_chop_hint"),
    "move_prob≥0.65 + smart≥50": lambda r: r.get("move_prob",0)>=0.65 and r.get("smart_score",0)>=50,
    "move_prob≥0.65 + xgb!FLAT + elite": lambda r: r.get("move_prob",0)>=0.65 and r.get("xgb_signal")!="FLAT" and r.get("ml_elite_ok"),
    "spread_aligned + move_prob≥0.50 + smart≥40": lambda r: (
        ((r.get("updr_score",0)-r.get("downdr_score",0))>0 and r.get("scorer_direction")=="BUY") or
        ((r.get("updr_score",0)-r.get("downdr_score",0))<0 and r.get("scorer_direction")=="SELL")
    ) and r.get("move_prob",0)>=0.50 and r.get("smart_score",0)>=40,
    "downdr≥0.20 + SELL + smart≥45": lambda r: r.get("downdr_score",0)>=0.20 and r.get("scorer_direction")=="SELL" and r.get("smart_score",0)>=45,
    "move_prob≥0.70 (any)": lambda r: r.get("move_prob",0)>=0.70,
    "FULL SNIPER: xgb!FLAT+prob≥0.55+elite+!chop+smart≥45+conf≥70": lambda r: (
        r.get("xgb_signal")!="FLAT" and r.get("move_prob",0)>=0.55 and r.get("ml_elite_ok") and
        not r.get("ml_chop_hint") and r.get("smart_score",0)>=45 and r.get("scorer_dir_confidence",0)>=70
    ),
}

for name, fn in combos.items():
    subset = [r for r in records if fn(r)]
    if len(subset) < 3:
        print(f"  {name}: n={len(subset)} (too few)")
        continue
    wins = sum(1 for r in subset if is_profitable(r, 0.30))
    wr = wins/len(subset)*100
    big_wins = sum(1 for r in subset if is_profitable(r, 0.50))
    big_wr = big_wins/len(subset)*100
    avg_fav = sum(r.get("max_favorable", 0) for r in subset)/len(subset)
    avg_adv = sum(abs(r.get("max_adverse", 0)) for r in subset)/len(subset)
    print(f"  {name}")
    print(f"    n={len(subset):4d}  WR≥0.3%={wr:5.1f}%  WR≥0.5%={big_wr:5.1f}%  avg_fav={avg_fav:.3f}%  avg_adv={avg_adv:.3f}%  edge={avg_fav-avg_adv:+.3f}%")

# 15. FORWARD PRICE CURVE — at which candle does the max move typically happen?
print(f"\n{'='*70}")
print("15. FORWARD PRICE CURVE — When does peak move happen?")
print(f"{'='*70}")
candle_peak = defaultdict(int)
candle_peak_profitable = defaultdict(int)
for r in records:
    fps = r.get("forward_prices", [])
    if not fps or len(fps) < 2:
        continue
    direction = r.get("scorer_direction", "BUY")
    best_candle = 1
    best_pct = 0
    for fp in fps[1:]:  # skip candle 1 (entry)
        pct = fp.get("pct_change", 0)
        if direction == "BUY" and pct > best_pct:
            best_pct = pct
            best_candle = fp["candle_n"]
        elif direction == "SELL" and pct < best_pct:
            best_pct = pct
            best_candle = fp["candle_n"]
    candle_peak[best_candle] += 1
    if abs(best_pct) >= 0.30:
        candle_peak_profitable[best_candle] += 1

for c_num in sorted(candle_peak):
    total = candle_peak[c_num]
    prof = candle_peak_profitable.get(c_num, 0)
    print(f"  Candle {c_num}: peak_at={total:4d}  profitable_at={prof:3d}  ({prof/total*100:.1f}% of peaks)")

# 16. CANDLE-BY-CANDLE CUMULATIVE WIN RATE — how quickly do trades become profitable?
print(f"\n{'='*70}")
print("16. CANDLE-BY-CANDLE CUMULATIVE WIN RATE — How fast do good trades move?")
print(f"{'='*70}")
for threshold in [0.20, 0.30, 0.50]:
    print(f"  Threshold ≥ {threshold}%:")
    for candle_n in range(2, 9):
        wins_by_candle = 0
        total_with_candle = 0
        for r in records:
            fps = r.get("forward_prices", [])
            available = [fp for fp in fps if fp["candle_n"] <= candle_n]
            if len(available) < 2:
                continue
            total_with_candle += 1
            direction = r.get("scorer_direction", "BUY")
            if direction == "BUY":
                max_move = max(fp.get("pct_change", 0) for fp in available[1:])
            else:
                max_move = max(abs(fp.get("pct_change", 0)) for fp in available[1:] if fp.get("pct_change", 0) <= 0) if any(fp.get("pct_change", 0) <= 0 for fp in available[1:]) else 0
            if max_move >= threshold:
                wins_by_candle += 1
        wr = wins_by_candle/total_with_candle*100 if total_with_candle else 0
        print(f"    by candle {candle_n}: {wins_by_candle}/{total_with_candle} = {wr:.1f}%")

print(f"\n{'='*70}")
print("DONE.")
