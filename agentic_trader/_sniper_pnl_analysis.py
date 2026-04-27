"""
Sniper (GMM_SNIPER) Trade Profitability Analysis from titan_state.db
"""
import json
from collections import defaultdict

def load_data():
    with open("sniper_trades_dump.json", "r") as f:
        return json.load(f)

def load_all():
    with open("all_trades_dump.json", "r", encoding="utf-8-sig") as f:
        return json.load(f)

def main():
    all_trades = load_all()
    
    # Filter by strategy type
    sniper = [t for t in all_trades if t.get("strategy_type") == "GMM_SNIPER"]
    non_sniper = [t for t in all_trades if t.get("strategy_type") != "GMM_SNIPER"]
    
    print(f"Total trades in DB: {len(all_trades)}")
    print(f"GMM_SNIPER trades: {len(sniper)}")
    print(f"Other trades: {len(non_sniper)}")
    
    # Strategy breakdown
    strat_counts = defaultdict(int)
    for t in all_trades:
        strat_counts[t.get("strategy_type", "?")] += 1
    print("\nStrategy breakdown:")
    for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {c}")
    
    closed = [t for t in sniper if t.get("status") == "CLOSED" and t.get("pnl") is not None]
    active = [t for t in sniper if t.get("status") != "CLOSED"]
    
    print(f"\nClosed sniper trades: {len(closed)}")
    print(f"Active sniper trades: {len(active)}")
    
    if not closed:
        print("No closed sniper trades to analyze.")
        return
    
    # ─── SECTION 1: Overall P&L ───
    print("\n" + "="*80)
    print("SECTION 1: OVERALL SNIPER P&L")
    print("="*80)
    
    total_pnl = sum(t["pnl"] for t in closed)
    winners = [t for t in closed if t["pnl"] > 0]
    losers = [t for t in closed if t["pnl"] < 0]
    breakeven = [t for t in closed if t["pnl"] == 0]
    
    avg_win = sum(t["pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(t["pnl"] for t in losers) / len(losers) if losers else 0
    
    print(f"  Total P&L: ₹{total_pnl:,.0f}")
    print(f"  Winners: {len(winners)}/{len(closed)} ({len(winners)/len(closed)*100:.1f}%)")
    print(f"  Losers: {len(losers)}/{len(closed)} ({len(losers)/len(closed)*100:.1f}%)")
    print(f"  Breakeven: {len(breakeven)}")
    print(f"  Avg win: ₹{avg_win:,.0f} | Avg loss: ₹{avg_loss:,.0f}")
    if avg_loss != 0:
        print(f"  Win/Loss ratio: {abs(avg_win/avg_loss):.2f}")
    
    # Avg PnL %
    pnl_pcts = [t.get("pnl_pct", 0) or 0 for t in closed]
    avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
    print(f"  Avg PnL %: {avg_pnl_pct:.2f}%")
    
    # ─── SECTION 2: By symbol ───
    print("\n" + "="*80)
    print("SECTION 2: BY STOCK — PROFITABLE vs LOSING")
    print("="*80)
    
    by_sym = defaultdict(list)
    for t in closed:
        sym = t.get("underlying", t.get("symbol", "?")).replace("NSE:", "")
        by_sym[sym].append(t)
    
    sym_stats = {}
    for sym, trades in by_sym.items():
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        sym_stats[sym] = {
            "trades": len(trades),
            "wins": wins,
            "win_rate": wins / len(trades) * 100 if trades else 0,
            "total_pnl": total,
            "avg_pnl": total / len(trades),
        }
    
    print(f"\n  ── TOP PROFITABLE STOCKS ──")
    print(f"  {'Symbol':<16} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for sym, st in sorted(sym_stats.items(), key=lambda x: -x[1]["total_pnl"])[:15]:
        print(f"  {sym:<16} {st['trades']:>6} {st['wins']:>5} {st['win_rate']:>7.1f}% ₹{st['total_pnl']:>10,.0f} ₹{st['avg_pnl']:>8,.0f}")
    
    print(f"\n  ── WORST LOSING STOCKS ──")
    print(f"  {'Symbol':<16} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*16} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for sym, st in sorted(sym_stats.items(), key=lambda x: x[1]["total_pnl"])[:15]:
        if st["total_pnl"] >= 0:
            break
        print(f"  {sym:<16} {st['trades']:>6} {st['wins']:>5} {st['win_rate']:>7.1f}% ₹{st['total_pnl']:>10,.0f} ₹{st['avg_pnl']:>8,.0f}")
    
    # ─── SECTION 3: By date ───
    print("\n" + "="*80)
    print("SECTION 3: DAY-BY-DAY P&L")
    print("="*80)
    
    by_date = defaultdict(list)
    for t in closed:
        by_date[t.get("date", "?")].append(t)
    
    print(f"\n  {'Date':<12} {'Trades':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for dt, trades in sorted(by_date.items()):
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {dt:<12} {len(trades):>6} {wins:>5} {wins/len(trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(trades):>8,.0f}")
    
    # ─── SECTION 4: By direction (BUY/SELL) ───
    print("\n" + "="*80)
    print("SECTION 4: BY DIRECTION")
    print("="*80)
    
    for direction in ["BUY", "SELL"]:
        dir_trades = [t for t in closed if t.get("direction") == direction]
        if not dir_trades:
            continue
        total = sum(t["pnl"] for t in dir_trades)
        wins = sum(1 for t in dir_trades if t["pnl"] > 0)
        avg = total / len(dir_trades)
        print(f"  {direction}: {len(dir_trades)} trades | Win rate: {wins}/{len(dir_trades)} ({wins/len(dir_trades)*100:.1f}%) | Total P&L: ₹{total:,.0f} | Avg: ₹{avg:,.0f}")
    
    # ─── SECTION 5: By smart score ───
    print("\n" + "="*80)
    print("SECTION 5: BY SMART SCORE BUCKET")
    print("="*80)
    
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    print(f"\n  {'Bucket':<12} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for lo, hi in buckets:
        bucket_trades = [t for t in closed if lo <= (t.get("smart_score") or 0) < hi]
        if not bucket_trades:
            continue
        total = sum(t["pnl"] for t in bucket_trades)
        wins = sum(1 for t in bucket_trades if t["pnl"] > 0)
        print(f"  {lo:>2}-{hi:<8} {len(bucket_trades):>6} {wins:>5} {wins/len(bucket_trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(bucket_trades):>8,.0f}")
    
    # ─── SECTION 6: By DR score ───
    print("\n" + "="*80)
    print("SECTION 6: BY DR SCORE BUCKET")
    print("="*80)
    
    dr_buckets = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    print(f"\n  {'DR Score':<12} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for lo, hi in dr_buckets:
        bucket_trades = [t for t in closed if lo <= (t.get("dr_score") or 0) < hi]
        if not bucket_trades:
            continue
        total = sum(t["pnl"] for t in bucket_trades)
        wins = sum(1 for t in bucket_trades if t["pnl"] > 0)
        print(f"  {lo:.2f}-{hi:<6.2f} {len(bucket_trades):>6} {wins:>5} {wins/len(bucket_trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(bucket_trades):>8,.0f}")
    
    # ─── SECTION 7: By ML move probability ───
    print("\n" + "="*80)
    print("SECTION 7: BY ML MOVE PROBABILITY")
    print("="*80)
    
    mp_buckets = [(0, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.0)]
    print(f"\n  {'MoveProb':<12} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for lo, hi in mp_buckets:
        bucket_trades = [t for t in closed if lo <= (t.get("ml_move_prob") or 0) < hi]
        if not bucket_trades:
            continue
        total = sum(t["pnl"] for t in bucket_trades)
        wins = sum(1 for t in bucket_trades if t["pnl"] > 0)
        print(f"  {lo:.2f}-{hi:<6.2f} {len(bucket_trades):>6} {wins:>5} {wins/len(bucket_trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(bucket_trades):>8,.0f}")
    
    # ─── SECTION 8: By exit type ───
    print("\n" + "="*80)
    print("SECTION 8: BY EXIT TYPE")
    print("="*80)
    
    by_exit = defaultdict(list)
    for t in closed:
        by_exit[t.get("exit_type", "?")].append(t)
    
    print(f"\n  {'ExitType':<20} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for exit_type, trades in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {str(exit_type):<20} {len(trades):>6} {wins:>5} {wins/len(trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(trades):>8,.0f}")
    
    # ─── SECTION 9: By lot multiplier ───
    print("\n" + "="*80)
    print("SECTION 9: BY LOT MULTIPLIER")
    print("="*80)
    
    by_lots = defaultdict(list)
    for t in closed:
        by_lots[t.get("lot_multiplier", 1.0)].append(t)
    
    print(f"\n  {'LotMult':<10} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for mult, trades in sorted(by_lots.items()):
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {mult:<10} {len(trades):>6} {wins:>5} {wins/len(trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(trades):>8,.0f}")
    
    # ─── SECTION 10: By score tier ───
    print("\n" + "="*80)
    print("SECTION 10: BY SCORE TIER")
    print("="*80)
    
    by_tier = defaultdict(list)
    for t in closed:
        by_tier[t.get("score_tier", "?")].append(t)
    
    print(f"\n  {'Tier':<15} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for tier, trades in sorted(by_tier.items(), key=lambda x: -sum(t["pnl"] for t in x[1])):
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {str(tier):<15} {len(trades):>6} {wins:>5} {wins/len(trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(trades):>8,.0f}")
    
    # ─── SECTION 11: By option type (CE/PE) ───
    print("\n" + "="*80)
    print("SECTION 11: BY OPTION TYPE (CE/PE)")
    print("="*80)
    
    for ot in ["CE", "PE"]:
        ot_trades = [t for t in closed if t.get("option_type") == ot]
        if not ot_trades:
            continue
        total = sum(t["pnl"] for t in ot_trades)
        wins = sum(1 for t in ot_trades if t["pnl"] > 0)
        print(f"  {ot}: {len(ot_trades)} trades | Win rate: {wins}/{len(ot_trades)} ({wins/len(ot_trades)*100:.1f}%) | Total P&L: ₹{total:,.0f} | Avg: ₹{total/len(ot_trades):,.0f}")
    
    # ─── SECTION 12: Hold time analysis ───
    print("\n" + "="*80)
    print("SECTION 12: BY HOLD TIME")
    print("="*80)
    
    hold_buckets = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 9999)]
    print(f"\n  {'HoldMins':<12} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*8} {'-'*12} {'-'*10}")
    for lo, hi in hold_buckets:
        bucket_trades = [t for t in closed if lo <= (t.get("hold_minutes") or 0) < hi]
        if not bucket_trades:
            continue
        total = sum(t["pnl"] for t in bucket_trades)
        wins = sum(1 for t in bucket_trades if t["pnl"] > 0)
        label = f"{lo}-{hi}min" if hi < 9999 else f"{lo}+min"
        print(f"  {label:<12} {len(bucket_trades):>6} {wins:>5} {wins/len(bucket_trades)*100:>7.1f}% ₹{total:>10,.0f} ₹{total/len(bucket_trades):>8,.0f}")
    
    # ─── SECTION 13: R-Multiple analysis ───
    print("\n" + "="*80)
    print("SECTION 13: R-MULTIPLE DISTRIBUTION")
    print("="*80)
    
    with_r = [t for t in closed if t.get("r_multiple") is not None]
    if with_r:
        r_mults = [t["r_multiple"] for t in with_r]
        print(f"  Avg R-multiple: {sum(r_mults)/len(r_mults):.2f}")
        print(f"  R > 1 (full target): {sum(1 for r in r_mults if r >= 1)}/{len(with_r)}")
        print(f"  R > 0 (any profit): {sum(1 for r in r_mults if r > 0)}/{len(with_r)}")
        print(f"  R < -1 (full SL hit): {sum(1 for r in r_mults if r <= -1)}/{len(with_r)}")
    
    # ─── SECTION 14: Max favorable excursion ───
    print("\n" + "="*80)
    print("SECTION 14: MAX FAVORABLE EXCURSION (MFE)")
    print("="*80)
    
    with_mfe = [t for t in closed if t.get("max_favorable") is not None]
    if with_mfe:
        mfes = [t["max_favorable"] for t in with_mfe]
        avg_mfe = sum(mfes) / len(mfes)
        print(f"  Avg MFE: {avg_mfe:.2f}%")
        
        # MFE for winners vs losers
        win_mfe = [t["max_favorable"] for t in with_mfe if t["pnl"] > 0]
        loss_mfe = [t["max_favorable"] for t in with_mfe if t["pnl"] <= 0]
        if win_mfe:
            print(f"  Winners avg MFE: {sum(win_mfe)/len(win_mfe):.2f}%")
        if loss_mfe:
            print(f"  Losers avg MFE: {sum(loss_mfe)/len(loss_mfe):.2f}%")
    
    # ─── SECTION 15: Each trade detail (last 20) ───
    print("\n" + "="*80)
    print("SECTION 15: RECENT 20 TRADES DETAIL")
    print("="*80)
    
    sorted_trades = sorted(closed, key=lambda t: t.get("entry_time", ""), reverse=True)
    print(f"\n  {'Date':<11} {'Symbol':<14} {'Dir':<5} {'OT':<3} {'Smart':>5} {'DR':>5} {'MovP':>5} {'P&L':>10} {'P&L%':>7} {'Exit':>12} {'Hold':>5}")
    print(f"  {'-'*11} {'-'*14} {'-'*5} {'-'*3} {'-'*5} {'-'*5} {'-'*5} {'-'*10} {'-'*7} {'-'*12} {'-'*5}")
    for t in sorted_trades[:20]:
        sym = (t.get("underlying") or t.get("symbol", "?")).replace("NSE:", "")
        print(f"  {t.get('date',''):<11} {sym:<14} {t.get('direction','?'):<5} {t.get('option_type','?'):<3} {t.get('smart_score',0):>5.0f} {t.get('dr_score',0):>5.2f} {t.get('ml_move_prob',0):>5.2f} ₹{t.get('pnl',0):>8,.0f} {t.get('pnl_pct',0):>6.1f}% {str(t.get('exit_type','?')):<12} {t.get('hold_minutes',0):>4}m")
    
    # ─── SECTION 16: ML Direction accuracy ───
    print("\n" + "="*80)
    print("SECTION 16: ML DIRECTION vs ACTUAL OUTCOME")
    print("="*80)
    
    by_ml_dir = defaultdict(list)
    for t in closed:
        by_ml_dir[t.get("ml_direction", "?")].append(t)
    
    print(f"\n  {'ML Dir':<15} {'Count':>6} {'Wins':>5} {'WinRate':>8} {'TotalPnL':>12}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*8} {'-'*12}")
    for ml_dir, trades in sorted(by_ml_dir.items(), key=lambda x: -len(x[1])):
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {str(ml_dir):<15} {len(trades):>6} {wins:>5} {wins/len(trades)*100:>7.1f}% ₹{total:>10,.0f}")
    
    # ─── SECTION 17: THP conversion ───
    print("\n" + "="*80)
    print("SECTION 17: THP (TRAILING) CONVERSION")
    print("="*80)
    
    thp_yes = [t for t in closed if t.get("thp_converted")]
    thp_no = [t for t in closed if not t.get("thp_converted")]
    
    for label, trades in [("THP Converted", thp_yes), ("No THP", thp_no)]:
        if not trades:
            continue
        total = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        print(f"  {label}: {len(trades)} trades | WR: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%) | P&L: ₹{total:,.0f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
