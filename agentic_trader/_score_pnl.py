"""Quick score-wise PnL analysis for today's trades."""
import json
from datetime import datetime

with open('trade_history.json', 'r') as f:
    trades = json.load(f)

today = '2026-02-16'
today_trades = [t for t in trades if t.get('timestamp','')[:10] == today or t.get('entry_time','')[:10] == today]
print(f"Total trades today ({today}): {len(today_trades)}")

if not today_trades:
    dates = set()
    for t in trades:
        ts = t.get('timestamp', t.get('entry_time', ''))[:10]
        if ts:
            dates.add(ts)
    print(f"Available dates: {sorted(dates)[-10:]}")
else:
    score_buckets = {}
    for t in today_trades:
        score = t.get('entry_score', t.get('score', 0))
        pnl = t.get('net_pnl', t.get('pnl', t.get('gross_pnl', 0)))
        brokerage = t.get('brokerage', 0)
        symbol = t.get('symbol', '?')
        status = t.get('status', t.get('exit_type', ''))

        if score >= 80:
            bucket = '80+'
        elif score >= 70:
            bucket = '70-79'
        elif score >= 60:
            bucket = '60-69'
        elif score >= 50:
            bucket = '50-59'
        else:
            bucket = '<50'

        if bucket not in score_buckets:
            score_buckets[bucket] = {'trades': 0, 'wins': 0, 'gross_pnl': 0, 'net_pnl': 0, 'brokerage': 0, 'details': []}

        score_buckets[bucket]['trades'] += 1
        score_buckets[bucket]['net_pnl'] += pnl
        score_buckets[bucket]['brokerage'] += brokerage
        if pnl > 0:
            score_buckets[bucket]['wins'] += 1
        score_buckets[bucket]['details'].append({
            'symbol': symbol[:20],
            'score': score,
            'pnl': pnl,
            'status': status,
        })

    header = f"{'Score':<10} {'Trades':<8} {'Wins':<6} {'WinRate':<10} {'Net PnL':>12} {'Brokerage':>12}"
    print(f"\n{header}")
    print("-" * 60)

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_brok = 0

    for bucket in ['80+', '70-79', '60-69', '50-59', '<50']:
        if bucket in score_buckets:
            b = score_buckets[bucket]
            wr = f"{(b['wins']/b['trades']*100):.0f}%" if b['trades'] > 0 else '0%'
            print(f"{bucket:<10} {b['trades']:<8} {b['wins']:<6} {wr:<10} {b['net_pnl']:>12,.0f} {b['brokerage']:>12,.0f}")
            total_pnl += b['net_pnl']
            total_trades += b['trades']
            total_wins += b['wins']
            total_brok += b['brokerage']

    print("-" * 60)
    wr_total = f"{(total_wins/total_trades*100):.0f}%" if total_trades > 0 else '0%'
    print(f"{'TOTAL':<10} {total_trades:<8} {total_wins:<6} {wr_total:<10} {total_pnl:>12,.0f} {total_brok:>12,.0f}")

    # Detail per bucket
    print("\n\n=== TRADE DETAILS BY SCORE ===")
    for bucket in ['80+', '70-79', '60-69', '50-59', '<50']:
        if bucket in score_buckets:
            b = score_buckets[bucket]
            print(f"\n--- Score {bucket} ({b['trades']} trades, Net: {b['net_pnl']:,.0f}) ---")
            for d in sorted(b['details'], key=lambda x: -x['pnl']):
                marker = "+" if d['pnl'] > 0 else ""
                print(f"  {d['symbol']:<22} Score:{d['score']:<4} PnL:{marker}{d['pnl']:>8,.0f}  [{d['status']}]")
