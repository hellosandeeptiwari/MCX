#!/usr/bin/env python3
"""
TITAN TRADE QUERY — CLI for instant trade analysis
====================================================
Usage:
    python trade_query.py                    # Today's summary
    python trade_query.py 2026-03-09         # Specific date
    python trade_query.py --week             # Last 7 trading days
    python trade_query.py --month            # Last 30 trading days
    python trade_query.py --source SNIPER    # Filter by strategy source
    python trade_query.py --underlying ONGC  # Filter by stock
    python trade_query.py --backfill         # Import JSONL → SQLite
    python trade_query.py --exit-types       # Breakdown by exit type
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state_db import get_state_db
from trade_ledger import get_trade_ledger


def fmt_pnl(v):
    return f"₹{v:>+10,.0f}" if v else "₹        0"


def print_daily(date_str=None, source=None, underlying=None):
    db = get_state_db()
    summary = db.daily_pnl_summary(date_str)

    if summary.get('total', 0) == 0:
        print(f"\nNo trades found for {date_str or 'today'}")
        return

    d = date_str or summary.get('date', 'today')
    total = summary['total']
    wins = summary.get('wins', 0)
    losses = summary.get('losses', 0)
    open_t = summary.get('open_trades', 0)
    total_pnl = summary.get('total_pnl', 0)
    avg_win = summary.get('avg_win', 0)
    avg_loss = summary.get('avg_loss', 0)
    best = summary.get('best_trade', 0)
    worst = summary.get('worst_trade', 0)
    thp = summary.get('thp_count', 0)
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    print(f"\n{'═' * 80}")
    print(f"  TITAN TRADE REPORT — {d}")
    print(f"{'═' * 80}")
    print(f"  Trades: {total} | W:{wins} L:{losses} Open:{open_t} | Win Rate: {wr:.0f}%")
    print(f"  P&L:   {fmt_pnl(total_pnl)}  | Avg Win: {fmt_pnl(avg_win)} | Avg Loss: {fmt_pnl(avg_loss)}")
    print(f"  Best:  {fmt_pnl(best)}  | Worst:  {fmt_pnl(worst)}  | THP Conversions: {thp}")
    print()

    # Strategy breakdown
    by_source = db.pnl_by_source(date_str)
    if by_source:
        print(f"  {'Source':<20} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'P&L':>12}")
        print(f"  {'─' * 20} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 12}")
        for row in by_source:
            src = row['source'] or 'UNKNOWN'
            wr_s = (row['wins'] / row['trades'] * 100) if row['trades'] > 0 else 0
            print(f"  {src:<20} {row['trades']:>6} {row['wins']:>5} {wr_s:>5.0f}% {fmt_pnl(row['pnl'])}")
        print()

    # Exit type breakdown
    by_exit = db.pnl_by_exit_type(date_str)
    if by_exit:
        print(f"  {'Exit Type':<35} {'Count':>5} {'P&L':>12}")
        print(f"  {'─' * 35} {'─' * 5} {'─' * 12}")
        for row in by_exit:
            print(f"  {(row['exit_type'] or '?'):<35} {row['trades']:>5} {fmt_pnl(row['pnl'])}")
        print()

    # Individual trades
    trades = db.query_trades(date_str, source=source, underlying=underlying)
    if trades:
        print(f"  {'#':>2} {'Time':<8} {'Underlying':<14} {'Dir':<4} {'Source':<16} {'Score':>5} {'P(mv)':>5} {'Entry':>8} {'Exit':>8} {'P&L':>10} {'Exit Type':<25} {'THP':>3}")
        print(f"  {'─' * 2} {'─' * 8} {'─' * 14} {'─' * 4} {'─' * 16} {'─' * 5} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 25} {'─' * 3}")

        for i, t in enumerate(trades, 1):
            ts = (t.get('entry_time') or '')
            if 'T' in ts:
                ts = ts.split('T')[1][:8]
            ul = (t.get('underlying') or t.get('symbol') or '?').replace('NSE:', '')
            dir_ = t.get('direction') or '?'
            src = t.get('source') or '?'
            sc = t.get('final_score') or t.get('smart_score') or 0
            pmv = t.get('ml_move_prob') or 0
            ep = t.get('entry_price') or 0
            xp = t.get('exit_price') or 0
            pnl = t.get('pnl') or 0
            et = t.get('exit_type') or t.get('status') or '?'
            thp_flag = '✓' if t.get('thp_converted') else ''
            print(f"  {i:>2} {ts:<8} {ul:<14} {dir_:<4} {src:<16} {sc:>5.0f} {pmv:>5.2f} {ep:>8.1f} {xp:>8.1f} {fmt_pnl(pnl)} {et:<25} {thp_flag:>3}")
    print(f"{'═' * 80}")


def print_multi_day(days=7):
    db = get_state_db()
    rows = db.multi_day_summary(days)

    if not rows:
        print(f"\nNo trade data found for last {days} days")
        return

    cumulative = 0
    print(f"\n{'═' * 70}")
    print(f"  TITAN P&L — Last {days} Trading Days")
    print(f"{'═' * 70}")
    print(f"  {'Date':<12} {'Trades':>6} {'Wins':>5} {'Losses':>6} {'WR%':>6} {'Day P&L':>12} {'Cumulative':>12}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 12} {'─' * 12}")

    for row in reversed(rows):  # oldest first
        wr = (row['wins'] / (row['wins'] + row['losses']) * 100) if (row['wins'] + row['losses']) > 0 else 0
        cumulative += row['pnl']
        print(f"  {row['date']:<12} {row['trades']:>6} {row['wins']:>5} {row['losses']:>6} {wr:>5.0f}% {fmt_pnl(row['pnl'])} {fmt_pnl(cumulative)}")

    print(f"  {'─' * 12} {'─' * 6} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 12} {'─' * 12}")
    total_trades = sum(r['trades'] for r in rows)
    total_wins = sum(r['wins'] for r in rows)
    total_losses = sum(r['losses'] for r in rows)
    total_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    print(f"  {'TOTAL':<12} {total_trades:>6} {total_wins:>5} {total_losses:>6} {total_wr:>5.0f}% {fmt_pnl(cumulative)} {fmt_pnl(cumulative)}")
    print(f"{'═' * 70}")


def do_backfill():
    print("Backfilling JSONL trade ledger → SQLite trades table...")
    ledger = get_trade_ledger()
    ledger.backfill_to_sqlite()
    print("\nVerifying:")
    print_multi_day(30)


if __name__ == '__main__':
    args = sys.argv[1:]

    if '--backfill' in args:
        do_backfill()
    elif '--week' in args:
        print_multi_day(7)
    elif '--month' in args:
        print_multi_day(30)
    elif '--exit-types' in args:
        date = next((a for a in args if a.startswith('20')), None)
        db = get_state_db()
        rows = db.pnl_by_exit_type(date)
        print(f"\nExit Type Breakdown ({date or 'today'}):")
        for r in rows:
            print(f"  {r['exit_type']:<35} {r['trades']:>3} trades  {fmt_pnl(r['pnl'])}")
    else:
        date = next((a for a in args if a.startswith('20')), None)
        source = None
        underlying = None
        for i, a in enumerate(args):
            if a == '--source' and i + 1 < len(args):
                source = args[i + 1]
            elif a == '--underlying' and i + 1 < len(args):
                underlying = args[i + 1]
        print_daily(date, source=source, underlying=underlying)
