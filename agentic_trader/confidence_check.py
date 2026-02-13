"""Quick confidence assessment for tomorrow's trading"""
import json, os
from collections import defaultdict

history = []
if os.path.exists('trade_history.json'):
    with open('trade_history.json', 'r') as f:
        history = json.load(f)

print(f"Total historical trades: {len(history)}")
wins = [t for t in history if t.get('pnl', 0) > 0]
losses = [t for t in history if t.get('pnl', 0) < 0]
print(f"Wins: {len(wins)} | Losses: {len(losses)}")
total_pnl = sum(t.get('pnl', 0) for t in history)
print(f"Total PnL: Rs {total_pnl:+,.0f}")
wsum = sum(t['pnl'] for t in wins) if wins else 0
lsum = sum(t['pnl'] for t in losses) if losses else 0
if wins:
    print(f"Avg winner: Rs {wsum/len(wins):+,.0f}")
if losses:
    print(f"Avg loser: Rs {lsum/len(losses):+,.0f}")
if wins and losses:
    print(f"Payoff ratio: {abs(wsum/len(wins) / (lsum/len(losses))):.2f}:1")
    wr = len(wins) / (len(wins) + len(losses)) * 100
    print(f"Win rate: {wr:.1f}%")

# Exit types
exit_types = {}
for t in history:
    et = t.get('result', 'UNKNOWN')
    exit_types[et] = exit_types.get(et, 0) + 1
print(f"\nExit types: {exit_types}")

# Daily breakdown
daily = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0})
for t in history:
    ts = t.get('closed_at', t.get('timestamp', ''))[:10]
    if ts:
        daily[ts]['pnl'] += t.get('pnl', 0)
        daily[ts]['trades'] += 1
        if t.get('pnl', 0) > 0:
            daily[ts]['wins'] += 1
        elif t.get('pnl', 0) < 0:
            daily[ts]['losses'] += 1

print(f"\nDaily breakdown (last 5 days):")
for date in sorted(daily.keys())[-5:]:
    d = daily[date]
    wr = d['wins'] / (d['wins'] + d['losses']) * 100 if (d['wins'] + d['losses']) > 0 else 0
    print(f"  {date}: {d['trades']}T  W{d['wins']}/L{d['losses']} ({wr:.0f}%)  Rs {d['pnl']:+,.0f}")

# Winning/losing days
winning_days = sum(1 for d in daily.values() if d['pnl'] > 0)
losing_days = sum(1 for d in daily.values() if d['pnl'] < 0)
print(f"\nWinning days: {winning_days} | Losing days: {losing_days}")

# Check what changes go live tomorrow
print("\n" + "=" * 60)
print("CHANGES GOING LIVE TOMORROW (NOT YET BATTLE-TESTED)")
print("=" * 60)
print("""
1. TIERED POSITION SIZING
   - Premium(score>=70): 5% risk, min 2 lots, 25% cap
   - Standard(score>=65): 3.5% risk, 20% cap  
   - Base: 2% risk, 15% cap
   - round() instead of int() for multiplier
   Simulation: Old Rs -2,220 -> New Rs +6,177

2. SMART GATES 8-12 (candle-data driven)
   - Gate 8: Follow-Through >= 2 (premium) / >= 1 (standard)
   - Gate 9: ADX >= 30 (premium) / >= 25 (standard)
   - Gate 10: ORB Overextension < 100%
   - Gate 11: Range Expansion < 0.60 ATR
   - Gate 12: Same-symbol re-entry prevention
   Simulation: Old Rs +3,381 -> New Rs +12,551 (7 losses prevented)

3. DEBIT SPREAD OVERHAUL
   - min_move: 2.5% -> 1.2%
   - R:R: 1:1.25 -> 1:2.67
   - max_lots: 2 -> 4
   - 5 candle-smart gates (D1-D5)
   - PROACTIVE scanning (not just fallback)
   Impact: 0 debit spreads historically -> expect 2-4/day

4. ENHANCED LOGGING
   - Entry: trade_id, score, tier, candle gates, sizing rationale
   - Exit: candles_held, R-multiple, max_favorable, exit_reason
   - Daily: daily_summaries/daily_summary_YYYY-MM-DD.json
   Impact: Full trade character analysis possible
""")

print("=" * 60)
print("CONFIDENCE ASSESSMENT")
print("=" * 60)
