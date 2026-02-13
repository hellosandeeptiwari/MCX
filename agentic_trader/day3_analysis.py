"""Day 3 (2026-02-10) Full Trade Analysis"""
import json
from datetime import datetime, timedelta
from collections import defaultdict

trades = json.load(open('trade_history.json'))
today = [t for t in trades if '2026-02-10' in t.get('closed_at', '')]

SESSION_CUTOFF = datetime(2026, 2, 10, 15, 15)
NO_ENTRY_AFTER = datetime(2026, 2, 10, 14, 45)  # sensible latest entry

print("=" * 80)
print(f"DAY 3 TRADE ANALYSIS — {len(today)} trades")
print("=" * 80)

# 1. P&L Verification
print("\n[1] P&L VERIFICATION")
pnl_issues = []
total_pnl = 0
for i, t in enumerate(today):
    entry = t['avg_price']
    exit_p = t['exit_price']
    qty = t['quantity']
    recorded = t['pnl']
    expected = round((exit_p - entry) * qty, 2)
    total_pnl += recorded
    diff = abs(expected - recorded)
    if diff > 1:
        pnl_issues.append((i+1, t['symbol'], expected, recorded, diff))
        print(f"  ❌ Trade {i+1} {t['symbol']}: expected={expected:+.2f} recorded={recorded:+.2f} DIFF={diff:.2f}")

if not pnl_issues:
    print(f"  ✅ All 27 trades P&L calculations verified correct")
print(f"  Total P&L: ₹{total_pnl:+,.2f}")

# 2. Direction/Option-Type Mismatch
print("\n[2] DIRECTION vs OPTION-TYPE CONSISTENCY")
dir_issues = []
for i, t in enumerate(today):
    direction = t.get('direction', '')
    opt_type = t.get('option_type', '')
    # BUY direction (bullish) should use CE, SELL direction (bearish) should use PE
    if direction == 'BUY' and opt_type == 'PE':
        dir_issues.append((i+1, t['symbol'], direction, opt_type))
    elif direction == 'SELL' and opt_type == 'CE':
        dir_issues.append((i+1, t['symbol'], direction, opt_type))

for idx, sym, d, o in dir_issues:
    t = today[idx-1]
    print(f"  ❌ Trade {idx}: {sym} — Direction={d} but Option={o}")
    print(f"     Entry={t['avg_price']} Exit={t['exit_price']} PnL={t['pnl']:+.2f}")
    print(f"     Rationale: {t.get('rationale','?')}")
if not dir_issues:
    print(f"  ✅ All directions consistent with option types")

# 3. Re-entry Analysis (same underlying traded multiple times)
print("\n[3] RE-ENTRY ANALYSIS (same underlying)")
by_underlying = defaultdict(list)
for i, t in enumerate(today):
    by_underlying[t.get('underlying', t['symbol'])].append((i+1, t))

for underlying, trade_list in sorted(by_underlying.items()):
    if len(trade_list) > 1:
        net = sum(t['pnl'] for _, t in trade_list)
        print(f"\n  {underlying}: {len(trade_list)} trades, Net={net:+,.2f}")
        for idx, t in trade_list:
            ts = t['timestamp'][11:16]
            ex = t['exit_time'][11:16] if t.get('exit_time') else '?'
            print(f"    #{idx} {t['symbol']} {ts}→{ex} {t['result']:20s} PnL={t['pnl']:+,.2f}")
        
        # Check if re-entered after loss
        losses_then_reentry = []
        for j in range(len(trade_list) - 1):
            if trade_list[j][1]['pnl'] < 0:
                losses_then_reentry.append(trade_list[j][0])
        if losses_then_reentry:
            print(f"    ⚠️  Re-entered after loss on trade(s): {losses_then_reentry}")

# 4. Late Entries
print("\n\n[4] LATE ENTRIES (after 14:45)")
late_entries = []
for i, t in enumerate(today):
    ts = datetime.fromisoformat(t['timestamp'])
    if ts > NO_ENTRY_AFTER:
        duration = (datetime.fromisoformat(t.get('exit_time', t.get('closed_at', ''))) - ts).total_seconds()
        late_entries.append((i+1, t, ts, duration))

for idx, t, ts, dur in late_entries:
    print(f"  ❌ Trade {idx}: {t['symbol']}")
    print(f"     Entered {ts.strftime('%H:%M:%S')}, held {dur:.0f}s ({dur/60:.1f}min)")
    print(f"     PnL={t['pnl']:+,.2f} Result={t['result']}")

if not late_entries:
    print(f"  ✅ No late entries")

# 5. Speed Gate Analysis (was the exit premature?)
print("\n[5] SPEED GATE EXIT ANALYSIS")
sg_trades = [(i+1, t) for i, t in enumerate(today) if t['result'] == 'OPTION_SPEED_GATE']
sg_wins = sum(1 for _, t in sg_trades if t['pnl'] > 0)
sg_total = sum(t['pnl'] for _, t in sg_trades)
print(f"  {len(sg_trades)} speed gate exits, {sg_wins}W-{len(sg_trades)-sg_wins}L, Net={sg_total:+,.2f}")
for idx, t in sg_trades:
    entry = t['avg_price']
    exit_p = t['exit_price']
    sl = t['stop_loss']
    target = t['target']
    # How close was it to SL vs Target?
    entry_to_exit = (exit_p - entry) / entry * 100
    entry_to_sl = (sl - entry) / entry * 100
    entry_to_tgt = (target - entry) / entry * 100
    hold_min = (datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(t['timestamp'])).total_seconds() / 60
    print(f"  #{idx:2d} {t['symbol']:35s} held {hold_min:5.1f}min  exit={entry_to_exit:+.1f}%  (SL={entry_to_sl:+.1f}% Tgt={entry_to_tgt:+.1f}%)  PnL={t['pnl']:+,.2f}")

# 6. Exit type distribution
print("\n[6] EXIT TYPE DISTRIBUTION")
by_exit = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
for t in today:
    r = t['result']
    by_exit[r]['count'] += 1
    by_exit[r]['pnl'] += t['pnl']
    if t['pnl'] > 0:
        by_exit[r]['wins'] += 1

for exit_type, stats in sorted(by_exit.items(), key=lambda x: -x[1]['count']):
    wr = stats['wins']/stats['count']*100 if stats['count'] else 0
    print(f"  {exit_type:25s} {stats['count']:2d} trades  {stats['wins']}W  WR={wr:.0f}%  PnL={stats['pnl']:+,.2f}")

# 7. Time-of-day analysis
print("\n[7] TIME-OF-DAY ENTRY ANALYSIS")
time_buckets = {
    '09:15-10:00 (ORB)': (datetime(2026,2,10,9,15), datetime(2026,2,10,10,0)),
    '10:00-11:00': (datetime(2026,2,10,10,0), datetime(2026,2,10,11,0)),
    '11:00-12:00': (datetime(2026,2,10,11,0), datetime(2026,2,10,12,0)),
    '12:00-13:00': (datetime(2026,2,10,12,0), datetime(2026,2,10,13,0)),
    '13:00-14:00': (datetime(2026,2,10,13,0), datetime(2026,2,10,14,0)),
    '14:00-15:00': (datetime(2026,2,10,14,0), datetime(2026,2,10,15,0)),
    '15:00-15:30': (datetime(2026,2,10,15,0), datetime(2026,2,10,15,30)),
}
for label, (start, end) in time_buckets.items():
    bucket = [t for t in today if start <= datetime.fromisoformat(t['timestamp']) < end]
    if bucket:
        pnl = sum(t['pnl'] for t in bucket)
        wins = sum(1 for t in bucket if t['pnl'] > 0)
        print(f"  {label:25s} {len(bucket):2d} trades  {wins}W-{len(bucket)-wins}L  PnL={pnl:+,.2f}")

# 8. Premium/Risk analysis
print("\n[8] POSITION SIZING ANALYSIS")
premiums = [t['total_premium'] for t in today]
print(f"  Avg premium per trade: ₹{sum(premiums)/len(premiums):,.0f}")
print(f"  Min premium: ₹{min(premiums):,.0f}")
print(f"  Max premium: ₹{max(premiums):,.0f}")
# How many concurrent positions max?
events = []
for t in today:
    events.append((datetime.fromisoformat(t['timestamp']), +1))
    events.append((datetime.fromisoformat(t.get('exit_time', t['closed_at'])), -1))
events.sort()
current = 0
max_concurrent = 0
for ts, delta in events:
    current += delta
    max_concurrent = max(max_concurrent, current)
print(f"  Max concurrent positions: {max_concurrent}")

# Total capital at risk at peak
# Sort by timestamp
sorted_trades = sorted(today, key=lambda t: t['timestamp'])
active = []
max_risk = 0
for t in sorted_trades:
    entry_time = datetime.fromisoformat(t['timestamp'])
    # Remove closed trades
    active = [a for a in active if datetime.fromisoformat(a.get('exit_time', a['closed_at'])) > entry_time]
    active.append(t)
    risk = sum(a['total_premium'] for a in active)
    if risk > max_risk:
        max_risk = risk
        max_risk_time = entry_time
print(f"  Max total premium at risk: ₹{max_risk:,.0f} at {max_risk_time.strftime('%H:%M')}")

# 9. Winner/loser stats
print("\n[9] WINNER vs LOSER STATISTICS")
winners = [t for t in today if t['pnl'] > 0]
losers = [t for t in today if t['pnl'] < 0]
even = [t for t in today if t['pnl'] == 0]
avg_win = sum(t['pnl'] for t in winners) / len(winners) if winners else 0
avg_loss = sum(t['pnl'] for t in losers) / len(losers) if losers else 0
print(f"  Winners: {len(winners)} | Avg win: ₹{avg_win:+,.2f}")
print(f"  Losers:  {len(losers)} | Avg loss: ₹{avg_loss:+,.2f}")
print(f"  Even:    {len(even)}")
print(f"  Win Rate: {len(winners)/len(today)*100:.1f}%")
print(f"  Reward:Risk ratio: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "")
print(f"  Expectancy: ₹{total_pnl/len(today):+,.2f} per trade")

# Top 3 / Bottom 3
sorted_by_pnl = sorted(today, key=lambda t: t['pnl'])
print(f"\n  WORST 3:")
for t in sorted_by_pnl[:3]:
    print(f"    {t['symbol']:35s} {t['pnl']:+,.2f} ({t['result']})")
print(f"  BEST 3:")
for t in sorted_by_pnl[-3:]:
    print(f"    {t['symbol']:35s} {t['pnl']:+,.2f} ({t['result']})")

print("\n" + "=" * 80)
print("SUMMARY OF DISCREPANCIES")
print("=" * 80)
