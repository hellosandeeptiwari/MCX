"""Day 3 Root Cause Analysis — WHY so many losses?"""
import json
from datetime import datetime
from collections import Counter, defaultdict

d = json.load(open('trade_history.json'))
today = [t for t in d if '2026-02-10' in t.get('closed_at','')]

losses = [t for t in today if t['pnl'] < 0]
wins = [t for t in today if t['pnl'] > 0]
print(f"TOTAL: {len(today)} trades, {len(losses)} losses, {len(wins)} wins")
print(f"Total PnL: +3,438 but WITHOUT MOTHERSON target hit: {sum(t['pnl'] for t in today) - 14022:+,.0f}")
print()

# ROOT CAUSE 1: SPEED GATE kills everything
print("=" * 70)
print("ROOT CAUSE 1: SPEED GATE — Exits too early, kills winners")
print("=" * 70)
sg = [t for t in today if t['result'] == 'OPTION_SPEED_GATE']
sg_loss = [t for t in sg if t['pnl'] < 0]
print(f"  {len(sg)} speed gate exits, {len(sg_loss)} were losses")
print(f"  Net impact: {sum(t['pnl'] for t in sg):+,.0f}")
print()
for t in sg:
    entry = t['avg_price']
    exit_p = t['exit_price']
    pct_move = (exit_p - entry) / entry * 100
    hold = (datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(t['timestamp'])).total_seconds() / 60
    print(f"  {t['symbol']:35s} {pct_move:+5.1f}% in {hold:4.0f}min  PnL={t['pnl']:+8,.0f}  {t.get('rationale','')[:40]}")
print()
print(f"  PROBLEM: Speed gate fires when premium moves <5% in 30min.")
print(f"  But options naturally have theta decay + bid-ask slippage.")
print(f"  A -3% move in 30min is NORMAL for an ATM option, not 'stalled'.")
print(f"  11 of 12 speed gate exits were LOSSES = they were exiting TOO EARLY.")

# ROOT CAUSE 2: RE-ENTRIES multiply losses
print()
print("=" * 70)
print("ROOT CAUSE 2: RE-ENTRIES — Same stock, same day, same result")
print("=" * 70)
by_underlying = defaultdict(list)
for i, t in enumerate(today):
    by_underlying[t['underlying']].append((i + 1, t))

total_reentry_losses = 0
for underlying, trades in sorted(by_underlying.items()):
    if len(trades) < 2:
        continue
    net = sum(t['pnl'] for _, t in trades)
    first_pnl = trades[0][1]['pnl']
    reentry_pnl = sum(t['pnl'] for _, t in trades[1:])
    if reentry_pnl < 0:
        total_reentry_losses += reentry_pnl
    
    print(f"\n  {underlying}: {len(trades)} trades, Net={net:+,.0f}")
    for idx, t in trades:
        ts = t['timestamp'][11:16]
        result = t['result']
        print(f"    #{idx:2d} {ts} {result:25s} PnL={t['pnl']:+8,.0f}  {'← RE-ENTRY' if idx != trades[0][0] else '← FIRST'}")
    
    if reentry_pnl < 0:
        print(f"    ⚠️ Re-entries cost: {reentry_pnl:+,.0f}")
    else:
        print(f"    ✅ Re-entries gained: {reentry_pnl:+,.0f}")

print(f"\n  TOTAL re-entry LOSSES (where re-entries made it worse): {total_reentry_losses:+,.0f}")

# ROOT CAUSE 3: TOO MANY TRADES
print()
print("=" * 70)
print("ROOT CAUSE 3: OVERTRADING — 27 trades in 1 day with 200K capital")
print("=" * 70)
unique = len(set(t['underlying'] for t in today))
print(f"  27 trades on {unique} unique underlyings")
print(f"  Avg premium per trade: ~20K = total capital turned over {27*20000/200000:.1f}x")
print(f"  Commissions lost to slippage (bid-ask ~0.5-1%): ~{27*20000*0.007:.0f}")
print()
print(f"  max_trades_per_day = 20, but bot placed 27!")
print(f"  This means limit wasn't enforced OR was raised during day.")

# ROOT CAUSE 4: Morning ORB traps
print()
print("=" * 70)
print("ROOT CAUSE 4: MORNING ORB TRAPS — Before 12:00 = disaster")
print("=" * 70)
morning = [t for t in today if datetime.fromisoformat(t['timestamp']).hour < 12]
afternoon = [t for t in today if datetime.fromisoformat(t['timestamp']).hour >= 12]
m_pnl = sum(t['pnl'] for t in morning)
a_pnl = sum(t['pnl'] for t in afternoon)
m_wins = sum(1 for t in morning if t['pnl'] > 0)
a_wins = sum(1 for t in afternoon if t['pnl'] > 0)
print(f"  Before 12:00: {len(morning)} trades, {m_wins}W-{len(morning)-m_wins}L, PnL={m_pnl:+,.0f}")
print(f"  After  12:00: {len(afternoon)} trades, {a_wins}W-{len(afternoon)-a_wins}L, PnL={a_pnl:+,.0f}")
print(f"  Difference: {a_pnl - m_pnl:+,.0f}")
print()
print(f"  10:00-10:30 ORB signals are false breakouts — stock hasn't settled.")
print(f"  BSE 3200CE entered 10:07, speed-gated at 10:36 = -8,231")
print(f"  TATASTEEL, JSWSTEEL, SWIGGY all trapped in first hour")

# ROOT CAUSE 5: BSE OBSESSION
print()
print("=" * 70)
print("ROOT CAUSE 5: BSE — Single stock lost ₹14,138 across 3 trades")
print("=" * 70)
bse = [t for t in today if t['underlying'] == 'NSE:BSE']
for t in bse:
    ts = t['timestamp'][11:16]
    print(f"  {ts} {t['symbol']:35s} {t['result']:25s} PnL={t['pnl']:+8,.0f}")
print(f"  Net BSE impact: {sum(t['pnl'] for t in bse):+,.0f}")
print(f"  Without BSE: day PnL = {sum(t['pnl'] for t in today) - sum(t['pnl'] for t in bse):+,.0f}")
print()
print(f"  BSE options have WIDE spreads (premium 98-106 = ~8% spread)")
print(f"  BSE lot size 375 x ₹100+ premium = ₹37K per trade — outsized risk")
print(f"  3 attempts, 3 losses. No re-entry guard blocked it.")

# SUMMARY
print()
print("=" * 70)
print("SUMMARY: WHY 15 LOSSES?")
print("=" * 70)
print(f"""
  1. SPEED GATE too aggressive: 11 of 12 were losses (-₹16,185)
     The 6-candle / 5% threshold still exits trades that just need TIME.
     
  2. RE-ENTRIES without guard: 7 symbols traded multiple times.
     No code prevents re-entering a symbol that just lost.
     
  3. MORNING ENTRIES trap: 7 trades before 12:00, 6 lost (-₹14,519)
     ORB signals in first 30-60min are unreliable false breakouts.
     
  4. TOO MANY TRADES: 27 trades burn capital on bid-ask slippage.
     Each trade costs ~0.5-1% in spread before it can profit.
     
  5. BSE ALONE cost ₹14,138 — wide spreads + high premium + 3 re-entries.
""")
print("RECOMMENDED FIXES FOR DAY 4:")
print(f"""
  FIX 1: Nerf speed gate further OR disable it.
         Current: exit if <5% in 6 candles (30min).
         Problem: ATM options with 30% IV naturally move slowly.
         Proposal: Raise to 10 candles (50min) and only exit if LOSING.
         
  FIX 2: Re-entry guard — block same underlying for 60min after loss.
         Would have saved: BSE re-entry #2 (-₹5,906), SWIGGY #2 (-₹715)
         
  FIX 3: Delay ORB entry to 10:30 (30min after open).
         Current: first trade at 10:07. Too early.
         
  FIX 4: Cap at 15 trades/day max (was 20, actually placed 27).
         Quality > quantity at ₹200K capital.
""")
