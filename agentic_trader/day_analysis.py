"""Full day analysis for Feb 16, 2026 — Titan Autonomous Trader"""
import json
from datetime import datetime
from collections import defaultdict

# Load all data
trades = json.load(open('trade_history.json'))
data = json.load(open('active_trades.json'))
active = data.get('active_trades', []) if isinstance(data, dict) else data

# Filter today
today_closed = [t for t in trades if '2026-02-16' in t.get('timestamp', '')]
today_active = [t for t in active if '2026-02-16' in t.get('timestamp', '')]
all_today = today_closed + today_active

print("=" * 80)
print(f"  TITAN BOT — FULL DAY ANALYSIS — February 16, 2026")
print("=" * 80)
print(f"\nClosed trades: {len(today_closed)} | Still active at close: {len(today_active)}")
print(f"Total trades: {len(all_today)}")

# === 1. STRATEGY BREAKDOWN ===
print(f"\n{'='*60}")
print(f"  1. STRATEGY BREAKDOWN")
print(f"{'='*60}")

categories = {
    'Naked Options': [],
    'Credit Spreads': [],
    'Debit Spreads': [],
    'Iron Condors': [],
    'Cash/Equity': []
}

for t in all_today:
    if t.get('is_credit_spread'):
        categories['Credit Spreads'].append(t)
    elif t.get('is_debit_spread'):
        categories['Debit Spreads'].append(t)
    elif t.get('is_iron_condor'):
        categories['Iron Condors'].append(t)
    elif t.get('is_option', False):
        categories['Naked Options'].append(t)
    else:
        categories['Cash/Equity'].append(t)

for cat, group in categories.items():
    if not group:
        continue
    pnl_trades = [t for t in group if t.get('pnl') is not None and t.get('pnl') != 0]
    wins = sum(1 for t in pnl_trades if t.get('pnl', 0) > 0)
    losses = sum(1 for t in pnl_trades if t.get('pnl', 0) < 0)
    flat = sum(1 for t in pnl_trades if t.get('pnl', 0) == 0)
    total_pnl = sum(t.get('pnl', 0) for t in pnl_trades)
    still_open = sum(1 for t in group if t.get('status') == 'OPEN')
    print(f"\n  {cat}: {len(group)} trades ({still_open} still open)")
    if pnl_trades:
        print(f"    W/L: {wins}W / {losses}L / {flat}F | WR: {wins/len(pnl_trades)*100:.0f}%")
        print(f"    Net P&L: ₹{total_pnl:+,.0f}")
        avg_win = sum(t['pnl'] for t in pnl_trades if t['pnl'] > 0) / max(wins, 1)
        avg_loss = sum(t['pnl'] for t in pnl_trades if t['pnl'] < 0) / max(losses, 1)
        print(f"    Avg Win: ₹{avg_win:+,.0f} | Avg Loss: ₹{avg_loss:+,.0f}")
        if avg_loss != 0:
            print(f"    R:R Ratio: {abs(avg_win/avg_loss):.2f}")

# === 2. EXIT TYPE ANALYSIS ===
print(f"\n{'='*60}")
print(f"  2. EXIT TYPE ANALYSIS")
print(f"{'='*60}")

exit_types = defaultdict(list)
for t in all_today:
    status = t.get('status', 'UNKNOWN')
    exit_types[status].append(t)

for etype, group in sorted(exit_types.items(), key=lambda x: -len(x[1])):
    pnl = sum(t.get('pnl', 0) for t in group)
    wins = sum(1 for t in group if t.get('pnl', 0) > 0)
    losses = sum(1 for t in group if t.get('pnl', 0) < 0)
    print(f"  {etype:25s}: {len(group):2d} trades | {wins}W/{losses}L | P&L: ₹{pnl:+,.0f}")

# === 3. INDIVIDUAL TRADE DETAILS ===
print(f"\n{'='*60}")
print(f"  3. ALL TRADES (sorted by P&L)")
print(f"{'='*60}")

sorted_trades = sorted(all_today, key=lambda t: t.get('pnl', 0) or 0, reverse=True)
for i, t in enumerate(sorted_trades):
    sym = t.get('symbol', '?')
    if '|' in sym:
        sym = sym.split('|')[0].replace('NFO:', '') + ' [SPREAD]'
    else:
        sym = sym.replace('NFO:', '').replace('NSE:', '')
    
    pnl = t.get('pnl', 0) or 0
    status = t.get('status', '?')
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0) or 0
    score = t.get('entry_score', t.get('score', '?'))
    side = t.get('side', '?')
    qty = t.get('quantity', 0)
    
    emoji = '✅' if pnl > 0 else '❌' if pnl < 0 else '⏳' if status == 'OPEN' else '➖'
    print(f"  {emoji} {sym:45s} | {status:18s} | ₹{pnl:+8,.0f} | Score:{score}")

# === 4. TIME ANALYSIS ===
print(f"\n{'='*60}")
print(f"  4. ENTRY TIME ANALYSIS")
print(f"{'='*60}")

time_buckets = defaultdict(list)
for t in all_today:
    ts = t.get('timestamp', '')
    try:
        dt = datetime.fromisoformat(ts)
        hour = dt.hour
        if hour < 10:
            bucket = "09:15-10:00"
        elif hour < 11:
            bucket = "10:00-11:00"
        elif hour < 12:
            bucket = "11:00-12:00"
        elif hour < 13:
            bucket = "12:00-13:00"
        elif hour < 14:
            bucket = "13:00-14:00"
        elif hour < 15:
            bucket = "14:00-15:00"
        else:
            bucket = "15:00-15:30"
        time_buckets[bucket].append(t)
    except:
        pass

for bucket in sorted(time_buckets.keys()):
    group = time_buckets[bucket]
    pnl = sum(t.get('pnl', 0) or 0 for t in group)
    wins = sum(1 for t in group if (t.get('pnl', 0) or 0) > 0)
    losses = sum(1 for t in group if (t.get('pnl', 0) or 0) < 0)
    print(f"  {bucket}: {len(group):2d} trades | {wins}W/{losses}L | P&L: ₹{pnl:+,.0f}")

# === 5. SCORE vs OUTCOME ===
print(f"\n{'='*60}")
print(f"  5. SCORE vs OUTCOME")
print(f"{'='*60}")

score_buckets = defaultdict(list)
for t in all_today:
    score = t.get('entry_score', t.get('score', 0))
    if score is None or score == '?':
        score = 0
    try:
        score = float(score)
    except:
        score = 0
    
    if score >= 80:
        bucket = "80+"
    elif score >= 70:
        bucket = "70-79"
    elif score >= 60:
        bucket = "60-69"
    elif score >= 50:
        bucket = "50-59"
    else:
        bucket = "<50"
    score_buckets[bucket].append(t)

for bucket in ["80+", "70-79", "60-69", "50-59", "<50"]:
    group = score_buckets.get(bucket, [])
    if not group:
        continue
    pnl_trades = [t for t in group if t.get('pnl') is not None and t.get('pnl') != 0]
    wins = sum(1 for t in pnl_trades if t.get('pnl', 0) > 0)
    losses = sum(1 for t in pnl_trades if t.get('pnl', 0) < 0)
    pnl = sum(t.get('pnl', 0) for t in pnl_trades)
    print(f"  Score {bucket:6s}: {len(group):2d} trades | {wins}W/{losses}L | P&L: ₹{pnl:+,.0f}")

# === 6. CREDIT SPREAD DEEP DIVE ===
print(f"\n{'='*60}")
print(f"  6. CREDIT SPREAD DEEP DIVE")
print(f"{'='*60}")

spreads = categories['Credit Spreads']
if spreads:
    for t in spreads:
        sym = t.get('symbol', '?')
        names = sym.replace('NFO:', '').split('|')
        short_name = names[0] if names else sym
        status = t.get('status', '?')
        pnl = t.get('pnl', 0) or 0
        credit = t.get('net_credit', 0)
        credit_total = t.get('net_credit_total', 0)
        max_risk = t.get('max_risk', 0)
        width = t.get('spread_width', 0)
        qty = t.get('quantity', 0)
        lots = t.get('lots', 0)
        delta = t.get('net_delta', 0)
        dte = t.get('dte', 0)
        credit_pct = t.get('credit_pct', 0)
        
        emoji = '✅' if pnl > 0 else '❌' if pnl < 0 else '⏳'
        print(f"  {emoji} {short_name:40s}")
        print(f"     Status: {status} | PnL: ₹{pnl:+,.0f}")
        print(f"     Credit: ₹{credit:.2f}/share (₹{credit_total:,.0f} total) | Width: ₹{width} | Risk: ₹{max_risk:,.0f}")
        print(f"     Lots: {lots} ({qty} qty) | DTE: {dte} | Delta: {delta:.0f} | Credit%: {credit_pct:.0f}%")

# === 7. BIGGEST WINNERS & LOSERS ===
print(f"\n{'='*60}")
print(f"  7. TOP 5 WINNERS & LOSERS")
print(f"{'='*60}")

with_pnl = [t for t in all_today if t.get('pnl') is not None and t.get('pnl') != 0]
winners = sorted(with_pnl, key=lambda t: t.get('pnl', 0), reverse=True)[:5]
losers = sorted(with_pnl, key=lambda t: t.get('pnl', 0))[:5]

print(f"\n  🏆 TOP WINNERS:")
for t in winners:
    sym = t.get('symbol', '?').replace('NFO:', '').split('|')[0]
    print(f"    +₹{t['pnl']:,.0f} | {sym} | {t.get('status','?')}")

print(f"\n  💀 TOP LOSERS:")
for t in losers:
    sym = t.get('symbol', '?').replace('NFO:', '').split('|')[0]
    print(f"    ₹{t['pnl']:,.0f} | {sym} | {t.get('status','?')}")

# === 8. SUMMARY STATS ===
print(f"\n{'='*60}")
print(f"  8. DAY SUMMARY")
print(f"{'='*60}")

total_pnl = sum(t.get('pnl', 0) or 0 for t in all_today)
realized = sum(t.get('pnl', 0) or 0 for t in today_closed)
still_open_count = len(today_active)
with_pnl = [t for t in all_today if t.get('pnl') is not None and t.get('pnl') != 0]
total_wins = sum(1 for t in with_pnl if t['pnl'] > 0)
total_losses = sum(1 for t in with_pnl if t['pnl'] < 0)
gross_profit = sum(t['pnl'] for t in with_pnl if t['pnl'] > 0)
gross_loss = sum(t['pnl'] for t in with_pnl if t['pnl'] < 0)
max_single_win = max((t['pnl'] for t in with_pnl), default=0)
max_single_loss = min((t['pnl'] for t in with_pnl), default=0)

print(f"  Total trades: {len(all_today)}")
print(f"  Win/Loss: {total_wins}W / {total_losses}L ({total_wins/(total_wins+total_losses)*100:.1f}% WR)" if total_wins+total_losses > 0 else "")
print(f"  Gross Profit: ₹{gross_profit:+,.0f}")
print(f"  Gross Loss: ₹{gross_loss:+,.0f}")
print(f"  NET P&L (realized): ₹{realized:+,.0f}")
print(f"  Still open: {still_open_count} positions")
print(f"  Max single win: ₹{max_single_win:+,.0f}")
print(f"  Max single loss: ₹{max_single_loss:+,.0f}")
print(f"  Profit Factor: {abs(gross_profit/gross_loss):.2f}" if gross_loss != 0 else "")

# === 9. DUPLICATE UNDERLYING CHECK ===
print(f"\n{'='*60}")
print(f"  9. SAME-STOCK DOUBLED TRADES (naked + spread on same underlying)")
print(f"{'='*60}")

naked_underlyings = set()
spread_underlyings = set()
for t in all_today:
    underlying = t.get('underlying', '')
    if not underlying:
        sym = t.get('symbol', '')
        # Extract underlying from option symbol
        parts = sym.replace('NFO:', '').split('|')[0]
        # crude extraction
        underlying = parts
    
    if t.get('is_credit_spread') or t.get('is_debit_spread'):
        spread_underlyings.add(t.get('underlying', t.get('symbol','')))
    elif t.get('is_option'):
        # Try to extract underlying
        und = t.get('underlying', '')
        if und:
            naked_underlyings.add(und)

doubled = naked_underlyings & spread_underlyings
if doubled:
    print(f"  ⚠️ {len(doubled)} stocks had BOTH naked option + credit spread:")
    for d in sorted(doubled):
        print(f"    {d}")
else:
    print(f"  None found")

print(f"\n{'='*80}")
print(f"  END OF ANALYSIS")
print(f"{'='*80}")
