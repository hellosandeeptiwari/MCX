import json, os

ledger_file = os.path.join(os.path.dirname(__file__), 'trade_ledger_2026-02-24.jsonl')

trades = []
with open(ledger_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

entries = [t for t in trades if t.get('event') == 'ENTRY']
exits = [t for t in trades if t.get('event') == 'EXIT']
sniper_entries = [t for t in entries if 'SNIPER' in t.get('setup', '')]
decisions = [t for t in trades if t.get('event') == 'DECISION']

print(f"Total events: {len(trades)}")
print(f"Entries: {len(entries)} | Exits: {len(exits)} | Sniper entries: {len(sniper_entries)}")
print()

# Analyze exits
wins = [e for e in exits if e.get('pnl', 0) > 0]
losses = [e for e in exits if e.get('pnl', 0) <= 0]
total_pnl = sum(e.get('pnl', 0) for e in exits)
n = len(wins) + len(losses)
print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/n*100:.1f}%" if n > 0 else "No exits")
print(f"Total P&L: Rs {total_pnl:+,.0f}")
if wins:
    avg_win = sum(e.get('pnl', 0) for e in wins) / len(wins)
    print(f"Avg Win: Rs {avg_win:+,.0f} | Largest: Rs {max(e.get('pnl',0) for e in wins):+,.0f}")
if losses:
    avg_loss = sum(e.get('pnl', 0) for e in losses) / len(losses)
    print(f"Avg Loss: Rs {avg_loss:+,.0f} | Largest: Rs {min(e.get('pnl',0) for e in losses):+,.0f}")
print()

# Exit reasons breakdown
print("=== EXIT REASONS ===")
reasons = {}
for e in exits:
    r = e.get('exit_reason', 'UNKNOWN')
    if r not in reasons:
        reasons[r] = {'count': 0, 'pnl': 0, 'trades': []}
    reasons[r]['count'] += 1
    reasons[r]['pnl'] += e.get('pnl', 0)
    reasons[r]['trades'].append(e)
for r, v in sorted(reasons.items(), key=lambda x: x[1]['pnl']):
    print(f"  {r}: {v['count']} trades, P&L: Rs {v['pnl']:+,.0f}")
print()

# Setup type breakdown
print("=== SETUP TYPES ===")
setups = {}
for e in exits:
    s = e.get('setup', 'UNKNOWN')
    if s not in setups:
        setups[s] = {'count': 0, 'pnl': 0, 'wins': 0}
    setups[s]['count'] += 1
    setups[s]['pnl'] += e.get('pnl', 0)
    if e.get('pnl', 0) > 0:
        setups[s]['wins'] += 1
for s, v in sorted(setups.items(), key=lambda x: x[1]['pnl']):
    wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
    print(f"  {s}: {v['count']} trades, P&L: Rs {v['pnl']:+,.0f}, WR: {wr:.0f}%")
print()

# Direction breakdown
print("=== DIRECTION ANALYSIS ===")
dirs = {}
for e in exits:
    d = e.get('direction', 'UNKNOWN')
    if d not in dirs:
        dirs[d] = {'count': 0, 'pnl': 0, 'wins': 0}
    dirs[d]['count'] += 1
    dirs[d]['pnl'] += e.get('pnl', 0)
    if e.get('pnl', 0) > 0:
        dirs[d]['wins'] += 1
for d, v in sorted(dirs.items(), key=lambda x: x[1]['pnl']):
    wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
    print(f"  {d}: {v['count']} trades, P&L: Rs {v['pnl']:+,.0f}, WR: {wr:.0f}%")
print()

# Sector breakdown
print("=== SECTOR ANALYSIS ===")
sectors = {}
for e in exits:
    sec = e.get('sector', 'UNKNOWN')
    if sec not in sectors:
        sectors[sec] = {'count': 0, 'pnl': 0, 'wins': 0}
    sectors[sec]['count'] += 1
    sectors[sec]['pnl'] += e.get('pnl', 0)
    if e.get('pnl', 0) > 0:
        sectors[sec]['wins'] += 1
for sec, v in sorted(sectors.items(), key=lambda x: x[1]['pnl']):
    wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
    print(f"  {sec}: {v['count']} trades, P&L: Rs {v['pnl']:+,.0f}, WR: {wr:.0f}%")
print()

# Chronological trade log (all exits)
print("=== ALL TRADES (Chronological) ===")
exits_sorted = sorted(exits, key=lambda e: e.get('timestamp', ''))
for i, e in enumerate(exits_sorted, 1):
    sym = e.get('underlying', e.get('symbol', '?'))
    pnl = e.get('pnl', 0)
    reason = e.get('exit_reason', '?')
    direction = e.get('direction', '?')
    setup = e.get('setup', '?')
    entry_t = e.get('entry_time', '?')
    exit_t = e.get('timestamp', '?')
    hold = e.get('hold_duration_min', '?')
    entry_px = e.get('entry_price', 0)
    exit_px = e.get('exit_price', 0)
    dr = e.get('ml_data', {}).get('dr_score', e.get('dr_score', '?'))
    smart = e.get('ml_data', {}).get('smart_score', e.get('smart_score', '?'))
    marker = 'WIN' if pnl > 0 else 'LOSS'
    print(f"  {i:2d}. [{marker:4s}] {sym:14s} {direction:4s} | P&L: Rs {pnl:+8,.0f} | {reason:12s} | setup={setup} | hold={hold}min | entry={entry_px} exit={exit_px} | dr={dr} smart={smart}")
print()

# Consecutive loss streaks
print("=== LOSS STREAKS ===")
streak = 0
max_streak = 0
streak_pnl = 0
streaks = []
for e in exits_sorted:
    if e.get('pnl', 0) <= 0:
        streak += 1
        streak_pnl += e.get('pnl', 0)
    else:
        if streak > 0:
            streaks.append((streak, streak_pnl))
        streak = 0
        streak_pnl = 0
if streak > 0:
    streaks.append((streak, streak_pnl))
for i, (s, p) in enumerate(streaks):
    print(f"  Streak {i+1}: {s} consecutive losses, P&L: Rs {p:+,.0f}")
print()

# Hold duration analysis
print("=== HOLD DURATION vs P&L ===")
dur_buckets = {'<5min': {'count': 0, 'pnl': 0, 'wins': 0}, '5-15min': {'count': 0, 'pnl': 0, 'wins': 0}, 
               '15-30min': {'count': 0, 'pnl': 0, 'wins': 0}, '30min+': {'count': 0, 'pnl': 0, 'wins': 0}}
for e in exits:
    dur = e.get('hold_duration_min', 0)
    if isinstance(dur, str):
        try:
            dur = float(dur)
        except:
            dur = 0
    if dur < 5:
        b = '<5min'
    elif dur < 15:
        b = '5-15min'
    elif dur < 30:
        b = '15-30min'
    else:
        b = '30min+'
    dur_buckets[b]['count'] += 1
    dur_buckets[b]['pnl'] += e.get('pnl', 0)
    if e.get('pnl', 0) > 0:
        dur_buckets[b]['wins'] += 1
for b, v in dur_buckets.items():
    wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
    print(f"  {b:10s}: {v['count']} trades, P&L: Rs {v['pnl']:+,.0f}, WR: {wr:.0f}%")

# ML direction conflict check
print()
print("=== ML DIRECTION CONFLICTS (scored_direction vs option type) ===")
conflicts = []
for e in exits:
    ml = e.get('ml_data', {})
    scored_dir = ml.get('scored_direction', e.get('direction', ''))
    option = e.get('option_type', e.get('symbol', ''))
    if scored_dir == 'SELL' and 'CE' in str(option):
        conflicts.append(e)
    elif scored_dir == 'BUY' and 'PE' in str(option):
        conflicts.append(e)
for c in conflicts:
    sym = c.get('underlying', '?')
    pnl = c.get('pnl', 0)
    scored_dir = c.get('ml_data', {}).get('scored_direction', c.get('direction', ''))
    opt = c.get('option_type', c.get('symbol', ''))
    print(f"  CONFLICT: {sym} scored={scored_dir} but option={opt} | P&L: Rs {pnl:+,.0f}")
if not conflicts:
    print("  None found")
