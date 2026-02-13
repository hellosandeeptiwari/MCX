import json
from datetime import datetime
d = json.load(open('trade_history.json'))
today = [t for t in d if '2026-02-10' in t.get('closed_at','')]

# Speed gate exits EXCLUDING BSE
sg = [t for t in today if t['result'] == 'OPTION_SPEED_GATE' and t['underlying'] != 'NSE:BSE']
sg_pnl = sum(t['pnl'] for t in sg)
print(f"SPEED GATE exits (excluding BSE): {len(sg)} trades, net = {sg_pnl:+,.0f}")
print()

for t in sg:
    e = t['avg_price']
    x = t['exit_price']
    sl = t['stop_loss']
    tgt = t['target']
    pct = (x - e) / e * 100
    sl_pct = (sl - e) / e * 100
    hold = (datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(t['timestamp'])).total_seconds() / 60
    still_alive = x > sl
    print(f"{t['symbol']:35s} held {hold:4.0f}min  PnL={t['pnl']:+8,.0f}")
    print(f"  Entry={e:7.2f}  Exit={x:7.2f} ({pct:+5.1f}%)  SL={sl:.2f} ({sl_pct:+.1f}%)")
    print(f"  Still above SL: {still_alive}  |  Exit was {abs(x-sl):.2f} from SL, {abs(tgt-x):.2f} from target")
    print()

# Non-speed-gate losses
print("=" * 60)
print("OTHER LOSSES (not speed gate, not BSE):")
print("=" * 60)
other_losses = [t for t in today if t['pnl'] < 0 and t['result'] != 'OPTION_SPEED_GATE' and t['underlying'] != 'NSE:BSE']
for t in other_losses:
    e = t['avg_price']
    x = t['exit_price']
    pct = (x - e) / e * 100
    hold = (datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(t['timestamp'])).total_seconds() / 60
    print(f"{t['symbol']:35s} {t['result']:20s} held {hold:4.0f}min  PnL={t['pnl']:+8,.0f}")
    print(f"  Entry={e:7.2f}  Exit={x:7.2f} ({pct:+5.1f}%)")
    print(f"  Rationale: {t.get('rationale','?')[:50]}")
    print()

print("=" * 60)
print("CLASSIFICATION OF ALL 14 LOSSES:")
print("=" * 60)
categories = {
    'BSE windfall (correct exit)': [],
    'Speed gate <3% move (premature)': [],
    'Speed gate 3-5% move (borderline)': [],
    'Speed gate >5% move (correct)': [],
    'TIME_STOP loss': [],
    'SESSION_CUTOFF loss': [],
}
for t in today:
    if t['pnl'] >= 0:
        continue
    pct = abs((t['exit_price'] - t['avg_price']) / t['avg_price'] * 100)
    if t['underlying'] == 'NSE:BSE':
        categories['BSE windfall (correct exit)'].append(t)
    elif t['result'] == 'OPTION_SPEED_GATE' and pct < 3:
        categories['Speed gate <3% move (premature)'].append(t)
    elif t['result'] == 'OPTION_SPEED_GATE' and pct < 5:
        categories['Speed gate 3-5% move (borderline)'].append(t)
    elif t['result'] == 'OPTION_SPEED_GATE':
        categories['Speed gate >5% move (correct)'].append(t)
    elif t['result'] == 'TIME_STOP':
        categories['TIME_STOP loss'].append(t)
    else:
        categories['SESSION_CUTOFF loss'].append(t)

for cat, trades in categories.items():
    if trades:
        cat_pnl = sum(t['pnl'] for t in trades)
        print(f"\n  {cat}: {len(trades)} trades, {cat_pnl:+,.0f}")
        for t in trades:
            pct = (t['exit_price'] - t['avg_price']) / t['avg_price'] * 100
            print(f"    {t['symbol']:35s} {pct:+5.1f}%  PnL={t['pnl']:+,.0f}")
