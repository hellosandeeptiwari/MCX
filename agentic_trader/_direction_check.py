import json
from datetime import datetime

h = json.load(open('trade_history.json'))

# Focus on option trades (where direction matters most)
opts = [t for t in h if t.get('is_option') and not t.get('is_credit_spread') and t.get('status') not in ['OPEN', None]]

print("=" * 80)
print("DIRECTIONAL ACCURACY ANALYSIS — NAKED OPTION TRADES")
print("=" * 80)

wrong_dir = 0
right_dir = 0

for t in opts:
    sym = t.get('symbol', '')[:40]
    side = t.get('side', '')
    direction = t.get('direction', '')
    opt_type = t.get('option_type', '')
    pnl = t.get('pnl', 0)
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    status = t.get('status', '')
    rationale = t.get('rationale', '')[:60]
    
    # Direction check: did price move in our favor?
    if side == 'BUY':
        moved_right = exit_p > entry
    else:
        moved_right = exit_p < entry
    
    pnl_pct = ((exit_p - entry) / entry * 100) if entry > 0 else 0
    if side == 'SELL':
        pnl_pct = -pnl_pct
    
    marker = "✅" if pnl > 0 else "❌"
    dir_marker = "→RIGHT" if moved_right else "→WRONG"
    
    if moved_right:
        right_dir += 1
    else:
        wrong_dir += 1
    
    print(f"{marker} {sym}")
    print(f"   {side} {opt_type} | Entry: {entry:.2f} → Exit: {exit_p:.2f} | {pnl_pct:+.1f}% | PnL: {pnl:+,.2f} | {status}")
    print(f"   Direction: {direction} | {dir_marker} | {rationale}")
    print()

total = right_dir + wrong_dir
print(f"\nDIRECTION ACCURACY: {right_dir}/{total} = {right_dir/max(total,1)*100:.1f}%")
print(f"Wrong direction: {wrong_dir}/{total} = {wrong_dir/max(total,1)*100:.1f}%")

# Now check: what if we FLIPPED every direction?
print("\n" + "=" * 80)
print("WHAT-IF: FLIPPED DIRECTION (CE→PE, PE→CE)")
print("=" * 80)
flip_wins = sum(1 for t in opts if t.get('pnl', 0) <= 0)  # Current losses would be wins
flip_losses = sum(1 for t in opts if t.get('pnl', 0) > 0)  # Current wins would be losses
print(f"If flipped: Wins={flip_wins} Losses={flip_losses} = {flip_wins/max(len(opts),1)*100:.1f}% win rate")
print(f"Current:    Wins={len(opts)-wrong_dir} Losses={wrong_dir} = {right_dir/max(total,1)*100:.1f}% win rate")

# Analyze which indicators led to wrong calls
print("\n" + "=" * 80)
print("WRONG DIRECTION TRADES — WHAT WENT WRONG")  
print("=" * 80)
for t in opts:
    entry = t.get('avg_price', 0)
    exit_p = t.get('exit_price', 0)
    side = t.get('side', '')
    moved_right = (exit_p > entry) if side == 'BUY' else (exit_p < entry)
    if not moved_right and t.get('pnl', 0) < -500:
        sym = t.get('symbol', '')[:40]
        rationale = t.get('rationale', '')
        print(f"❌ {sym} | PnL: {t.get('pnl',0):+,.2f}")
        print(f"   Rationale: {rationale[:120]}")
        print()

# Today's open positions - are they going right?
print("\n" + "=" * 80)
print("TODAY'S OPEN POSITIONS — DIRECTION CHECK")
print("=" * 80)
d = json.load(open('active_trades.json'))
for t in d.get('active_trades', []):
    if t.get('is_credit_spread'):
        continue
    sym = t.get('symbol', '')[:45]
    side = t.get('side', '')
    opt_type = t.get('option_type', '')
    direction = t.get('direction', '')
    entry = t.get('avg_price', 0)
    rationale = t.get('rationale', '')[:100]
    print(f"{sym}")
    print(f"   Agent direction: {direction} → Chose: {opt_type} ({side})")
    print(f"   Entry: {entry:.2f}")
    print(f"   Rationale: {rationale}")
    print()
