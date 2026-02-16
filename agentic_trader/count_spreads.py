import json

# Trade history (closed trades)
trades = json.load(open('trade_history.json'))
today_hist = [t for t in trades if '2026-02-16' in t.get('timestamp', '')]

# Active trades (still open)
data = json.load(open('active_trades.json'))
active = data.get('active_trades', []) if isinstance(data, dict) else data

print(f"=== TODAY Feb 16 ===")
print(f"Closed trades in history: {len(today_hist)}")
print(f"Active trades: {len(active)}")
print()

# Categorize ALL trades
all_trades = today_hist + active
naked = []
credit = []
debit = []
ic = []
cash = []

for t in all_trades:
    if t.get('is_credit_spread'):
        credit.append(t)
    elif t.get('is_debit_spread'):
        debit.append(t)
    elif t.get('is_iron_condor'):
        ic.append(t)
    elif t.get('is_option', False):
        naked.append(t)
    else:
        cash.append(t)

print(f"Naked Options: {len(naked)}")
print(f"Credit Spreads: {len(credit)}")
print(f"Debit Spreads: {len(debit)}")
print(f"Iron Condors: {len(ic)}")
print(f"Cash/Equity: {len(cash)}")
print(f"TOTAL: {len(all_trades)}")

print(f"\n=== SPREAD DETAILS ===")
for t in credit:
    sym = t.get('symbol', '?')
    status = t.get('status', '?')
    pnl = t.get('pnl', 0) or 0
    src = 'ACTIVE' if t in active else 'CLOSED'
    print(f"  CR | {sym[:55]} | {status} | pnl={pnl:+,.0f} | {src}")

for t in debit:
    sym = t.get('symbol', '?')
    status = t.get('status', '?')
    pnl = t.get('pnl', 0) or 0
    src = 'ACTIVE' if t in active else 'CLOSED'
    print(f"  DB | {sym[:55]} | {status} | pnl={pnl:+,.0f} | {src}")

for t in ic:
    sym = t.get('symbol', '?')
    status = t.get('status', '?')
    pnl = t.get('pnl', 0) or 0
    src = 'ACTIVE' if t in active else 'CLOSED'
    print(f"  IC | {sym[:55]} | {status} | pnl={pnl:+,.0f} | {src}")

# Check for duplicates (same spread in both active and history)
hist_syms = set(t.get('symbol', '') for t in today_hist if t.get('is_credit_spread') or t.get('is_debit_spread') or t.get('is_iron_condor'))
active_syms = set(t.get('symbol', '') for t in active if t.get('is_credit_spread') or t.get('is_debit_spread') or t.get('is_iron_condor'))
dupes = hist_syms & active_syms
if dupes:
    print(f"\n⚠️ DUPLICATES (in both active + history): {len(dupes)}")
    for d in dupes:
        print(f"  {d}")
