import json

# Check order IDs from VOLUME_SURGE watcher log placements
vol_surge_oids = [
    'OPTION_PAPER_952700',   # DLF
    'OPTION_PAPER_888204',   # IREDA
    'OPTION_PAPER_821585',   # UNOMINDA
    'OPTION_PAPER_515242',   # RBLBANK
    'OPTION_PAPER_509920',   # LODHA
    'OPTION_PAPER_420731',   # ADANIGREEN
    'OPTION_PAPER_132600',   # KPITTECH
    'OPTION_PAPER_343605',   # CAMS
    'OPTION_PAPER_621429',   # BSE
    'OPTION_PAPER_945768',   # KALYANIKJIL
    'OPTION_PAPER_436475',   # TVSMOTOR (latest code)
    'OPTION_PAPER_907242',   # KFINTECH (latest code)
]

print("=== TRACKING VOLUME_SURGE ORDER IDs IN LEDGER ===")
for oid in vol_surge_oids:
    found = False
    for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
        t = json.loads(line)
        if t.get('order_id') == oid:
            found = True
            print(f"{oid}: event={t.get('event')} source={t.get('source','?')} sym={t.get('underlying','?')}")
            if t.get('event') == 'ENTRY':
                print(f"  rationale: {t.get('rationale','')[:120]}")
            if t.get('event') == 'EXIT':
                print(f"  pnl={t.get('pnl',0):+.0f} exit={t.get('exit_type','')}")
    if not found:
        print(f"{oid}: NOT FOUND IN LEDGER")

# Count ALL watcher entries by looking at rationale
print("\n=== ALL WATCHER ENTRIES (by rationale) ===")
for line in open('trade_ledger/trade_ledger_2026-03-17.jsonl'):
    t = json.loads(line)
    if t.get('event') == 'ENTRY' and 'WATCHER' in t.get('rationale', ''):
        rat_start = t.get('rationale', '')[:80]
        src = t.get('source', '?')
        sym = t.get('underlying', '').replace('NSE:', '')
        print(f"  {src:<30} {sym:<14} {rat_start}")

# Check if VOLUME_SURGE triggers got stuck in risk governor
print("\n=== CHECKING BOT_DEBUG.LOG FOR VOLUME_SURGE BLOCKS ===")
try:
    with open('bot_debug.log') as f:
        for line in f:
            if 'VOLUME_SURGE' in line or ('VOLUME' in line and 'block' in line.lower()):
                print(f"  {line.strip()[:180]}")
except:
    pass
