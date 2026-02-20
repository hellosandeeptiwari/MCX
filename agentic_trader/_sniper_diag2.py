import json
data = json.load(open('trade_history.json'))
snipers = [t for t in data if t.get('setup_type') == 'GMM_SNIPER' or t.get('is_sniper')]
print(f"Total sniper trades: {len(snipers)}")
for s in snipers:
    print(json.dumps(s, indent=2, default=str)[:600])
    print("---")

# Now check: what DR scores are available today?
# Read from scan_decisions - get entries with extra.dr_score
decisions = json.load(open('scan_decisions.json'))
# Get all from today
today_d = [d for d in decisions if '2026-02-20' in d.get('timestamp', '')]
print(f"\nToday's decision entries: {len(today_d)}")

# Check which entries have dr_score at all
with_dr = [d for d in today_d if d.get('extra', {}).get('dr_score') is not None]
print(f"Entries with DR score: {len(with_dr)}")

# How many have low DR?
low_dr = [d for d in with_dr if d['extra']['dr_score'] <= 0.10]
print(f"Entries with DR <= 0.10: {len(low_dr)}")
for d in low_dr[:10]:
    sym = d.get('symbol', '?')
    dr = d['extra']['dr_score']
    outcome = d.get('outcome', '?')
    print(f"  {sym} dr={dr:.4f} outcome={outcome}")

# Most are SCORED_LOW — sniper uses ML results directly, not scan_decisions
# Let's check the unique outcomes
from collections import Counter
outcomes = Counter(d.get('outcome') for d in today_d)
print(f"\nOutcome distribution: {dict(outcomes)}")
