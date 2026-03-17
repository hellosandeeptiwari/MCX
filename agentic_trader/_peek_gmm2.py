import json, os, glob
from collections import Counter

# Check all available ledger files
files = sorted(glob.glob('trade_ledger/trade_ledger_*.jsonl'))
print(f"Available ledger files: {files}")

for f in files:
    setups = Counter()
    total = 0
    for line in open(f):
        t = json.loads(line.strip())
        setups[t.get('setup_type', 'NONE')] += 1
        total += 1
    print(f"\n{f}: {total} trades")
    for k, v in setups.most_common():
        print(f"  {k}: {v}")

# Now find TEST_GMM trades specifically and show their rationale field
print("\n=== TEST_GMM TRADE SAMPLES ===")
count = 0
for f in files:
    for line in open(f):
        t = json.loads(line.strip())
        if t.get('setup_type') == 'TEST_GMM':
            count += 1
            sym = t.get('underlying', t.get('symbol', ''))
            pnl = t.get('pnl', 0)
            rat = t.get('rationale', '')[:120]
            entry = t.get('entry_time', '')[:16]
            # Check all keys at top level
            if count <= 3:
                print(f"\nTrade {count} keys: {list(t.keys())}")
                ml = t.get('ml_data', {})
                if ml:
                    print(f"  ml_data keys: {list(ml.keys())}")
                    gmm = ml.get('gmm_model', {})
                    if gmm:
                        print(f"  gmm_model: {gmm}")
            print(f"{sym:<20} PNL={pnl:>+8.0f} {entry} | {rat}")

if count == 0:
    # Maybe it's a different field name - search for "GMM" in rationale
    print("\n=== SEARCHING FOR GMM IN RATIONALE ===")
    for f in files:
        for line in open(f):
            t = json.loads(line.strip())
            rat = t.get('rationale', '')
            setup = t.get('setup_type', '')
            if 'GMM' in rat.upper() or 'GMM' in setup.upper():
                sym = t.get('underlying', t.get('symbol', ''))
                pnl = t.get('pnl', 0)
                print(f"  {setup:<20} {sym:<20} PNL={pnl:>+8.0f} | {rat[:100]}")

print(f"\nTotal TEST_GMM trades found: {count}")
