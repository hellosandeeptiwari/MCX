import re

with open('/home/ubuntu/titan/logs/titan.log') as f:
    all_lines = f.readlines()

# Get the strength just before FIRING vs all evaluated
passed = []
for l in all_lines:
    if 'OI_WATCHER' in l and 'str=' in l and 'FIRING' in l:
        m = re.search(r'str=([0-9.]+)', l)
        if m:
            passed.append(float(m.group(1)))

print(f'FIRED: {len(passed)} trades')
for s in passed:
    print(f'  str={s:.3f}')
if passed:
    print(f'Average FIRED: {sum(passed)/len(passed):.3f}')

# Conviction distribution
convictions = {}
for l in all_lines:
    if 'OI_WATCHER' in l:
        m = re.search(r'(\d)/4 factors', l)
        if m:
            k = m.group(1) + '/4'
            convictions[k] = convictions.get(k, 0) + 1
        m2 = re.search(r'(\d)/4 conviction', l)
        if m2:
            k = m2.group(1) + '/4'
            convictions[k] = convictions.get(k, 0) + 1

print(f'\nConviction distribution: {convictions}')

# Strength by conviction level
print('\nStrength grouped by conviction:')
by_conv = {}
for l in all_lines:
    if 'OI_WATCHER' in l and 'str=' in l and 'factors' in l:
        ms = re.search(r'str=([0-9.]+)', l)
        mc = re.search(r'(\d)/4 factors', l)
        if ms and mc:
            conv = mc.group(1) + '/4'
            s = float(ms.group(1))
            by_conv.setdefault(conv, []).append(s)

for conv in sorted(by_conv.keys()):
    vals = by_conv[conv]
    print(f'  {conv}: n={len(vals)}, avg={sum(vals)/len(vals):.3f}, range=[{min(vals):.3f}, {max(vals):.3f}]')
