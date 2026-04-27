"""Remove temp DIAG lines from watcher_pipeline.py + syntax check both files"""

# Remove DIAG from watcher_pipeline.py
wp = '/home/ubuntu/titan/agentic_trader/watcher_pipeline.py'
with open(wp, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if '# TEMP DIAG: per-stock scoring input/output' in line:
        skip = True
        continue
    if skip and ('_diag_' in line or 'DIAG [' in line):
        continue
    skip = False
    new_lines.append(line)

with open(wp, 'w') as f:
    f.writelines(new_lines)
print(f"Removed DIAG lines from watcher_pipeline.py ({len(lines)} -> {len(new_lines)} lines)")

# Syntax check both files
for fpath in [wp, '/home/ubuntu/titan/agentic_trader/autonomous_trader.py']:
    try:
        compile(open(fpath).read(), fpath, 'exec')
        print(f"SYNTAX OK: {fpath.split('/')[-1]}")
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {fpath}: {e}")
        exit(1)

print("\nAll checks passed!")
