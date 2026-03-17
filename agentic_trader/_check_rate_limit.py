import re

log_path = '/home/ubuntu/titan/agentic_trader/bot_debug.log'
restart_line = 0
errors_after = []

with open(log_path, 'r', errors='replace') as f:
    for i, line in enumerate(f, 1):
        if 'AutonomousTrader created' in line:
            restart_line = i
            errors_after = []
        if restart_line and i > restart_line:
            if 'Too many requests' in line or 'Error checking NFO:' in line:
                errors_after.append(line.strip())

print(f"Last restart at line: {restart_line}")
print(f"Errors since restart: {len(errors_after)}")
for e in errors_after[:10]:
    print(f"  {e}")
if not errors_after:
    print("  NONE - fix is working!")
