import sys
p = '/home/ubuntu/titan/agentic_trader/config.py'
c = open(p).read()
n = 0
old1 = '"start": "09:15",  # Market open'
new1 = '"start": "09:16",  # Scan starts 09:16 sharp'
if old1 in c:
    c = c.replace(old1, new1, 1)
    n += 1
    print(f"OK: TRADING_HOURS start -> 09:16")
else:
    print(f"SKIP: TRADING_HOURS start not found")

old2 = '"watcher_start": "09:20"'
new2 = '"watcher_start": "09:16"'
if old2 in c:
    c = c.replace(old2, new2, 1)
    n += 1
    print(f"OK: watcher_start -> 09:16")
else:
    print(f"SKIP: watcher_start not found")

open(p, 'w').write(c)
import py_compile
py_compile.compile(p, doraise=True)
print(f"{n} changes applied, compile OK")
