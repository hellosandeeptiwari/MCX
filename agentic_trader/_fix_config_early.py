import re
p = "config.py"
txt = open(p).read()
txt = re.sub(r'"early_market_min_score":\s*\d+', '"early_market_min_score": 35', txt)
open(p, "w").write(txt)
print("Done: config.py early_market_min_score=35")
