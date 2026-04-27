import json
d = json.load(open('/tmp/_status.json'))
ps = d.get('positions', [])
print(len(ps), 'positions')
for p in ps:
    print(p.get('symbol'), '| setup=', p.get('setup'), '| is_option=', p.get('is_option'), '| opt=', p.get('option_type'), '| K=', p.get('strike'), '| qty=', p.get('quantity'))
