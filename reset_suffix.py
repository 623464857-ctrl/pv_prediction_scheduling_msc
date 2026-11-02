import re
with open('experiments/common.py', encoding='utf-8') as f:
    c = f.read()
c = re.sub(r'^_CFG_SUFFIX = .*$', '_CFG_SUFFIX = ""  # run_4configs', c, flags=re.MULTILINE)
with open('experiments/common.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Reset _CFG_SUFFIX to empty')
