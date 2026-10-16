import os,subprocess,importlib.util
from pathlib import Path
print('vllm importable',bool(importlib.util.find_spec('vllm')))
print(subprocess.check_output(['ps','-eo','pid,args']).decode()[-6000:])
p=Path('/content/r5q035/setup.log');print(p.read_text()[-1500:])
