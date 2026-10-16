import subprocess,os,sys
from pathlib import Path
r=Path('/content/r5q035')
if (r/'error.txt').exists():(r/'error.txt').rename(r/'initial_launch_error.txt')
f=(r/'infer.log').open('a')
p=subprocess.Popen([sys.executable,'-u',str(r/'infer.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(r/'pid').write_text(str(p.pid));print('relaunch',p.pid)
