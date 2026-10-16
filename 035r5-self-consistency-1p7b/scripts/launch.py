import subprocess,os,sys
from pathlib import Path
r=Path('/content/r5q035')
pid=r/'pid'
alive=False
if pid.exists():
    try:
        n=int(pid.read_text());os.kill(n,0)
        alive=Path(f'/proc/{n}/stat').read_text().split()[2]!='Z'
    except ProcessLookupError:pass
if not alive:
    f=(r/'infer.log').open('a')
    p=subprocess.Popen([sys.executable,'-u',str(r/'infer.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    pid.write_text(str(p.pid))
    print('launched',p.pid)
else:print('already running',pid.read_text())
