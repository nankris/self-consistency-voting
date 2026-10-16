"""Idempotent recovery entry point. Reuses only the one owned A100 session.
Run: source /workspace/tools/setup.sh; python scripts/resume.py
"""
import subprocess,json,time,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'scripts';O=R/'results'
s=json.loads((R/'state.json').read_text())
if len(list(O.glob('b[0-9][0-9][0-9][0-9].jsonl')))==8:
 if s['status']!='stopped':
  subprocess.run(['colab','stop','-s',s['session']],check=True)
  s.update(status='stopped',stop=time.time());s['cu_upper']=(s['stop']-s['start'])*s['rate_upper']/3600
  (R/'state.json').write_text(json.dumps(s,indent=2))
 subprocess.run([sys.executable,str(S/'analyze.py')],check=True);raise SystemExit
if s['status']=='stopped':
 raise SystemExit('One-session allocation ended. Preserve existing samples; do not restart the experiment. See ledger and writeup.')
assert s['owned'] and s['session']=='r5q035'
if time.time()>s['deadline']:
 subprocess.run(['colab','stop','-s',s['session']],check=True)
 raise SystemExit('Saved budget deadline expired; owned session stopped. Analyze preserved samples.')
if (R/'.keys/colab').exists():
 subprocess.run([sys.executable,str(S/'ssh_drive.py')],check=True)
 subprocess.run([sys.executable,str(S/'analyze.py')],check=True)
 raise SystemExit
def cli(args):subprocess.run(['/workspace/tools/uv-tools/google-colab-cli/bin/python',str(S/'colab_long.py'),*args],check=True,timeout=90)
cli(['status','-s',s['session']])
# Restore only files missing remotely. Never replace a live, partially written batch.
restore=S/'restore.py'
restore.write_text("from pathlib import Path\nPath('/content/r5q035').mkdir(exist_ok=True)\nprint('READY')\n")
cli(['exec','-s',s['session'],'-f',str(restore),'--timeout','25'])
cli(['upload','-s',s['session'],str(S/'infer.py'),'/content/r5q035/infer.py'])
for p in sorted(O.glob('b[0-9][0-9][0-9][0-9].jsonl')):
 assert len(p.read_text().splitlines())==50
 cli(['upload','-s',s['session'],str(p),'/content/r5q035/'+p.name])
subprocess.Popen([sys.executable,str(S/'watchdog.py')],stdout=(O/'watchdog.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
cli(['exec','-s',s['session'],'-f',str(S/'launch.py'),'--timeout','25'])
subprocess.run([sys.executable,str(S/'drive.py')],check=True)
if list(O.glob('b*.jsonl')):subprocess.run([sys.executable,str(S/'analyze.py')],check=True)
