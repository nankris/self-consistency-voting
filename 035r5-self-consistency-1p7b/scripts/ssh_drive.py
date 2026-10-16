"""Control-connection fallback: Colab SSH, same session, same inference process.
Copies each completed JSONL by cat before acknowledgement, with no data changes.
"""
import json,subprocess,time,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'results'
s=json.loads((R/'state.json').read_text());assert s['owned'] and s['status']=='allocated'
key=R/'.keys/colab'
base=['ssh','-o',f'ProxyCommand=colab ssh --proxy-mode -s r5q035 -i {key}','-i',str(key),'-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','ConnectTimeout=20','root@colab-runtime']
def ledger(event,**kw):
 with (R/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),event=event,**kw))+'\n')
def remote(cmd,check=True):
 p=subprocess.run(base+[cmd],capture_output=True,text=True,timeout=45)
 if check and p.returncode:raise RuntimeError(p.stderr[-1000:])
 return p
ledger('ssh_control_recovery',detail='same remote pid 5368; only notebook control endpoint bypassed using authorized Colab SSH')
try:
 for start in range(0,400,50):
  name=f'b{start:04d}.jsonl';dest=O/name
  while not dest.exists():
   if time.time()>s['deadline']-60:raise TimeoutError('budget deadline')
   p=remote('cat /content/r5q035/'+name,False)
   if p.returncode==0:
    rows=[json.loads(l) for l in p.stdout.splitlines()]
    if len(rows)==50 and [r['pos'] for r in rows]==list(range(start,start+50)):
     tmp=dest.with_suffix('.partial');tmp.write_text(p.stdout);tmp.replace(dest)
     ledger('batch_mirrored',start=start,rows=50,transport='colab SSH cat');break
   time.sleep(3)
  remote(f'touch /content/r5q035/ack{start:04d}')
  print('CHECKPOINT',start+50,flush=True)
 for name in ['metadata.json','questions.json','prompt.txt','progress.json','infer.log','setup.log','initial_launch_error.txt']:
  p=remote('cat /content/r5q035/'+name,False)
  if p.returncode==0:(O/name).write_text(p.stdout)
 ledger('all_400_saved',transport='colab SSH cat')
finally:
 p=subprocess.run(['colab','stop','-s','r5q035'],capture_output=True,text=True,timeout=60)
 (O/'stop.txt').write_text(p.stdout+p.stderr)
 s.update(stop=time.time(),status='stopped' if p.returncode==0 else 'stop_failed');s['cu_upper']=(s['stop']-s['start'])*s['rate_upper']/3600
 (R/'state.json').write_text(json.dumps(s,indent=2));ledger('stop',**s)
 p=subprocess.run(['colab','usage'],capture_output=True,text=True,timeout=45);(O/'usage_after.txt').write_text(p.stdout+p.stderr)
