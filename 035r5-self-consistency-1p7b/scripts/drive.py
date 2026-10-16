"""Continue an owned session and mirror each batch before acknowledging it.
Run after source /workspace/tools/setup.sh. State survives sandbox reboot.
"""
import json,time,subprocess,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'results'; S=R/'scripts'
state=json.loads((R/'state.json').read_text()); session=state['session']
def ledger(event,**kw):
 with (R/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),event=event,**kw))+'\n');f.flush()
def run(args,timeout=120,check=True):
 try:
  p=subprocess.run(['/workspace/tools/uv-tools/google-colab-cli/bin/python',str(S/'colab_long.py'),*args],text=True,capture_output=True,timeout=timeout)
 except subprocess.TimeoutExpired as e:
  ledger('control_timeout',args=args)
  if check:raise
  return subprocess.CompletedProcess(args,124,stdout='',stderr='control timeout; retry')
 with (O/'driver.log').open('a') as f:f.write(str(args)+'\n'+p.stdout+p.stderr+'\n')
 if check and p.returncode:raise RuntimeError((p.stdout+p.stderr)[-2000:])
 return p

def download(name,required=True):
 tmp=O/(name+'.partial'); dest=O/name
 p=run(['download','-s',session,'/content/r5q035/'+name,str(tmp)],check=required)
 if p.returncode==0:
  if name.endswith('.jsonl'):
   rows=[json.loads(l) for l in tmp.read_text().splitlines()];assert len(rows)==50
  tmp.replace(dest);return True
 return False

def execute(code):
 p=S/'control.py';p.write_text(code)
 return run(['exec','-s',session,'-f',str(p),'--timeout','25'])
def main():
 if state.get('status')=='stopped':
  print('Session already stopped; analyze saved results.');return
 assert state['owned']
 ledger('driver_resumed')
 try:
  for start in range(0,400,50):
   dest=O/f'b{start:04d}.jsonl'
   while not dest.exists():
    if time.time()>state['deadline']:raise TimeoutError('conservative 5 CU budget deadline')
    p=run(['exec','-s',session,'-f',str(S/'status.py'),'--timeout','25'],check=False)
    (O/'status_latest.txt').write_text(p.stdout+p.stderr)
    if 'error.txt ' in p.stdout:raise RuntimeError('remote inference failed; see status_latest.txt')
    if f'b{start:04d}.jsonl' in p.stdout:
     if download(f'b{start:04d}.jsonl',required=False):break
    time.sleep(12)
   ledger('batch_mirrored',start=start,rows=50)
   for name in ['questions.json','metadata.json','progress.json','prompt.txt']:
    if not (O/name).exists() or name=='progress.json': download(name,required=False)
   execute(f"from pathlib import Path\nPath('/content/r5q035/ack{start:04d}').touch()\nprint('ack {start}')\n")
   print('CHECKPOINT',start+50,flush=True)
  ledger('all_400_saved')
 finally:
  for name in ['infer.log','setup.log','error.txt','metadata.json','progress.json']: 
   try:download(name,required=False)
   except Exception:pass
  stop=run(['stop','-s',session],timeout=60,check=False)
  (O/'stop.txt').write_text(stop.stdout+stop.stderr)
  state['stop']=time.time();state['status']='stopped' if stop.returncode==0 else 'stop_failed'
  state['cu_upper']=(state['stop']-state['start'])*state['rate_upper']/3600
  (R/'state.json').write_text(json.dumps(state,indent=2));ledger('stop',**state)
  p=run(['usage'],check=False);(O/'usage_after.txt').write_text(p.stdout+p.stderr)
if __name__=='__main__':main()
