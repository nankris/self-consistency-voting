import psutil,subprocess,time,os
p=psutil.Process(4224)
assert p.cmdline()[-1]=='/content/infer.py', 'Unexpected process; do not touch'
children=p.children(recursive=True)
p.terminate()
for c in children:
 try:c.terminate()
 except psutil.NoSuchProcess:pass
_,alive=psutil.wait_procs([p]+children,timeout=10)
for c in alive:
 try:c.kill()
 except psutil.NoSuchProcess:pass
p=subprocess.Popen(['python','/content/infer_stream.py'],stdout=open('/content/infer.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('CONTINUOUS PID',p.pid)
