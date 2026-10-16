import subprocess,sys,os
os.makedirs('/content/r5q035',exist_ok=True)
f=open('/content/r5q035/setup.log','a')
p=subprocess.Popen([sys.executable,'-m','pip','install','-q','vllm','datasets'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('setup pid',p.pid)
