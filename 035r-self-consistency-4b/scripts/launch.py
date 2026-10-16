import subprocess,os
# infer.py is uploaded separately; detach so progress can be monitored.
p=subprocess.Popen(['python','/content/infer.py'],stdout=open('/content/infer.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('PID',p.pid)
