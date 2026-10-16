from pathlib import Path
r=Path('/content/r5q035')
for n in ['setup.log','infer.log','error.txt','progress.json']:
 p=r/n
 if p.exists():print(n,p.read_text()[-3500:])
print('files',[p.name for p in r.glob('*')])
import subprocess
p=subprocess.run(['ps','-C','ninja,cc1plus,nvcc','-o','comm,pcpu,etime'],capture_output=True,text=True)
print('compiler processes',p.stdout[-1000:])
