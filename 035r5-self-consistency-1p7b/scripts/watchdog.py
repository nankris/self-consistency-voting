"""Independent local hard-budget stop, including during setup/controller failure."""
import time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=json.loads((R/'state.json').read_text())
while time.time()<s['deadline']:
 time.sleep(min(10,max(.1,s['deadline']-time.time())))
 s=json.loads((R/'state.json').read_text())
 if s.get('status')=='stopped':raise SystemExit
p=subprocess.run(['colab','stop','-s',s['session']],capture_output=True,text=True,timeout=60)
(R/'results'/'watchdog_stop.txt').write_text(p.stdout+p.stderr)
with (R/'attempts.jsonl').open('a') as f:f.write(json.dumps({'event':'watchdog_stop','time':time.time(),'returncode':p.returncode})+'\n')
