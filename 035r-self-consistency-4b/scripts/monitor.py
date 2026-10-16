from pathlib import Path
import json,time,subprocess
p=Path('/content/infer.log')
print(p.read_text()[-180:] if p.exists() else 'No log')
p=Path('/content/035r/samples.jsonl')
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 print('COMPLETED',len(rows),'mode-question pairs;', {m:sum(r['mode']==m for r in rows) for m in ['off','on']})
 print('TOKENS',sum(s['tokens'] for r in rows for s in r['samples']))
 print('TRUNCATED',sum(s['finish_reason']=='length' for r in rows for s in r['samples']))
print(subprocess.getoutput('nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv'))
print('UTC',time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))

print('INFERENCE_PIDS',subprocess.getoutput("pgrep -f '^python3? /content/infer'"))
