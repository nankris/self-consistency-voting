import pathlib,json,hashlib,subprocess,torch
root=pathlib.Path('/content/035r')
info=dict(gpu=subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv'),cuda=torch.version.cuda)
for label,pattern in [('model','models--Qwen--Qwen3-4B'),('dataset','datasets--openai--gsm8k')]:
 p=pathlib.Path.home()/'.cache/huggingface/hub'/pattern/'snapshots'
 info[label+'_snapshot_ids']=[x.name for x in p.iterdir()] if p.exists() else []
q=root/'questions.json'
if q.exists():info['questions_sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
(root/'provenance.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
