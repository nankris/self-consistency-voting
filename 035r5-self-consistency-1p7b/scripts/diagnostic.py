"""Post-hoc formatting diagnostic. Does not change primary parsing or verdict."""
import json,re,sys
from pathlib import Path
import numpy as np
from parser import parse_answer,canon,vote
R=Path(__file__).resolve().parents[1];O=R/'results'
rows=[json.loads(l) for p in sorted(O.glob('b*.jsonl')) for l in p.read_text().splitlines()]
def clean(t):return re.sub(r'(?i)(answer\s*:\s*\**\s*)NUMBER\s+',r'\1',t)
def parse(t):return parse_answer(clean(t))
g=np.array([parse(r['greedy']['text'])==canon(r['gold']) for r in rows],dtype=int)
v=np.array([vote([parse(x['text']) for x in r['samples']],5)==canon(r['gold']) for r in rows],dtype=int)
d=v-g;rng=np.random.default_rng(35);boot=d[rng.integers(0,len(rows),(20000,len(rows)))].mean(axis=1)
s={'label':'Post-hoc diagnostic only: remove literal NUMBER following Answer label, then unchanged parser','greedy_correct':int(g.sum()),'vote_correct':int(v.sum()),'delta':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'ci90':np.quantile(boot,[.05,.95]).tolist(),'literal_NUMBER_greedy':sum(clean(r['greedy']['text'])!=r['greedy']['text'] for r in rows),'literal_NUMBER_samples':sum(clean(x['text'])!=x['text'] for r in rows for x in r['samples'])}
(O/'format_diagnostic.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
