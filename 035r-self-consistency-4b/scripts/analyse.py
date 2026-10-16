"""Question-cluster bootstrap; emits missing-data chart if allocation failed."""
import json,re,collections
from pathlib import Path
from decimal import Decimal,InvalidOperation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def parse_strict(text,mode):
 if mode=='on':
  if '</think>' not in text:return None
  text=text.rsplit('</think>',1)[1]
 matches=re.findall(r'Answer:\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)',text,re.I)
 if not matches:return None
 try:return str(Decimal(matches[-1].replace(',','')).normalize())
 except InvalidOperation:return None

def parse(text,mode):
 if mode=='on':
  if '</think>' not in text:return None
  text=text.rsplit('</think>',1)[1]
 number=r'([-+]?\d[\d,]*(?:\.\d+)?)'
 # Explicit numeric boxed answers are final-answer markers, never a last-number fallback.
 boxed=list(re.finditer(r'\\boxed\s*\{\s*\$?\s*'+number+r'\s*\}',text))
 labels=list(re.finditer(r'(?:final\s+)?answer\s*:',text,re.I))
 if labels:
  tail=text[labels[-1].end():]
  # Remove only formatting wrappers at the start of the final answer field.
  tail=re.sub(r'^(?:\s|\*|\$|`|\}|\\\[|\\\(|\\boxed\s*\{)+','',tail)
  match=re.match(number,tail)
  if match:
   try:return str(Decimal(match.group(1).replace(',','')).normalize())
   except InvalidOperation:pass
 if boxed:
  try:return str(Decimal(boxed[-1].group(1).replace(',','')).normalize())
  except InvalidOperation:pass
 return None

def vote(a,k):
 valid=[v for v in a[:k] if v is not None]
 if not valid:return None
 counts=collections.Counter(valid);best=max(counts.values())
 return next(v for v in valid if counts[v]==best)

def main(strict=False):
 parser=parse_strict if strict else parse
 path=ROOT/'results/samples.jsonl'
 if not path.exists():
  fig,ax=plt.subplots(figsize=(8,4));ax.axis('off')
  ax.text(.5,.65,'035r: accuracy curve unavailable',ha='center',fontsize=18)
  ax.text(.5,.4,'L4 allocation refused; 0 questions evaluated\nNo accuracy estimates or confidence intervals',ha='center',fontsize=12)
  fig.savefig(ROOT/'chart.png',dpi=160,bbox_inches='tight');return
 rows=[json.loads(x) for x in path.read_text().splitlines()];by={(r['mode'],r['index']):r for r in rows}
 ids=sorted({r['index'] for r in rows})
 expected=json.loads((ROOT/'results/metadata.json').read_text())['n_questions']
 assert len(rows)==2*expected and len(by)==len(rows), 'Missing or duplicate mode-question rows'
 assert len(ids)==expected, f'Expected {expected} questions, got {len(ids)}'
 assert all((m,i) in by for m in ['off','on'] for i in ids),'Incomplete paired data'
 rng=np.random.default_rng(3502);draw=rng.integers(0,len(ids),size=(20000,len(ids)))
 acc=np.zeros((len(ids),2,3));stats=[]
 for j,m in enumerate(['off','on']):
  failures=truncations=0
  for h,i in enumerate(ids):
   r=by[m,i];assert len(r['samples'])==10
   a=[parser(s['text'],m) for s in r['samples']];gold=str(Decimal(r['gold'].replace(',','')).normalize())
   failures+=sum(v is None for v in a);truncations+=sum(s['finish_reason']=='length' for s in r['samples'])
   for t,k in enumerate([1,5,10]):acc[h,j,t]=vote(a,k)==gold
  for t,k in enumerate([1,5,10]):
   values=acc[:,j,t];ci=np.quantile(values[draw].mean(1),[.025,.975])
   delta=values-acc[:,j,0];dci=np.quantile(delta[draw].mean(1),[.025,.975])
   token_mean=np.mean([sum(s['tokens'] for s in by[m,i]['samples'][:k]) for i in ids])
   stats.append(dict(mean_output_tokens_per_question=float(token_mean),mode=m,k=k,correct=int(values.sum()),n=len(ids),accuracy=values.mean(),ci=ci.tolist(),delta=delta.mean(),delta_ci=dci.tolist(),parse_failures=failures,truncations=truncations))
 delta=(acc[:,:,2]-acc[:,:,0]).mean(1);ci=np.quantile(delta[draw].mean(1),[.0125,.9875])
 verdict='PROVEN' if ci[0]>0 else 'DISPROVEN' if ci[1]<0 else 'STILL INCONCLUSIVE'
 result=dict(parser='strict' if strict else 'normalized explicit final answer',verdict=verdict,primary_delta=delta.mean(),primary_ci_97_5=ci.tolist(),stats=stats)
 (ROOT/('results/summary_strict.json' if strict else 'results/summary.json')).write_text(json.dumps(result,indent=2))
 fig,ax=plt.subplots(figsize=(8,5))
 for m in ['off','on']:
  s=[r for r in stats if r['mode']==m];y=np.array([r['accuracy'] for r in s]);cis=np.array([r['ci'] for r in s])
  ax.errorbar([1,5,10],100*y,yerr=100*np.array([y-cis[:,0],cis[:,1]-y]),marker='o',capsize=5,label='Thinking '+m)
 ax.set(xlabel='Number of samples (plurality vote)',ylabel='GSM8K accuracy (%)',xticks=[1,5,10],title=f'Qwen3-4B, {len(ids)} questions; 95% bootstrap CIs');ax.legend();fig.tight_layout();fig.savefig(ROOT/('chart_strict.png' if strict else 'chart.png'),dpi=160)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 import sys
 main(strict='--strict' in sys.argv)
