"""Single fixed paired bootstrap analysis; no adaptive scoring."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parser import canon,parse_answer,parse_strict,vote
R=Path(__file__).resolve().parents[1];O=R/'results'
rows=[]
for f in sorted(O.glob('b*.jsonl')): rows.extend(json.loads(l) for l in f.read_text().splitlines())
assert rows and [r['pos'] for r in rows]==list(range(len(rows)))
assert all(len(r['samples'])==5 and r['temperature']==.7 and r['top_p']==.95 and r['max_tokens']==512 and r['thinking']==False for r in rows)
n=len(rows)
assert all(r['seed']==35000+r['pos'] for r in rows)
def score(parser):
 g=np.array([parser(r['greedy']['text'])==canon(r['gold']) for r in rows],dtype=int)
 v=np.array([vote([parser(s['text']) for s in r['samples']],5)==canon(r['gold']) for r in rows],dtype=int)
 return g,v
g,v=score(parse_answer);d=v-g
rng=np.random.default_rng(35)
indices=rng.integers(0,n,size=(20000,n))
means=d[indices].mean(axis=1)
gci=np.quantile(g[indices].mean(axis=1),[.025,.975]);vci=np.quantile(v[indices].mean(axis=1),[.025,.975])
lo,hi=np.quantile(means,[.025,.975]);l90,u90=np.quantile(means,[.05,.95])
verdict='PROVEN' if lo>0 else 'DISPROVEN' if hi<0 or (-.02<l90 and u90<.02) else 'STILL INCONCLUSIVE'
reason='superiority' if lo>0 else 'inferiority' if hi<0 else 'equivalent within ±2 points' if (-.02<l90 and u90<.02) else 'fixed evidence budget exhausted; neither superiority, inferiority nor equivalence established'
b=int(((v==1)&(g==0)).sum());c=int(((v==0)&(g==1)).sum())
strict=score(parse_strict)
s={'n':n,'greedy_correct':int(g.sum()),'vote_correct':int(v.sum()),'greedy_accuracy':float(g.mean()),'vote_accuracy':float(v.mean()),'delta':float(d.mean()),'ci95':[float(lo),float(hi)],'ci90':[float(l90),float(u90)],'vote_only':b,'greedy_only':c,'mcnemar_exact_p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.,'verdict':verdict,'reason':reason,'strict_greedy_correct':int(strict[0].sum()),'strict_vote_correct':int(strict[1].sum()),'parse_missing_greedy':sum(parse_answer(r['greedy']['text']) is None for r in rows),'parse_missing_samples':sum(parse_answer(x['text']) is None for r in rows for x in r['samples']),'truncated_greedy':sum(r['greedy']['finish_reason']=='length' for r in rows),'truncated_samples':sum(x['finish_reason']=='length' for r in rows for x in r['samples'])}
s.update(greedy_ci95=gci.tolist(),vote_ci95=vci.tolist())
(O/'summary.json').write_text(json.dumps(s,indent=2))
with (O/'scored.jsonl').open('w') as f:
 for r,a,z in zip(rows,g,v):f.write(json.dumps(dict(pos=r['pos'],gold=canon(r['gold']),greedy=parse_answer(r['greedy']['text']),samples=[parse_answer(x['text']) for x in r['samples']],vote=vote([parse_answer(x['text']) for x in r['samples']],5),greedy_correct=int(a),vote_correct=int(z)))+'\n')
fig,ax=plt.subplots(1,2,figsize=(9,4))
ax[0].bar(['Greedy','Vote of 5'],100*np.array([g.mean(),v.mean()]),color=['#426b9b','#e99a35']);ax[0].set_ylim(0,100);ax[0].set_ylabel('Locked-parser accuracy (%)')
for i,a in enumerate([g,v]):ax[0].text(i,100*a.mean()+1,f'{a.sum()}/{n}',ha='center')
ax[1].axhspan(-2,2,color='green',alpha=.12,label='±2-point equivalence margin');ax[1].axhline(0,color='black',lw=1)
ax[1].errorbar([0],[100*d.mean()],yerr=[[100*(d.mean()-lo)],[100*(hi-d.mean())]],fmt='o',capsize=6,label='Paired 95% CI')
ax[1].plot([0,0],[100*l90,100*u90],lw=5,alpha=.4,label='90% CI');ax[1].set_xticks([0],['Vote minus greedy']);ax[1].set_ylabel('Accuracy difference (points)');ax[1].legend(fontsize=8)
fig.suptitle(f'Qwen3-1.7B • thinking off • GSM8K first {n}\n{verdict}');fig.text(.5,.01,'Formatting caveat: >55% missing parses. Post-hoc literal NUMBER repair: +8.75 points [5.25, 12.25].',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(R/'chart.png',dpi=160)
state=json.loads((R/'state.json').read_text());cu=state.get('cu_upper')
text=f'''# Self-consistency, round 5\n\n**{verdict} — {reason}.**\n\n| Method | Correct | Accuracy | 95% bootstrap CI |\n|---|---:|---:|---:|\n| Greedy | {g.sum()}/{n} | {100*g.mean():.2f}% | [{100*gci[0]:.2f}, {100*gci[1]:.2f}]% |\n| Vote of five | {v.sum()}/{n} | {100*v.mean():.2f}% | [{100*vci[0]:.2f}, {100*vci[1]:.2f}]% |\n\nPrimary paired difference: **{100*d.mean():+.2f} percentage points**, percentile bootstrap 95% CI **[{100*lo:.2f}, {100*hi:.2f}]**, 90% CI [{100*l90:.2f}, {100*u90:.2f}]. 20,000 question-paired resamples, seed 35. Vote-only correct: {b}; greedy-only correct: {c}. Supplementary exact McNemar p={s['mcnemar_exact_p']:.6g}; this does not replace the preregistered decision.\n\nAll {n} questions use first-in-file order, one greedy and five independent-temperature samples, with thinking disabled, temperature .7, top_p .95, max_tokens 512. Voting uses unchanged round-3 parser/plurality tie policy. Parse failures: greedy {s['parse_missing_greedy']}/{n}, stochastic {s['parse_missing_samples']}/{5*n}. Token-limit finishes: greedy {s['truncated_greedy']}, stochastic {s['truncated_samples']}. Strict label-only sensitivity: greedy {s['strict_greedy_correct']}, vote {s['strict_vote_correct']} correct. The 512-token cap and parsing define the measured task.\n\nGPU: NVIDIA A100-SXM4-40GB, session r5q035, stopped. Conservative CU upper bound: {cu:.3f} (entire account observed rate × session elapsed; another session ran concurrently, so this overcounts this experiment). See state.json, attempts.jsonl, results/usage_*.txt and stop.txt for audit. Results were copied locally and acknowledged every 50 questions before inference continued. No outcome-dependent stopping or additional model calls.\n\nThis estimates performance on a fixed public GSM8K subset, not a random population sample; question bootstrap describes variation under question resampling. Training contamination, correlated model errors and a modest fixed sample limit generalization. Margin and n were fixed by Hark. A non-significant result would not by itself establish equivalence.\n\nReproduce analysis: `python scripts/analyze.py` with numpy, scipy, matplotlib installed. Raw generations: results/b0000.jsonl through b0350.jsonl; question text and reference answers: results/questions.json. Original and earlier rerun folders were read only.\n'''
if (O/'format_diagnostic.json').exists():
 diag=json.loads((O/'format_diagnostic.json').read_text())
 text+=f'''\n## Formatting artifact and post-hoc diagnostic\n\nThe prompt's literal `Answer: NUMBER` wording was a mistake: 208 greedy and 1,037 stochastic outputs copied the placeholder before their numeric answer. The inherited parser requires the number immediately after the label, so it rejects these. Primary parse failures are 55.75% (greedy) and 55.80% (stochastic). Voting can recover a parsable answer from multiple attempts, so **the primary +15.5-point gain mixes formatting reliability with mathematical correctness**. The preregistered parser and verdict were not changed.\n\nA post-hoc, gold-independent diagnostic removes only literal `NUMBER` after an Answer label (allowing Markdown asterisks), then applies the original parser and voting unchanged. It scores greedy **{diag['greedy_correct']}/400 (68.50%)** and vote **{diag['vote_correct']}/400 (77.25%)**, difference **+8.75 points**, paired bootstrap 95% CI **[+5.25, +12.25]**, 90% CI [+5.75, +11.75]. This is exploratory, not a second confirmatory verdict. It supports a positive voting effect after that narrow formatting repair, but the primary magnitude should not be presented as a pure reasoning gain. No new model calls or altered raw outputs were used.\n\nEstimated experiment consumption is **{state.get('cu_estimate',0):.3f} CU**, using the observed rate drop from 10.60 to 5.30 CU/hour when only r5q035 stopped, over {(state['stop']-state['start'])/60:.2f} allocation minutes. The **{cu:.3f}-CU upper bound** charges the whole observed shared-account rate. The account balance moved 171.05 to 169.06 CU; that includes concurrent work and billing update lag, so it is not experiment-specific billing.\n'''
(R/'writeup.md').write_text(text)
print(json.dumps(s,indent=2))
