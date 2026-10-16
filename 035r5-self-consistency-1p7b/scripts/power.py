import json
from scipy.stats import norm
out=[]
for d,q in [(.07,.15),(.07,.20),(0,.05),(0,.02)]:
 se=((q-d*d)/400)**.5
 superiority=norm.cdf(d/se-1.96)+norm.cdf(-d/se-1.96)
 half=1.6448536269514722*se
 equivalence=max(0,float(norm.cdf((.02-half-d)/se)-norm.cdf((-.02+half-d)/se))) if half<.02 else 0
 out.append(dict(delta=d,discordance=q,se=se,superiority_power=float(superiority),equivalence_power=equivalence))
print(json.dumps(out,indent=2))
