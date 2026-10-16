import json, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eq = lambda p, g: p is not None and abs(float(p) - float(g)) < 1e-6
L = [json.loads(l) for l in open(f"{D}/results/local_samples.jsonl")]
G = json.load(open(f"{D}/results/grok_answers.json")); gc = pd.read_csv(f"{D}/results/grok_calls.csv")
n = len(L); ids = {r["idx"] for r in L}
vote_acc = np.mean([eq(r["vote"], r["gold"]) for r in L])
single_acc = np.mean([np.mean([eq(p, r["gold"]) for p in r["preds"]]) for r in L])   # expected accuracy of 1 sample
first_acc = np.mean([eq(r["preds"][0], r["gold"]) for r in L])
oracle = np.mean([any(eq(p, r["gold"]) for p in r["preds"]) for r in L])           # any of 5 correct
unanimous = np.mean([len(set(r["preds"])) == 1 for r in L])
g_sub = [g for g in G if g["idx"] in ids]
s = dict(questions=n, local_model="Qwen/Qwen3-0.6B (fp32, CPU 2 threads, non-thinking, T=0.7 top_p=0.8 top_k=20)",
         local_vote5_acc=round(vote_acc, 4), local_single_sample_acc=round(single_acc, 4), local_first_sample_acc=round(first_acc, 4),
         local_any_of_5_acc=round(oracle, 4), local_unanimous_share=round(unanimous, 4),
         local_wall_s_total=round(sum(r["wall_s"] for r in L), 1), local_wall_s_per_q=round(np.mean([r["wall_s"] for r in L]), 1),
         local_gen_tokens_per_q=round(np.mean([r["gen_tokens"] for r in L])),
         grok_acc=round(np.mean([g["correct"] for g in g_sub]), 4), grok_n=len(g_sub), grok_model=sorted(set(gc.model)),
         grok_cost_total_usd=round(gc.cost_usd.sum(), 4), grok_cost_per_q_usd=round(gc.cost_usd.mean(), 5),
         grok_latency_mean_s=round(gc.latency_s.mean(), 2), grok_wall_s_total_5_parallel=None,
         grok_in_tokens_mean=round((gc.input_tokens + gc.cache_read_input_tokens).mean()), grok_out_tokens_mean=round(gc.output_tokens.mean()))
ts = pd.to_datetime(gc.ts_ist); s["grok_wall_s_total_5_parallel"] = round((ts.max() - ts.min()).total_seconds() + gc.latency_s.iloc[0], 1)
s["local_wrong_vote"] = [dict(idx=r["idx"], gold=r["gold"], preds=r["preds"]) for r in L if not eq(r["vote"], r["gold"])]
json.dump(s, open(f"{D}/results/summary.json", "w"), indent=1); print(json.dumps(s, indent=1))
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
names = ["Qwen3-0.6B\n1 sample", "Qwen3-0.6B\nvote of 5", "Qwen3-0.6B\nany of 5\n(oracle)", "Grok CLI\n1 call"]
vals = [single_acc, vote_acc, oracle, s["grok_acc"]]; cols = ["#C4C4C4", "#DD8452", "#EED8C4", "#4C72B0"]
a.bar(names, [v * 100 for v in vals], color=cols)
for i, v in enumerate(vals): a.text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center")
a.set_ylim(0, 110); a.set_ylabel(f"accuracy on {n} GSM8K test questions (%)"); a.set_title("Accuracy")
b.bar(["Qwen3-0.6B\n5 samples, CPU", "Grok CLI\n1 call"], [s["local_wall_s_per_q"], s["grok_latency_mean_s"]], color=["#DD8452", "#4C72B0"])
b.set_ylabel("seconds per question"); b.set_title(f"Time per question\nGrok {s['grok_cost_per_q_usd']*100:.2f}¢ each (API-equiv); local $0 marginal")
for i, v in enumerate([s["local_wall_s_per_q"], s["grok_latency_mean_s"]]): b.text(i, v + 0.5, f"{v:.1f}s", ha="center")
fig.suptitle("Can a crowd of small models beat one big one?\n5 votes from Qwen3-0.6B on a 2-core CPU vs one Grok call (grok-4.7-build)")
plt.tight_layout(); plt.savefig(f"{D}/self_consistency.png", dpi=130)
