# One Grok call per GSM8K question (lean config: plain chat, web search off, no subagents).
import json, os, sys, re, tempfile, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(__file__))
from grok_log import run_grok, QuotaHit
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = json.load(open(f"{D}/results/questions.json")); CSV = f"{D}/results/grok_calls.csv"
LEAN = ["--system-prompt-override", "You are a careful math tutor.", "--disable-web-search", "--no-subagents", "--tools", "read_file"]
def extract(t):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"(-?[\d,]*\.?\d+)", t)
    return m[-1].replace(",", "").rstrip(".") if m else None
def job(q):
    p = q["question"] + "\n\nSolve it step by step briefly. End with a final line 'Answer: <number>'."
    r = run_grok(p, f"q{q['idx']}", CSV, "035-grok", 1, cwd=tempfile.mkdtemp(), extra=LEAN)
    pred = extract(r["text"])
    ok = pred is not None and abs(float(pred) - float(q["answer"])) < 1e-6
    return dict(idx=q["idx"], gold=q["answer"], pred=pred, correct=ok, cost=r["cost_usd"], latency=r["latency_s"], text=r["text"])
out = []
with cf.ThreadPoolExecutor(5) as ex:
    for f in cf.as_completed([ex.submit(job, q) for q in Q]):
        try: o = f.result(); out.append(o); print(o["idx"], o["correct"], o["cost"], flush=True)
        except QuotaHit as e: print("QUOTA HIT", e); break
json.dump(sorted(out, key=lambda o: o["idx"]), open(f"{D}/results/grok_answers.json", "w"), indent=1)
print("acc", sum(o["correct"] for o in out), "/", len(out))
