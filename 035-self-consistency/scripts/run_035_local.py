# Qwen3-0.6B on CPU (2 threads, fp32, non-thinking mode): 5 sampled answers per GSM8K question, majority vote.
import json, os, re, time, torch, collections
from transformers import AutoTokenizer, AutoModelForCausalLM
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MODEL", "/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work/Qwen3-0.6B")  # local copy of the files of Qwen/Qwen3-0.6B from HF
N, QPB, MAXNEW = 5, 2, 384
torch.set_num_threads(2); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
if os.environ.get("DEL_AFTER_LOAD") and MODEL.startswith("/dev/shm/"):  # RAM-disk copy: free it once weights are in memory
    os.remove(os.path.join(MODEL, "model.safetensors"))
Q = json.load(open(f"{D}/results/questions.json")); OUT = f"{D}/results/local_samples.jsonl"
done = {json.loads(l)["idx"] for l in open(OUT)} if os.path.exists(OUT) else set()


def extract(t):
    a = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", t) or re.findall(r"(-?[\d,]*\.?\d+)", t)
    return a[-1].replace(",", "").rstrip(".") if a else None


todo = [q for q in Q if q["idx"] not in done]
for b in range(0, len(todo), QPB):
    qs = todo[b:b + QPB]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["question"] + "\n\nSolve it step by step briefly. End with a final line 'Answer: <number>'."}],
               add_generation_prompt=True, enable_thinking=False, tokenize=False) for q in qs]
    enc = tok(prompts, return_tensors="pt", padding=True)
    t0 = time.time()
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                       num_return_sequences=N, pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    gen = o[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    ntok = int((gen != tok.pad_token_id).sum())
    for k, q in enumerate(qs):
        s = texts[k * N:(k + 1) * N]
        preds = [extract(t) for t in s]
        vote = collections.Counter([p for p in preds if p is not None]).most_common(1)
        rec = dict(idx=q["idx"], gold=q["answer"], preds=preds, vote=vote[0][0] if vote else None,
                   wall_s=round(dt / len(qs), 2), gen_tokens=ntok // len(qs), texts=s)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(q["idx"], preds, "gold", q["answer"], f"{dt/len(qs):.1f}s", flush=True)
