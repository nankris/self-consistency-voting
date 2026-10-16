# Take the first 40 questions of the public GSM8K test split (openai/gsm8k, 'main').
import json, re, os
from datasets import load_dataset
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ds = load_dataset("openai/gsm8k", "main", split="test")
out = []
for i in range(40):
    ex = ds[i]; ans = ex["answer"].split("####")[-1].strip().replace(",", "")
    out.append({"idx": i, "question": ex["question"], "answer": ans})
json.dump(out, open(os.path.join(D, "results", "questions.json"), "w"), indent=1)
print(len(out), out[0])
