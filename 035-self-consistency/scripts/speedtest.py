import time, torch, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(2)
for name, dt in [("/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work/Qwen3-0.6B", torch.float32), ("/workspace/projects/4e0e7b68-80db-40d3-a6d5-b21cf99d82b6/work/Qwen3-1.7B", torch.bfloat16)]:
    tok = AutoTokenizer.from_pretrained(name); t0=time.time()
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dt); m.eval(); print(name, "load", time.time()-t0, flush=True)
    msgs=[{"role":"user","content":"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Answer step by step, end with 'Answer: <number>'."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        t0=time.time(); o=m.generate(**ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.8, num_return_sequences=5)
    dt_=time.time()-t0; n=(o.shape[1]-ids["input_ids"].shape[1])*5
    print(name, f"batch5 {n} tok in {dt_:.1f}s = {n/dt_:.1f} tok/s", flush=True)
    del m
