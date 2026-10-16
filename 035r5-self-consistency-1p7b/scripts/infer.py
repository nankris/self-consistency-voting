"""Resumable vLLM inference; never score results in the GPU process."""
import os,json,time,subprocess,sys,traceback
from pathlib import Path
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
R=Path('/content/r5q035')
def main():
    # Setup can still be replacing packages after notebook reconnects.
    while subprocess.run(['pgrep','-f','^/usr/bin/python3 -m pip install -q vllm datasets$'],capture_output=True).returncode==0: time.sleep(3)
    from datasets import load_dataset
    from vllm import LLM,SamplingParams
    import torch,importlib.metadata
    qs=load_dataset('openai/gsm8k','main',split='test')
    data=[dict(pos=i,question=qs[i]['question'],answer=qs[i]['answer'],gold=qs[i]['answer'].split('####')[-1].strip()) for i in range(400)]
    (R/'questions.json').write_text(json.dumps(data))
    llm=LLM(model='Qwen/Qwen3-1.7B',dtype='bfloat16',max_model_len=2048,max_num_seqs=256,gpu_memory_utilization=.85,seed=35,disable_log_stats=True)
    tok=llm.get_tokenizer()
    suffix='\nSolve briefly. End with exactly: Answer: NUMBER'
    metadata={'model':'Qwen/Qwen3-1.7B','gpu':torch.cuda.get_device_name(),'thinking':False,'suffix':suffix,'versions':{p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','datasets']},'start':time.time(),'model_commit':getattr(llm.llm_engine.model_config.hf_config,'_commit_hash',None)}
    (R/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for start in range(0,400,50):
        dest=R/f'b{start:04d}.jsonl'
        existing=[]
        if dest.exists():
            for line in dest.read_text().splitlines():
                try: existing.append(json.loads(line))
                except json.JSONDecodeError: break
            dest.write_text(''.join(json.dumps(r)+'\n' for r in existing))
        batch=data[start+len(existing):start+50]
        if batch:
            prompts=[tok.apply_chat_template([{'role':'user','content':q['question']+suffix}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for q in batch]
            if start==0: (R/'prompt.txt').write_text(prompts[0])
            t=time.time()
            stochastic=llm.generate(prompts,[SamplingParams(n=5,temperature=.7,top_p=.95,top_k=-1,max_tokens=512,seed=35000+q['pos']) for q in batch],use_tqdm=False)
            greedy=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=512),use_tqdm=False)
            def pack(x): return {'text':x.text,'tokens':len(x.token_ids),'finish_reason':x.finish_reason}
            with dest.open('a') as f:
                for q,s,g in zip(batch,stochastic,greedy):
                    row={**q,'samples':[pack(x) for x in s.outputs],'greedy':pack(g.outputs[0]),'temperature':.7,'top_p':.95,'max_tokens':512,'thinking':False,'seed':35000+q['pos'],'batch_seconds':time.time()-t}
                    f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print('BATCH',start+50,'seconds',time.time()-t,flush=True)
        (R/'progress.json').write_text(json.dumps({'completed':start+50,'waiting_ack':start,'time':time.time()}))
        while not (R/f'ack{start:04d}').exists(): time.sleep(1)
    (R/'complete').write_text(str(time.time()))
if __name__=='__main__':
    try: main()
    except Exception:
        (R/'error.txt').write_text(traceback.format_exc());raise
