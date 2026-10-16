"""Run on Colab after setup_remote.py; resumable per 10-question chunk."""
import os,json,time,random,subprocess,importlib.metadata
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'

def main():
 from datasets import load_dataset
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 import torch
 root='/content/035r';os.makedirs(root,exist_ok=True)
 data=load_dataset('openai/gsm8k','main',split='test')
 ids=list(range(len(data)));random.Random(3501).shuffle(ids)
 n=int(os.environ.get('N_QUESTIONS','200'))
 questions=[dict(index=i,**data[i]) for i in ids[:n]]
 json.dump(questions,open(root+'/questions.json','w'))
 tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
 metadata=dict(start=time.time(),gpu=torch.cuda.get_device_name(),model='Qwen/Qwen3-4B',model_revision=tok.init_kwargs.get('_commit_hash'),n_questions=n,versions={p:importlib.metadata.version(p) for p in ['vllm','transformers','torch','datasets']})
 json.dump(metadata,open(root+'/metadata.json','w'))
 llm=LLM(model='Qwen/Qwen3-4B',dtype='bfloat16',max_model_len=9216,gpu_memory_utilization=.90,max_num_seqs=64,seed=3503)
 path=root+'/samples.jsonl';done=set()
 if os.path.exists(path):
  for line in open(path):
   r=json.loads(line);done.add((r['mode'],r['index']))
 for thinking in [False,True]:
  mode='on' if thinking else 'off'
  for start in range(0,n,10):
   batch=[q for q in questions[start:start+10] if (mode,q['index']) not in done]
   if not batch:continue
   prompts=[tok.apply_chat_template([{'role':'user','content':q['question']+"\nSolve it step by step. Finish with a final line 'Answer: <number>'."}],tokenize=False,add_generation_prompt=True,enable_thinking=thinking) for q in batch]
   params=SamplingParams(n=10,skip_special_tokens=False,temperature=.6 if thinking else .7,top_p=.95 if thinking else .8,top_k=20,max_tokens=8192 if thinking else 2048,seed=3503+start+10000*thinking)
   t=time.time();outs=llm.generate(prompts,params,use_tqdm=False)
   with open(path,'a') as f:
    for q,out in zip(batch,outs):
     f.write(json.dumps(dict(index=q['index'],mode=mode,gold=q['answer'].split('####')[-1].strip(),samples=[dict(text=o.text,tokens=len(o.token_ids),finish_reason=o.finish_reason) for o in out.outputs],batch_seconds=time.time()-t))+'\n');f.flush()
   print(mode,start+len(batch),'/',n,flush=True)
 metadata['end']=time.time();json.dump(metadata,open(root+'/metadata.json','w'))
if __name__=='__main__':main()
