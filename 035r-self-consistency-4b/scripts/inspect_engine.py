import inspect
from vllm import LLM
from vllm.v1.engine.llm_engine import LLMEngine
print('ADD',inspect.signature(LLMEngine.add_request))
print('STEP',inspect.signature(LLMEngine.step))
print('PENDING',hasattr(LLMEngine,'has_unfinished_requests'))
print(inspect.getsource(LLM._run_engine)[:10000])
