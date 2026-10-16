# #035 Self-consistency: can 5 votes from a small local model beat one Grok call?

**Question.** Self-consistency means sampling a small model several times and taking the majority answer. On grade-school maths word problems, does that close the gap to one call to a frontier model? What does each cost in time and money?

**Answer, short.** No, not even close at this size. On the first 40 GSM8K test questions:
- **Qwen3-0.6B on CPU:** 50.5% with one sample, **57.5% with a majority vote of 5** (+7 points). It took **79 s per question** on a 2-core CPU, about 53 min for all 40.
- **One Grok call:** **40/40 (100%)**, about 9 s each, **0.38¢ per question** API-equivalent. That's $0.15 for the whole set.

## Setup
- **Data:** the first 40 items of the public GSM8K test split (`openai/gsm8k`, config `main`, rows 0–39). The gold answer is the number after `####`. File: `results/questions.json`.
- **Prompt (both models):** `<question>` followed by "Solve it step by step briefly. End with a final line 'Answer: <number>'." The answer is parsed from the last `Answer: <n>`, falling back to the last number in the reply, and compared numerically.
- **Local model:** `Qwen/Qwen3-0.6B`, fp32, PyTorch CPU (transformers 5.19) with `torch.set_num_threads(2)` on a 2-vCPU arm64 sandbox with 7.9 GB RAM. Non-thinking mode (`enable_thinking=False`), sampling with T=0.7, top_p=0.8, top_k=20 (Qwen's recommended non-thinking settings), `max_new_tokens=384`. 5 samples per question (`num_return_sequences=5`), 2 questions per batch, majority vote over parsed answers.
- **Why 0.6B and not 1.7B or 4B:** Qwen3-1.7B in bf16 didn't finish a 5×64-token test generation in about 6 minutes (bf16 matmuls are very slow on this CPU). In fp32 it needs ~6.8 GB, which doesn't fit beside everything else in 7.9 GB RAM. 0.6B was the largest Qwen3 that ran in reasonable time.
- **Grok:** Grok Build CLI 1.0.52 (SuperGrok), reported model `grok-4.7-build`. One call per question, 5 in parallel, with a lean config: `grok --no-auto-update -p "<prompt>" --output-format json --system-prompt-override "You are a careful math tutor." --disable-web-search --no-subagents --tools read_file`. Run on 2026-10-14, 19:39–19:41 IST.

```bash
source /workspace/tools/setup.sh
python scripts/prep.py              # -> results/questions.json
python scripts/run_035_grok.py      # -> results/grok_calls.csv, results/grok_answers.json
python scripts/run_035_local.py     # resumable; MODEL=<path to Qwen3-0.6B files> -> results/local_samples.jsonl
python scripts/analyse_035.py       # -> results/summary.json, self_consistency.png
```

## Results (40 questions)

| method | accuracy | time per question | cost |
|---|---|---|---|
| Qwen3-0.6B, 1 sample (mean over the 5 samples) | 50.5% | ~16 s (1/5 of the batch) | $0 marginal |
| Qwen3-0.6B, first sample only | 55.0% | | |
| **Qwen3-0.6B, majority vote of 5** | **57.5%** (23/40) | **78.9 s** | $0 marginal |
| Qwen3-0.6B, "any of 5 correct" (oracle upper bound) | 75.0% | | |
| **Grok CLI, 1 call** | **100%** (40/40) | **9.0 s** latency (all 40 done in 83 s with 5 in parallel) | **0.38¢** each, $0.152 total |

Other numbers:
- Local generation: ~785 tokens per question across the 5 samples, ~10 tok/s overall throughput. All 40 took 3,156 s wall (52.6 min).
- All 5 local samples agreed on 35% of questions.
- Grok: ~5.1K input tokens per call (mostly the CLI's fixed overhead), 459 output tokens, of which 371 were reasoning.
- The vote fixed some questions, but on 17/40 the majority was still wrong. In several of those (e.g. idx 3: 180, 180, 180, 180, 180 against gold 540) the model was confidently, consistently wrong.

Chart: `self_consistency.png`. Wrong-vote cases are listed in `results/summary.json`; every sample's full text is in `results/local_samples.jsonl`.

## Caveats
- 40 questions, so each question is 2.5 points. The +7-point gain from voting is 3 questions and well within noise at this size.
- GSM8K is old and public, and frontier models have almost certainly seen it in training. Grok's 100% probably overstates its accuracy on new problems.
- The plan said "a local 4B model". The CPU-only, 2-core, 7.9 GB sandbox couldn't run one in reasonable time, so this tests 0.6B. A 4B model on a laptop GPU would be far closer.
- Wall time is on a shared sandbox. The sandbox restarted twice mid-run (the script resumed from where it stopped), and other processes shared the CPU at times. Treat the local timings as rough.
- "$0 marginal" for local ignores hardware and electricity. Grok's cost is the CLI's API-equivalent `total_cost_usd`; on SuperGrok the actual limit is the usage window, not dollars.
- Thinking mode for Qwen3 was off. Thinking mode would likely raise accuracy but multiply tokens, and CPU time with it.
