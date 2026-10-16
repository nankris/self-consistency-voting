# 035r3: Does a majority vote help Qwen3-4B on GSM8K?

Preregistered before GPU allocation and before any accuracy is computed. 2026-10-16.

## Hypothesis

On grade-school maths word problems, sampling Qwen3-4B ten times with thinking switched off and taking a majority vote raises accuracy relative to the first of those samples.

This is a paired comparison on one fixed set of public GSM8K test questions. It is not a comparison with a frontier model, and it is not a test of thinking mode.

## What changed, and why

Round 1 (`035-self-consistency`) ran Qwen3-0.6B on CPU, 40 questions, 5 samples. The write-up's +7 points compared the vote (57.5%) with the average of the five samples (50.5%). The paired contrast that matches a real choice — first sample versus the vote — was 55% versus 57.5%, one discordant question out of 40. Forty questions cannot separate that from noise.

Round 2 (`035r-self-consistency-4b`) specified Qwen3-4B, both thinking modes, 200 questions, and a 97.5% interval that could call the effect positive or negative but could not call a small effect "no practical gain." The confirmatory analysis was never completed. A later L4 session did write 200 unscored non-thinking generations. Those rows are not reused. They used a different question shuffle (seed 3501) and a batch-level seed, and they sat beside an iterated parser. This round draws a fresh 500. The only number taken from that file is a planning rate: mean 257 generated tokens, every sample finished on its own, about 11 minutes to generate 200 questions × 10 samples on an L4.

This round changes four things.

1. Thinking stays off. Round 2 split the budget across a long thinking cap and never reached a verdict. Thinking-off is the round-1 policy, it is what was requested for this round, and the token rate above fits 500 × 10 samples inside 5 CU.
2. The sample is 500 questions, one look, no expansion. Round 2's optional jump from 200 to 600 would have been a second look.
3. The single-sample arm is the first draw, not the average of the ten draws. Majority-of-5 and majority-of-10 are nested in those same ten draws.
4. A practical margin is locked in advance, so "voting does not help by enough to matter" can be a result. Round 2 could only reject the effect by proving it was negative.

No frontier-model arm. Round 1's Grok run scored 40/40 on public GSM8K, which does not tell us whether voting helps the local model.

## Locked decision rule

Primary estimand: the mean of D = (majority of 10 correct) − (sample 1 correct), one number per question, on the planned 500.

Sample 1 is generation index 0. Majority of 5 uses indices 0–4. Majority of 10 uses indices 0–9. The vote is a plurality. Ties go to whichever tied answer appeared first. A sample that does not parse is wrong and cannot win the vote. If nothing parses, the vote is wrong.

Uncertainty: paired question bootstrap, 20,000 resamples, seed 35036, percentile intervals. The same resamples are used for the accuracy of each arm and for D. Exact McNemar (two-sided binomial on the discordant pairs) is reported and does not decide the verdict.

Practical margin: 5 percentage points. Ten samples cost about ten times the decode of one sample. A gain below 5 points is smaller than the round-1 headline and is not a reason to spend that compute by default on this benchmark. Equivalence does not mean the difference is exactly zero.

Apply this once, to the final saved prefix of questions:

1. **PROVEN** if the 95% interval for D lies entirely above 0. A reliable gain of any size, including one below 5 points, is superiority. If the lower bound is also above 5 points, say so as a magnitude note. It does not change the word.
2. Otherwise **DISPROVEN (equivalent within ±5 points)** if the 90% interval lies entirely inside (−5, +5) points. That is a bootstrap TOST at α = .05 on each side.
3. Otherwise **DISPROVEN (no practically meaningful advantage)** if the 95% upper bound is below +5 points. This includes an interval that lies entirely below 0 (the vote is worse). The reason string says "inferior" in that case.
4. Otherwise the interval includes 0 and still reaches +5 points or more. The word is **STILL INCONCLUSIVE**. If that happens because the run stopped before 500 questions, the cause is the budget or a dead session. If it happens on all 500, the estimate landed in this indeterminate band. Nonsignificance alone is not DISPROVEN.

Superiority is checked first, so a clearly positive interval that is also narrower than 5 points is PROVEN, not equivalence.

Majority-of-5 uses the same summaries and the same numerical thresholds in the table. It does not change the verdict. "Any of 10 correct" is an oracle upper bound, not a method, and is descriptive only.

A strict label-only parser (no boxed-only fallback) is a sensitivity column. It does not change the verdict.

The run does not look at accuracy between batches. Three protocol gates look only at the first saved batch, and none of them reads the gold answer. Stop if more than 20% of samples hit the 2048-token cap, or if the mean length exceeds 1000 tokens. Either one means thinking-off was not actually in force. Also stop if more than half the questions in that batch have identical sample texts, which means the ten draws did not actually vary. Those generations are discarded and are not scored.

## Power at n = 500 and a 5-point margin

Normal approximation to the bootstrap interval, 100,000 draws, D-hat ~ Normal(Δ, (q − Δ²) / 500), where q is the fraction of questions on which sample 1 and the vote disagree. Code: `scripts/power.py`. Output: `results/power.json`.

| True paired gap | Discordance q | Approx. 95% half-width | P(PROVEN) | P(DISPROVEN) | P(indeterminate) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.05 | 2.0 pts | 0.03 | 0.97 | 0.00 |
| 0 | 0.10 | 2.8 pts | 0.03 | 0.97 | 0.00 |
| 0 | 0.15 | 3.4 pts | 0.03 | 0.89 | 0.08 |
| 0 | 0.20 | 3.9 pts | 0.03 | 0.80 | 0.17 |
| 0 | 0.25 | 4.4 pts | 0.02 | 0.72 | 0.25 |
| +2.5 pts (one-way, round-1 paired size) | 0.025 | 1.4 pts | 0.95 | 0.05 | 0.00 |
| +5 pts | 0.08 | 2.4 pts | 0.98 | 0.02 | 0.00 |
| +5 pts | 0.15 | 3.4 pts | 0.83 | 0.05 | 0.12 |
| +7 pts | 0.15 | 3.3 pts | 0.98 | 0.00 | 0.01 |
| +7 pts | 0.20 | 3.9 pts | 0.94 | 0.00 | 0.05 |

False PROVEN under a true zero is about 2.5%, the upper tail of a 95% interval. Under a true zero and discordance up to 0.20, DISPROVEN is the likely call. Under a paired gain of 5 points or of the round-1 one-directional 2.5 points, PROVEN is the likely call. The indeterminate band is the middle, where the data still allow both 0 and +5 points. That is why the margin is 5 points rather than 3: at n = 500 a 3-point bound is not reliably cleared when discordance is 0.15 or higher. GSM8K's test split has 1,319 questions. Five hundred is the pre-set sample. It is not expanded after seeing D.

## Protocol

- Model: `Qwen/Qwen3-4B`, vLLM, bfloat16 on L4 or A100, float16 on T4. Thinking off via `apply_chat_template(..., enable_thinking=False)`.
- Data: Hugging Face `openai/gsm8k`, config `main`, split `test`. `random.Random(35035)` shuffles the indices and the first 500 are kept. The question text and the gold string after `####` are saved in `results/questions.json` on the first successful prepare, and reused if the job resumes.
- Prompt suffix, both rounds of the model call: newline + `Solve it step by step. Finish with a final line 'Answer: <number>'.`
- Sampler, official non-thinking recommendation: temperature 0.7, top_p 0.8, top_k 20, max 2048 new tokens, 10 samples. Request seed is `35035 + question position`. Sample order is vLLM's output order for that request.
- Parser: last explicit `Answer:` / `Final Answer:` whose tail starts with a number after Markdown or LaTeX wrappers are stripped; if none, the last `\boxed{number}`. Text after `</think>` only, if that tag appears. No last-number fallback. `1,000` and `1000` and `1000.0` are the same answer. Details and synthetic tests: `scripts/parser.py`, `scripts/test_protocol.py`.
- Batches of 10 questions (4 on a T4). Each batch is an atomic jsonl file. The driver downloads it before the next batch is required for the analysis. A resumed job skips positions already saved.
- Constants: `scripts/protocol.json`. The analysis refuses rows whose temperature, top-p, top-k, max tokens, seed, or thinking flag differ from that file.

## Budget and jobs

Account check first. Do not start if the balance is below 50 CU. Session name `r3035` only. Prefer L4, then T4, then A100. On "allocation refused", wait 5 minutes and retry: 15 times for L4, then 8 for T4, then 6 for A100. Never stop a session this experiment did not start.

Hard cap 5 CU. The driver stops at 4.8 CU using this session's marginal usage rate (account rate after launch minus account rate before, when the assignment count goes up by one). If that difference cannot be isolated, use 1.54 CU/h for L4 (the rate measured on earlier L4 jobs), 1.20 for T4, and 4.00 for A100. A watchdog stops `r3035` at that same deadline if the driver dies. Wall time target is under about 4 hours. Planning time on an L4 is roughly 10 minutes of setup plus about half an hour of generation.

Allocation attempts at 00:39–00:54 UTC on 16 Oct were refused while two other assignments were active. No generations were saved. The decision rule above was not changed.

GPU type: pending, filled in after the run.
CU used: pending, filled in after the run.

## How to run

```bash
source /workspace/tools/setup.sh
python3 scripts/test_protocol.py
python3 scripts/power.py
bash scripts/run_experiment.sh
python3 scripts/analyze.py
```

`scripts/drive.py` is resumable. It copies each batch into `results/batches/` and stops `r3035` on success, failure, or the CU deadline.
