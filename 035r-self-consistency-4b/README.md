# 035r: Qwen3-4B self-consistency

## Preregistered hypothesis and decision rule (before execution)
Hypothesis: majority vote of 10 improves Qwen3-4B GSM8K accuracy over a single stochastic sample, averaged equally across thinking on and off.
Round 1: 200 GSM8K test questions sampled without replacement with seed 3501; generate 10 independent samples per question and thinking mode. Report nested vote-of-1, 5, 10 curves. Single is sample 1, not the average of ten. Vote means plurality; ties resolve to the earliest sampled tied answer. Missing answers count wrong and cannot form a valid winning answer. Parse only the final answer outside thinking.
Primary endpoint: paired question-level mean of (vote10 correct minus sample1 correct) across both modes. Bootstrap questions (20,000 replicates, seed 3502), preserving modes and samples. PROVEN if its 95% percentile CI is strictly above zero; DISPROVEN if strictly below zero; otherwise inconclusive. Secondary per-mode and vote5 CIs are descriptive, not additional primary tests.
If inconclusive, redesign once: add 400 disjoint randomly selected test questions (600 total), keep inference unchanged, and use a 97.5% bootstrap CI for the final primary endpoint to limit repeated-look optimism (round 1 will also use 97.5% for its verdict; 95% curves remain descriptive). If budget prevents this, report STILL INCONCLUSIVE. Stop after two rounds. No effect claim from nonsignificance.

Model: Qwen/Qwen3-4B, bf16 vLLM on L4 (Q4 GGUF fallback only if required). Official recommended sampling per mode: thinking T=.6/p=.95/k=20; nonthinking T=.7/p=.8/k=20. Token cap 8192 thinking, 2048 nonthinking; truncations and parse failures reported. Same prompt in both modes: solve step by step and finish with `Answer: <number>`. No frontier-model comparison. Public data only. Retain raw outputs and exact environment metadata. Budget hard limit 5 CU; monitor runtime and stop session on completion/failure.

GPU and CU: pending execution. Account balances can be shared; session rate × allocation duration will be reported separately from account balance change.

## Execution outcome
STILL INCONCLUSIVE — infrastructure blocked, no inference rounds completed. Three `colab new --gpu L4 -s tig035r` attempts were refused with `precondition failed`. This is not a statistical result. The second-round contingency was not triggered because round 1 could not run. The original folder was not modified.
GPU requested: NVIDIA L4; GPU allocated: none. Experiment CU used: **0.00**. Before and after balance **181.82 CU**, account usage rate **3.08 CU/hr**, two unrelated active assignments. `colab stop -s tig035r` was called and reported no such session. Allocation log: `results/allocation_log.txt`.

## Prepared reproduction
After capacity becomes available, source `/workspace/tools/setup.sh`, create only `tig035r` with `colab new --gpu L4 -s tig035r`, execute `scripts/setup_remote.py`, upload `scripts/infer.py` to `/content/infer.py`, and execute `scripts/launch.py`. Monitor `/content/infer.log`, `colab usage`, and elapsed allocation time; download `/content/035r` results before stopping. Enforce a 5-CU session budget including setup. Run `python scripts/analyse.py` locally once paired results are present. If inconclusive and enough budget remains, set `N_QUESTIONS=600` for the resumed inference process (existing samples are preserved), then analyse again. Always stop the session, including on failure. The inference/analysis scripts are prepared but GPU execution is unvalidated.
Sampling recommendations and thinking switch follow the [official model card](https://huggingface.co/Qwen/Qwen3-4B). `chart.png` explicitly shows missing observations; it is not a measured accuracy curve.

## Resumed allocation attempt — 2026-10-15
Before running inference, reaffirm the hypothesis and decision rule above without changes: paired vote10 minus first-sample accuracy averaged across modes; 97.5% bootstrap CI strictly above/below zero determines PROVEN/DISPROVEN, otherwise one expansion from 200 to 600 questions, maximum two rounds. All six accuracy points receive descriptive 95% bootstrap CIs. Existing artifacts and original experiment are preserved. Starting account balance: 180.02 CU; unrelated assignments: 1, rate 1.54 CU/hr. New session name: tig035r-retry. Maximum experiment allocation budget: 5 CU.

Implementation clarification before inference: retain special tokens in decoded outputs so the closing thinking delimiter remains available to the preregistered final-answer parser.

## Current resumption — 2026-10-15
No inference data exists at resumption. Reaffirm the preregistered hypothesis and decision rule above before any execution. Starting account balance 179.32 CU; one unrelated L4 assignment, which will not be touched. Session `tig035r-final`; maximum allocated runtime 5/1.54 hours at the observed L4 rate, with a conservative stop before 3 hours. Allocation refusals will be retried at five-minute intervals up to six retries. GPU type and actual session cost will be recorded after execution.

## Resumption at 2026-10-15 (current attempt)
Before inference, reaffirm the hypothesis, 97.5% primary paired bootstrap decision rule, and at most two rounds above. No measured samples exist. Starting balance: 178.36 CU; active assignments: 0. Session name: tig035r-oct15. Maximum budget 5 CU including setup.

### Parser clarification before accuracy analysis
A startup checkpoint of 60 nonthinking questions showed 48/600 outputs rejected by the prepared strict regex, including mathematically explicit `Final Answer: \boxed{2}`. No accuracy or effect estimate was calculated. Final-answer extraction will normalize common Markdown/LaTeX wrappers and accept the last numeric boxed final answer, still only outside thinking. This corrects a formatting confound without changing questions, generation, endpoint, or decision thresholds. Strict-label scoring is retained as a sensitivity analysis. No gold-dependent extraction or last-number-in-reasoning fallback is permitted.

### Scheduling optimization (no endpoint change)
After the first 10 thinking questions were saved, switch from synchronous 10-question chunks to continuous vLLM engine scheduling, with one checkpoint per completed question. Existing question-mode pairs are skipped; sample order is output index; seeds remain `3503 + 10*floor(position/10) + 10000*thinking`. No prompts, temperatures, token caps, or model weights change. An unfinished chunk may be discarded and regenerated with the same seed. This runtime optimization avoids waiting for each chunk’s longest completion and does not constitute a second statistical round.
