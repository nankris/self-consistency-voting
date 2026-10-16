# Experiment 035, round 5: self-consistency

Preregistered 2026-10-16 before allocation or inference. Hypothesis: majority/plurality of five stochastic Qwen/Qwen3-1.7B answers improves GSM8K accuracy over one greedy answer.

## Fixed design and decision rule

First 400 GSM8K main test questions in original file order, no shuffle. Qwen/Qwen3-1.7B, thinking disabled via chat template. One A100 session r5q035, vLLM. Five samples per question: temperature 0.7, top_p 0.95, max_tokens 512; separate greedy sample temperature 0, max_tokens 512. Seeds 35000 + question index. Prompt asks for brief working and final `Answer: NUMBER`. Reuse unchanged round-3 explicit-label/boxed parser; ignore missing parses in voting, all missing is incorrect, ties resolve to earliest sample. This is the conventional plurality interpretation of majority-of-5. No outcome-dependent prompt or parser changes.

Primary: question-paired accuracy difference (vote minus greedy). Percentile paired bootstrap, 20,000 resamples, NumPy default_rng seed 35, two-sided 95% and 90% intervals. PROVEN if 95% lower bound > 0. Otherwise DISPROVEN if 95% upper bound < 0, or 90% CI strictly inside [-0.02,+0.02]. Two points means at most eight answers in 400, a small practical gain for fivefold sampling cost. No supplementary threshold will replace this fixed rule. If none holds, report STILL INCONCLUSIVE honestly; fixed sample size can exhaust the authorized evidence budget even with all 400 complete. No optional stopping on results; only analyze after completion or resource exhaustion.

## Power calculation (before running)

For paired delta d and discordance q, SE ≈ sqrt((q-d²)/400). At anticipated +7 points and q=.15, SE=1.905 points and normal-approximation two-sided superiority power ≈95.7%; at q=.20 power ≈88.6%. A 7-point effect with q=.20 needs about 313 questions for 80% power. These are planning assumptions, not estimates from this run. Equivalence at zero is difficult: for q=.05 the 90% half-width is 1.84 points, with only about 11% chance to land fully within ±2; at q=.02 half-width=1.16 points and power ≈76.5%. At q≥.06 even a perfectly centered normal CI cannot fit. The fixed n=400 is likely decisive for the earlier +7-point hypothesis, but cannot guarantee either outcome. We retain Hark's exact n and margin rather than claim false power or enlarge the sample.

## Changes and prior lessons

Read-only sources: ../035-self-consistency, ../035r-self-consistency-4b, ../035r3-self-consistency. Earlier 0.6B CPU pilot had only 40 questions; later 4B attempts failed on GPU allocation/timeouts. Round 5 uses smaller 1.7B, goes directly to A100, caps output at 512, batches 50 questions, and compares to a separate greedy response. First 400 in file order replaces round-3 shuffled sampling. No frontier model calls needed. Public benchmark only; possible training contamination limits generalization.

## Resumption and budget

scripts/drive.py will record allocation ownership before starting, enforce a hard time cap conservatively below 5 CU, mirror every 50-question batch before acknowledging it remotely, and stop only the owned r5q035 session in finally. Remote inference waits for acknowledgement before generating the next batch. Append-only attempts.jsonl is the recovery ledger. A rerun checks saved state and completed batches, restores them and never discards samples. Allocation refusal: wait five minutes and retry at most 15 times. Never start if account balance <50 CU. Stop r5q035 at success or failure. Setup time counts. Maximum total experiment allocation 5 CU and wall time under four hours.

GPU: NVIDIA A100-SXM4-40GB. CU used: estimated 1.430, conservative upper bound 2.860; account balance change is reported separately because other experiments ran concurrently. Full execution details follow below.

Recovery entry point: `source /workspace/tools/setup.sh; .venv/bin/python scripts/resume.py`. It checks the saved deadline and session ownership; completed experiments are analyzed without allocating again. For the already active controller, do not start a duplicate. `results/controller.log` and `results/status_latest.txt` show progress. Analysis dependencies are pinned in requirements-analysis.txt.

## Completed execution (2026-10-16)

**PROVEN under the fixed parser:** 400/400 questions, 2,400 generations; greedy 114/400 (28.50%), vote 176/400 (44.00%); paired +15.50 points, 95% bootstrap CI [+11.50,+19.50], 90% [+12.00,+19.00]. Exact McNemar p=4.374e-13. All eight checkpoints were saved before acknowledgment. GPU was **NVIDIA A100-SXM4-40GB**, one session **r5q035**, successfully stopped. Allocation duration 16.19 minutes. **CU used: estimated 1.430; conservative upper bound 2.860**, below 5. Rate estimate comes from 10.60 CU/hour with two sessions minus 5.30 with only the unrelated session remaining; exact per-session billing is unavailable. Account balance delta 1.99 CU includes concurrent work and billing lag.

Infrastructure recovery stayed in the same GPU session: notebook kernel restarted during pip installation; one premature launch failed before generation; HTTP control repeatedly timed out at ten seconds, then sporadically at sixty seconds. An isolated timeout wrapper and ultimately Colab SSH recovered control. First 250 questions were mirrored by `colab download`; last 150 by `cat` over `colab ssh --proxy-mode`, with identical checkpoint-before-ack behavior. No completed generation was discarded or repeated. `results/initial_launch_error.txt` and the ledger preserve failures. The unrelated session was never stopped.

**Material limitation:** the prompt `Answer: NUMBER` caused many outputs to copy literal NUMBER. The locked parser rejects these (55.75% greedy, 55.80% stochastic missing parses), so the primary gain includes formatting reliability. A clearly post-hoc diagnostic strips only that literal placeholder before applying the same parser: greedy 274/400 (68.50%), vote 309/400 (77.25%); +8.75 points, 95% CI [+5.25,+12.25]. It supports a gain after this narrow repair, but does not replace the registered verdict or make the primary 15.5-point magnitude a pure reasoning effect. See writeup.md. No post-hoc inference rerun was performed.

The original pilot's +7-point headline compared voting to average sampled accuracy, not greedy, so it was only a hypothetical planning effect. Round 5's actual comparator is the required separate greedy draw.

Exact GPU package versions and model revision are in results/metadata.json and requirements-gpu.txt. Reproduce primary: `.venv/bin/python scripts/analyze.py`; optional post-hoc diagnostic: `.venv/bin/python scripts/diagnostic.py`. All numerical results derive from preserved raw generations. Analysis scripts have no GPU/model calls.
