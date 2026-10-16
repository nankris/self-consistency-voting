# Self-consistency, round 5

**PROVEN — superiority.**

| Method | Correct | Accuracy | 95% bootstrap CI |
|---|---:|---:|---:|
| Greedy | 114/400 | 28.50% | [24.25, 33.00]% |
| Vote of five | 176/400 | 44.00% | [39.25, 48.75]% |

Primary paired difference: **+15.50 percentage points**, percentile bootstrap 95% CI **[11.50, 19.50]**, 90% CI [12.00, 19.00]. 20,000 question-paired resamples, seed 35. Vote-only correct: 71; greedy-only correct: 9. Supplementary exact McNemar p=4.37398e-13; this does not replace the preregistered decision.

All 400 questions use first-in-file order, one greedy and five independent-temperature samples, with thinking disabled, temperature .7, top_p .95, max_tokens 512. Voting uses unchanged round-3 parser/plurality tie policy. Parse failures: greedy 223/400, stochastic 1116/2000. Token-limit finishes: greedy 2, stochastic 13. Strict label-only sensitivity: greedy 113, vote 174 correct. The 512-token cap and parsing define the measured task.

GPU: NVIDIA A100-SXM4-40GB, session r5q035, stopped. Conservative CU upper bound: 2.860 (entire account observed rate × session elapsed; another session ran concurrently, so this overcounts this experiment). See state.json, attempts.jsonl, results/usage_*.txt and stop.txt for audit. Results were copied locally and acknowledged every 50 questions before inference continued. No outcome-dependent stopping or additional model calls.

This estimates performance on a fixed public GSM8K subset, not a random population sample; question bootstrap describes variation under question resampling. Training contamination, correlated model errors and a modest fixed sample limit generalization. Margin and n were fixed by Hark. A non-significant result would not by itself establish equivalence.

Reproduce analysis: `python scripts/analyze.py` with numpy, scipy, matplotlib installed. Raw generations: results/b0000.jsonl through b0350.jsonl; question text and reference answers: results/questions.json. Original and earlier rerun folders were read only.

## Formatting artifact and post-hoc diagnostic

The prompt's literal `Answer: NUMBER` wording was a mistake: 208 greedy and 1,037 stochastic outputs copied the placeholder before their numeric answer. The inherited parser requires the number immediately after the label, so it rejects these. Primary parse failures are 55.75% (greedy) and 55.80% (stochastic). Voting can recover a parsable answer from multiple attempts, so **the primary +15.5-point gain mixes formatting reliability with mathematical correctness**. The preregistered parser and verdict were not changed.

A post-hoc, gold-independent diagnostic removes only literal `NUMBER` after an Answer label (allowing Markdown asterisks), then applies the original parser and voting unchanged. It scores greedy **274/400 (68.50%)** and vote **309/400 (77.25%)**, difference **+8.75 points**, paired bootstrap 95% CI **[+5.25, +12.25]**, 90% CI [+5.75, +11.75]. This is exploratory, not a second confirmatory verdict. It supports a positive voting effect after that narrow formatting repair, but the primary magnitude should not be presented as a pure reasoning gain. No new model calls or altered raw outputs were used.

Estimated experiment consumption is **1.430 CU**, using the observed rate drop from 10.60 to 5.30 CU/hour when only r5q035 stopped, over 16.19 allocation minutes. The **2.860-CU upper bound** charges the whole observed shared-account rate. The account balance moved 171.05 to 169.06 CU; that includes concurrent work and billing update lag, so it is not experiment-specific billing.
