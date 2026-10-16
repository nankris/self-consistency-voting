# 035r results

**Verdict: STILL INCONCLUSIVE.** The requested experiment could not run because Colab refused all three L4 allocation attempts. No inference round completed; no statistical claim can be made about self-consistency. The original +7-point result remains unresolved.

| Thinking | Samples | Questions evaluated | Accuracy | 95% bootstrap CI |
|---|---:|---:|---|---|
| Off | 1 | 0 | unavailable | unavailable |
| Off | 5 | 0 | unavailable | unavailable |
| Off | 10 | 0 | unavailable | unavailable |
| On | 1 | 0 | unavailable | unavailable |
| On | 5 | 0 | unavailable | unavailable |
| On | 10 | 0 | unavailable | unavailable |

The redesign replaces the original 0.6B/40-question test with Qwen3-4B, 200 randomly selected public GSM8K test questions, and paired 1/5/10-sample curves in both thinking modes. The hypothesis, parsing, tie handling, bootstrap, and one allowed sample-size expansion to 600 were written in README.md before allocation attempts. The primary endpoint is the mode-averaged paired gain of vote10 over sample1, using a 97.5% interval at both possible looks; curve intervals are descriptive 95% intervals. This tests improvement on this benchmark and decoding configuration, not uncontaminated mathematical reasoning or superiority over a frontier model.

No confidence intervals can be computed without observations. A second statistical round was not run because the first never started. Allocation retries are infrastructure attempts, not experimental rounds. `chart.png` displays this absence of data explicitly. Scripts are ready for a capacity-available rerun but are not GPU-validated.

GPU requested: L4; allocated: none. CU consumed: **0.00**. Account balance before/after: **181.82 / 181.82 CU**. The account had two other active assignments; neither was touched. Cleanup `colab stop -s tig035r` confirmed session not found. See `results/allocation_log.txt` for the exact failure. A100 was not attempted because this was an allocation failure, not an L4 memory failure.
