# Can a crowd of small models beat one big one? I made a 0.6B model vote five times

*Draft for Tech in General: real numbers, 14 Oct 2026.*

There's a well-known trick called **self-consistency**: instead of trusting one answer, sample a model several times and take the majority. It's cheap if the model is small and running on your own machine. So can a "crowd" of small local models match one call to a frontier model?

I tested it on maths word problems, the kind where one slip ruins the answer.

## The test

- **40 questions** from GSM8K, the public grade-school maths benchmark (the first 40 of the test set)
- **Local:** Qwen3-0.6B on a plain 2-core CPU, no GPU. 5 samples per question, majority vote.
- **Cloud:** one call per question to Grok via Grok Build CLI (`grok-4.7-build`)

I wanted a 4B model, as in the original plan. On a 2-core machine with 8 GB of RAM, even the 1.7B version barely generated a token a second. 0.6B was the biggest that finished in reasonable time.

## Results

| | accuracy | time per question | cost per question |
|---|---|---|---|
| Qwen3-0.6B, 1 answer | 50.5% | ~16 s | $0 |
| **Qwen3-0.6B, vote of 5** | **57.5%** | **79 s** | $0 |
| **Grok, 1 call** | **100%** | **9 s** | **0.38¢** |

Voting helped: **+7 points**, from about a coin flip to 57.5%. But it cost 5× the compute and still left the small model 42 points behind. The whole 40-question run took **53 minutes** on my CPU. Grok finished all 40 in **83 seconds** for **15 cents** (at API-equivalent rates).

## Why voting didn't save it

Self-consistency only helps when the model is *sometimes* right. On 75% of questions at least one of the five samples was correct, so the right answer was often in the crowd. But the crowd often agreed on the wrong one. On one question all five samples said **180**; the answer was **540**. If a model makes the same mistake every time, voting just confirms it.

## So when does the small-model crowd make sense?

- **When privacy or offline use matters more than accuracy.** 57% is not a maths tutor.
- **With a bigger local model.** A 4B to 8B model on a laptop GPU is a different story. I'd like to rerun this on real hardware.
- **Not for maths on a CPU.** One cloud call at a third of a cent was 9× faster and right every time.

## Caveats

- 40 questions is small: the +7 points is 3 questions.
- GSM8K is old and public. Big models have almost certainly seen it, so Grok's 100% flatters it.
- The local model ran in Qwen3's fast non-thinking mode. Thinking mode would probably score higher and take much longer.
- "$0" ignores the computer and the electricity. Grok's price is API-equivalent; on a SuperGrok plan what really limits me is the usage window.

*Scripts, every sample and the raw Grok logs are in the experiment repo.*
