"""Fill writeup.md from results/summary.json. Does not rescore."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci(pair: list[float], scale: float = 100, signed: bool = False) -> str:
    fmt = "{:+.2f}" if signed else "{:.2f}"
    return f"[{fmt.format(scale * pair[0])}, {fmt.format(scale * pair[1])}]"


def main() -> None:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    budget = summary.get("budget") or {}
    meta = summary.get("metadata") or {}
    arms = {a["name"]: a for a in summary["arms"]}
    one = arms["1 sample"]
    five = arms["majority of 5"]
    ten = arms["majority of 10"]
    oracle = arms["any of 10 (oracle)"]
    gpu = meta.get("gpu") or budget.get("allocations", [{}])[-1].get("gpu", "unknown")
    cu = budget.get("cu_used")
    cu_s = f"{cu:.2f}" if isinstance(cu, (int, float)) else "unknown"
    versions = meta.get("versions") or {}
    strict = summary["strict_parser"]
    n = summary["n"]
    lines = [
        "# 035r3 results: majority vote on Qwen3-4B, thinking off",
        "",
        f"**Verdict: {summary['verdict']}.** {summary['reason']}",
        "",
        "Primary contrast, locked before the run: on each question, majority-of-10 correct minus the first sample correct. Paired question bootstrap, 20,000 resamples, seed 35036. PROVEN if the 95% interval lies entirely above 0. Otherwise DISPROVEN if the 90% interval lies inside ±5 points (bootstrap TOST) or the 95% upper bound is below +5 points. Majority-of-5 is reported with the same intervals and does not decide the verdict.",
        "",
        f"Model `{meta.get('model', 'Qwen/Qwen3-4B')}`, thinking off, {gpu}. "
        f"vLLM {versions.get('vllm', '?')}, torch {versions.get('torch', '?')}. "
        f"Sample: {n} of {summary['planned_n']} GSM8K test questions, seed 35035. "
        f"Temperature 0.7, top_p 0.8, top_k 20, 10 samples, 2048-token cap.",
        "",
        "## Results",
        "",
        "| Arm | Correct | Accuracy | 95% CI | Versus sample 1 | 95% CI for the difference | 90% CI | McNemar p |",
        "|---|---:|---:|---|---:|---|---|---:|",
    ]
    for arm in (one, five, ten, oracle):
        lines.append(
            "| {name} | {correct}/{n} | {acc} | {aci} | {d} | {dci} | {d90} | {p:.4g} |".format(
                name=arm["name"],
                correct=arm["correct"],
                n=arm["n"],
                acc=pct(arm["accuracy"]),
                aci=ci(arm["accuracy_ci95"]),
                d=pts(arm["delta_vs_sample1"]),
                dci=ci(arm["delta_ci95"], signed=True),
                d90=ci(arm["delta_ci90"], signed=True),
                p=arm["mcnemar_p"],
            )
        )
    lines += [
        "",
        f"Parse failures: {summary['parse_failures']} of {n * 10} samples ({pct(summary['parse_failure_rate'])}). "
        f"Truncations at 2048 tokens: {summary['truncations']} ({pct(summary['truncation_rate'])}). "
        f"Mean generated tokens per sample: {summary['mean_tokens_per_sample']:.1f}.",
        "",
        f"Strict label-only parser, sensitivity only: sample 1 {pct(strict['accuracy_sample1'])}, "
        f"majority of 5 {pct(strict['accuracy_vote5'])}, majority of 10 {pct(strict['accuracy_vote10'])}, "
        f"difference {pts(strict['delta'])} with 95% CI {ci(strict['delta_ci95'], signed=True)}. "
        f"If that parser had decided the call, the word would have been {strict['verdict_if_used']}. "
        f"Parse-failure rate under the strict parser: {pct(strict['parse_failure_rate'])}.",
        "",
        f"GPU: {gpu}. Compute units charged to this session at the marginal rate: {cu_s} CU. "
        f"Account balance before the run: {budget.get('allocations', [{}])[0].get('balance_before', 'unknown')} CU. "
        f"Balance after: {budget.get('balance_after', 'unknown')} CU.",
        "",
        "Chart: `chart.png`. Per-question scores: `results/paired.jsonl`. Raw generations: `results/batches/`. Full machine summary: `results/summary.json`.",
        "",
        "## What this does and does not say",
        "",
        "The comparison is one stochastic sample against a majority of samples from the same model, on one public benchmark, with thinking switched off. It is not a test of thinking mode, and it is not a comparison with a frontier model. Round 1's +7 point headline was a 0.6B CPU model on 40 questions, and it compared the vote with the average of five samples. The paired first-versus-vote gap there was one question.",
        "",
        "Round 2 never reached a confirmatory analysis. A later file of 200 non-thinking generations was not reused: different question order, a shared seed inside each block of questions, and a parser that was still being revised. This round drew a fresh 500.",
        "",
    ]
    (ROOT / "writeup.md").write_text("\n".join(lines))
    print("wrote", ROOT / "writeup.md")


if __name__ == "__main__":
    main()
