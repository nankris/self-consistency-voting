"""One confirmatory analysis of the paired self-consistency contrast.

Reads results/batches/*.jsonl. Does not generate text. Verdict rules live in
decide() and match README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from parser import canon, parse_answer, parse_strict, vote

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text())


def intervals(values: np.ndarray, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 5
    draw = rng.integers(0, n, size=(reps, n))
    means = values[draw].mean(axis=1)
    est = float(values.mean())
    l95, u95 = (float(x) for x in np.quantile(means, [0.025, 0.975]))
    l90, u90 = (float(x) for x in np.quantile(means, [0.05, 0.95]))
    return est, l95, u95, l90, u90


def mcnemar(single: np.ndarray, other: np.ndarray):
    b = int(np.sum((other == 1) & (single == 0)))
    c = int(np.sum((other == 0) & (single == 1)))
    if b + c == 0:
        return b, c, 1.0
    p = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return b, c, p


def decide(l95: float, u95: float, l90: float, u90: float, margin: float):
    """Superiority first. Else equivalence (TOST) or an upper bound below +margin."""
    if l95 > 0:
        return "PROVEN", "superiority: paired bootstrap 95% CI for vote10 minus sample 1 is entirely above 0"
    if -margin < l90 and u90 < margin:
        return (
            "DISPROVEN",
            "equivalent within the margin: 90% CI lies inside +/-5 percentage points (bootstrap TOST at alpha .05)",
        )
    if u95 < 0:
        return "DISPROVEN", "inferior: paired bootstrap 95% CI is entirely below 0"
    if u95 < margin:
        return (
            "DISPROVEN",
            "no practically meaningful advantage: 95% CI upper bound is below +5 percentage points",
        )
    return (
        "INDETERMINATE",
        "95% CI includes 0 and extends to at least +5 percentage points",
    )


def load_rows() -> list[dict]:
    rows = []
    paths = sorted((ROOT / "results" / "batches").glob("b*.jsonl"))
    seen = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pos = row["pos"]
            if pos in seen:
                raise SystemExit(f"duplicate pos {pos} in {path.name}")
            seen.add(pos)
            rows.append(row)
    rows.sort(key=lambda r: r["pos"])
    if not rows:
        raise SystemExit("no batch rows")
    prefix = []
    for row in rows:
        if row["pos"] != len(prefix):
            break
        prefix.append(row)
    dropped = len(rows) - len(prefix)
    if dropped:
        print(f"warning: ignoring {dropped} rows after a hole at pos {len(prefix)}")
    for row in prefix:
        if len(row["samples"]) != int(PROTOCOL["n_samples"]):
            raise SystemExit(f"pos {row['pos']} has {len(row['samples'])} samples")
        if abs(row["temperature"] - float(PROTOCOL["temperature"])) > 1e-9:
            raise SystemExit("temperature does not match protocol.json")
        if row["top_p"] != float(PROTOCOL["top_p"]) or row["top_k"] != int(PROTOCOL["top_k"]):
            raise SystemExit("top_p/top_k do not match protocol.json")
        if row["max_tokens"] != int(PROTOCOL["max_tokens"]):
            raise SystemExit("max_tokens does not match protocol.json")
        if row["seed"] != int(PROTOCOL["sample_seed_base"]) + row["pos"]:
            raise SystemExit(f"seed mismatch at pos {row['pos']}")
        if row.get("thinking") not in (False, 0):
            raise SystemExit("thinking was not off")
    return prefix, dropped


def score(rows: list[dict], parser) -> dict:
    n = len(rows)
    parsed = []
    golds = []
    trunc = 0
    tokens = []
    for row in rows:
        golds.append(canon(row["gold"]))
        parsed.append([parser(sample["text"]) for sample in row["samples"]])
        trunc += sum(sample["finish_reason"] == "length" for sample in row["samples"])
        tokens.append([sample["tokens"] for sample in row["samples"]])
    golds_a = np.array(golds, dtype=object)
    single = np.array([1 if vote(p, 1) == g else 0 for p, g in zip(parsed, golds)], dtype=float)
    five = np.array([1 if vote(p, 5) == g else 0 for p, g in zip(parsed, golds)], dtype=float)
    ten = np.array([1 if vote(p, 10) == g else 0 for p, g in zip(parsed, golds)], dtype=float)
    oracle = np.array(
        [1 if any(v == g for v in p if v is not None) else 0 for p, g in zip(parsed, golds)],
        dtype=float,
    )
    failures = sum(v is None for p in parsed for v in p)
    return {
        "n": n,
        "single": single,
        "five": five,
        "ten": ten,
        "oracle": oracle,
        "parsed": parsed,
        "golds": golds_a,
        "parse_failures": failures,
        "parse_failure_rate": failures / (n * 10),
        "truncations": trunc,
        "truncation_rate": trunc / (n * 10),
        "tokens": np.array(tokens),
    }


def arm_record(name: str, correct: np.ndarray, single: np.ndarray, reps: int, seed: int) -> dict:
    est, l95, u95, l90, u90 = intervals(correct, reps, seed)
    delta = correct - single
    d_est, d_l95, d_u95, d_l90, d_u90 = intervals(delta, reps, seed)
    b, c, p = mcnemar(single, correct)
    return {
        "name": name,
        "correct": int(correct.sum()),
        "n": int(len(correct)),
        "accuracy": est,
        "accuracy_ci95": [l95, u95],
        "delta_vs_sample1": d_est,
        "delta_ci95": [d_l95, d_u95],
        "delta_ci90": [d_l90, d_u90],
        "mcnemar_b_vote_only": b,
        "mcnemar_c_single_only": c,
        "mcnemar_p": p,
    }


def write_chart(summary: dict, path: Path) -> None:
    arms = [a for a in summary["arms"] if a["name"] != "any of 10 (oracle)"]
    oracle = next(a for a in summary["arms"] if a["name"].startswith("any"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    xs = np.arange(len(arms))
    acc = np.array([a["accuracy"] * 100 for a in arms])
    lo = np.array([a["accuracy_ci95"][0] * 100 for a in arms])
    hi = np.array([a["accuracy_ci95"][1] * 100 for a in arms])
    axes[0].bar(xs, acc, color=colors, width=0.72)
    axes[0].errorbar(xs, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor="black", capsize=4, lw=1)
    for i, a in enumerate(arms):
        axes[0].text(i, hi[i] + 0.6, f"{a['accuracy']*100:.1f}", ha="center", va="bottom", fontsize=9)
    floor = max(0, min(lo) - 8)
    axes[0].set_ylim(floor, min(100, max(hi) + 6))
    axes[0].set_xticks(xs, ["1 sample", "majority of 5", "majority of 10"])
    axes[0].set_ylabel("GSM8K accuracy (%)")
    axes[0].set_title("Accuracy, marginal 95% CI")
    axes[0].axhline(oracle["accuracy"] * 100, color="#9e9e9e", ls=":", lw=1)
    axes[0].text(
        0.0,
        oracle["accuracy"] * 100 + 0.4,
        f"any of 10 correct (oracle) {oracle['accuracy']*100:.1f}%",
        color="#666666",
        fontsize=8,
    )

    deltas = arms[1:]
    dx = np.arange(len(deltas))
    dest = np.array([a["delta_vs_sample1"] * 100 for a in deltas])
    dlo = np.array([a["delta_ci95"][0] * 100 for a in deltas])
    dhi = np.array([a["delta_ci95"][1] * 100 for a in deltas])
    axes[1].axhspan(-5, 5, color="#54A24B", alpha=0.08, label="equivalence margin")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].axhline(5, color="#54A24B", ls="--", lw=1)
    axes[1].axhline(-5, color="#54A24B", ls="--", lw=1)
    axes[1].errorbar(
        dx,
        dest,
        yerr=[dest - dlo, dhi - dest],
        fmt="o",
        color="#E45756",
        ms=8,
        capsize=4,
        lw=1.5,
    )
    for i, a in enumerate(deltas):
        axes[1].text(i, dhi[i] + 0.25, f"{a['delta_vs_sample1']*100:+.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(dx, ["majority of 5\nminus 1 sample", "majority of 10\nminus 1 sample"])
    axes[1].set_ylabel("Paired accuracy difference (points)")
    axes[1].set_title("Confirmatory contrast is the right point")
    span = max(6, np.max(np.abs(np.concatenate([dlo, dhi]))) + 1.5)
    axes[1].set_ylim(-span, span)
    n = summary["n"]
    fig.suptitle(
        f"Qwen3-4B, thinking off, {n} GSM8K test questions\n{summary['verdict']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    rows, dropped = load_rows()
    reps = int(PROTOCOL["bootstrap_reps"])
    seed = int(PROTOCOL["bootstrap_seed"])
    margin = float(PROTOCOL["margin"])
    primary = score(rows, parse_answer)
    strict = score(rows, parse_strict)
    arms = [
        arm_record("1 sample", primary["single"], primary["single"], reps, seed),
        arm_record("majority of 5", primary["five"], primary["single"], reps, seed),
        arm_record("majority of 10", primary["ten"], primary["single"], reps, seed),
        arm_record("any of 10 (oracle)", primary["oracle"], primary["single"], reps, seed),
    ]
    # Sample-1 vs itself is not a contrast. Zero the delta fields for the first arm.
    arms[0]["delta_vs_sample1"] = 0.0
    arms[0]["delta_ci95"] = [0.0, 0.0]
    arms[0]["delta_ci90"] = [0.0, 0.0]
    arms[0]["mcnemar_b_vote_only"] = 0
    arms[0]["mcnemar_c_single_only"] = 0
    arms[0]["mcnemar_p"] = 1.0

    ten = next(a for a in arms if a["name"] == "majority of 10")
    l95, u95 = ten["delta_ci95"]
    l90, u90 = ten["delta_ci90"]
    base, reason = decide(l95, u95, l90, u90, margin)
    budget_path = ROOT / "results" / "budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    stopped_early = bool(budget.get("stopped_for_budget")) or primary["n"] < int(PROTOCOL["n_questions"])
    if base == "INDETERMINATE":
        verdict = "STILL INCONCLUSIVE"
        if primary["n"] < int(PROTOCOL["n_questions"]):
            why = "The budget cap" if budget.get("stopped_for_budget") else "The run"
            reason = (
                f"{why} stopped before 500 questions, with the estimate still inside the indeterminate band. "
                + reason
            )
        else:
            reason = (
                "The full planned sample landed in the pre-registered indeterminate band. " + reason
            )
    else:
        verdict = base
        if stopped_early:
            reason = (
                f"Criteria met on the {primary['n']} questions saved before the run stopped. " + reason
            )

    strict_ten_delta = strict["ten"] - strict["single"]
    s_est, s_l95, s_u95, s_l90, s_u90 = intervals(strict_ten_delta, reps, seed)
    s_verdict, s_reason = decide(s_l95, s_u95, s_l90, s_u90, margin)
    meta_path = ROOT / "results" / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tok = primary["tokens"]
    summary = {
        "verdict": verdict,
        "reason": reason,
        "n": primary["n"],
        "dropped_after_hole": dropped,
        "planned_n": int(PROTOCOL["n_questions"]),
        "stopped_for_budget": bool(budget.get("stopped_for_budget")),
        "stopped_early": stopped_early,
        "margin": margin,
        "primary_delta": ten["delta_vs_sample1"],
        "primary_delta_ci95": ten["delta_ci95"],
        "primary_delta_ci90": ten["delta_ci90"],
        "primary_mcnemar_p": ten["mcnemar_p"],
        "parse_failures": primary["parse_failures"],
        "parse_failure_rate": primary["parse_failure_rate"],
        "truncations": primary["truncations"],
        "truncation_rate": primary["truncation_rate"],
        "mean_tokens_per_sample": float(tok.mean()),
        "arms": arms,
        "strict_parser": {
            "verdict_if_used": s_verdict,
            "reason": s_reason,
            "delta": s_est,
            "delta_ci95": [s_l95, s_u95],
            "accuracy_sample1": float(strict["single"].mean()),
            "accuracy_vote5": float(strict["five"].mean()),
            "accuracy_vote10": float(strict["ten"].mean()),
            "parse_failure_rate": strict["parse_failure_rate"],
            "note": "Sensitivity only. The verdict uses the primary parser.",
        },
        "metadata": metadata,
        "budget": budget,
        "protocol": PROTOCOL,
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    paired_path = ROOT / "results" / "paired.jsonl"
    with paired_path.open("w") as handle:
        for i, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "pos": row["pos"],
                        "index": row["index"],
                        "gold": primary["golds"][i],
                        "preds": primary["parsed"][i],
                        "correct1": int(primary["single"][i]),
                        "correct5": int(primary["five"][i]),
                        "correct10": int(primary["ten"][i]),
                        "oracle10": int(primary["oracle"][i]),
                    }
                )
                + "\n"
            )
    write_chart(summary, ROOT / "chart.png")
    print(json.dumps({k: summary[k] for k in (
        "verdict", "reason", "n", "primary_delta", "primary_delta_ci95",
        "primary_delta_ci90", "primary_mcnemar_p", "parse_failure_rate", "truncation_rate",
    )}, indent=2))
    for arm in arms:
        print(
            f"{arm['name']:22} acc={arm['accuracy']*100:6.2f} "
            f"CI95=({arm['accuracy_ci95'][0]*100:6.2f},{arm['accuracy_ci95'][1]*100:6.2f}) "
            f"d={arm['delta_vs_sample1']*100:+6.2f} "
            f"dCI=({arm['delta_ci95'][0]*100:+6.2f},{arm['delta_ci95'][1]*100:+6.2f}) "
            f"p={arm['mcnemar_p']:.4g}"
        )


if __name__ == "__main__":
    main()
