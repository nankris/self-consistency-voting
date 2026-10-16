"""Planning power for the locked decision rule. No model outputs are read."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text())


def decide(hat: float, se: float, margin: float) -> str:
    z95, z90 = 1.95996398454, 1.64485362695
    l95, u95 = hat - z95 * se, hat + z95 * se
    l90, u90 = hat - z90 * se, hat + z90 * se
    if l95 > 0:
        return "PROVEN"
    if -margin < l90 and u90 < margin:
        return "DISPROVEN"
    if u95 < margin:
        return "DISPROVEN"
    return "INDETERMINATE"


def simulate(delta: float, q: float, n: int, margin: float, reps: int, rng) -> dict:
    var = q - delta ** 2
    if var < 0:
        raise ValueError(f"impossible q={q} delta={delta}")
    se = np.sqrt(var / n)
    hats = rng.normal(delta, se, size=reps)
    labels = [decide(float(h), float(se), margin) for h in hats]
    return {
        "delta": delta,
        "q": q,
        "se": se,
        "ci95_halfwidth": 1.95996398454 * se,
        "proven": labels.count("PROVEN") / reps,
        "disproven": labels.count("DISPROVEN") / reps,
        "indeterminate": labels.count("INDETERMINATE") / reps,
    }


def main() -> None:
    n = int(PROTOCOL["n_questions"])
    margin = float(PROTOCOL["margin"])
    rng = np.random.default_rng(35034)
    reps = 100000
    scenarios = []
    for q in (0.05, 0.10, 0.15, 0.20, 0.25):
        scenarios.append(simulate(0.0, q, n, margin, reps, rng))
    for delta, q in (
        (0.025, 0.025),
        (0.025, 0.08),
        (0.05, 0.08),
        (0.05, 0.15),
        (0.07, 0.10),
        (0.07, 0.15),
        (0.07, 0.20),
    ):
        scenarios.append(simulate(delta, q, n, margin, reps, rng))
    # Round-1 paired facts, for the write-up of why n=500. Not 4B data.
    round1 = {
        "n": 40,
        "first_sample_accuracy": 0.55,
        "vote5_accuracy": 0.575,
        "mean_of_samples_accuracy": 0.505,
        "paired_vote5_minus_first": 0.025,
        "headline_vote5_minus_mean_of_samples": 0.07,
        "discordant_pairs_first_vs_vote5": 1,
        "q_first_vs_vote5": 0.025,
        "note": "The +7 point headline compares the vote to the average of five samples, not to one pre-specified sample. The paired first-vs-vote contrast was +2.5 points, 1 discordant question of 40.",
    }
    out = {
        "n": n,
        "margin": margin,
        "reps": reps,
        "approximation": "Delta-hat ~ Normal(delta, (q-delta^2)/n). Intervals are normal approximations to the paired percentile bootstrap. PROVEN if the 95% lower bound exceeds 0. Otherwise DISPROVEN if the 90% interval lies inside +/- margin (TOST) or the 95% upper bound is below +margin. Else indeterminate.",
        "round1_paired_facts": round1,
        "scenarios": scenarios,
    }
    dest = ROOT / "results" / "power.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"{'delta':>7} {'q':>6} {'SE':>8} {'half':>8} {'PROVEN':>8} {'DISPROVEN':>10} {'GRAY':>8}")
    for row in scenarios:
        print(
            f"{row['delta']:7.3f} {row['q']:6.3f} {row['se']:8.4f} {row['ci95_halfwidth']:8.4f} "
            f"{row['proven']:8.3f} {row['disproven']:10.3f} {row['indeterminate']:8.3f}"
        )


if __name__ == "__main__":
    main()
