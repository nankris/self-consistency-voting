"""Synthetic tests for the parser and the locked verdict rule. No model outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import decide, intervals, mcnemar  # noqa: E402
from parser import canon, parse_answer, parse_strict, vote  # noqa: E402

PROTOCOL = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text())


def check(name: str, cond: bool) -> None:
    if not cond:
        raise SystemExit(f"FAIL {name}")
    print("ok", name)


def test_parser() -> None:
    check("plain", parse_answer("Step.\nAnswer: 42") == "42")
    check("final", parse_answer("### Final Answer: 2") == "2")
    check("bold", parse_answer("**Answer: 11**") == "11")
    check("commas", parse_answer("Answer: 1,000") == "1000")
    check("dollar", parse_answer("Final Answer: $18") == "18")
    check("boxed label", parse_answer("Final Answer: \\boxed{2}") == "2")
    check(
        "boxed after blank final",
        parse_answer("### Final Answer:\n\n$$\n\\boxed{3}\n$$") == "3",
    )
    check("boxed only", parse_answer("Therefore\n\\boxed{60}") == "60")
    check("last label", parse_answer("Answer: 4\nWait.\nAnswer: 9") == "9")
    check(
        "walk back",
        parse_answer("Answer: 5\nFinal Answer: the result above") == "5",
    )
    check("no marker", parse_answer("The total is 33 then 2.") is None)
    check("scientific not truncated", parse_answer("Answer: 1e3") is None)
    check("negative", parse_answer("Answer: -3") == "-3")
    check("decimal", parse_answer("Answer: 2.50") == "2.5")
    check("think stripped", parse_answer("<think>Answer: 1</think>\nAnswer: 8") == "8")
    check("strict boxed", parse_strict("\\boxed{4}") is None)
    check("strict label", parse_strict("Answer: 4") == "4")
    check("canon gold", canon("1,000") == canon("1000.0") == "1000")
    parsed = ["2", "2", None, "3", "2"]
    check("vote5", vote(parsed, 5) == "2")
    check("tie earliest", vote(["3", "4", "4", "3"], 4) == "3")
    check("missing", vote([None, None], 2) is None)
    check("prefix", vote(["9", "1", "1", "1"], 1) == "9")


def test_decide() -> None:
    margin = float(PROTOCOL["margin"])
    reps = int(PROTOCOL["bootstrap_reps"])
    seed = int(PROTOCOL["bootstrap_seed"])
    # 40 one-way fixes out of 500: clear superiority.
    d = np.array([1] * 40 + [0] * 460, dtype=float)
    est, l95, u95, l90, u90 = intervals(d, reps, seed)
    verdict, _ = decide(l95, u95, l90, u90, margin)
    check("proven case", verdict == "PROVEN" and l95 > 0)
    # All ties: exact zero, upper bound below the margin.
    d0 = np.zeros(500)
    est, l95, u95, l90, u90 = intervals(d0, reps, seed)
    verdict, reason = decide(l95, u95, l90, u90, margin)
    check("zero disproven", verdict == "DISPROVEN" and u95 < margin)
    check("zero estimate", est == 0)
    # Balanced discordance, net zero, q=0.12. Upper bound should be under 5 points.
    d_bal = np.array([1] * 30 + [-1] * 30 + [0] * 440, dtype=float)
    est, l95, u95, l90, u90 = intervals(d_bal, reps, seed)
    verdict, _ = decide(l95, u95, l90, u90, margin)
    check("balanced disproven", verdict == "DISPROVEN" and l95 <= 0)
    # Medium noisy effect: about +3 points with wide discordance. Indeterminate or a real call.
    # b=48, c=33 -> delta=0.03, q=0.162. Expect L<=0 and U>~0.05, so indeterminate
    # if the bootstrap matches the normal approximation. Allow either DISPROVEN or
    # INDETERMINATE, but not PROVEN: this fixture must not count as superiority.
    d_gray = np.array([1] * 48 + [-1] * 33 + [0] * 419, dtype=float)
    est, l95, u95, l90, u90 = intervals(d_gray, reps, seed)
    verdict, _ = decide(l95, u95, l90, u90, margin)
    check("gray not proven", verdict != "PROVEN")
    check("gray bounds", l95 <= 0)
    b, c, p = mcnemar(np.array([0] * 10 + [1] * 10), np.array([1] * 10 + [1] * 10))
    check("mcnemar", b == 10 and c == 0 and p < 0.05)
    b0, c0, p0 = mcnemar(np.ones(20), np.ones(20))
    check("mcnemar ties", b0 == 0 and c0 == 0 and p0 == 1.0)
    print("gray fixture", verdict, round(est, 4), round(l95, 4), round(u95, 4))


if __name__ == "__main__":
    test_parser()
    test_decide()
    print("ALL PASSED")
