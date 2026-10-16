"""Final-answer parser for GSM8K generations.

Locked before scoring. Explicit final-answer labels and boxed numbers only.
No last-number-in-the-trace fallback: intermediate arithmetic must not count.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)(?![0-9A-Za-z])")
_LABEL = re.compile(r"(?:final\s+)?answer\s*:", re.IGNORECASE)
_BOXED = re.compile(
    r"\\boxed\s*\{\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)(?![0-9A-Za-z])"
)
_LEAD = re.compile(r"^(?:\s|\*|\$|`|\}|\\\[|\\\(|\\boxed\s*\{)+")


def canon(value) -> str | None:
    """Canonical fixed-point string. 1,000, 1000, and 1000.0 are the same key."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if text == "":
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in ("", "-"):
        rendered = "0"
    return rendered


def _number_at(text: str) -> str | None:
    cleaned = _LEAD.sub("", text)
    match = _NUMBER.match(cleaned)
    if not match:
        return None
    return canon(match.group(1))


def _after_think(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def parse_answer(text: str, allow_boxed: bool = True) -> str | None:
    """Return the canonical final number, or None if no explicit final answer.

    Labels are tried from last to first. A label whose tail does not start
    with a number (after Markdown/LaTeX wrappers) is skipped. If no label
    parses and allow_boxed is set, the last \\boxed{number} is used.
    """
    if not text:
        return None
    body = _after_think(text)
    labels = list(_LABEL.finditer(body))
    for label in reversed(labels):
        parsed = _number_at(body[label.end() :])
        if parsed is not None:
            return parsed
    if allow_boxed:
        boxes = list(_BOXED.finditer(body))
        if boxes:
            return canon(boxes[-1].group(1))
    return None


def parse_strict(text: str) -> str | None:
    """Sensitivity parser: explicit Answer labels only, no boxed-only fallback."""
    return parse_answer(text, allow_boxed=False)


def vote(parsed: list, k: int):
    """Plurality of the first k parses. Ties go to the earliest sample.

    Unparseable samples are wrong and cannot win a tie. If nothing parsed,
    the vote is missing (scored incorrect).
    """
    window = list(parsed[:k])
    counts = {}
    for value in window:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    for value in window:
        if value is not None and counts[value] == best:
            return value
    return None
