"""Score AirCopBench answer results with deadline-aware metrics."""

from __future__ import annotations

import re
from typing import Any


ANSWER_CHOICES = {"A", "B", "C", "D"}

# Keep AirCopBench task grouping in one shared place for later aggregates.
CATEGORY_MAP = {
    "Scene Understanding": [
        "Scene Description",
        "Scene Comparison",
        "Observing Posture",
    ],
    "Object Understanding": [
        "Object Recognition",
        "Object Counting",
        "Object Grounding",
        "Object Matching",
    ],
    "Perception Assessment": [
        "Quality Assessment",
        "Usability Assessment",
        "Causal Assessment",
    ],
    "Collaborative Decision": [
        "When to Collaborate",
        "What to Collaborate",
        "Who to Collaborate",
        "Why to Collaborate",
    ],
}


ANSWER_PATTERNS = [
    re.compile(r'"ANSWER"\s*:\s*"([ABCD])"'),
    re.compile(r"'ANSWER'\s*:\s*'([ABCD])'"),
    re.compile(r"\bANSWER\s*:\s*([ABCD])\b"),
    re.compile(r"\bANSWER\s*-\s*([ABCD])\b"),
    re.compile(r"\bTHE ANSWER IS\s*([ABCD])\b"),
    re.compile(r"\(([ABCD])\)"),
    re.compile(r"^([ABCD])\s*:", re.MULTILINE),
    re.compile(r"\bOPTION\s*([ABCD])\b"),
    # Final fallback only accepts an answer letter at the end, not any stray A-D.
    re.compile(r"\b([ABCD])\s*\.?\s*$"),
]


def parse_answer(raw_response: str | None) -> str | None:
    """Extract an A/B/C/D answer from a model response, or None if invalid."""
    if raw_response is None:
        return None

    text = str(raw_response).strip().upper()
    if text in ANSWER_CHOICES:
        return text

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)

    return None


def is_correct(parsed_answer: str | None, correct_answer: str) -> bool:
    return parsed_answer == correct_answer


def is_stale(latency_ms: int | float, deadline_ms: int | float) -> bool:
    return latency_ms > deadline_ms


def score_answer(
    raw_answer: str | None,
    correct_answer: str,
    latency_ms: int | float,
    deadline_ms: int | float,
) -> dict[str, Any]:
    """Score a raw answer without requiring AirCopBench case metadata."""
    # Generic scoring stays independent from dataset-specific case fields.
    parsed_answer = parse_answer(raw_answer)
    parsed_correct_answer = parse_answer(correct_answer)
    parse_failed = parsed_answer is None
    correct = parsed_answer is not None and parsed_answer == parsed_correct_answer
    stale = is_stale(latency_ms, deadline_ms)

    return {
        "raw_answer": raw_answer,
        "parsed_answer": parsed_answer,
        "correct_answer": parsed_correct_answer,
        "parse_failed": parse_failed,
        "latency_ms": latency_ms,
        "deadline_ms": deadline_ms,
        "correct": correct,
        "stale": stale,
        "actionable": correct and not stale,
    }


def score_result(
    case: dict[str, Any],
    raw_response: str | None,
    latency_ms: int | float,
    deadline_ms: int | float,
) -> dict[str, Any]:
    """Return answer correctness plus stale/actionable deadline metrics."""
    # AirCopBench wrapper adds case provenance around the generic score.
    answer_score = score_answer(raw_response, case["correct_answer"], latency_ms, deadline_ms)

    return {
        "question_id": case["question_id"],
        "source_group": case.get("source_group"),
        "source_split": case.get("source_split"),
        "question_type": case["question_type"],
        **answer_score,
    }
