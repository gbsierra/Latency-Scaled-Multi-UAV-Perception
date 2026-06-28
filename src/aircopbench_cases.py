"""Load AirCopBench VQA test metadata without transforming cases."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


DEFAULT_RAW_TEST_DIR = Path("data/raw/aircopbench/test")

# These fields are the minimum contract needed for answer scoring.
REQUIRED_FIELDS = {
    "question_id",
    "question_type",
    "question",
    "options",
    "correct_answer",
    "uav_paths",
}


class AirCopBenchCaseError(ValueError):
    """Raised when an AirCopBench case file does not match the expected schema."""


def source_group_from_file(path: Path) -> str:
    """Return Real2, Sim3, Sim5, or Sim6 from filenames like Sim5_VQA_test.json."""
    return path.name.removesuffix("_VQA_test.json")


def validate_case(case: dict[str, Any], source_file: Path, index: int) -> None:
    # Fail before inference if the downloaded metadata is incomplete.
    missing = sorted(REQUIRED_FIELDS - case.keys())
    if missing:
        case_id = case.get("question_id", f"index {index}")
        raise AirCopBenchCaseError(
            f"{source_file}: case {case_id} missing required fields: {', '.join(missing)}"
        )

    options = case["options"]
    if not isinstance(options, dict):
        raise AirCopBenchCaseError(
            f"{source_file}: case {case['question_id']} options must be an object"
        )

    missing_options = [letter for letter in ("A", "B", "C", "D") if letter not in options]
    if missing_options:
        raise AirCopBenchCaseError(
            f"{source_file}: case {case['question_id']} missing options: {', '.join(missing_options)}"
        )

    if case["correct_answer"] not in options:
        raise AirCopBenchCaseError(
            f"{source_file}: case {case['question_id']} correct_answer is not one of options"
        )


def load_cases(raw_test_dir: Path | str = DEFAULT_RAW_TEST_DIR) -> Iterator[dict[str, Any]]:
    """Yield AirCopBench test cases with only source metadata added."""
    raw_test_path = Path(raw_test_dir)
    # Keep file order stable so result ordering is reproducible.
    for source_file in sorted(raw_test_path.glob("*_VQA_test.json")):
        with source_file.open(encoding="utf-8") as file:
            cases = json.load(file)

        if not isinstance(cases, list):
            raise AirCopBenchCaseError(f"{source_file}: expected a JSON list")

        source_group = source_group_from_file(source_file)
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                raise AirCopBenchCaseError(f"{source_file}: case index {index} must be an object")

            validate_case(case, source_file, index)
            yield {
                **case,
                # Add provenance without rewriting the upstream case schema.
                "source_group": source_group,
                "source_split": "test",
                "source_file": str(source_file),
            }
