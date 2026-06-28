"""Load selected benchmark cases and resolve their UAV image files."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from src.aircopbench_cases import AirCopBenchCaseError, validate_case


DEFAULT_SELECTED_CASES_PATH = Path("data/selected/aircopbench/selection/core_30_cases.json")
DEFAULT_SELECTED_DATASET_ROOT = Path("data/selected/aircopbench")


class BenchmarkCaseError(ValueError):
    """Raised when the selected benchmark slice is invalid or incomplete."""


def load_selected_cases(path: Path | str = DEFAULT_SELECTED_CASES_PATH) -> Iterator[dict[str, Any]]:
    """Yield validated cases from a fixed selected-case JSON file."""
    case_path = Path(path)
    with case_path.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise BenchmarkCaseError(f"{case_path}: expected a JSON list")

    # Reuse the raw AirCopBench contract so the selected slice cannot drift.
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise BenchmarkCaseError(f"{case_path}: case index {index} must be an object")

        try:
            validate_case(case, case_path, index)
        except AirCopBenchCaseError as error:
            raise BenchmarkCaseError(str(error)) from error

        yield case


def resolve_uav_image_paths(
    case: Mapping[str, Any],
    dataset_root: Path | str = DEFAULT_SELECTED_DATASET_ROOT,
) -> dict[str, Path]:
    """Return existing image paths for every UAV view in a selected case."""
    uav_paths = case.get("uav_paths")
    case_id = case.get("question_id", "unknown case")

    if not isinstance(uav_paths, Mapping) or not uav_paths:
        raise BenchmarkCaseError(f"{case_id}: uav_paths must be a non-empty object")

    root = Path(dataset_root)
    resolved_paths: dict[str, Path] = {}
    missing_paths: list[str] = []

    # Resolve every view before model calls so missing images fail early.
    for uav_id, relative_path in uav_paths.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise BenchmarkCaseError(f"{case_id}: {uav_id} image path must be a non-empty string")

        image_path = root / relative_path
        if not image_path.is_file():
            missing_paths.append(str(image_path))
        resolved_paths[str(uav_id)] = image_path

    if missing_paths:
        raise BenchmarkCaseError(f"{case_id}: missing UAV images: {', '.join(missing_paths)}")

    return resolved_paths
