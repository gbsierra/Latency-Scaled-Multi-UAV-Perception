"""Persist scored result rows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class ResultFormatError(ValueError):
    """Raised when a result file contains invalid JSONL rows."""


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write scored result rows as newline-delimited JSON."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("w", encoding="utf-8") as file:
        for row in rows:
            # Preserve row fields exactly; do not invent benchmark outputs here.
            file.write(json.dumps(dict(row), sort_keys=True))
            file.write("\n")


def append_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append scored result rows as newline-delimited JSON."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), sort_keys=True))
            file.write("\n")


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read newline-delimited JSON result rows."""
    result_path = Path(path)
    with result_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ResultFormatError(f"{result_path}: invalid JSON on line {line_number}") from error

            if not isinstance(row, dict):
                raise ResultFormatError(f"{result_path}: line {line_number} is not a JSON object")

            yield row
