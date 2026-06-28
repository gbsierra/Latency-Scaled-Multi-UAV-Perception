"""Persist scored result rows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


class ResultFormatError(ValueError):
    """Raised when a result file contains invalid JSONL rows."""


def to_jsonable(value: Any) -> Any:
    """Convert common SDK and Python objects into JSON-serializable values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    # Pydantic-style SDK models often expose a dict conversion method.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return to_jsonable(as_dict())

    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))

    return str(value)


def jsonl_line(row: Mapping[str, Any]) -> str:
    """Serialize one result row for JSONL persistence."""
    return json.dumps(to_jsonable(row), sort_keys=True)


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write scored result rows as newline-delimited JSON."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("w", encoding="utf-8") as file:
        for row in rows:
            # Preserve row fields exactly; do not invent benchmark outputs here.
            file.write(jsonl_line(row))
            file.write("\n")


def append_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append scored result rows as newline-delimited JSON."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with result_path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(jsonl_line(row))
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
