"""Summarize workflow result JSONL files across deadline thresholds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from src.results import read_jsonl, write_jsonl


DEFAULT_DEADLINES_MS = (3000, 4000, 5000)


class ResultSummaryError(ValueError):
    """Raised when result rows cannot be summarized."""


def _completed_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if not row.get("error")]


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def summarize_results(
    rows: Iterable[Mapping[str, Any]],
    deadlines_ms: Sequence[int] = DEFAULT_DEADLINES_MS,
) -> list[dict[str, Any]]:
    """Return aggregate metrics by workflow and deadline."""
    completed = _completed_rows(rows)
    if not completed:
        raise ResultSummaryError("no completed result rows to summarize")

    by_workflow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        workflow = row.get("workflow")
        if not isinstance(workflow, str) or not workflow:
            raise ResultSummaryError("result row missing workflow")
        if not isinstance(row.get("latency_ms"), int | float):
            raise ResultSummaryError(f"{row.get('question_id')}: result row missing latency_ms")
        by_workflow[workflow].append(row)

    summaries: list[dict[str, Any]] = []
    for workflow in sorted(by_workflow):
        workflow_rows = by_workflow[workflow]
        total = len(workflow_rows)
        correct_count = sum(1 for row in workflow_rows if row.get("correct") is True)
        latencies = [float(row["latency_ms"]) for row in workflow_rows]
        for deadline_ms in deadlines_ms:
            stale_count = sum(1 for latency in latencies if latency > deadline_ms)
            actionable_count = sum(
                1 for row in workflow_rows if row.get("correct") is True and float(row["latency_ms"]) <= deadline_ms
            )
            summaries.append(
                {
                    "workflow": workflow,
                    "deadline_ms": deadline_ms,
                    "n": total,
                    "accuracy": _rate(correct_count, total),
                    "median_latency_ms": median(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "stale_rate": _rate(stale_count, total),
                    "actionable_accuracy": _rate(actionable_count, total),
                }
            )

    return summaries


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write summary rows to CSV."""
    if not rows:
        raise ResultSummaryError("no summary rows to write")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_summary_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a compact terminal table for summary rows."""
    lines = ["workflow deadline_ms n accuracy median_latency_ms stale_rate actionable_accuracy"]
    for row in rows:
        lines.append(
            f"{row['workflow']} {row['deadline_ms']} {row['n']} "
            f"{row['accuracy']:.3f} {row['median_latency_ms']:.1f} "
            f"{row['stale_rate']:.3f} {row['actionable_accuracy']:.3f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize AirCopBench workflow JSONL results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--deadline-ms", action="append", type=int, default=[])
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--jsonl-output", "--json-output", dest="jsonl_output", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    deadlines = tuple(args.deadline_ms) if args.deadline_ms else DEFAULT_DEADLINES_MS
    summaries = summarize_results(read_jsonl(args.input), deadlines_ms=deadlines)

    if args.csv_output:
        write_csv(args.csv_output, summaries)
    if args.jsonl_output:
        write_jsonl(args.jsonl_output, summaries)

    print(format_summary_table(summaries))


if __name__ == "__main__":
    main()
