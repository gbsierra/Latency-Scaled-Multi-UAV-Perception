import csv

import pytest

from src.result_summary import (
    ResultSummaryError,
    format_summary_table,
    summarize_results,
    write_csv,
)


def summary_rows() -> list[dict]:
    return [
        {
            "workflow": "global_single",
            "question_id": "case-1",
            "latency_ms": 2500,
            "correct": True,
        },
        {
            "workflow": "global_single",
            "question_id": "case-2",
            "latency_ms": 3500,
            "correct": False,
        },
        {
            "workflow": "parallel_uav_fusion",
            "question_id": "case-1",
            "latency_ms": 3400,
            "correct": True,
        },
        {
            "workflow": "parallel_uav_fusion",
            "question_id": "case-2",
            "latency_ms": 4200,
            "correct": True,
        },
    ]


def test_summarize_results_recomputes_actionability_by_deadline():
    summary = summarize_results(summary_rows(), deadlines_ms=(3000, 4000))

    global_3000 = next(row for row in summary if row["workflow"] == "global_single" and row["deadline_ms"] == 3000)
    fusion_4000 = next(row for row in summary if row["workflow"] == "parallel_uav_fusion" and row["deadline_ms"] == 4000)

    assert global_3000["n"] == 2
    assert global_3000["accuracy"] == 0.5
    assert global_3000["median_latency_ms"] == 3000.0
    assert global_3000["stale_rate"] == 0.5
    assert global_3000["actionable_accuracy"] == 0.5
    assert fusion_4000["accuracy"] == 1.0
    assert fusion_4000["stale_rate"] == 0.5
    assert fusion_4000["actionable_accuracy"] == 0.5


def test_summarize_results_ignores_error_rows():
    summary = summarize_results(
        [
            {"workflow": "global_single", "latency_ms": 100, "correct": True},
            {"workflow": "global_single", "error": True, "error_message": "failed"},
        ],
        deadlines_ms=(400,),
    )

    assert summary[0]["n"] == 1
    assert summary[0]["accuracy"] == 1.0


def test_summarize_results_fails_without_completed_rows():
    with pytest.raises(ResultSummaryError, match="no completed"):
        summarize_results([{"workflow": "global_single", "error": True}])


def test_format_summary_table_includes_core_metrics():
    table = format_summary_table(summarize_results(summary_rows(), deadlines_ms=(3000,)))

    assert "workflow deadline_ms n accuracy median_latency_ms stale_rate actionable_accuracy" in table
    assert "global_single 3000 2 0.500 3000.0 0.500 0.500" in table


def test_write_csv_persists_summary_rows(tmp_path):
    path = tmp_path / "summary.csv"
    rows = summarize_results(summary_rows(), deadlines_ms=(3000,))

    write_csv(path, rows)

    with path.open(encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))

    assert csv_rows[0]["workflow"] == "global_single"
    assert csv_rows[0]["deadline_ms"] == "3000"
