import pytest

from src.results import read_jsonl
from src.workflow_runner import (
    BOTH_WORKFLOWS,
    WorkflowRunnerError,
    build_error_row,
    expand_workflow_names,
    run_workflow_batch,
    select_cases,
    summarize_row,
)
from src.workflows import GLOBAL_SINGLE_WORKFLOW, PARALLEL_UAV_FUSION_WORKFLOW


def runner_cases() -> list[dict]:
    return [
        {
            "question_id": "real-1",
            "source_group": "Real2",
            "source_split": "test",
            "question_type": "What to Collaborate",
            "correct_answer": "A",
        },
        {
            "question_id": "sim5-1",
            "source_group": "Sim5",
            "source_split": "test",
            "question_type": "Object Matching",
            "correct_answer": "B",
        },
        {
            "question_id": "sim3-1",
            "source_group": "Sim3",
            "source_split": "test",
            "question_type": "Object Matching",
            "correct_answer": "C",
        },
    ]


def test_expand_workflow_names_expands_both_in_stable_order():
    assert expand_workflow_names(BOTH_WORKFLOWS) == (
        GLOBAL_SINGLE_WORKFLOW,
        PARALLEL_UAV_FUSION_WORKFLOW,
    )
    assert expand_workflow_names(GLOBAL_SINGLE_WORKFLOW) == (GLOBAL_SINGLE_WORKFLOW,)


def test_select_cases_preserves_explicit_question_id_order():
    selected = select_cases(
        runner_cases(),
        question_ids=("sim5-1", "real-1"),
    )

    assert [case["question_id"] for case in selected] == ["sim5-1", "real-1"]


def test_select_cases_filters_by_source_question_type_and_limit():
    selected = select_cases(
        runner_cases(),
        all_cases=True,
        source_groups=("Sim5", "Sim3"),
        question_types=("Object Matching",),
        limit=1,
    )

    assert [case["question_id"] for case in selected] == ["sim5-1"]


def test_select_cases_requires_selection_criteria():
    with pytest.raises(WorkflowRunnerError, match="select cases"):
        select_cases(runner_cases())


def test_select_cases_fails_for_missing_question_id():
    with pytest.raises(WorkflowRunnerError, match="selected case not found: missing"):
        select_cases(runner_cases(), question_ids=("missing",))


def test_run_workflow_batch_writes_rows_incrementally(tmp_path):
    output_path = tmp_path / "results" / "rows.jsonl"

    def fake_workflow(case, **kwargs):
        return {
            "experiment_id": kwargs["experiment_id"],
            "workflow": GLOBAL_SINGLE_WORKFLOW,
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "question_id": case["question_id"],
            "latency_ms": 10.0,
            "parsed_answer": "A",
            "correct": True,
            "actionable": True,
        }

    rows = run_workflow_batch(
        [runner_cases()[0]],
        [GLOBAL_SINGLE_WORKFLOW],
        output_path,
        provider="local",
        model="test-model",
        deadline_ms=400,
        experiment_id="exp-1",
        append=False,
        workflow_functions={GLOBAL_SINGLE_WORKFLOW: fake_workflow},
    )

    assert rows == list(read_jsonl(output_path))
    assert rows[0]["experiment_id"] == "exp-1"
    assert rows[0]["question_id"] == "real-1"


def test_run_workflow_batch_can_continue_with_error_row(tmp_path):
    output_path = tmp_path / "rows.jsonl"

    def failing_workflow(_case, **_kwargs):
        raise RuntimeError("provider failed")

    rows = run_workflow_batch(
        [runner_cases()[0]],
        [GLOBAL_SINGLE_WORKFLOW],
        output_path,
        provider="local",
        deadline_ms=400,
        continue_on_error=True,
        workflow_functions={GLOBAL_SINGLE_WORKFLOW: failing_workflow},
    )

    assert rows == list(read_jsonl(output_path))
    assert rows[0]["error"] is True
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["question_id"] == "real-1"


def test_run_workflow_batch_raises_without_continue_on_error(tmp_path):
    def failing_workflow(_case, **_kwargs):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        run_workflow_batch(
            [runner_cases()[0]],
            [GLOBAL_SINGLE_WORKFLOW],
            tmp_path / "rows.jsonl",
            provider="local",
            workflow_functions={GLOBAL_SINGLE_WORKFLOW: failing_workflow},
        )


def test_summarize_row_reports_success_and_error():
    success = {
        "workflow": GLOBAL_SINGLE_WORKFLOW,
        "question_id": "case-1",
        "latency_ms": 12.5,
        "parsed_answer": "A",
        "correct": True,
        "actionable": False,
    }
    error = build_error_row(
        runner_cases()[0],
        GLOBAL_SINGLE_WORKFLOW,
        RuntimeError("bad"),
        provider="local",
        model=None,
        deadline_ms=400,
        run_index=1,
        experiment_id="exp",
    )

    assert summarize_row(success) == "global_single case-1 latency_ms=12.5 parsed=A correct=True actionable=False"
    assert summarize_row(error).startswith("ERROR global_single real-1 RuntimeError: bad")
