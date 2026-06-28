"""Command-line runner for benchmark-compatible AirCopBench workflows."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.benchmark_cases import (
    DEFAULT_SELECTED_CASES_PATH,
    DEFAULT_SELECTED_DATASET_ROOT,
    load_selected_cases,
)
from src.cerebras_models import CEREBRAS_PROVIDER
from src.results import append_jsonl, write_jsonl
from src.workflows import (
    DEFAULT_EXPERIMENT_ID,
    GLOBAL_SINGLE_WORKFLOW,
    PARALLEL_UAV_FUSION_WORKFLOW,
    run_global_single,
    run_parallel_uav_fusion,
)


BOTH_WORKFLOWS = "both"
DEFAULT_OUTPUT_PATH = Path("results/workflow_iteration_v0.jsonl")
WORKFLOW_ORDER = (GLOBAL_SINGLE_WORKFLOW, PARALLEL_UAV_FUSION_WORKFLOW)

WorkflowFunction = Callable[..., dict[str, Any]]
WorkflowRegistry = Mapping[str, WorkflowFunction]

DEFAULT_WORKFLOW_FUNCTIONS: dict[str, WorkflowFunction] = {
    GLOBAL_SINGLE_WORKFLOW: run_global_single,
    PARALLEL_UAV_FUSION_WORKFLOW: run_parallel_uav_fusion,
}


class WorkflowRunnerError(ValueError):
    """Raised when runner inputs cannot define a valid benchmark run."""


def expand_workflow_names(workflow: str) -> tuple[str, ...]:
    """Return concrete workflow names from a CLI workflow selector."""
    if workflow == BOTH_WORKFLOWS:
        return WORKFLOW_ORDER

    if workflow not in WORKFLOW_ORDER:
        raise WorkflowRunnerError(f"unsupported workflow: {workflow}")

    return (workflow,)


def select_cases(
    cases: Iterable[Mapping[str, Any]],
    question_ids: Sequence[str] = (),
    all_cases: bool = False,
    source_groups: Sequence[str] = (),
    question_types: Sequence[str] = (),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select benchmark cases by explicit IDs or metadata filters."""
    if limit is not None and limit <= 0:
        raise WorkflowRunnerError("limit must be positive")

    case_list = [dict(case) for case in cases]
    if question_ids:
        by_id = {case["question_id"]: case for case in case_list}
        missing = [question_id for question_id in question_ids if question_id not in by_id]
        if missing:
            raise WorkflowRunnerError(f"selected case not found: {', '.join(missing)}")
        selected = [by_id[question_id] for question_id in question_ids]
    else:
        if not all_cases and not source_groups and not question_types:
            raise WorkflowRunnerError("select cases with --question-id, --all, --source-group, or --question-type")
        selected = case_list

    if source_groups:
        allowed_sources = set(source_groups)
        selected = [case for case in selected if case.get("source_group") in allowed_sources]

    if question_types:
        allowed_types = set(question_types)
        selected = [case for case in selected if case.get("question_type") in allowed_types]

    if limit is not None:
        selected = selected[:limit]

    if not selected:
        raise WorkflowRunnerError("case selection matched no cases")

    return selected


def build_error_row(
    case: Mapping[str, Any],
    workflow: str,
    error: Exception,
    provider: str,
    model: str | None,
    deadline_ms: int,
    run_index: int,
    experiment_id: str,
) -> dict[str, Any]:
    """Return a result-like row for failed workflow calls."""
    return {
        "experiment_id": experiment_id,
        "workflow": workflow,
        "provider": provider,
        "model": model,
        "run_index": run_index,
        "question_id": case.get("question_id"),
        "source_group": case.get("source_group"),
        "source_split": case.get("source_split"),
        "question_type": case.get("question_type"),
        "deadline_ms": deadline_ms,
        "error": True,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def run_case_workflow(
    case: Mapping[str, Any],
    workflow: str,
    provider: str,
    model: str | None,
    deadline_ms: int,
    run_index: int,
    experiment_id: str,
    dataset_root: Path | str,
    workflow_functions: WorkflowRegistry = DEFAULT_WORKFLOW_FUNCTIONS,
) -> dict[str, Any]:
    """Run one workflow for one case."""
    try:
        workflow_function = workflow_functions[workflow]
    except KeyError as error:
        raise WorkflowRunnerError(f"unsupported workflow: {workflow}") from error

    return workflow_function(
        case,
        provider=provider,
        model=model,
        deadline_ms=deadline_ms,
        run_index=run_index,
        experiment_id=experiment_id,
        dataset_root=dataset_root,
    )


def run_workflow_batch(
    cases: Sequence[Mapping[str, Any]],
    workflows: Sequence[str],
    output_path: Path | str,
    provider: str,
    model: str | None = None,
    deadline_ms: int = 400,
    run_index: int = 1,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    dataset_root: Path | str = DEFAULT_SELECTED_DATASET_ROOT,
    append: bool = True,
    continue_on_error: bool = False,
    workflow_functions: WorkflowRegistry = DEFAULT_WORKFLOW_FUNCTIONS,
) -> list[dict[str, Any]]:
    """Run selected cases and persist each row as soon as it finishes."""
    if not append:
        write_jsonl(output_path, [])

    rows: list[dict[str, Any]] = []
    for case in cases:
        for workflow in workflows:
            try:
                row = run_case_workflow(
                    case,
                    workflow,
                    provider=provider,
                    model=model,
                    deadline_ms=deadline_ms,
                    run_index=run_index,
                    experiment_id=experiment_id,
                    dataset_root=dataset_root,
                    workflow_functions=workflow_functions,
                )
            except Exception as error:
                if not continue_on_error:
                    raise
                row = build_error_row(
                    case,
                    workflow,
                    error,
                    provider=provider,
                    model=model,
                    deadline_ms=deadline_ms,
                    run_index=run_index,
                    experiment_id=experiment_id,
                )

            rows.append(row)
            append_jsonl(output_path, [row])

    return rows


def summarize_row(row: Mapping[str, Any]) -> str:
    """Return one concise terminal summary line for a workflow result."""
    if row.get("error"):
        return (
            f"ERROR {row.get('workflow')} {row.get('question_id')} "
            f"{row.get('error_type')}: {row.get('error_message')}"
        )

    return (
        f"{row['workflow']} {row['question_id']} "
        f"latency_ms={row['latency_ms']:.1f} parsed={row['parsed_answer']} "
        f"correct={row['correct']} actionable={row['actionable']}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser separately so tests can exercise runner behavior."""
    parser = argparse.ArgumentParser(description="Run AirCopBench workflow benchmarks.")
    parser.add_argument("--provider", default=CEREBRAS_PROVIDER)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--workflow",
        choices=(GLOBAL_SINGLE_WORKFLOW, PARALLEL_UAV_FUSION_WORKFLOW, BOTH_WORKFLOWS),
        default=BOTH_WORKFLOWS,
    )
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--all", action="store_true", dest="all_cases")
    parser.add_argument("--source-group", action="append", default=[])
    parser.add_argument("--question-type", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deadline-ms", type=int, default=400)
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--cases-path", default=str(DEFAULT_SELECTED_CASES_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_SELECTED_DATASET_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    workflows = expand_workflow_names(args.workflow)
    cases = select_cases(
        load_selected_cases(args.cases_path),
        question_ids=args.question_id,
        all_cases=args.all_cases,
        source_groups=args.source_group,
        question_types=args.question_type,
        limit=args.limit,
    )

    if args.dry_run:
        for case in cases:
            print(
                f"{case['question_id']} source={case.get('source_group')} "
                f"type={case.get('question_type')} workflows={','.join(workflows)}"
            )
        return

    rows = run_workflow_batch(
        cases,
        workflows,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        deadline_ms=args.deadline_ms,
        run_index=args.run_index,
        experiment_id=args.experiment_id,
        dataset_root=args.dataset_root,
        append=not args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    for row in rows:
        print(summarize_row(row))


if __name__ == "__main__":
    main()
