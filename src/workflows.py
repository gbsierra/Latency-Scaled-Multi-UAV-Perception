"""Run benchmark-compatible AirCopBench model workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

from src.benchmark_cases import DEFAULT_SELECTED_DATASET_ROOT, resolve_uav_image_paths
from src.cerebras_models import CEREBRAS_PROVIDER, GEMMA_4_31B_MODEL, call_cerebras
from src.github_models import GITHUB_MODELS_PROVIDER, PHI_4_MULTIMODAL_MODEL, call_github_models
from src.model_clients import (
    ModelCall,
    ModelRequest,
    ModelResponse,
    call_model,
    provider_reported_latency_ms,
)
from src.prompts import (
    FUSION_COMPARER_PROMPT_ID,
    GLOBAL_SINGLE_PROMPT_ID,
    PER_UAV_OBSERVATION_PROMPT_ID,
    build_fusion_prompt,
    build_global_single_prompt,
    build_per_uav_observation_prompt,
)
from src.scoring import score_result


GLOBAL_SINGLE_WORKFLOW = "global_single"
PARALLEL_UAV_FUSION_WORKFLOW = "parallel_uav_fusion"
DEFAULT_EXPERIMENT_ID = "workflow_iteration_v0"

PROVIDER_DEFAULT_MODELS = {
    GITHUB_MODELS_PROVIDER: PHI_4_MULTIMODAL_MODEL,
    CEREBRAS_PROVIDER: GEMMA_4_31B_MODEL,
}
PROVIDER_CALLS: dict[str, ModelCall] = {
    GITHUB_MODELS_PROVIDER: call_github_models,
    CEREBRAS_PROVIDER: call_cerebras,
}

Timer = Callable[[], float]


class WorkflowError(ValueError):
    """Raised when a workflow cannot be constructed or executed."""


def _model_for_provider(provider: str, model: str | None) -> str:
    if model:
        return model

    try:
        return PROVIDER_DEFAULT_MODELS[provider]
    except KeyError as error:
        raise WorkflowError(f"unsupported provider: {provider}") from error


def _model_call_for_provider(provider: str, model_call: ModelCall | None) -> ModelCall:
    if model_call is not None:
        return model_call

    try:
        return PROVIDER_CALLS[provider]
    except KeyError as error:
        raise WorkflowError(f"unsupported provider: {provider}") from error


def _provider_timing(provider_metadata: Mapping[str, Any]) -> dict[str, Any]:
    client_timing = provider_metadata.get("client_timing")
    if not isinstance(client_timing, Mapping):
        client_timing = {}

    return {
        "payload_build_ms": client_timing.get("payload_build_ms"),
        "provider_call_ms": client_timing.get("provider_call_ms"),
        "response_parse_ms": client_timing.get("response_parse_ms"),
    }


def _image_payload_stats(image_paths: tuple[Path, ...]) -> dict[str, int]:
    raw_image_bytes = sum(path.stat().st_size for path in image_paths)
    return {
        "raw_image_bytes": raw_image_bytes,
        "estimated_base64_image_bytes": sum(((path.stat().st_size + 2) // 3) * 4 for path in image_paths),
    }


def build_workflow_row(
    case: Mapping[str, Any],
    request: ModelRequest,
    raw_text: str,
    latency_ms: float,
    deadline_ms: int,
    workflow: str,
    prompt_template_id: str,
    run_index: int = 1,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    provider_metadata: dict[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine final workflow output with scoring fields for result persistence."""
    metadata = provider_metadata or {}
    row = {
        "experiment_id": experiment_id,
        "workflow": workflow,
        "provider": request.provider,
        "model": request.model,
        "num_images": len(request.image_paths),
        "prompt_template_id": prompt_template_id,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "run_index": run_index,
        "client_wall_clock_latency_ms": latency_ms,
        "provider_reported_latency_ms": provider_reported_latency_ms(metadata),
        "provider_metadata": metadata,
        **_provider_timing(metadata),
        **_image_payload_stats(request.image_paths),
        **score_result(dict(case), raw_text, latency_ms=latency_ms, deadline_ms=deadline_ms),
    }
    if extra_fields:
        row.update(dict(extra_fields))
    return row


def run_global_single(
    case: Mapping[str, Any],
    provider: str = CEREBRAS_PROVIDER,
    model: str | None = None,
    deadline_ms: int = 400,
    run_index: int = 1,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    dataset_root: Path | str = DEFAULT_SELECTED_DATASET_ROOT,
    model_call: ModelCall | None = None,
) -> dict[str, Any]:
    """Run one case through the all-images single-call workflow."""
    selected_model = _model_for_provider(provider, model)
    adapter = _model_call_for_provider(provider, model_call)
    image_paths = tuple(resolve_uav_image_paths(case, dataset_root).values())
    request = ModelRequest(
        provider=provider,
        model=selected_model,
        prompt=build_global_single_prompt(str(case["question"]), case["options"]),
        image_paths=image_paths,
        max_tokens=16,
        temperature=0.0,
        metadata={"question_id": case["question_id"], "workflow": GLOBAL_SINGLE_WORKFLOW},
    )
    response = call_model(request, adapter)

    return build_workflow_row(
        case,
        request,
        response.raw_text,
        response.latency_ms,
        deadline_ms,
        workflow=GLOBAL_SINGLE_WORKFLOW,
        prompt_template_id=GLOBAL_SINGLE_PROMPT_ID,
        run_index=run_index,
        experiment_id=experiment_id,
        provider_metadata=response.provider_metadata,
    )


def _run_observer(
    case: Mapping[str, Any],
    uav_id: str,
    image_path: Path,
    provider: str,
    model: str,
    adapter: ModelCall,
) -> dict[str, Any]:
    request = ModelRequest(
        provider=provider,
        model=model,
        prompt=build_per_uav_observation_prompt(str(uav_id), str(case["question"]), case["options"]),
        image_paths=(image_path,),
        max_tokens=160,
        temperature=0.0,
        metadata={
            "question_id": case["question_id"],
            "workflow": PARALLEL_UAV_FUSION_WORKFLOW,
            "uav_id": uav_id,
            "prompt_template_id": PER_UAV_OBSERVATION_PROMPT_ID,
        },
    )
    response = call_model(request, adapter)
    return {
        "uav_id": uav_id,
        "raw_observation": response.raw_text,
        "latency_ms": response.latency_ms,
        "provider_metadata": response.provider_metadata,
        **_provider_timing(response.provider_metadata),
    }


def run_parallel_uav_fusion(
    case: Mapping[str, Any],
    provider: str = CEREBRAS_PROVIDER,
    model: str | None = None,
    deadline_ms: int = 400,
    run_index: int = 1,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    dataset_root: Path | str = DEFAULT_SELECTED_DATASET_ROOT,
    model_call: ModelCall | None = None,
    timer: Timer = perf_counter,
) -> dict[str, Any]:
    """Run one case through parallel one-image observers and a fusion call."""
    selected_model = _model_for_provider(provider, model)
    adapter = _model_call_for_provider(provider, model_call)
    resolved_images = resolve_uav_image_paths(case, dataset_root)
    workflow_started_at = timer()

    observer_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(resolved_images)) as executor:
        futures = {
            executor.submit(_run_observer, case, uav_id, image_path, provider, selected_model, adapter): uav_id
            for uav_id, image_path in resolved_images.items()
        }
        for future in as_completed(futures):
            observer_rows.append(future.result())

    observer_rows.sort(key=lambda row: list(resolved_images).index(row["uav_id"]))
    observations = {row["uav_id"]: row["raw_observation"] for row in observer_rows}
    fusion_request = ModelRequest(
        provider=provider,
        model=selected_model,
        prompt=build_fusion_prompt(str(case["question"]), case["options"], observations),
        max_tokens=16,
        temperature=0.0,
        metadata={
            "question_id": case["question_id"],
            "workflow": PARALLEL_UAV_FUSION_WORKFLOW,
            "prompt_template_id": FUSION_COMPARER_PROMPT_ID,
        },
    )
    fusion_response: ModelResponse = call_model(fusion_request, adapter)
    workflow_finished_at = timer()
    workflow_latency_ms = (workflow_finished_at - workflow_started_at) * 1000

    return build_workflow_row(
        case,
        ModelRequest(
            provider=provider,
            model=selected_model,
            prompt=fusion_request.prompt,
            image_paths=tuple(resolved_images.values()),
            max_tokens=fusion_request.max_tokens,
            temperature=fusion_request.temperature,
            metadata=fusion_request.metadata,
        ),
        fusion_response.raw_text,
        workflow_latency_ms,
        deadline_ms,
        workflow=PARALLEL_UAV_FUSION_WORKFLOW,
        prompt_template_id=FUSION_COMPARER_PROMPT_ID,
        run_index=run_index,
        experiment_id=experiment_id,
        provider_metadata=fusion_response.provider_metadata,
        extra_fields={
            "observer_prompt_template_id": PER_UAV_OBSERVATION_PROMPT_ID,
            "observer_outputs": observer_rows,
            "fusion_latency_ms": fusion_response.latency_ms,
            "observer_max_latency_ms": max(row["latency_ms"] for row in observer_rows),
            "observer_total_latency_ms": sum(row["latency_ms"] for row in observer_rows),
        },
    )
