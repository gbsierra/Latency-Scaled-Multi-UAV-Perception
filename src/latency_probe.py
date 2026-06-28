"""Run one selected AirCopBench case through a provider for latency probing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.benchmark_cases import load_selected_cases, resolve_uav_image_paths
from src.cerebras_models import CEREBRAS_PROVIDER, GEMMA_4_31B_MODEL, call_cerebras
from src.github_models import GITHUB_MODELS_PROVIDER, PHI_4_MULTIMODAL_MODEL, call_github_models
from src.model_clients import (
    ModelCall,
    ModelRequest,
    call_model,
    provider_reported_latency_ms as model_provider_reported_latency_ms,
)
from src.prompts import build_answer_prompt
from src.results import write_jsonl
from src.scoring import score_result


DEFAULT_PROBE_CASE_ID = "MDMT_OM_UAV5_27"
DEFAULT_OUTPUT_PATH = Path("results/latency_probe_v0.jsonl")
GLOBAL_SINGLE_WORKFLOW = "global_single"
PROVIDER_DEFAULT_MODELS = {
    GITHUB_MODELS_PROVIDER: PHI_4_MULTIMODAL_MODEL,
    CEREBRAS_PROVIDER: GEMMA_4_31B_MODEL,
}
PROVIDER_CALLS: dict[str, ModelCall] = {
    GITHUB_MODELS_PROVIDER: call_github_models,
    CEREBRAS_PROVIDER: call_cerebras,
}


def image_payload_stats(image_paths: tuple[Path, ...]) -> dict[str, int]:
    """Return input-image size diagnostics for latency analysis."""
    raw_image_bytes = sum(path.stat().st_size for path in image_paths)
    return {
        "raw_image_bytes": raw_image_bytes,
        # Base64 encodes each 3-byte block into 4 bytes, rounded up per file.
        "estimated_base64_image_bytes": sum(((path.stat().st_size + 2) // 3) * 4 for path in image_paths),
    }


def provider_reported_latency_ms(provider_metadata: dict[str, Any]) -> int | float | None:
    """Return provider total duration when provider metadata includes it."""
    return model_provider_reported_latency_ms(provider_metadata)


def find_selected_case(question_id: str) -> dict[str, Any]:
    """Return one selected benchmark case by AirCopBench question id."""
    for case in load_selected_cases():
        if case["question_id"] == question_id:
            return case

    raise ValueError(f"selected case not found: {question_id}")


def build_probe_prompt(case: dict[str, Any]) -> str:
    """Build the global-single prompt used by the one-case latency probe."""
    instruction = (
        "You are answering a multiple-choice AirCopBench UAV perception question.\n"
        "Use the provided UAV images, question, and answer choices.\n"
        "Return only one letter: A, B, C, or D.\n\n"
    )
    return instruction + build_answer_prompt(case["question"], case["options"])


def build_probe_row(
    case: dict[str, Any],
    request: ModelRequest,
    raw_text: str,
    latency_ms: float,
    deadline_ms: int,
    run_index: int,
    provider_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine model output with scoring fields for JSONL persistence."""
    provider_metadata = provider_metadata or {}
    scored = score_result(case, raw_text, latency_ms=latency_ms, deadline_ms=deadline_ms)
    client_timing = provider_metadata.get("client_timing")
    if not isinstance(client_timing, dict):
        client_timing = {}

    return {
        "experiment_id": "latency_probe_v0_single_case",
        "workflow": GLOBAL_SINGLE_WORKFLOW,
        "provider": request.provider,
        "model": request.model,
        "num_images": len(request.image_paths),
        "run_index": run_index,
        "client_wall_clock_latency_ms": latency_ms,
        "payload_build_ms": client_timing.get("payload_build_ms"),
        "provider_call_ms": client_timing.get("provider_call_ms"),
        "response_parse_ms": client_timing.get("response_parse_ms"),
        "provider_reported_latency_ms": provider_reported_latency_ms(provider_metadata),
        "provider_metadata": provider_metadata,
        **image_payload_stats(request.image_paths),
        **scored,
    }


def run_latency_probe(
    question_id: str = DEFAULT_PROBE_CASE_ID,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    deadline_ms: int = 400,
    run_index: int = 1,
    provider: str = GITHUB_MODELS_PROVIDER,
    model: str | None = None,
) -> dict[str, Any]:
    """Run one provider latency probe and save its scored row."""
    if provider not in PROVIDER_CALLS:
        raise ValueError(f"unsupported provider: {provider}")

    case = find_selected_case(question_id)
    image_paths = tuple(resolve_uav_image_paths(case).values())
    request = ModelRequest(
        provider=provider,
        model=model or PROVIDER_DEFAULT_MODELS[provider],
        prompt=build_probe_prompt(case),
        image_paths=image_paths,
        max_tokens=16,
        temperature=0.0,
        metadata={"question_id": question_id, "workflow": GLOBAL_SINGLE_WORKFLOW},
    )
    response = call_model(request, PROVIDER_CALLS[provider])
    row = build_probe_row(
        case,
        request,
        response.raw_text,
        response.latency_ms,
        deadline_ms,
        run_index,
        provider_metadata=response.provider_metadata,
    )
    write_jsonl(output_path, [row])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one provider latency probe.")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_CALLS),
        default=GITHUB_MODELS_PROVIDER,
    )
    parser.add_argument("--question-id", default=DEFAULT_PROBE_CASE_ID)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--deadline-ms", type=int, default=400)
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    row = run_latency_probe(
        question_id=args.question_id,
        output_path=args.output,
        deadline_ms=args.deadline_ms,
        run_index=args.run_index,
        provider=args.provider,
        model=args.model,
    )
    print(
        f"{row['provider']} {row['model']} {row['question_id']} "
        f"latency_ms={row['latency_ms']:.1f} parsed={row['parsed_answer']} "
        f"correct={row['correct']} actionable={row['actionable']}"
    )


if __name__ == "__main__":
    main()
