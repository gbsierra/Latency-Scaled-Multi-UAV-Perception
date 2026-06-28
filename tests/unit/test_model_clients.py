from pathlib import Path

import pytest

from src.model_clients import (
    ModelClientError,
    ModelOutput,
    ModelRequest,
    call_model,
    provider_reported_latency_ms,
    validate_model_request,
)


def valid_request() -> ModelRequest:
    # Shared request keeps validation and timing tests focused on one boundary.
    return ModelRequest(
        provider="local",
        model="test-model",
        prompt="Answer with the option letter only.",
        image_paths=(Path("image.png"),),
    )


def test_validate_model_request_accepts_text_only_fusion_requests():
    # Fusion calls may use per-UAV observations instead of raw images.
    request = ModelRequest(
        provider="local",
        model="fusion-model",
        prompt="Fuse observations and answer A/B/C/D.",
    )

    validate_model_request(request)


def test_validate_model_request_fails_without_provider():
    request = valid_request()
    # Provider must be present so benchmark rows are attributable.
    request = ModelRequest(
        provider=" ",
        model=request.model,
        prompt=request.prompt,
        image_paths=request.image_paths,
    )

    with pytest.raises(ModelClientError, match="provider is required"):
        validate_model_request(request)


def test_validate_model_request_fails_when_image_paths_are_not_paths():
    # Workflows should pass resolved Path objects, not raw strings from metadata.
    request = ModelRequest(
        provider="local",
        model="test-model",
        prompt="Question",
        image_paths=("image.png",),  # type: ignore[arg-type]
    )

    with pytest.raises(ModelClientError, match="image_paths must contain pathlib.Path values"):
        validate_model_request(request)


def test_call_model_returns_normalized_response_with_latency():
    request = valid_request()
    # Deterministic timer proves latency math without making the test slow.
    times = iter([10.0, 10.125])

    def model_call(received_request: ModelRequest) -> str:
        # The adapter receives the exact request workflows will construct.
        assert received_request == request
        return "Answer: A"

    response = call_model(request, model_call, timer=lambda: next(times))

    assert response.provider == "local"
    assert response.model == "test-model"
    assert response.raw_text == "Answer: A"
    assert response.latency_ms == 125.0
    assert response.provider_metadata == {}


def test_call_model_preserves_adapter_metadata():
    request = valid_request()
    times = iter([1.0, 1.25])

    def model_call(_received_request: ModelRequest) -> ModelOutput:
        return ModelOutput(
            raw_text="Answer: C",
            provider_metadata={"usage": {"total_tokens": 24}},
        )

    response = call_model(request, model_call, timer=lambda: next(times))

    assert response.raw_text == "Answer: C"
    assert response.latency_ms == 250.0
    assert response.provider_metadata == {"usage": {"total_tokens": 24}}


def test_call_model_converts_adapter_metadata_to_json_safe_values():
    request = valid_request()
    times = iter([1.0, 1.25])

    class Usage:
        def __init__(self):
            self.total_tokens = 24

    def model_call(_received_request: ModelRequest) -> ModelOutput:
        return ModelOutput(raw_text="Answer: C", provider_metadata={"usage": Usage()})

    response = call_model(request, model_call, timer=lambda: next(times))

    assert response.provider_metadata == {"usage": {"total_tokens": 24}}


def test_provider_reported_latency_ms_reads_github_and_cerebras_shapes():
    assert provider_reported_latency_ms(
        {"usage": {"latency_checkpoint": {"total_duration_ms": 210}}}
    ) == 210
    assert provider_reported_latency_ms({"time_info": {"total_time_ms": 125.5}}) == 125.5
    assert provider_reported_latency_ms({"time_info": {"total_time": 0.18703985214233398}}) == 187.03985214233398
    assert provider_reported_latency_ms({"usage": {"total_tokens": 24}}) is None


def test_call_model_fails_when_adapter_returns_non_text_response():
    # Scoring expects raw text, so provider adapters must normalize SDK objects first.
    def model_call(_request: ModelRequest) -> str:
        return None  # type: ignore[return-value]

    with pytest.raises(ModelClientError, match="model_call must return raw response text or ModelOutput"):
        call_model(valid_request(), model_call)


def test_call_model_fails_when_adapter_metadata_is_not_an_object():
    def model_call(_request: ModelRequest) -> ModelOutput:
        return ModelOutput(raw_text="A", provider_metadata=["not", "an", "object"])  # type: ignore[arg-type]

    with pytest.raises(ModelClientError, match="provider_metadata must be an object"):
        call_model(valid_request(), model_call)
