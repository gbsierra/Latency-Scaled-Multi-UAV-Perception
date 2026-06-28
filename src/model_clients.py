"""Provider-neutral model request and response boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any


class ModelClientError(ValueError):
    """Raised when a model request or response cannot be used for benchmarking."""


@dataclass(frozen=True)
class ModelRequest:
    """Inputs every provider adapter must be able to receive."""

    # Keep provider/model explicit so result rows can compare backends fairly.
    provider: str
    model: str
    prompt: str
    image_paths: tuple[Path, ...] = ()
    max_tokens: int = 128
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Provider output normalized for workflow scoring and result export."""

    provider: str
    model: str
    raw_text: str
    latency_ms: float


ModelCall = Callable[[ModelRequest], str]
# Injectable timer keeps latency measurement testable without sleeping.
Timer = Callable[[], float]


def validate_model_request(request: ModelRequest) -> None:
    """Fail before provider calls when benchmark inputs are incomplete."""
    if not request.provider.strip():
        raise ModelClientError("provider is required")

    if not request.model.strip():
        raise ModelClientError("model is required")

    if not request.prompt.strip():
        raise ModelClientError("prompt is required")

    if request.max_tokens <= 0:
        raise ModelClientError("max_tokens must be positive")

    if request.temperature < 0:
        raise ModelClientError("temperature must be non-negative")

    # Existence checks stay with benchmark_cases; adapters may handle remote paths later.
    for image_path in request.image_paths:
        if not isinstance(image_path, Path):
            raise ModelClientError("image_paths must contain pathlib.Path values")


def call_model(
    request: ModelRequest,
    model_call: ModelCall,
    timer: Timer = perf_counter,
) -> ModelResponse:
    """Call a provider adapter and measure end-to-end model-call latency."""
    validate_model_request(request)

    # The provider adapter owns SDK details; this boundary owns timing and shape.
    started_at = timer()
    raw_text = model_call(request)
    finished_at = timer()

    if not isinstance(raw_text, str):
        raise ModelClientError("model_call must return raw response text")

    return ModelResponse(
        provider=request.provider,
        model=request.model,
        raw_text=raw_text,
        latency_ms=(finished_at - started_at) * 1000,
    )
