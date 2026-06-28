"""Cerebras adapter for OpenAI-compatible chat completions."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Protocol

from src.image_payloads import ImagePayloadError, image_data_url
from src.model_clients import ModelClientError, ModelOutput, ModelRequest


CEREBRAS_PROVIDER = "cerebras"
GEMMA_4_31B_MODEL = "gemma-4-31b"


class CerebrasModelsError(ModelClientError):
    """Raised when Cerebras cannot return usable response text."""


class CerebrasChatCompletions(Protocol):
    """Small SDK surface needed by this adapter."""

    def create(self, **kwargs: Any) -> Any:
        """Create a Cerebras chat completion."""


class CerebrasClient(Protocol):
    """Small SDK client surface needed by this adapter."""

    chat: Any


def build_cerebras_payload(request: ModelRequest) -> dict[str, Any]:
    """Build the Cerebras chat-completions payload from a benchmark request."""
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

    # Preserve the same image order used by other providers for fair comparisons.
    for image_path in request.image_paths:
        try:
            url = image_data_url(image_path)
        except ImagePayloadError as error:
            raise CerebrasModelsError(str(error)) from error

        content.append({"type": "image_url", "image_url": {"url": url}})

    return {
        "model": request.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }


def response_value(response: Any, key: str) -> Any:
    """Read SDK response fields whether they are dict-like or object-like."""
    if isinstance(response, dict):
        return response.get(key)

    return getattr(response, key, None)


def json_safe(value: Any) -> Any:
    """Convert SDK response objects into JSON-serializable values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    # Pydantic-style SDK models often expose a dict conversion method.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json_safe(model_dump())

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return json_safe(as_dict())

    if hasattr(value, "__dict__"):
        return json_safe(vars(value))

    return str(value)


def parse_cerebras_response(response: Any) -> ModelOutput:
    """Extract assistant text and useful metadata from a Cerebras SDK response."""
    try:
        choices = response_value(response, "choices")
        choice = choices[0]
        message = response_value(choice, "message")
        raw_text = response_value(message, "content")
    except (IndexError, TypeError) as error:
        raise CerebrasModelsError("Cerebras response did not contain assistant text") from error

    if not isinstance(raw_text, str):
        raise CerebrasModelsError("Cerebras assistant content was not text")

    return ModelOutput(
        raw_text=raw_text,
        provider_metadata={
            "response_id": json_safe(response_value(response, "id")),
            "response_model": json_safe(response_value(response, "model")),
            "created": json_safe(response_value(response, "created")),
            "system_fingerprint": json_safe(response_value(response, "system_fingerprint")),
            "usage": json_safe(response_value(response, "usage")),
            "finish_reason": json_safe(response_value(choice, "finish_reason")),
        },
    )


def create_cerebras_client(api_key: str) -> CerebrasClient:
    """Create the official Cerebras SDK client lazily."""
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError as error:
        raise CerebrasModelsError(
            "cerebras-cloud-sdk is required; install with `python -m pip install -r requirements.txt`"
        ) from error

    return Cerebras(api_key=api_key)


def call_cerebras(request: ModelRequest, client: CerebrasClient | None = None) -> ModelOutput:
    """Call Cerebras through the official SDK and return normalized output."""
    token = os.environ.get("CEREBRAS_API_KEY")
    if client is None and not token:
        raise CerebrasModelsError("CEREBRAS_API_KEY is required")

    if request.provider != CEREBRAS_PROVIDER:
        raise CerebrasModelsError(f"unsupported provider for Cerebras: {request.provider}")

    client = client or create_cerebras_client(token or "")
    payload_started_at = perf_counter()
    payload = build_cerebras_payload(request)
    payload_finished_at = perf_counter()

    try:
        provider_started_at = perf_counter()
        response = client.chat.completions.create(**payload)
        provider_finished_at = perf_counter()
    except Exception as error:
        raise CerebrasModelsError(f"Cerebras SDK request failed: {error}") from error

    parse_started_at = perf_counter()
    output = parse_cerebras_response(response)
    parse_finished_at = perf_counter()
    output.provider_metadata["client_timing"] = {
        "payload_build_ms": (payload_finished_at - payload_started_at) * 1000,
        "provider_call_ms": (provider_finished_at - provider_started_at) * 1000,
        "response_parse_ms": (parse_finished_at - parse_started_at) * 1000,
    }
    return output
