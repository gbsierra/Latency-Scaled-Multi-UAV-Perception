"""GitHub Models adapter for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.image_payloads import ImagePayloadError, image_data_url
from src.model_clients import ModelClientError, ModelOutput, ModelRequest


GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference/chat/completions"
GITHUB_MODELS_PROVIDER = "github_models"
PHI_4_MULTIMODAL_MODEL = "microsoft/Phi-4-multimodal-instruct"


class GitHubModelsError(ModelClientError):
    """Raised when GitHub Models cannot return usable response text."""


def build_github_models_payload(request: ModelRequest) -> dict[str, Any]:
    """Build the provider payload while keeping benchmark request fields intact."""
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

    # Preserve UAV order from the resolved tuple so runs are comparable.
    for image_path in request.image_paths:
        try:
            url = image_data_url(image_path)
        except ImagePayloadError as error:
            raise GitHubModelsError(str(error)) from error

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )

    return {
        "model": request.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }


def parse_github_models_response(response_body: bytes) -> ModelOutput:
    """Extract assistant text and provider metadata from a GitHub Models response."""
    try:
        payload = json.loads(response_body.decode("utf-8"))
        raw_text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise GitHubModelsError("GitHub Models response did not contain assistant text") from error

    if not isinstance(raw_text, str):
        raise GitHubModelsError("GitHub Models assistant content was not text")

    # Preserve provider timing, usage, resolved model id, and filter fields for analysis.
    return ModelOutput(
        raw_text=raw_text,
        provider_metadata={
            "response_id": payload.get("id"),
            "response_model": payload.get("model"),
            "created": payload.get("created"),
            "service_tier": payload.get("service_tier"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage"),
            "prompt_filter_results": payload.get("prompt_filter_results"),
            "content_filter_results": payload["choices"][0].get("content_filter_results"),
            "finish_reason": payload["choices"][0].get("finish_reason"),
        },
    )


def call_github_models(request: ModelRequest) -> ModelOutput:
    """Call GitHub Models and return assistant text plus provider metadata."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GitHubModelsError("GITHUB_TOKEN is required")

    if request.provider != GITHUB_MODELS_PROVIDER:
        raise GitHubModelsError(f"unsupported provider for GitHub Models: {request.provider}")

    payload_started_at = perf_counter()
    body = json.dumps(build_github_models_payload(request)).encode("utf-8")
    payload_finished_at = perf_counter()
    http_request = Request(
        GITHUB_MODELS_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        provider_started_at = perf_counter()
        with urlopen(http_request, timeout=120) as response:
            response_body = response.read()
        provider_finished_at = perf_counter()
        parse_started_at = perf_counter()
        output = parse_github_models_response(response_body)
        parse_finished_at = perf_counter()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise GitHubModelsError(f"GitHub Models HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise GitHubModelsError(f"GitHub Models request failed: {error.reason}") from error

    output.provider_metadata["client_timing"] = {
        "payload_build_ms": (payload_finished_at - payload_started_at) * 1000,
        "provider_call_ms": (provider_finished_at - provider_started_at) * 1000,
        "response_parse_ms": (parse_finished_at - parse_started_at) * 1000,
        "request_body_bytes": len(body),
    }
    return output
