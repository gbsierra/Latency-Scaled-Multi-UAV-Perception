import json
from pathlib import Path

import pytest

from src.github_models import GitHubModelsError, build_github_models_payload, parse_github_models_response
from src.model_clients import ModelRequest


def test_build_github_models_payload_keeps_prompt_and_image_order(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    request = ModelRequest(
        provider="github_models",
        model="microsoft/Phi-4-multimodal-instruct",
        prompt="Question",
        image_paths=(first_image, second_image),
        max_tokens=16,
    )

    payload = build_github_models_payload(request)

    content = payload["messages"][0]["content"]
    assert payload["model"] == "microsoft/Phi-4-multimodal-instruct"
    assert content[0] == {"type": "text", "text": "Question"}
    assert content[1]["image_url"]["url"].endswith("Zmlyc3Q=")
    assert content[2]["image_url"]["url"].endswith("c2Vjb25k")


def test_parse_github_models_response_extracts_assistant_text():
    response_body = json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "gpt-4.1-2025-04-14",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Answer: B"},
                    "content_filter_results": {"violence": {"filtered": False}},
                }
            ],
            "usage": {
                "total_tokens": 24,
                "latency_checkpoint": {"total_duration_ms": 568, "engine_ttft_ms": 46},
            },
        }
    ).encode("utf-8")

    output = parse_github_models_response(response_body)

    assert output.raw_text == "Answer: B"
    assert output.provider_metadata["response_id"] == "chatcmpl-test"
    assert output.provider_metadata["response_model"] == "gpt-4.1-2025-04-14"
    assert output.provider_metadata["usage"]["latency_checkpoint"]["total_duration_ms"] == 568
    assert output.provider_metadata["finish_reason"] == "stop"


def test_parse_github_models_response_fails_without_assistant_text():
    with pytest.raises(GitHubModelsError, match="did not contain assistant text"):
        parse_github_models_response(b"{}")
