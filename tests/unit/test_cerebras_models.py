import pytest

from src.cerebras_models import (
    CEREBRAS_PROVIDER,
    GEMMA_4_31B_MODEL,
    CerebrasModelsError,
    build_cerebras_payload,
    call_cerebras,
    json_safe,
    parse_cerebras_response,
)
from src.model_clients import ModelRequest


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return {
            "id": "chatcmpl-cerebras",
            "model": GEMMA_4_31B_MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": "Answer: C"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
        }


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 2
        self.total_tokens = 102


def test_build_cerebras_payload_keeps_prompt_and_image_order(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    request = ModelRequest(
        provider=CEREBRAS_PROVIDER,
        model=GEMMA_4_31B_MODEL,
        prompt="Question",
        image_paths=(first_image, second_image),
        max_tokens=16,
    )

    payload = build_cerebras_payload(request)

    content = payload["messages"][0]["content"]
    assert payload["model"] == GEMMA_4_31B_MODEL
    assert content[0] == {"type": "text", "text": "Question"}
    assert content[1]["image_url"]["url"].endswith("Zmlyc3Q=")
    assert content[2]["image_url"]["url"].endswith("c2Vjb25k")


def test_parse_cerebras_response_extracts_assistant_text_and_metadata():
    response = {
        "id": "chatcmpl-cerebras",
        "model": GEMMA_4_31B_MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": "Answer: C"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102},
    }

    output = parse_cerebras_response(response)

    assert output.raw_text == "Answer: C"
    assert output.provider_metadata["response_id"] == "chatcmpl-cerebras"
    assert output.provider_metadata["response_model"] == GEMMA_4_31B_MODEL
    assert output.provider_metadata["usage"]["total_tokens"] == 102
    assert output.provider_metadata["finish_reason"] == "stop"


def test_parse_cerebras_response_converts_sdk_usage_object_to_json_safe_metadata():
    response = {
        "id": "chatcmpl-cerebras",
        "model": GEMMA_4_31B_MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": "Answer: C"}}],
        "usage": FakeUsage(),
    }

    output = parse_cerebras_response(response)

    assert output.provider_metadata["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "total_tokens": 102,
    }


def test_parse_cerebras_response_fails_without_assistant_text():
    with pytest.raises(CerebrasModelsError, match="did not contain assistant text"):
        parse_cerebras_response({})


def test_call_cerebras_uses_sdk_client_and_payload(tmp_path):
    image_path = tmp_path / "uav.png"
    image_path.write_bytes(b"image")
    request = ModelRequest(
        provider=CEREBRAS_PROVIDER,
        model=GEMMA_4_31B_MODEL,
        prompt="Question",
        image_paths=(image_path,),
        max_tokens=16,
    )
    client = FakeClient()

    output = call_cerebras(request, client=client)

    assert output.raw_text == "Answer: C"
    assert client.chat.completions.kwargs["model"] == GEMMA_4_31B_MODEL
    assert client.chat.completions.kwargs["messages"][0]["content"][0]["text"] == "Question"


def test_json_safe_uses_model_dump_when_available():
    class ModelDumpUsage:
        def model_dump(self):
            return {"total_tokens": 12}

    assert json_safe(ModelDumpUsage()) == {"total_tokens": 12}
