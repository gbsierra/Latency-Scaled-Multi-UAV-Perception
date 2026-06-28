from pathlib import Path

from src.cerebras_models import CEREBRAS_PROVIDER, GEMMA_4_31B_MODEL
from src.github_models import GITHUB_MODELS_PROVIDER, PHI_4_MULTIMODAL_MODEL
from src.latency_probe import (
    GLOBAL_SINGLE_WORKFLOW,
    PROVIDER_DEFAULT_MODELS,
    build_probe_prompt,
    build_probe_row,
    image_payload_stats,
    provider_reported_latency_ms,
)
from src.model_clients import ModelRequest


def probe_case() -> dict:
    return {
        "question_id": "case-1",
        "source_group": "Sim5",
        "source_split": "test",
        "question_type": "Object Matching",
        "question": "Which object matches?",
        "options": {"A": "Car", "B": "Bus", "C": "Truck", "D": "None"},
        "correct_answer": "A",
    }


def test_build_probe_prompt_adds_latency_probe_instruction():
    prompt = build_probe_prompt(probe_case())

    assert "AirCopBench UAV perception question" in prompt
    assert "Return only one letter" in prompt
    assert "Which object matches?" in prompt
    assert "A. Car" in prompt


def test_build_probe_row_adds_provider_workflow_and_score_fields(tmp_path):
    first_image = tmp_path / "uav1.png"
    second_image = tmp_path / "uav2.png"
    first_image.write_bytes(b"123")
    second_image.write_bytes(b"1234")
    request = ModelRequest(
        provider=GITHUB_MODELS_PROVIDER,
        model=PHI_4_MULTIMODAL_MODEL,
        prompt="Question",
        image_paths=(first_image, second_image),
    )

    row = build_probe_row(
        probe_case(),
        request,
        raw_text="A",
        latency_ms=250.0,
        deadline_ms=400,
        run_index=2,
        provider_metadata={
            "client_timing": {
                "payload_build_ms": 10,
                "provider_call_ms": 200,
                "response_parse_ms": 1,
            },
            "usage": {
                "latency_checkpoint": {
                    "total_duration_ms": 210,
                    "service_ttft_ms": 50,
                }
            }
        },
    )

    assert row["experiment_id"] == "latency_probe_v0_single_case"
    assert row["workflow"] == GLOBAL_SINGLE_WORKFLOW
    assert row["provider"] == GITHUB_MODELS_PROVIDER
    assert row["model"] == PHI_4_MULTIMODAL_MODEL
    assert row["num_images"] == 2
    assert row["run_index"] == 2
    assert row["payload_build_ms"] == 10
    assert row["provider_call_ms"] == 200
    assert row["response_parse_ms"] == 1
    assert row["raw_image_bytes"] == 7
    assert row["estimated_base64_image_bytes"] == 12
    assert row["provider_reported_latency_ms"] == 210
    assert row["provider_metadata"]["usage"]["latency_checkpoint"]["service_ttft_ms"] == 50
    assert row["correct"] is True
    assert row["actionable"] is True


def test_image_payload_stats_reports_raw_and_base64_sizes(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(b"123")
    second_image.write_bytes(b"1234")

    stats = image_payload_stats((first_image, second_image))

    assert stats == {
        "raw_image_bytes": 7,
        "estimated_base64_image_bytes": 12,
    }


def test_provider_reported_latency_ms_returns_none_without_checkpoint():
    assert provider_reported_latency_ms({}) is None
    assert provider_reported_latency_ms({"usage": {"total_tokens": 24}}) is None


def test_provider_default_models_include_github_and_cerebras():
    assert PROVIDER_DEFAULT_MODELS[GITHUB_MODELS_PROVIDER] == PHI_4_MULTIMODAL_MODEL
    assert PROVIDER_DEFAULT_MODELS[CEREBRAS_PROVIDER] == GEMMA_4_31B_MODEL
