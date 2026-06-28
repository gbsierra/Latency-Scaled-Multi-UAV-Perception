from pathlib import Path

from src.model_clients import ModelOutput, ModelRequest
from src.workflows import (
    GLOBAL_SINGLE_WORKFLOW,
    PARALLEL_UAV_FUSION_WORKFLOW,
    build_workflow_row,
    run_global_single,
    run_parallel_uav_fusion,
)


def workflow_case() -> dict:
    return {
        "question_id": "case-1",
        "source_group": "Sim3",
        "source_split": "test",
        "question_type": "Object Matching",
        "question": "Which UAV sees the matching vehicle?",
        "options": {"A": "UAV1", "B": "UAV2", "C": "UAV3", "D": "None"},
        "correct_answer": "B",
        "uav_paths": {
            "UAV1": "Sim_3_UAVs/Samples/UAV1/example-uav1.png",
            "UAV2": "Sim_3_UAVs/Samples/UAV2/example-uav2.png",
        },
    }


def write_case_images(tmp_path, case: dict) -> None:
    for relative_path in case["uav_paths"].values():
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")


def test_build_workflow_row_scores_final_answer_and_keeps_run_fields(tmp_path):
    image_path = tmp_path / "uav1.png"
    image_path.write_bytes(b"1234")
    request = ModelRequest(
        provider="local",
        model="test-model",
        prompt="Question",
        image_paths=(image_path,),
        max_tokens=16,
        temperature=0.0,
    )

    row = build_workflow_row(
        workflow_case(),
        request,
        raw_text="B",
        latency_ms=250,
        deadline_ms=400,
        workflow=GLOBAL_SINGLE_WORKFLOW,
        prompt_template_id="prompt-v1",
        run_index=3,
        experiment_id="exp-1",
        provider_metadata={
            "client_timing": {"provider_call_ms": 200},
            "time_info": {"total_time_ms": 225},
        },
    )

    assert row["experiment_id"] == "exp-1"
    assert row["workflow"] == GLOBAL_SINGLE_WORKFLOW
    assert row["prompt_template_id"] == "prompt-v1"
    assert row["run_index"] == 3
    assert row["provider_call_ms"] == 200
    assert row["provider_reported_latency_ms"] == 225
    assert row["raw_image_bytes"] == 4
    assert row["correct"] is True
    assert row["actionable"] is True


def test_run_global_single_uses_all_case_images_and_scores_answer(tmp_path):
    case = workflow_case()
    write_case_images(tmp_path, case)
    received_requests: list[ModelRequest] = []

    def model_call(request: ModelRequest) -> str:
        received_requests.append(request)
        return "Answer: B"

    row = run_global_single(
        case,
        provider="local",
        model="test-model",
        deadline_ms=400,
        dataset_root=tmp_path,
        model_call=model_call,
    )

    assert row["workflow"] == GLOBAL_SINGLE_WORKFLOW
    assert row["num_images"] == 2
    assert row["prompt_template_id"] == "global_single_v1"
    assert row["correct"] is True
    assert "AirCopBench UAV perception question" in received_requests[0].prompt
    assert received_requests[0].image_paths == (
        tmp_path / "Sim_3_UAVs/Samples/UAV1/example-uav1.png",
        tmp_path / "Sim_3_UAVs/Samples/UAV2/example-uav2.png",
    )


def test_run_parallel_uav_fusion_records_observers_and_scores_workflow_latency(tmp_path):
    case = workflow_case()
    write_case_images(tmp_path, case)
    requests: list[ModelRequest] = []
    workflow_times = iter([1.0, 2.75])

    def model_call(request: ModelRequest) -> ModelOutput:
        requests.append(request)
        if request.image_paths:
            uav_id = request.metadata["uav_id"]
            return ModelOutput(raw_text=f"{uav_id} evidence", provider_metadata={})
        return ModelOutput(
            raw_text="Answer: B",
            provider_metadata={
                "client_timing": {"provider_call_ms": 10},
                "time_info": {"total_time_ms": 25},
            },
        )

    row = run_parallel_uav_fusion(
        case,
        provider="local",
        model="test-model",
        deadline_ms=2000,
        dataset_root=tmp_path,
        model_call=model_call,
        timer=lambda: next(workflow_times),
    )

    assert row["workflow"] == PARALLEL_UAV_FUSION_WORKFLOW
    assert row["prompt_template_id"] == "fusion_comparer_v2"
    assert row["observer_prompt_template_id"] == "per_uav_observation_v2"
    assert row["latency_ms"] == 1750.0
    assert row["client_wall_clock_latency_ms"] == 1750.0
    assert row["fusion_latency_ms"] >= 0
    assert row["provider_call_ms"] == 10
    assert row["provider_reported_latency_ms"] == 25
    assert row["correct"] is True
    assert row["actionable"] is True
    assert [output["uav_id"] for output in row["observer_outputs"]] == ["UAV1", "UAV2"]
    assert [request.metadata["workflow"] for request in requests].count(PARALLEL_UAV_FUSION_WORKFLOW) == 3
    assert len([request for request in requests if request.image_paths]) == 2
    assert len([request for request in requests if not request.image_paths]) == 1
    assert "UAV1 evidence" in requests[-1].prompt
    assert "UAV2 evidence" in requests[-1].prompt
