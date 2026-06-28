import json

import pytest

from src.benchmark_cases import BenchmarkCaseError, load_selected_cases, resolve_uav_image_paths


def selected_case() -> dict:
    # Minimal selected-slice case with the same fields needed by model workflows.
    return {
        "question_id": "case-1",
        "question_type": "Object Matching",
        "question": "Which object matches across views?",
        "options": {
            "A": "Car",
            "B": "Truck",
            "C": "Bus",
            "D": "None",
        },
        "correct_answer": "A",
        "source_group": "Sim3",
        "source_split": "test",
        "uav_paths": {
            "UAV1": "Sim_3_UAVs/Samples/UAV1/example-uav1.png",
            "UAV2": "Sim_3_UAVs/Samples/UAV2/example-uav2.png",
        },
    }


def write_selected_cases(tmp_path, cases) -> None:
    # Tests use a temporary selected-slice file instead of gitignored data.
    path = tmp_path / "core_30_cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")


def test_load_selected_cases_validates_aircopbench_case_fields(tmp_path):
    write_selected_cases(tmp_path, [selected_case()])

    cases = list(load_selected_cases(tmp_path / "core_30_cases.json"))

    assert len(cases) == 1
    assert cases[0]["question_id"] == "case-1"
    assert cases[0]["uav_paths"]["UAV1"].endswith("example-uav1.png")


def test_load_selected_cases_fails_when_slice_is_not_a_list(tmp_path):
    path = tmp_path / "core_30_cases.json"
    path.write_text(json.dumps({"question_id": "case-1"}), encoding="utf-8")

    with pytest.raises(BenchmarkCaseError, match="expected a JSON list"):
        list(load_selected_cases(path))


def test_load_selected_cases_fails_when_case_is_missing_required_field(tmp_path):
    case = selected_case()
    del case["correct_answer"]
    write_selected_cases(tmp_path, [case])

    with pytest.raises(BenchmarkCaseError, match="missing required fields: correct_answer"):
        list(load_selected_cases(tmp_path / "core_30_cases.json"))


def test_resolve_uav_image_paths_returns_existing_paths(tmp_path):
    case = selected_case()
    # Real benchmark runs need files on disk, not just metadata references.
    for relative_path in case["uav_paths"].values():
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")

    resolved = resolve_uav_image_paths(case, tmp_path)

    assert resolved["UAV1"] == tmp_path / "Sim_3_UAVs/Samples/UAV1/example-uav1.png"
    assert resolved["UAV2"].is_file()


def test_resolve_uav_image_paths_fails_when_image_is_missing(tmp_path):
    case = selected_case()

    with pytest.raises(BenchmarkCaseError, match="missing UAV images"):
        resolve_uav_image_paths(case, tmp_path)
