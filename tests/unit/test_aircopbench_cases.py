import json
from pathlib import Path

import pytest

from src.aircopbench_cases import AirCopBenchCaseError, load_cases, source_group_from_file


def valid_case() -> dict:
    # Minimal valid AirCopBench-style case used by loader tests.
    return {
        "question_id": "case-1",
        "question_type": "Object Grounding",
        "question": "Where is the vehicle?",
        "options": {
            "A": "Left",
            "B": "Right",
            "C": "Center",
            "D": "Not visible",
        },
        "correct_answer": "A",
        "uav_paths": {"UAV1": "Sim_3_UAVs/Samples/UAV1/example.png"},
    }


def write_cases(tmp_path, filename: str, cases) -> None:
    # Write a temporary raw metadata file shaped like the downloaded dataset.
    path = tmp_path / filename
    path.write_text(json.dumps(cases), encoding="utf-8")


def test_source_group_from_file_uses_dataset_group_name():
    # Dataset group is encoded in the raw AirCopBench filename.
    assert source_group_from_file(Path("Sim5_VQA_test.json")) == "Sim5"


def test_load_cases_adds_source_metadata(tmp_path):
    # Loaded cases keep original fields and gain provenance fields.
    write_cases(tmp_path, "Sim3_VQA_test.json", [valid_case()])

    cases = list(load_cases(tmp_path))

    assert len(cases) == 1
    assert cases[0]["question_id"] == "case-1"
    assert cases[0]["source_group"] == "Sim3"
    assert cases[0]["source_split"] == "test"
    assert cases[0]["source_file"].endswith("Sim3_VQA_test.json")


def test_load_cases_fails_when_required_field_is_missing(tmp_path):
    # Bad metadata should fail before model runs or scoring.
    case = valid_case()
    del case["question"]
    write_cases(tmp_path, "Real2_VQA_test.json", [case])

    with pytest.raises(AirCopBenchCaseError, match="missing required fields: question"):
        list(load_cases(tmp_path))


def test_load_cases_fails_when_correct_answer_is_not_an_option(tmp_path):
    # Gold answers must match one of the provided multiple-choice options.
    case = valid_case()
    case["correct_answer"] = "E"
    write_cases(tmp_path, "Sim6_VQA_test.json", [case])

    with pytest.raises(AirCopBenchCaseError, match="correct_answer is not one of options"):
        list(load_cases(tmp_path))
