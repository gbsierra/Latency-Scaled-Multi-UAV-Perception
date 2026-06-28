from collections import Counter
from pathlib import Path

import pytest

from src.aircopbench_cases import DEFAULT_RAW_TEST_DIR, load_cases
from src.prompts import ANSWER_INSTRUCTION, build_answer_prompt
from src.results import read_jsonl, write_jsonl
from src.scoring import score_answer, score_result


EXPECTED_GROUP_COUNTS = {
    "Real2": 223,
    "Sim3": 524,
    "Sim5": 147,
    "Sim6": 131,
}


def raw_metadata_available() -> bool:
    # Integration tests require the gitignored AirCopBench metadata download.
    return Path(DEFAULT_RAW_TEST_DIR).exists()


@pytest.mark.skipif(not raw_metadata_available(), reason="AirCopBench raw metadata not downloaded")
def test_real_aircopbench_metadata_loads_expected_case_counts():
    # Case counts catch partial or wrong Hugging Face downloads.
    cases = list(load_cases())
    group_counts = Counter(case["source_group"] for case in cases)

    assert len(cases) == 1025
    assert group_counts == EXPECTED_GROUP_COUNTS


@pytest.mark.skipif(not raw_metadata_available(), reason="AirCopBench raw metadata not downloaded")
def test_real_aircopbench_answers_score_as_actionable_when_on_time():
    # Every gold answer should parse and score correctly under the deadline.
    for case in load_cases():
        result = score_answer(
            raw_answer=case["correct_answer"],
            correct_answer=case["correct_answer"],
            latency_ms=1.0,
            deadline_ms=400.0,
        )

        assert result["parse_failed"] is False
        assert result["correct"] is True
        assert result["stale"] is False
        assert result["actionable"] is True


@pytest.mark.skipif(not raw_metadata_available(), reason="AirCopBench raw metadata not downloaded")
def test_real_aircopbench_cases_build_answer_prompts():
    # Real metadata must format into prompts before model workflows exist.
    for case in load_cases():
        prompt = build_answer_prompt(case["question"], case["options"])

        assert case["question"] in prompt
        assert "A." in prompt
        assert "B." in prompt
        assert "C." in prompt
        assert "D." in prompt
        assert prompt.endswith(ANSWER_INSTRUCTION)


@pytest.mark.skipif(not raw_metadata_available(), reason="AirCopBench raw metadata not downloaded")
def test_real_aircopbench_score_can_be_saved_and_read(tmp_path):
    # One real case proves loader, scorer, and persistence boundaries connect.
    case = next(load_cases())
    row = score_result(case, case["correct_answer"], latency_ms=1.0, deadline_ms=400.0)
    path = tmp_path / "results.jsonl"

    write_jsonl(path, [row])

    assert list(read_jsonl(path)) == [row]
