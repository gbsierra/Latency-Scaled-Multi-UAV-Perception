import pytest

from src.prompts import (
    ANSWER_INSTRUCTION,
    PromptFormatError,
    build_answer_prompt,
    build_fusion_prompt,
    build_global_single_prompt,
    build_per_uav_observation_prompt,
    format_options,
)


def test_format_options_uses_stable_answer_order():
    # Prompt option order must not depend on input dictionary order.
    options = {
        "C": "Center",
        "A": "Left",
        "D": "Not visible",
        "B": "Right",
    }

    assert format_options(options) == "A. Left\nB. Right\nC. Center\nD. Not visible"


def test_format_options_fails_when_choice_is_missing():
    # Bad prompts should fail before reaching model workflows.
    with pytest.raises(PromptFormatError, match="missing options: D"):
        format_options({"A": "Left", "B": "Right", "C": "Center"})


def test_build_answer_prompt_combines_question_options_and_instruction():
    # The answer prompt is shared scaffolding for future model calls.
    prompt = build_answer_prompt(
        "Where is the vehicle?",
        {
            "A": "Left",
            "B": "Right",
            "C": "Center",
            "D": "Not visible",
        },
    )

    assert prompt == (
        "Where is the vehicle?\n"
        "A. Left\n"
        "B. Right\n"
        "C. Center\n"
        "D. Not visible\n\n"
        f"{ANSWER_INSTRUCTION}"
    )


def test_build_answer_prompt_fails_when_question_is_empty():
    # Empty questions indicate malformed metadata or caller misuse.
    with pytest.raises(PromptFormatError, match="question is empty"):
        build_answer_prompt("  ", {"A": "Left", "B": "Right", "C": "Center", "D": "Not visible"})


def test_build_global_single_prompt_adds_aircopbench_instruction():
    prompt = build_global_single_prompt(
        "Where is the vehicle?",
        {"A": "Left", "B": "Right", "C": "Center", "D": "Not visible"},
    )

    assert "AirCopBench UAV perception question" in prompt
    assert "Return only one letter" in prompt
    assert "A. Left" in prompt


def test_build_per_uav_observation_prompt_limits_context_to_one_view():
    prompt = build_per_uav_observation_prompt(
        "UAV2",
        "Which view is useful?",
        {"A": "UAV1", "B": "UAV2", "C": "UAV3", "D": "None"},
    )

    assert "only UAV2's UAV image" in prompt
    assert "Do not infer from other UAVs" in prompt
    assert "Which view is useful?" in prompt
    assert "1-3 short bullet points" in prompt


def test_build_fusion_prompt_includes_observations_and_final_answer_instruction():
    prompt = build_fusion_prompt(
        "Which view is useful?",
        {"A": "UAV1", "B": "UAV2", "C": "UAV3", "D": "None"},
        {"UAV1": "blurred image", "UAV2": "clear vehicle visible"},
    )

    assert "fusion/comparer" in prompt
    assert "UAV1: blurred image" in prompt
    assert "UAV2: clear vehicle visible" in prompt
    assert "Final answer letter only" in prompt


def test_build_fusion_prompt_fails_without_observations():
    with pytest.raises(PromptFormatError, match="observations are required"):
        build_fusion_prompt(
            "Question?",
            {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            {},
        )
