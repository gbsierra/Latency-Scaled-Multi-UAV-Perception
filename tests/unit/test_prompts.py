import pytest

from src.prompts import ANSWER_INSTRUCTION, PromptFormatError, build_answer_prompt, format_options


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
