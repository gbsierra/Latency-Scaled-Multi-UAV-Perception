"""Build AirCopBench multiple-choice prompts."""

from __future__ import annotations

from collections.abc import Mapping


OPTION_ORDER = ("A", "B", "C", "D")
ANSWER_INSTRUCTION = "Answer with the option letter only."


class PromptFormatError(ValueError):
    """Raised when a case cannot be formatted into a valid prompt."""


def format_options(options: Mapping[str, str]) -> str:
    """Return answer choices in stable A/B/C/D order."""
    missing = [letter for letter in OPTION_ORDER if letter not in options]
    if missing:
        raise PromptFormatError(f"missing options: {', '.join(missing)}")

    # Stable option order keeps prompts comparable across model runs.
    return "\n".join(f"{letter}. {options[letter]}" for letter in OPTION_ORDER)


def build_answer_prompt(question: str, options: Mapping[str, str]) -> str:
    """Build the single-pass answer prompt without model-specific tuning."""
    question_text = question.strip()
    if not question_text:
        raise PromptFormatError("question is empty")

    return f"{question_text}\n{format_options(options)}\n\n{ANSWER_INSTRUCTION}"
