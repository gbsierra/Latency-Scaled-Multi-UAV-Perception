"""Build AirCopBench multiple-choice workflow prompts."""

from __future__ import annotations

from collections.abc import Mapping


OPTION_ORDER = ("A", "B", "C", "D")
ANSWER_INSTRUCTION = "Answer with the option letter only."
GLOBAL_SINGLE_PROMPT_ID = "global_single_v1"
PER_UAV_OBSERVATION_PROMPT_ID = "per_uav_observation_v2"
FUSION_COMPARER_PROMPT_ID = "fusion_comparer_v2"


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


def build_global_single_prompt(question: str, options: Mapping[str, str]) -> str:
    """Build the all-images single-call AirCopBench answer prompt."""
    instruction = (
        "You are answering a multiple-choice AirCopBench UAV perception question.\n"
        "Use the provided UAV images, question, and answer choices.\n"
        "Return only one letter: A, B, C, or D.\n\n"
    )
    return instruction + build_answer_prompt(question, options)


def build_per_uav_observation_prompt(
    uav_id: str,
    question: str,
    options: Mapping[str, str],
) -> str:
    """Build a compact one-view observer prompt for parallel fusion workflows."""
    uav_text = uav_id.strip()
    if not uav_text:
        raise PromptFormatError("uav_id is empty")

    question_text = question.strip()
    if not question_text:
        raise PromptFormatError("question is empty")

    return (
        f"You are inspecting only {uav_text}'s UAV image.\n"
        "Use only this image. Do not infer from other UAVs.\n"
        "Do not choose A, B, C, or D as the final answer. Return visual evidence only.\n"
        "For supports, name the option letter directly supported by this image, or none, or uncertain.\n\n"
        f"Question: {question_text}\n"
        f"{format_options(options)}\n\n"
        "Return exactly this structure:\n"
        "visibility: clear | degraded | blocked | uncertain\n"
        "usefulness: high | medium | low | uncertain\n"
        "evidence:\n"
        "- ...\n"
        "- ...\n"
        "supports: A | B | C | D | none | uncertain"
    )


def build_fusion_prompt(
    question: str,
    options: Mapping[str, str],
    observations: Mapping[str, str],
) -> str:
    """Build the text-only fusion prompt that returns one final option letter."""
    question_text = question.strip()
    if not question_text:
        raise PromptFormatError("question is empty")

    if not observations:
        raise PromptFormatError("observations are required")

    observation_lines: list[str] = []
    for uav_id, observation in observations.items():
        observation_text = str(observation).strip()
        if not observation_text:
            raise PromptFormatError(f"{uav_id} observation is empty")
        observation_lines.append(f"{uav_id}: {observation_text}")

    return (
        "You are the fusion/comparer for an AirCopBench multi-UAV question.\n"
        "Use the structured per-UAV observations to compare evidence against the same answer choices.\n"
        "Prefer high-usefulness direct evidence.\n"
        "Discount degraded, blocked, or uncertain observations.\n"
        "If observations conflict, choose the option best supported by the clearest useful views.\n"
        "Return only one final option letter: A, B, C, or D.\n\n"
        f"Question: {question_text}\n"
        f"{format_options(options)}\n\n"
        "Per-UAV observations:\n"
        + "\n".join(observation_lines)
        + "\n\nFinal answer letter only."
    )
