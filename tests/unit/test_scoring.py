from src.scoring import parse_answer, score_answer, score_result


def test_parse_answer_accepts_common_answer_formats():
    # Accept common concise model response formats.
    assert parse_answer("A") == "A"
    assert parse_answer("Answer: B") == "B"
    assert parse_answer('{"answer": "C"}') == "C"
    assert parse_answer("The answer is D") == "D"


def test_parse_answer_returns_none_for_missing_or_invalid_answers():
    # Invalid responses stay invalid instead of becoming random guesses.
    assert parse_answer(None) is None
    assert parse_answer("unclear") is None


def test_score_answer_marks_correct_on_time_answer_actionable():
    # Correct answers count only when they arrive before the deadline.
    result = score_answer("Answer: A", "A", latency_ms=100, deadline_ms=400)

    assert result["parse_failed"] is False
    assert result["correct"] is True
    assert result["stale"] is False
    assert result["actionable"] is True


def test_score_answer_marks_correct_late_answer_stale_not_actionable():
    # Correct-but-late answers are stale and not actionable.
    result = score_answer("B", "B", latency_ms=500, deadline_ms=400)

    assert result["correct"] is True
    assert result["stale"] is True
    assert result["actionable"] is False


def test_score_answer_marks_parse_failure_not_correct():
    # Parse failures are tracked separately and never counted correct.
    result = score_answer("unclear", "C", latency_ms=100, deadline_ms=400)

    assert result["parsed_answer"] is None
    assert result["parse_failed"] is True
    assert result["correct"] is False
    assert result["actionable"] is False


def test_score_result_adds_case_metadata_to_answer_score():
    # AirCopBench wrapper preserves case provenance around generic scoring.
    case = {
        "question_id": "case-1",
        "source_group": "Sim5",
        "source_split": "test",
        "question_type": "Causal Assessment",
        "correct_answer": "D",
    }

    result = score_result(case, "Option D", latency_ms=1, deadline_ms=400)

    assert result["question_id"] == "case-1"
    assert result["source_group"] == "Sim5"
    assert result["source_split"] == "test"
    assert result["question_type"] == "Causal Assessment"
    assert result["correct"] is True
    assert result["actionable"] is True
