import pytest

from src.results import ResultFormatError, append_jsonl, read_jsonl, write_jsonl
from src.scoring import score_answer


def test_write_and_read_jsonl_preserves_scored_result_row(tmp_path):
    # Persist a real score dict, not a pretend model result.
    row = score_answer("Answer: A", "A", latency_ms=100, deadline_ms=400)
    path = tmp_path / "results" / "scores.jsonl"

    write_jsonl(path, [row])

    assert list(read_jsonl(path)) == [row]


def test_write_jsonl_creates_parent_directories(tmp_path):
    # Result writers should work before output folders exist.
    path = tmp_path / "nested" / "scores.jsonl"

    write_jsonl(path, [score_answer("B", "B", latency_ms=1, deadline_ms=400)])

    assert path.exists()


def test_append_jsonl_preserves_existing_rows(tmp_path):
    path = tmp_path / "scores.jsonl"
    write_jsonl(path, [{"question_id": "case-1"}])

    append_jsonl(path, [{"question_id": "case-2"}])

    assert list(read_jsonl(path)) == [{"question_id": "case-1"}, {"question_id": "case-2"}]


def test_read_jsonl_skips_blank_lines(tmp_path):
    # Blank lines should not create empty result rows.
    path = tmp_path / "scores.jsonl"
    path.write_text('{"correct": true}\n\n{"correct": false}\n', encoding="utf-8")

    assert list(read_jsonl(path)) == [{"correct": True}, {"correct": False}]


def test_read_jsonl_fails_when_line_is_not_json_object(tmp_path):
    # Result rows must be objects so downstream metrics can use named fields.
    path = tmp_path / "scores.jsonl"
    path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ResultFormatError, match="line 1 is not a JSON object"):
        list(read_jsonl(path))
