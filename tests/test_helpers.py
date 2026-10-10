import pytest

pytest.importorskip("langchain_core")

from scripts.eval_rag import keyword_hit, parse_scores  # noqa: E402
from scripts.make_pairs import parse_questions, split_sources  # noqa: E402


def test_parse_questions_handles_code_fences_and_junk():
    assert parse_questions('```json\n["How many leave days do I get?", "x"]\n```') == ["How many leave days do I get?"]
    assert parse_questions("not json") == []


def test_split_sources_is_disjoint_and_deterministic():
    train, held = split_sources(["a", "b", "c", "d", "a"], 0.25)
    assert train.isdisjoint(held) and len(held) == 1 and len(train) == 3
    assert split_sources(["a", "b", "c", "d"], 0.25) == (train, held)


def test_split_sources_needs_two_documents():
    with pytest.raises(SystemExit):
        split_sources(["only.md"], 0.25)


def test_keyword_hit_and_parse_scores():
    assert keyword_hit("You get 24 days. Carry over 10 until 31 March.", ["24", "31 march"])
    assert not keyword_hit("You get 24 days.", ["24", "10"])
    assert parse_scores('{"faithfulness": 5, "relevance": 4}') == (5, 4)
    assert parse_scores("garbage") is None
