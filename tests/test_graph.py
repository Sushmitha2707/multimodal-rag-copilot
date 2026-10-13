import pytest

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from copilot.graph import FALLBACK, GraphRAG, clean_query, is_yes  # noqa: E402

DOC = Document(
    page_content="Annual leave is 24 days.", metadata={"source": "leave.md", "modality": "text", "locator": "full"}
)


class FakeStore:
    """Returns the leave document only for queries containing 'leave'; records every query."""

    def __init__(self):
        self.queries = []

    def search(self, query, k=None):
        self.queries.append(query)
        return [(DOC, 0.1)] if "leave" in query.lower() else []


def fake_llm_factory(grade_replies):
    """Fake chat model. Grader replies come from the list; rewriter returns 'annual leave days'."""
    replies = iter(grade_replies)

    def respond(prompt_value):
        text = prompt_value.to_string()
        if "Reply with exactly one word" in text:
            return AIMessage(content=next(replies))
        if "Rewrite the user's question" in text:
            return AIMessage(content='"annual leave days"')
        return AIMessage(content="24 days [1]")

    return lambda provider: RunnableLambda(respond)


def test_relevant_passages_go_straight_to_generate():
    store = FakeStore()
    out = GraphRAG(store, llm_factory=fake_llm_factory(["YES"])).answer("How much leave do I get?")
    assert out["answer"] == "24 days [1]"
    assert [t.split(":")[0] for t in out["trace"]] == ["retrieve", "grade", "generate"]
    assert out["sources"][0]["source"] == "leave.md" and out["context"].startswith("[1]")


def test_grader_says_no_then_rewritten_query_succeeds():
    store = FakeStore()
    question = "What is the time-off leave allowance?"  # retrieves a passage, but the grader rejects it first
    out = GraphRAG(store, llm_factory=fake_llm_factory(["NO", "YES"])).answer(question)
    assert out["answer"] == "24 days [1]"
    assert store.queries == [question, "annual leave days"]  # the retry uses the rewritten query
    assert [t.split(":")[0] for t in out["trace"]] == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]


def test_empty_first_retrieval_skips_grader_and_retries_with_rewrite():
    store = FakeStore()
    out = GraphRAG(store, llm_factory=fake_llm_factory(["YES"])).answer("How many vacation days?")
    assert out["answer"] == "24 days [1]"
    assert store.queries == ["How many vacation days?", "annual leave days"]
    assert [t.split(":")[0] for t in out["trace"]] == ["retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]


def test_unanswerable_question_ends_in_fallback_after_max_attempts():
    store = FakeStore()
    out = GraphRAG(store, llm_factory=fake_llm_factory(["NO", "NO"]), max_attempts=2).answer("Capital of France?")
    assert out["answer"] == FALLBACK
    assert len(store.queries) == 2 and out["trace"][-1] == "fallback"


def test_empty_retrieval_skips_the_grader():
    class EmptyStore:
        def search(self, query, k=None):
            return []

    # grader replies list is empty: calling the grader would raise StopIteration
    out = GraphRAG(EmptyStore(), llm_factory=fake_llm_factory([]), max_attempts=1).answer("anything at all")
    assert out["answer"] == FALLBACK and out["sources"] == []


def test_helpers():
    assert is_yes("YES") and is_yes(" yes, it does") and is_yes('"Yes"')
    assert not is_yes("NO") and not is_yes("yesterday I went") and not is_yes("")
    assert clean_query('"annual leave days"\nextra line', "x") == "annual leave days"
    assert clean_query("   ", "fallback q") == "fallback q"
