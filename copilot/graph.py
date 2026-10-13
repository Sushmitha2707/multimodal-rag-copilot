"""LangGraph workflow: corrective RAG.

    retrieve -> grade -+-> generate -> END                (passages can answer the question)
                       +-> rewrite -> retrieve ...        (not relevant, retries left: rewrite the query)
                       +-> fallback -> END                (not relevant, no retries left: "I don't know")

The extra grading step stops the model from answering out of weak or irrelevant context. The cost is one
more LLM call per question (two or three for questions the documents cannot answer).
"""

from __future__ import annotations

import re
import time
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from copilot.llm import get_chat_model
from copilot.prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT, format_context
from copilot.vectorstore import VectorStore

FALLBACK = "I don't know based on the provided documents."

ANSWER_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)])

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check whether retrieved passages contain the information needed to answer a question. "
            "Reply with exactly one word: YES if they do, NO if they do not.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{context}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question as a short, specific search query for a document search engine. "
            "Use likely keywords from the documents. Return ONLY the query.",
        ),
        ("human", "{question}"),
    ]
)


class GraphState(TypedDict, total=False):
    question: str
    provider: str | None
    k: int | None
    query: str  # rewritten search query (empty on the first attempt)
    hits: list  # [(Document, score)]
    context: str
    relevant: bool
    attempts: int
    answer: str
    trace: list[str]  # which nodes ran, for debugging and the UI


def is_yes(verdict: str) -> bool:
    return re.match(r"\W*yes\b", verdict.strip(), re.I) is not None


def clean_query(text: str, fallback: str) -> str:
    lines = [ln.strip(" \t\"'`*-") for ln in text.strip().splitlines() if ln.strip()]
    return lines[0] if lines else fallback


class GraphRAG:
    def __init__(self, store: VectorStore | None = None, max_attempts: int = 2, llm_factory=get_chat_model):
        self.store = store or VectorStore()
        self.max_attempts = max_attempts
        self._llm_factory = llm_factory
        self._llms: dict = {}
        self.graph = self._build()

    # ---- helpers -----------------------------------------------------------------
    def _llm(self, provider):
        if provider not in self._llms:
            self._llms[provider] = self._llm_factory(provider)
        return self._llms[provider]

    def _run(self, prompt, provider, **variables) -> str:
        return (prompt | self._llm(provider) | StrOutputParser()).invoke(variables)

    # ---- nodes -------------------------------------------------------------------
    def _retrieve(self, state: GraphState) -> dict:
        query = state.get("query") or state["question"]
        hits = self.store.search(query, k=state.get("k"))
        context = format_context([d for d, _ in hits]) if hits else ""
        trace = state.get("trace", []) + [f"retrieve: {query}"]
        return {"hits": hits, "context": context, "attempts": state.get("attempts", 0) + 1, "trace": trace}

    def _grade(self, state: GraphState) -> dict:
        if not state["hits"]:
            relevant = False
        else:
            verdict = self._run(
                GRADE_PROMPT, state.get("provider"), question=state["question"], context=state["context"]
            )
            relevant = is_yes(verdict)
        return {"relevant": relevant, "trace": state["trace"] + [f"grade: {'yes' if relevant else 'no'}"]}

    def _rewrite(self, state: GraphState) -> dict:
        raw = self._run(REWRITE_PROMPT, state.get("provider"), question=state["question"])
        return {"query": clean_query(raw, state["question"]), "trace": state["trace"] + ["rewrite"]}

    def _generate(self, state: GraphState) -> dict:
        answer = self._run(ANSWER_PROMPT, state.get("provider"), context=state["context"], question=state["question"])
        return {"answer": answer, "trace": state["trace"] + ["generate"]}

    def _fallback(self, state: GraphState) -> dict:
        return {"answer": FALLBACK, "trace": state["trace"] + ["fallback"]}

    def _route(self, state: GraphState) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state["attempts"] < self.max_attempts else "fallback"

    # ---- graph -------------------------------------------------------------------
    def _build(self):
        g = StateGraph(GraphState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("fallback", self._fallback)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", self._route, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"}
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)
        g.add_edge("fallback", END)
        return g.compile()

    # ---- public API (same shape as RAGPipeline.answer, plus a trace) --------------
    def answer(self, question: str, provider: str | None = None, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        final = self.graph.invoke({"question": question, "provider": provider, "k": k, "attempts": 0, "trace": []})
        hits = final.get("hits", [])
        return {
            "answer": final["answer"],
            "sources": [
                {
                    "n": i,
                    "source": d.metadata.get("source"),
                    "modality": d.metadata.get("modality"),
                    "locator": d.metadata.get("locator"),
                    "score": float(score),
                    "snippet": d.page_content[:240],
                }
                for i, (d, score) in enumerate(hits, start=1)
            ],
            "context": final.get("context", ""),
            "latency_s": round(time.perf_counter() - t0, 3),
            "provider": provider,
            "trace": final.get("trace", []),
        }
