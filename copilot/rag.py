"""RAG pipeline (LangChain LCEL): retrieve -> grounded prompt -> LLM -> cited answer."""

from __future__ import annotations

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from copilot.llm import get_chat_model
from copilot.prompts import HUMAN_TEMPLATE, SYSTEM_PROMPT, format_context
from copilot.vectorstore import VectorStore

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)])


class RAGPipeline:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()

    def answer(self, question: str, provider: str | None = None, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        hits = self.store.search(question, k=k)
        docs = [d for d, _ in hits]
        if not docs:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "context": "",
                "latency_s": 0.0,
                "provider": provider,
            }
        context = format_context(docs)
        chain = PROMPT | get_chat_model(provider) | StrOutputParser()
        text = chain.invoke({"context": context, "question": question})
        return {
            "answer": text,
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
            "context": context,
            "latency_s": round(time.perf_counter() - t0, 3),
            "provider": provider,
        }
