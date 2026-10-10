"""One interface over two vector databases: FAISS (local) and Pinecone (hosted)."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document

from copilot.config import get_settings
from copilot.embeddings import get_embeddings


class VectorStore:
    def __init__(self, backend: str | None = None, embedding_model: str | None = None):
        self.s = get_settings()
        self.backend = (backend or self.s.vector_backend).lower()
        self.emb = get_embeddings(embedding_model)
        self._store = None
        self._load()

    # ---- setup ---------------------------------------------------------------
    def _faiss_dir(self) -> Path:
        return Path(self.s.index_dir)

    def _load(self) -> None:
        if self.backend == "faiss":
            from langchain_community.vectorstores import FAISS

            d = self._faiss_dir()
            if (d / "index.faiss").exists():
                self._store = FAISS.load_local(str(d), self.emb, allow_dangerous_deserialization=True)
        elif self.backend == "pinecone":
            from langchain_pinecone import PineconeVectorStore

            self._ensure_pinecone_index()
            self._store = PineconeVectorStore(
                index_name=self.s.pinecone_index, embedding=self.emb, namespace=self.s.pinecone_namespace
            )
        else:
            raise ValueError(f"Unknown VECTOR_BACKEND: {self.backend}")

    def _ensure_pinecone_index(self) -> None:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        names = [i["name"] for i in pc.list_indexes()]
        if self.s.pinecone_index not in names:
            dim = len(self.emb.embed_query("dimension probe"))
            pc.create_index(
                name=self.s.pinecone_index,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    # ---- operations ----------------------------------------------------------
    def add(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        if self.backend == "faiss":
            from langchain_community.vectorstores import FAISS

            if self._store is None:
                self._store = FAISS.from_documents(chunks, self.emb)
            else:
                self._store.add_documents(chunks)
            self._faiss_dir().mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(self._faiss_dir()))
        else:
            self._store.add_documents(chunks)
        return len(chunks)

    def is_empty(self) -> bool:
        return self._store is None

    def search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        if self._store is None:
            return []
        return self._store.similarity_search_with_score(query, k=k or self.s.top_k)
