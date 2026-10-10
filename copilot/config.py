"""Central configuration, read from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    vector_backend: str = field(default_factory=lambda: os.getenv("VECTOR_BACKEND", "faiss"))
    index_dir: str = field(default_factory=lambda: os.getenv("INDEX_DIR", "data/index"))
    pinecone_index: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX", "copilot-index"))
    pinecone_namespace: str = field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "default"))

    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))

    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "openai/whisper-base"))
    caption_model: str = field(
        default_factory=lambda: os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    )
    feedback_db: str = field(default_factory=lambda: os.getenv("FEEDBACK_DB", "data/feedback.sqlite"))
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "data/uploads"))


def get_settings() -> Settings:
    return Settings()
