"""Embedding model wrapper. Works with a Hugging Face model id OR a local fine-tuned folder."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from copilot.config import get_settings

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=4)
def get_embeddings(model_name_or_path: str | None = None) -> "HuggingFaceEmbeddings":
    from langchain_huggingface import HuggingFaceEmbeddings  # lazy: heavy import

    name = model_name_or_path or get_settings().embedding_model
    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})
