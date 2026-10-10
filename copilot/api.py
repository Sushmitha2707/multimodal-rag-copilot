"""FastAPI backend: /ingest, /ask, /feedback, /health.

    uvicorn copilot.api:app --reload
Optional protection: set API_KEY and send it in the `x-api-key` header.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from copilot import feedback
from copilot.config import get_settings

app = FastAPI(title="Multimodal RAG Copilot", version="1.0")
_pipeline = None


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from copilot.rag import RAGPipeline  # imported lazily: loads embedding model + vector store

        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    provider: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: int
    corrected_answer: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest) -> dict:
    try:
        out = get_pipeline().answer(req.question, provider=req.provider, k=req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    out["interaction_id"] = feedback.log_interaction(
        req.question, out["answer"], out["context"], out["sources"], req.provider or get_settings().llm_provider
    )
    out.pop("context", None)  # the context is stored server-side for DPO; clients only need sources
    return out


@app.post("/feedback", dependencies=[Depends(require_key)])
def post_feedback(req: FeedbackRequest) -> dict:
    try:
        ok = feedback.set_feedback(req.interaction_id, req.rating, req.corrected_answer)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown interaction_id")
    return {"saved": True}


@app.post("/ingest", dependencies=[Depends(require_key)])
def ingest(files: list[UploadFile] = File(...)) -> dict:
    from copilot.ingest import SUPPORTED_EXT, chunk_documents, load_file

    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    indexed, errors = 0, {}
    for f in files:
        name = Path(f.filename or "upload").name  # strip any path components
        if Path(name).suffix.lower() not in SUPPORTED_EXT:
            errors[name] = "unsupported file type"
            continue
        dest = upload_dir / name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            indexed += get_pipeline().store.add(chunk_documents(load_file(dest)))
        except Exception as e:  # report per-file failures instead of failing the whole batch
            errors[name] = str(e)
    return {"chunks_indexed": indexed, "errors": errors}
