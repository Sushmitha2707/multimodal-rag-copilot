"""Multimodal ingestion: PDF / text / image / audio / video  ->  LangChain Documents.

Every Document carries metadata: source, modality and a locator (page or time range)
so answers can cite exactly where a fact came from.
"""

from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from copilot.config import get_settings

TEXT_EXT = {".txt", ".md"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
SUPPORTED_EXT = TEXT_EXT | PDF_EXT | IMAGE_EXT | AUDIO_EXT | VIDEO_EXT


# ----------------------------------------------------------------------------- loaders
def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "modality": "text", "locator": "full"})]


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # scanned pages with no text layer are skipped (add OCR if you need them)
            docs.append(
                Document(page_content=text, metadata={"source": path.name, "modality": "pdf", "locator": f"page {i}"})
            )
    return docs


@lru_cache(maxsize=1)
def _captioner():
    from transformers import pipeline

    return pipeline("image-to-text", model=get_settings().caption_model)


def load_image(path: Path) -> list[Document]:
    """Caption with BLIP (what is in the image) + OCR with Tesseract (text inside the image, if available)."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    caption = _captioner()(img)[0]["generated_text"]
    parts = [f"[Image caption] {caption}"]
    try:
        import pytesseract

        ocr = pytesseract.image_to_string(img).strip()
        if ocr:
            parts.append(f"[Text in image] {ocr}")
    except Exception:  # tesseract binary not installed -> caption only
        pass
    return [
        Document(page_content="\n".join(parts), metadata={"source": path.name, "modality": "image", "locator": "image"})
    ]


@lru_cache(maxsize=1)
def _asr():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=get_settings().whisper_model,
        chunk_length_s=30,
        device=device,
    )


def _fmt(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def group_segments(segments: list[dict], window_s: float = 45.0) -> list[dict]:
    """Merge Whisper timestamped segments into ~window_s windows. Pure function (unit tested)."""
    groups, cur, start, end = [], [], None, None
    for seg in segments:
        s, e = seg["timestamp"]
        s = 0.0 if s is None else s
        e = s if e is None else e
        if start is None:
            start = s
        cur.append(seg["text"].strip())
        end = e
        if end - start >= window_s:
            groups.append({"start": start, "end": end, "text": " ".join(cur)})
            cur, start = [], None
    if cur:
        groups.append({"start": start, "end": end, "text": " ".join(cur)})
    return groups


def load_audio(path: Path, modality: str = "audio") -> list[Document]:
    out = _asr()(str(path), return_timestamps=True)
    segments = [{"timestamp": c["timestamp"], "text": c["text"]} for c in out.get("chunks", [])]
    if not segments:
        segments = [{"timestamp": (0.0, 0.0), "text": out["text"]}]
    return [
        Document(
            page_content=g["text"],
            metadata={
                "source": path.name,
                "modality": modality,
                "locator": f"{_fmt(g['start'])}-{_fmt(g['end'])}",
            },
        )
        for g in group_segments(segments)
    ]


def load_video(path: Path) -> list[Document]:
    """Extract the audio track with ffmpeg, then transcribe it."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
            capture_output=True,
        )
        docs = load_audio(wav, modality="video")
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return load_text(path)
    if ext in PDF_EXT:
        return load_pdf(path)
    if ext in IMAGE_EXT:
        return load_image(path)
    if ext in AUDIO_EXT:
        return load_audio(path)
    if ext in VIDEO_EXT:
        return load_video(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_directory(directory: str | Path) -> list[Document]:
    docs: list[Document] = []
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            docs.extend(load_file(p))
    return docs


# ----------------------------------------------------------------------------- chunking
def chunk_documents(docs: list[Document]) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
