# Multimodal Knowledge Copilot (RAG + embedding fine-tuning + preference feedback)

A production-style Generative AI application: ask questions over **PDFs, text, images, audio and video** and get
**cited** answers. It includes the full LLM-engineering loop - retrieval, evaluation, fine-tuning and feedback.

```
 files ──► ingest ───────────────► chunk ──► embed (HF / fine-tuned) ──► vector DB (FAISS | Pinecone)
          PDF: pypdf                                                         │
          image: BLIP caption + OCR                                          ▼
          audio/video: Whisper (+ffmpeg)      question ──► retrieve top-k ──► LangChain LCEL prompt
                                                                              │
                                      OpenAI | Anthropic | Gemini  ◄──────────┘
                                                  │
                          cited answer ──► FastAPI ──► Streamlit chat UI ──► 👍/👎 + correction
                                                                              │
                                      SQLite feedback ──► preference pairs ──► LoRA + DPO tuning
```

## What is in the box

| Capability | Where |
|---|---|
| Multimodal ingestion (PDF / text / image / audio / video) with page + timestamp citations | `copilot/ingest.py` |
| Embeddings (Hugging Face, PyTorch) and **fine-tuning** with MultipleNegativesRankingLoss | `scripts/train_embeddings.py` |
| Synthetic training data from your corpus, with a leak-free split by source document | `scripts/make_pairs.py` |
| **FAISS and Pinecone** behind one interface | `copilot/vectorstore.py` |
| **LangChain** RAG chain, grounded prompt, "I don't know" fallback | `copilot/rag.py`, `copilot/prompts.py` |
| **Provider-switchable LLM** (OpenAI, Anthropic, Gemini) | `copilot/llm.py` |
| Retrieval evaluation (Recall@k, MRR) base vs fine-tuned | `scripts/eval_retrieval.py` |
| RAG evaluation (keyword accuracy, LLM-judge faithfulness and relevance, latency) per provider | `scripts/eval_rag.py` |
| Feedback loop + **DPO preference tuning** (LoRA, loss written in plain PyTorch) | `copilot/feedback.py`, `copilot/dpo.py`, `scripts/train_dpo.py` |
| FastAPI service (API-key option) and Streamlit chat UI | `copilot/api.py`, `app.py` |
| Docker, CI (ruff + pytest), unit tests | `Dockerfile`, `.github/workflows/ci.yml`, `tests/` |

## Quickstart (about 10 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add at least one LLM key (e.g. GOOGLE_API_KEY)

python -m scripts.ingest_dir data/sample_docs      # index the 4 sample documents
uvicorn copilot.api:app --reload                   # terminal 1: API on :8000
streamlit run app.py                               # terminal 2: chat UI
```

Linux/macOS system packages for multimodal input: `ffmpeg` (video/audio) and optionally `tesseract-ocr` (text inside images).
Whisper and BLIP are downloaded from Hugging Face on first use; they are slow on CPU, so try short clips first.

## The experiment workflow (this produces the numbers for your resume)

The 4 sample documents are only a smoke test. For meaningful results use **30+ real documents from one domain**
(for example public HR/leave policies, insurance or banking circulars, product manuals, open-source docs, textbook chapters).

```bash
# 1. Put your corpus in data/my_corpus/ (PDFs, md, txt ...)

# 2. Generate (question, passage) pairs; held-out documents are never used for training
python -m scripts.make_pairs --corpus data/my_corpus --provider gemini

# 3. Baseline vs fine-tuned embeddings  (use a GPU, e.g. free Colab T4, if you can)
python -m scripts.train_embeddings --pairs data/pairs/train.jsonl --out models/embedding-ft
python -m scripts.eval_retrieval --pairs data/pairs/heldout.jsonl \
    --models sentence-transformers/all-MiniLM-L6-v2 models/embedding-ft
#   -> results/retrieval_eval.json  (Recall@1/3/5, MRR@10 for both models)

# 4. Re-index with the tuned model (set EMBEDDING_MODEL=models/embedding-ft in .env), then compare LLM providers
rm -rf data/index && python -m scripts.ingest_dir data/my_corpus --embedding-model models/embedding-ft
python -m scripts.eval_rag --providers gemini openai anthropic --judge-provider gemini
#   -> results/rag_eval.json   (extend eval/qa_set.jsonl to 50+ questions about YOUR corpus first)

# 5. Pinecone vs FAISS: set PINECONE_API_KEY, re-run ingest with --backend pinecone, compare latency + recall

# 6. Feedback + preference tuning: use the UI, click 👎 and type a better answer for ~50+ answers, then
pip install -r requirements-dpo.txt
python -m scripts.train_dpo --model Qwen/Qwen2.5-0.5B-Instruct
#   -> results/dpo_eval.json (preference accuracy on held-out pairs, before vs after)
```

### Results (fill in with YOUR measured numbers - do not publish placeholders)

| Embedding model | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| all-MiniLM-L6-v2 (base) | _ | _ | _ |
| fine-tuned | _ | _ | _ |

| Provider | keyword_acc | faithfulness | relevance | latency (s) |
|---|---|---|---|---|
| Gemini | _ | _ | _ | _ |
| OpenAI | _ | _ | _ | _ |
| Anthropic | _ | _ | _ | _ |

## Deploy to Cloud Run

Cloud Run's disk is ephemeral, so use **Pinecone** as the vector store (`VECTOR_BACKEND=pinecone`) and treat the
SQLite feedback file as temporary (move it to Cloud SQL/Firestore for real use).

```bash
gcloud run deploy knowledge-copilot --source . --region asia-south1 --memory 4Gi --cpu 2 --timeout 900 \
  --set-env-vars VECTOR_BACKEND=pinecone,LLM_PROVIDER=gemini,API_KEY=choose-a-secret \
  --set-secrets GOOGLE_API_KEY=google-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest
```

Deploy the Streamlit UI separately (or run it locally) and point it to the service with `API_URL` and `API_KEY`.

## Honest limitations

- **DPO here is lightweight preference optimisation, not full RLHF** (no reward model, no PPO). Describe it as
  "preference-feedback loop with DPO". With only dozens of pairs it demonstrates the pipeline; it will not make a large
  quality jump, so report the held-out preference accuracy exactly as measured.
- Synthetic questions are written by an LLM, so held-out numbers measure retrieval on that question style. The
  hand-written `eval/qa_set.jsonl` (extend it!) is the more realistic end-to-end check.
- Scanned PDFs with no text layer are skipped (add OCR per page if you need them). Video support uses the audio track only.
- Model names in `.env.example` change over time; check each provider's documentation for current ones.
- The API is unauthenticated unless `API_KEY` is set. Do not expose `/ingest` publicly without it.

## Resume bullets - use ONLY after you have run it and have real numbers

- Built a multimodal RAG copilot (LangChain, FAISS/Pinecone, FastAPI, Streamlit) over PDFs, images and audio/video using
  Whisper and BLIP, with page/timestamp-cited answers across OpenAI, Anthropic and Gemini backends.
- Fine-tuned a Hugging Face sentence-transformer (PyTorch, MultipleNegativesRankingLoss) on N synthetic query-passage pairs,
  improving held-out Recall@5 from X to Y and MRR@10 from A to B; published the model on the Hugging Face Hub.
- Evaluated RAG quality on a Q-question set (keyword accuracy, LLM-judge faithfulness/relevance, latency) and compared
  three LLM providers; reached F faithfulness at L s latency.
- Implemented a feedback-driven preference-tuning loop (LoRA + DPO in PyTorch) from M user corrections; preference accuracy
  on held-out pairs moved from P to Q.
- Containerized with Docker, CI via GitHub Actions (ruff, pytest), deployed on GCP Cloud Run.

## Tests

```bash
pip install -r requirements-dev.txt fastapi httpx python-multipart pydantic langchain-core langchain-text-splitters
pytest -q
```
