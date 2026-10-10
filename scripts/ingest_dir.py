"""Index a folder of documents (text / PDF / images / audio / video).

python -m scripts.ingest_dir data/sample_docs
"""

import argparse

from copilot.ingest import chunk_documents, load_directory
from copilot.vectorstore import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--backend", default=None, help="faiss | pinecone (default from .env)")
    ap.add_argument("--embedding-model", default=None, help="HF id or local path (e.g. models/embedding-ft)")
    args = ap.parse_args()

    docs = load_directory(args.directory)
    chunks = chunk_documents(docs)
    store = VectorStore(backend=args.backend, embedding_model=args.embedding_model)
    n = store.add(chunks)
    mods = sorted({d.metadata["modality"] for d in docs})
    print(f"Loaded {len(docs)} documents ({', '.join(mods)}) -> indexed {n} chunks in {store.backend}.")


if __name__ == "__main__":
    main()
