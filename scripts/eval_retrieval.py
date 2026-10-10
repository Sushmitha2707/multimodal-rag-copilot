"""Compare embedding models on held-out pairs (Recall@k, MRR@10).

    python -m scripts.eval_retrieval --pairs data/pairs/heldout.jsonl \
        --models sentence-transformers/all-MiniLM-L6-v2 models/embedding-ft
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from copilot.metrics import rank_metrics


def evaluate_model(name: str, queries: list[str], passages: list[str], relevant: list[set[int]]) -> dict:
    model = SentenceTransformer(name)
    q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    p = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
    return rank_metrics(np.asarray(q) @ np.asarray(p).T, relevant, ks=(1, 3, 5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/heldout.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default="results/retrieval_eval.json")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.pairs)]
    passages = sorted({r["passage"] for r in rows})
    index = {p: i for i, p in enumerate(passages)}
    queries = [r["query"] for r in rows]
    relevant = [{index[r["passage"]]} for r in rows]
    print(f"{len(queries)} queries over {len(passages)} passages")

    results = {m: evaluate_model(m, queries, passages, relevant) for m in args.models}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    cols = ["recall@1", "recall@3", "recall@5", "mrr@10"]
    print("\n| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for m, r in results.items():
        print(f"| {m} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
