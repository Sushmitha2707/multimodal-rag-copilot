"""Fine-tune a Hugging Face sentence-transformer (PyTorch) on (question, passage) pairs.

Loss: MultipleNegativesRankingLoss - every other passage in the batch acts as a negative.

    python -m scripts.train_embeddings --pairs data/pairs/train.jsonl --out models/embedding-ft
"""

import argparse
import json
import math

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/train.jsonl")
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="models/embedding-ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.pairs)]
    print(f"{len(rows)} training pairs")
    examples = [InputExample(texts=[r["query"], r["passage"]]) for r in rows]

    try:  # avoids duplicate passages in one batch acting as false negatives
        from sentence_transformers.datasets import NoDuplicatesDataLoader

        loader = NoDuplicatesDataLoader(examples, batch_size=args.batch_size)
    except ImportError:
        loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)

    model = SentenceTransformer(args.base)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = math.ceil(len(loader) * args.epochs * 0.1)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        output_path=args.out,
        show_progress_bar=True,
    )
    model.save(args.out)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
