"""Create (question, passage) training pairs for embedding fine-tuning using an LLM.

The corpus is split BY SOURCE FILE into train / held-out so the held-out evaluation never sees
passages from documents the model was trained on (no leakage).

    python -m scripts.make_pairs --corpus data/my_corpus --provider gemini --max-chunks 300 --sleep 4
"""

import argparse
import json
import random
import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from copilot.ingest import chunk_documents, load_directory
from copilot.llm import get_chat_model

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write search questions for an enterprise knowledge base. Given a passage, write {n} diverse "
            "questions a real employee might type that this passage answers. Use different wording than the "
            "passage (paraphrase, do not copy sentences). Return ONLY a JSON list of strings.",
        ),
        ("human", "Passage:\n{passage}"),
    ]
)


def parse_questions(text: str) -> list[str]:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in data if isinstance(q, str) and len(q.strip()) > 8]


def split_sources(sources: list[str], heldout_frac: float, seed: int = 42) -> tuple[set[str], set[str]]:
    srcs = sorted(set(sources))
    if len(srcs) < 2:
        raise SystemExit("Need at least 2 source documents to make a train/held-out split.")
    random.Random(seed).shuffle(srcs)
    n_held = max(1, round(len(srcs) * heldout_frac))
    return set(srcs[n_held:]), set(srcs[:n_held])


def ask_with_retry(chain, payload, tries: int = 5) -> str:
    """Call the LLM; on rate-limit or transient errors wait and retry instead of crashing."""
    for attempt in range(tries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  LLM error ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out-dir", default="data/pairs")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--questions-per-chunk", type=int, default=3)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--max-chunks", type=int, default=0, help="cap the number of chunks (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between LLM calls")
    args = ap.parse_args()

    chunks = chunk_documents(load_directory(args.corpus))
    if args.max_chunks:
        chunks = random.Random(1).sample(chunks, min(args.max_chunks, len(chunks)))
    train_src, held_src = split_sources([c.metadata["source"] for c in chunks], args.heldout_frac)
    print(f"{len(chunks)} chunks | train sources: {len(train_src)} | held-out sources: {len(held_src)}")

    chain = PROMPT | get_chat_model(args.provider, temperature=0.4) | StrOutputParser()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "heldout": 0}
    with open(out / "train.jsonl", "w") as ftr, open(out / "heldout.jsonl", "w") as fho:
        for i, c in enumerate(chunks, start=1):
            qs = parse_questions(ask_with_retry(chain, {"n": args.questions_per_chunk, "passage": c.page_content}))
            time.sleep(args.sleep)
            split = "train" if c.metadata["source"] in train_src else "heldout"
            f = ftr if split == "train" else fho
            for q in qs:
                f.write(json.dumps({"query": q, "passage": c.page_content, "source": c.metadata["source"]}) + "\n")
                counts[split] += 1
            if i % 20 == 0:
                print(f"  {i}/{len(chunks)} chunks processed")
    print(f"Wrote {counts['train']} train pairs and {counts['heldout']} held-out pairs to {out}/")


if __name__ == "__main__":
    main()