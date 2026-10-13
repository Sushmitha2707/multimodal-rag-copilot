"""End-to-end RAG evaluation + provider comparison.

Metrics per provider:
  keyword_acc  - fraction of answers containing ALL gold keywords (deterministic)
  faithfulness - 1-5, is every claim supported by the retrieved context? (LLM judge)
  relevance    - 1-5, does the answer address the question? (LLM judge)
  latency_s    - mean seconds per question (retrieval + generation)

    python -m scripts.eval_rag --providers gemini --judge-provider gemini --sleep 4
    python -m scripts.eval_rag --providers gemini --sleep 6 --mode graph --out results/rag_eval_graph.json
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from copilot.llm import get_chat_model
from copilot.rag import RAGPipeline

JUDGE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator of RAG answers. Score two things from 1 to 5.\n"
            "faithfulness: 5 = every claim is supported by the context; 1 = mostly unsupported/invented.\n"
            "relevance: 5 = fully answers the question; 1 = does not answer it.\n"
            'Reply in exactly this format: {{"faithfulness": <int>, "relevance": <int>}}',
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def parse_scores(text: str) -> tuple[int, int] | None:
    """Pull the two 1-5 scores out of the judge's reply, even if the JSON is malformed."""
    f = re.search(r"faithfulness\D{0,6}([1-5])", text, re.I)
    r = re.search(r"relevance\D{0,6}([1-5])", text, re.I)
    return (int(f.group(1)), int(r.group(1))) if f and r else None


def retry(fn, *args, tries: int = 5, **kwargs):
    """Run fn(*args, **kwargs); on rate-limit or transient errors wait and retry. None if all tries fail."""
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"  error ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_set.jsonl")
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--out", default="results/rag_eval.json")
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds to wait between LLM calls")
    ap.add_argument(
        "--mode", choices=["simple", "graph"], default="simple", help="graph = LangGraph corrective RAG workflow"
    )
    args = ap.parse_args()

    qa = [json.loads(line) for line in open(args.qa)]
    pipe = RAGPipeline()
    if args.mode == "graph":
        from copilot.graph import GraphRAG

        pipe = GraphRAG(store=pipe.store)
    judge = JUDGE | get_chat_model(args.judge_provider) | StrOutputParser()

    results = {}
    for prov in args.providers:
        kw, faith, rel, lat, errors = [], [], [], [], 0
        for item in qa:
            out = retry(pipe.answer, item["question"], provider=prov)
            time.sleep(args.sleep)
            if out is None:
                errors += 1
                continue
            kw.append(keyword_hit(out["answer"], item["answer_keywords"]))
            lat.append(out["latency_s"])
            verdict = retry(
                judge.invoke, {"question": item["question"], "context": out["context"], "answer": out["answer"]}
            )
            time.sleep(args.sleep)
            s = parse_scores(verdict) if verdict else None
            if s:
                faith.append(s[0])
                rel.append(s[1])
        results[prov] = {
            "n": len(qa),
            "answered": len(kw),
            "errors": errors,
            "keyword_acc": sum(kw) / len(kw) if kw else None,
            "faithfulness": statistics.mean(faith) if faith else None,
            "relevance": statistics.mean(rel) if rel else None,
            "latency_s": statistics.mean(lat) if lat else None,
        }
        print(f"{prov}: {results[prov]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    def fmt(v, spec):
        return format(v, spec) if v is not None else "n/a"

    print("\n| provider | keyword_acc | faithfulness | relevance | latency_s | errors |\n|---|---|---|---|---|---|")
    for p, r in results.items():
        print(
            f"| {p} | {fmt(r['keyword_acc'], '.2f')} | {fmt(r['faithfulness'], '.2f')} | "
            f"{fmt(r['relevance'], '.2f')} | {fmt(r['latency_s'], '.2f')} | {r['errors']} |"
        )


if __name__ == "__main__":
    main()
