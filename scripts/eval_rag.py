"""End-to-end RAG evaluation + provider comparison.

Metrics per provider:
  keyword_acc  - fraction of answers containing ALL gold keywords (deterministic)
  faithfulness - 1-5, is every claim supported by the retrieved context? (LLM judge)
  relevance    - 1-5, does the answer address the question? (LLM judge)
  latency_s    - mean seconds per question (retrieval + generation)

    python -m scripts.eval_rag --providers gemini openai anthropic --judge-provider gemini
"""

import argparse
import json
import statistics
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
            'Return ONLY JSON: {{"faithfulness": <int>, "relevance": <int>}}',
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"),
    ]
)


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


def parse_scores(text: str) -> tuple[int, int] | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        d = json.loads(text)
        return int(d["faithfulness"]), int(d["relevance"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_set.jsonl")
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--out", default="results/rag_eval.json")
    args = ap.parse_args()

    qa = [json.loads(line) for line in open(args.qa)]
    pipe = RAGPipeline()
    judge = JUDGE | get_chat_model(args.judge_provider) | StrOutputParser()

    results = {}
    for prov in args.providers:
        kw, faith, rel, lat = [], [], [], []
        for item in qa:
            out = pipe.answer(item["question"], provider=prov)
            kw.append(keyword_hit(out["answer"], item["answer_keywords"]))
            lat.append(out["latency_s"])
            s = parse_scores(
                judge.invoke({"question": item["question"], "context": out["context"], "answer": out["answer"]})
            )
            if s:
                faith.append(s[0])
                rel.append(s[1])
        results[prov] = {
            "n": len(qa),
            "keyword_acc": sum(kw) / len(kw),
            "faithfulness": statistics.mean(faith) if faith else None,
            "relevance": statistics.mean(rel) if rel else None,
            "latency_s": statistics.mean(lat),
        }
        print(f"{prov}: {results[prov]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\n| provider | keyword_acc | faithfulness | relevance | latency_s |\n|---|---|---|---|---|")
    for p, r in results.items():
        f = f"{r['faithfulness']:.2f}" if r["faithfulness"] else "n/a"
        v = f"{r['relevance']:.2f}" if r["relevance"] else "n/a"
        print(f"| {p} | {r['keyword_acc']:.2f} | {f} | {v} | {r['latency_s']:.2f} |")


if __name__ == "__main__":
    main()
