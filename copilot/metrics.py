"""Retrieval metrics (pure numpy, unit tested)."""

from __future__ import annotations

import numpy as np


def rank_metrics(sim: np.ndarray, relevant: list[set[int]], ks=(1, 3, 5, 10)) -> dict:
    """
    sim:      (n_queries, n_passages) similarity matrix (higher = more similar)
    relevant: for each query, the set of relevant passage indices
    Returns Recall@k, Hit@k and MRR@10 averaged over queries.
    """
    n = sim.shape[0]
    assert n == len(relevant), "one relevant-set per query"
    order = np.argsort(-sim, axis=1)
    recall = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    rr = []
    for qi in range(n):
        rel = relevant[qi]
        ranked = order[qi]
        for k in ks:
            found = len(rel.intersection(ranked[:k].tolist()))
            recall[k].append(found / len(rel))
            hit[k].append(1.0 if found else 0.0)
        r = 0.0
        for pos, idx in enumerate(ranked[:10], start=1):
            if idx in rel:
                r = 1.0 / pos
                break
        rr.append(r)
    out = {f"recall@{k}": float(np.mean(recall[k])) for k in ks}
    out.update({f"hit@{k}": float(np.mean(hit[k])) for k in ks})
    out["mrr@10"] = float(np.mean(rr))
    return out
