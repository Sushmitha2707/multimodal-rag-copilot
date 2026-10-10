import numpy as np

from copilot.metrics import rank_metrics


def test_perfect_ranking():
    sim = np.eye(3)
    m = rank_metrics(sim, [{0}, {1}, {2}], ks=(1, 3))
    assert m["recall@1"] == 1.0 and m["mrr@10"] == 1.0 and m["hit@3"] == 1.0


def test_second_place_gives_half_mrr():
    sim = np.array([[0.5, 0.9, 0.1]])  # relevant passage 0 is ranked 2nd
    m = rank_metrics(sim, [{0}], ks=(1, 3))
    assert m["recall@1"] == 0.0
    assert m["recall@3"] == 1.0
    assert m["mrr@10"] == 0.5


def test_multiple_relevant_recall_is_fractional():
    sim = np.array([[0.9, 0.8, 0.1, 0.0]])
    m = rank_metrics(sim, [{0, 3}], ks=(2,))
    assert m["recall@2"] == 0.5
