"""Minimal assert-based self-checks (no framework, no fixtures)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vecbench.benchmark.metrics import percentiles, recall_at_k
from vecbench.data.generator import compute_ground_truth, generate_vectors


def test_recall_at_k_perfect_match():
    truth = np.array([[1, 2, 3], [4, 5, 6]])
    predicted = [[1, 2, 3], [4, 5, 6]]
    assert recall_at_k(predicted, truth) == 1.0


def test_recall_at_k_partial_match():
    truth = np.array([[1, 2, 3, 4]])
    predicted = [[1, 2, 99, 98]]
    assert recall_at_k(predicted, truth) == 0.5


def test_percentiles_monotonic():
    p = percentiles([1, 2, 3, 4, 5, 100])
    assert p["p50"] <= p["p95"] <= p["p99"]


def test_faiss_flat_recovers_ground_truth_exactly():
    from vecbench.adapters.faiss_adapter import FaissFlat

    vectors = generate_vectors(500, 32, seed=0)
    queries = generate_vectors(10, 32, seed=1)
    truth = compute_ground_truth(vectors, queries, k=5)

    adapter = FaissFlat()
    adapter.setup(32)
    adapter.insert(vectors, list(range(len(vectors))))
    predicted = adapter.batch_query(queries, 5)
    assert recall_at_k(predicted, truth) == 1.0, "exact index must match its own ground truth exactly"
    adapter.teardown()


if __name__ == "__main__":
    test_recall_at_k_perfect_match()
    test_recall_at_k_partial_match()
    test_percentiles_monotonic()
    test_faiss_flat_recovers_ground_truth_exactly()
    print("all tests passed")
