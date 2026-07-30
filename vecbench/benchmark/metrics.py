from __future__ import annotations

import numpy as np


def recall_at_k(predicted: list[list[int]], ground_truth: np.ndarray) -> float:
    hits = 0
    total = 0
    for pred, truth in zip(predicted, ground_truth):
        truth_set = set(int(t) for t in truth if t != -1)
        hits += len(set(pred) & truth_set)
        total += len(truth_set)
    return hits / total if total else 0.0


def percentiles(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
