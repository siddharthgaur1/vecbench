from __future__ import annotations

import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from vecbench.benchmark.metrics import percentiles, recall_at_k


@dataclass
class ExperimentResult:
    db: str
    size: int
    dim: int
    k: int
    insert_time_s: float
    insert_throughput: float
    recall_at_k: float
    latency_ms: dict
    qps_by_concurrency: dict
    peak_memory_mb: float
    index_size_mb: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "db": self.db, "size": self.size, "dim": self.dim, "k": self.k,
            "insert_time_s": self.insert_time_s, "insert_throughput": self.insert_throughput,
            "recall_at_k": self.recall_at_k, "peak_memory_mb": self.peak_memory_mb,
            "index_size_mb": self.index_size_mb,
        }
        d.update({f"latency_{k}": v for k, v in self.latency_ms.items()})
        d.update({f"qps_c{c}": v for c, v in self.qps_by_concurrency.items()})
        return d


def run_experiment(adapter, vectors: np.ndarray, ground_truth: np.ndarray, queries: np.ndarray,
                    k: int, n_threads: list[int] | None = None, n_query_latency: int = 100) -> ExperimentResult:
    n_threads = n_threads or [1]
    dim = vectors.shape[1]

    tracemalloc.start()
    adapter.setup(dim)
    insert_time = adapter.insert(vectors, list(range(len(vectors))))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for q in queries[:10]:
        adapter.query(q, k)  # warmup, not measured

    latency_sample = queries[:n_query_latency]
    lat_ms = []
    for q in latency_sample:
        start = time.perf_counter()
        adapter.query(q, k)
        lat_ms.append((time.perf_counter() - start) * 1000)

    predicted = adapter.batch_query(latency_sample, k)
    recall = recall_at_k(predicted, ground_truth[:len(latency_sample)])

    qps_by_concurrency = {}
    for threads in n_threads:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(lambda q: adapter.query(q, k), latency_sample))
        elapsed = time.perf_counter() - start
        qps_by_concurrency[threads] = len(latency_sample) / elapsed if elapsed > 0 else 0.0

    result = ExperimentResult(
        db=adapter.name, size=len(vectors), dim=dim, k=k,
        insert_time_s=insert_time, insert_throughput=len(vectors) / insert_time if insert_time > 0 else 0.0,
        recall_at_k=recall, latency_ms=percentiles(lat_ms), qps_by_concurrency=qps_by_concurrency,
        peak_memory_mb=peak / (1024 * 1024), index_size_mb=adapter.get_index_size_mb(),
    )
    adapter.teardown()
    return result
