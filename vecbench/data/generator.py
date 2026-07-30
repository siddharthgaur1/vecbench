from __future__ import annotations

import numpy as np


def generate_vectors(size: int, dim: int, seed: int = 0) -> np.ndarray:
    """Random unit vectors — realistic enough for benchmarking the geometric
    operations (distance, nearest-neighbor search) a vector DB actually does;
    see METHODOLOGY.md for why this is a valid proxy for embedding search."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(size, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k neighbor ids via brute-force L2, used as recall ground truth."""
    import faiss

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    _, ids = index.search(queries, k)
    return ids
