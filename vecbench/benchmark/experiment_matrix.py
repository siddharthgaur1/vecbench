from __future__ import annotations

import itertools

QUICK_CONFIG = {
    "dbs": ["chroma", "faiss_flat", "faiss_ivf", "faiss_hnsw"],
    "sizes": [2_000, 10_000],
    "dims": [128],
    "k_values": [10],
    "n_queries": 100,
    "concurrency": [1, 4],
}


def build_matrix(config: dict) -> list[dict]:
    return [
        {"db": db, "size": size, "dim": dim, "k": k}
        for db, size, dim, k in itertools.product(
            config["dbs"], config["sizes"], config["dims"], config["k_values"]
        )
    ]


ADAPTER_REGISTRY = {}


def _register():
    from vecbench.adapters.chroma_adapter import ChromaAdapter
    from vecbench.adapters.faiss_adapter import FaissFlat, FaissHNSW, FaissIVF

    ADAPTER_REGISTRY.update({
        "chroma": ChromaAdapter,
        "faiss_flat": FaissFlat,
        "faiss_ivf": FaissIVF,
        "faiss_hnsw": FaissHNSW,
    })


_register()
