from __future__ import annotations

import time

import faiss
import numpy as np

from vecbench.adapters.base import BaseAdapter


class _FaissAdapterBase(BaseAdapter):
    def setup(self, dim: int) -> None:
        self.dim = dim
        self.index = self._build_index(dim)

    def _build_index(self, dim: int):
        raise NotImplementedError

    def insert(self, vectors: np.ndarray, ids: list[int]) -> float:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        start = time.perf_counter()
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add_with_ids(vectors, np.array(ids, dtype="int64"))
        return time.perf_counter() - start

    def query(self, vector: np.ndarray, k: int) -> list[int]:
        vector = np.ascontiguousarray(vector, dtype="float32").reshape(1, -1)
        _, ids = self.index.search(vector, k)
        return [i for i in ids[0] if i != -1]

    def batch_query(self, vectors: np.ndarray, k: int) -> list[list[int]]:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        _, ids = self.index.search(vectors, k)
        return [[i for i in row if i != -1] for row in ids]

    def get_index_size_mb(self) -> float:
        return self.index.ntotal * self.dim * 4 / (1024 * 1024)  # float32, approx (no disk serialization)


class FaissFlat(_FaissAdapterBase):
    name = "faiss_flat"

    def _build_index(self, dim: int):
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))


class FaissIVF(_FaissAdapterBase):
    name = "faiss_ivf"

    def _build_index(self, dim: int):
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 100)
        index.nprobe = 10  # default of 1 cluster searched tanks recall; 10/100 trades some speed for it
        return faiss.IndexIDMap(index)


class FaissHNSW(_FaissAdapterBase):
    name = "faiss_hnsw"

    def _build_index(self, dim: int):
        return faiss.IndexIDMap(faiss.IndexHNSWFlat(dim, 32))
