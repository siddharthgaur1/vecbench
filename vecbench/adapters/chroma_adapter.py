from __future__ import annotations

import shutil
import tempfile
import time

import chromadb
import numpy as np

from vecbench.adapters.base import BaseAdapter


class ChromaAdapter(BaseAdapter):
    name = "chroma"

    def setup(self, dim: int) -> None:
        self.dim = dim
        self._tmpdir = tempfile.mkdtemp(prefix="vecbench_chroma_")
        self.client = chromadb.PersistentClient(path=self._tmpdir)
        self.collection = self.client.create_collection("bench", metadata={"hnsw:space": "l2"})

    def insert(self, vectors: np.ndarray, ids: list[int]) -> float:
        start = time.perf_counter()
        batch = 1000
        str_ids = [str(i) for i in ids]
        for i in range(0, len(vectors), batch):
            self.collection.add(
                embeddings=vectors[i:i + batch].tolist(),
                ids=str_ids[i:i + batch],
            )
        return time.perf_counter() - start

    def query(self, vector: np.ndarray, k: int) -> list[int]:
        res = self.collection.query(query_embeddings=[vector.tolist()], n_results=k)
        return [int(i) for i in res["ids"][0]]

    def get_index_size_mb(self) -> float:
        return self.collection.count() * self.dim * 4 / (1024 * 1024)

    def teardown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)
