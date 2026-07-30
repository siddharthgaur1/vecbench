from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def setup(self, dim: int) -> None: ...

    @abstractmethod
    def insert(self, vectors: np.ndarray, ids: list[int]) -> float:
        """Returns wall-clock insert time in seconds."""

    @abstractmethod
    def query(self, vector: np.ndarray, k: int) -> list[int]: ...

    def batch_query(self, vectors: np.ndarray, k: int) -> list[list[int]]:
        return [self.query(v, k) for v in vectors]

    def get_index_size_mb(self) -> float:
        return 0.0

    def teardown(self) -> None:
        pass
