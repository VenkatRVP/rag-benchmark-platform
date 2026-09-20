import numpy as np
from abc import ABC, abstractmethod

from models.chunk import Chunk


class BaseVectorStore(ABC):

    @abstractmethod
    def add(
        self,
        chunks: list[Chunk]
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 3
    ) -> list[tuple[Chunk, float]]:
        pass