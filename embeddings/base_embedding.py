import numpy as np
from abc import ABC, abstractmethod
from models.chunk import Chunk

class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, chunks : list[Chunk]) -> list[Chunk]:
        pass
    
    @abstractmethod
    def embed_query(self, query : str) -> np.ndarray:
        pass