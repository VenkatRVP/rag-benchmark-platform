from abc import ABC, abstractmethod
from models.chunk import Chunk

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k : int = 3)-> list[tuple[Chunk, float]]:
        pass