from abc import ABC, abstractmethod
from models.page import Page
from models.chunk import Chunk

class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, pages: list[Page])-> list[Chunk]:
        pass