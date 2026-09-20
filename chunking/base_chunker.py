from abc import ABC, abstractmethod

from models.chunk import Chunk
from models.page import Page


class BaseChunker(ABC):

    @abstractmethod
    def chunk(
        self,
        pages: list[Page]
    ) -> list[Chunk]:
        pass