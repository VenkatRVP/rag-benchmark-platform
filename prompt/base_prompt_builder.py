from abc import ABC, abstractmethod

from models.chunk import Chunk


class BasePromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        user_query: str,
        retrieved_chunks: list[tuple[Chunk, float]]
    ) -> str:
        pass