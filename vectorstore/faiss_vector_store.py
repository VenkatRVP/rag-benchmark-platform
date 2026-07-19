import faiss
import numpy as np

from models.chunk import Chunk
from vectorstore import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):

    def __init__(self, index: faiss.Index):
        self.index = index
        self._chunk_registry: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:

        if not chunks:
            return

        expected_dimension = self.index.d

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' has no embedding."
                )

            if len(chunk.embedding) != expected_dimension:
                raise ValueError(
                    f"Expected embedding dimension {expected_dimension}, "
                    f"got {len(chunk.embedding)}."
                )

        vectors = np.asarray(
            [chunk.embedding for chunk in chunks],
            dtype=np.float32,
        )

        self.index.add(vectors)
        self._chunk_registry.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 3,
    ) -> list[tuple[Chunk, float]]:

        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector.")

        query_matrix = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(1, -1)

        distances, indices = self.index.search(query_matrix, k)

        results: list[tuple[Chunk, float]] = []

        for chunk_index, distance in zip(indices[0], distances[0]):

            if chunk_index == -1:
                continue

            results.append(
                (
                    self._chunk_registry[chunk_index],
                    float(distance),
                )
            )

        return results