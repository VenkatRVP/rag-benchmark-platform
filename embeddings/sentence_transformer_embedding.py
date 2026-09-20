import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings.base_embedding import BaseEmbedding
from models.chunk import Chunk


class SentenceTransformerEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def embed(
        self,
        chunks: list[Chunk]
    ) -> list[Chunk]:

        if not chunks:
            return []

        texts = [
            chunk.chunk_text
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):
            chunk.embedding = embedding

        return chunks

    def embed_query(
        self,
        query: str
    ) -> np.ndarray:

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        return embedding

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()