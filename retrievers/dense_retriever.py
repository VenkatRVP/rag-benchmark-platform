from retrievers.base_retriever import BaseRetriever
from embeddings.base_embedding import BaseEmbedding
from vectorstore.base_vector_store import BaseVectorStore
from models.chunk import Chunk


class DenseRetriever(BaseRetriever):

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def index(
        self,
        chunks: list[Chunk]
    ) -> None:
        chunks = self.embedding.embed(chunks)
        self.vector_store.add(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> list[tuple[Chunk, float]]:
        query_vector = self.embedding.embed_query(query)
        return self.vector_store.search(
            query_vector,
            k
        )