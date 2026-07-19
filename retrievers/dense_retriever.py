from retrievers import BaseRetriever
from embeddings import BaseEmbedding
from vectorstore import BaseVectorStore
from models.chunk import Chunk

class DenseRetriever(BaseRetriever):

    def __init__(self, embedding : BaseEmbedding, vector_store : BaseVectorStore) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def retrieve(self, query: str, k : int = 3)-> list[tuple[Chunk, float]]:
        query_vector = self.embedding.embed_query(query)
        return self.vector_store.search(query_vector, k)