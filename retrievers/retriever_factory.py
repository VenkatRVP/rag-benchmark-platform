from retrievers.base_retriever import BaseRetriever
from embeddings.base_embedding import BaseEmbedding
from vectorstore.base_vector_store import BaseVectorStore
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.hybrid_retriever import HybridRetriever
class RetrieverFactory:
    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store

    def create(self, retriever_type: str) -> BaseRetriever:

        retriever_type = retriever_type.lower()

        if retriever_type == "bm25":
            return BM25Retriever()

        elif retriever_type == "dense":
            return DenseRetriever(
                embedding=self.embedding,
                vector_store=self.vector_store
            )
        elif retriever_type == "hybrid":
            bm25_retriever = BM25Retriever()
            dense_retriever = DenseRetriever(
                embedding=self.embedding,
                vector_store=self.vector_store
            )
            return HybridRetriever(dense_retriever=dense_retriever, bm25_retriever=bm25_retriever)

        raise ValueError(
            f"Unsupported retriever type: {retriever_type}"
        )