from retrievers.base_retriever import BaseRetriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.bm25_retriever import BM25Retriever
from models.chunk import Chunk


class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        candidate_k: int = 10,
        rrf_constant: int = 60
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def index(
        self,
        chunks: list[Chunk]
    ) -> None:
        self.dense_retriever.index(chunks)
        self.bm25_retriever.index(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> list[tuple[Chunk, float]]:

        dense_results = self.dense_retriever.retrieve(
            query=query,
            k=self.candidate_k
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            k=self.candidate_k
        )

        rrf_scores = {}
        chunks_by_id = {}

        for rank, (chunk, _) in enumerate(
            dense_results,
            start=1
        ):
            chunks_by_id[chunk.chunk_id] = chunk

            score = 1 / (
                self.rrf_constant + rank
            )

            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0) + score
            )

        for rank, (chunk, _) in enumerate(
            bm25_results,
            start=1
        ):
            chunks_by_id[chunk.chunk_id] = chunk

            score = 1 / (
                self.rrf_constant + rank
            )

            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0) + score
            )

        ranked_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            (chunks_by_id[chunk_id], score)
            for chunk_id, score in ranked_results[:k]
        ]