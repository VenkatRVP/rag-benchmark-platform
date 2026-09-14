from rank_bm25 import BM25Okapi

from retrievers.base_retriever import BaseRetriever
from models.chunk import Chunk


class BM25Retriever(BaseRetriever):

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Chunks cannot be empty.");
        self.chunks = chunks
    
        tokenized_chunks = [
            chunk.chunk_text.lower().split()
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(
        self,
        query: str,
        k: int = 3
    ) -> list[tuple[Chunk, float]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if k<=0:
            raise ValueError("k should be greater than 0")

        count =  min(k, len(self.chunks))

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[-count:][::-1]

        results = []

        for index in top_indices:
            results.append(
                (self.chunks[index], float(scores[index]))
            )

        return results