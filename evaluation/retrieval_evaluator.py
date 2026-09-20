from models.chunk import Chunk
from models.evaluation_result import EvaluationResult


class RetrievalEvaluator:

    def evaluate(
        self,
        retrieved_chunks: list[Chunk],
        relevant_chunk_ids: list[str],
        k: int
    ) -> EvaluationResult:

        recall = self.recall_at_k(
            retrieved_chunks,
            relevant_chunk_ids,
            k
        )

        precision = self.precision_at_k(
            retrieved_chunks,
            relevant_chunk_ids,
            k
        )

        hit_rate = self.hit_rate_at_k(
            retrieved_chunks,
            relevant_chunk_ids,
            k
        )

        reciprocal_rank = self.reciprocal_rank_at_k(
            retrieved_chunks,
            relevant_chunk_ids,
            k
        )

        return EvaluationResult(
            recall=recall,
            precision=precision,
            hit_rate=hit_rate,
            reciprocal_rank=reciprocal_rank
        )

    def recall_at_k(
        self,
        retrieved_chunks: list[Chunk],
        relevant_chunk_ids: list[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not relevant_chunk_ids:
            raise ValueError(
                "Relevant chunk IDs cannot be empty."
            )

        retrieved_ids = {
            chunk.chunk_id
            for chunk in retrieved_chunks[:k]
        }

        relevant_ids = set(relevant_chunk_ids)

        hits = retrieved_ids.intersection(relevant_ids)

        return len(hits) / len(relevant_ids)

    def precision_at_k(
        self,
        retrieved_chunks: list[Chunk],
        relevant_chunk_ids: list[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not relevant_chunk_ids:
            raise ValueError(
                "Relevant chunk IDs cannot be empty."
            )

        if not retrieved_chunks:
            return 0.0

        retrieved_ids = {
            chunk.chunk_id
            for chunk in retrieved_chunks[:k]
        }

        relevant_ids = set(relevant_chunk_ids)

        hits = retrieved_ids.intersection(relevant_ids)

        return len(hits) / min(k, len(retrieved_chunks))

    def hit_rate_at_k(
        self,
        retrieved_chunks: list[Chunk],
        relevant_chunk_ids: list[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not relevant_chunk_ids:
            raise ValueError(
                "Relevant chunk IDs cannot be empty."
            )

        if not retrieved_chunks:
            return 0.0

        retrieved_ids = {
            chunk.chunk_id
            for chunk in retrieved_chunks[:k]
        }

        relevant_ids = set(relevant_chunk_ids)

        hits = retrieved_ids.intersection(relevant_ids)

        return 1.0 if hits else 0.0

    def reciprocal_rank_at_k(
        self,
        retrieved_chunks: list[Chunk],
        relevant_chunk_ids: list[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        if not relevant_chunk_ids:
            raise ValueError(
                "Relevant chunk IDs cannot be empty."
            )

        if not retrieved_chunks:
            return 0.0

        relevant_ids = set(relevant_chunk_ids)

        for rank, chunk in enumerate(
            retrieved_chunks[:k],
            start=1
        ):
            if chunk.chunk_id in relevant_ids:
                return 1 / rank

        return 0.0