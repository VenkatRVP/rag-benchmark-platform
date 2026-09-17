from retrievers.base_retriever import BaseRetriever
from evaluation.evaluation_dataset import EvaluationDataset
from evaluation.retrieval_evaluator import RetrievalEvaluator
from models.evaluation_result import EvaluationResult
from models.query_evaluation_result import QueryEvaluationResult
from models.benchmark_result import BenchmarkResult
from models.chunk import Chunk


class BenchmarkRunner:

    def __init__(
        self,
        retriever: BaseRetriever,
        evaluation_dataset: EvaluationDataset,
        evaluator: RetrievalEvaluator,
        chunks: list[Chunk]
    ) -> None:
        self.retriever = retriever
        self.evaluation_dataset = evaluation_dataset
        self.evaluator = evaluator
        self.chunks = chunks
        self.results: list[QueryEvaluationResult] = []

    def run(self, k: int) -> BenchmarkResult:

        # Clear previous results so every run starts fresh
        self.results.clear()

        for i, sample in enumerate(
            self.evaluation_dataset.samples,
            start=1
        ):

            retrieved_results = self.retriever.retrieve(
                sample.question,
                k
            )

            retrieved_chunks = [
                result[0]
                for result in retrieved_results
            ]

            relevant_chunk_ids = (
                self.evaluation_dataset.get_relevant_chunk_ids(
                    sample.question
                )
            )

            print(f"\n{'=' * 60}")
            print(f"Question {i}: {sample.question}")

            print("\nRetrieved Chunks:")

            for rank, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):
                print(f"\nRank {rank}")
                print(f"Chunk ID: {chunk.chunk_id}")
                print(f"Text: {chunk.chunk_text}")

            print("\nRelevant Chunks:")

            for chunk_id in relevant_chunk_ids:
                chunk_text = self.get_chunk_text(
                    self.chunks,
                    chunk_id
                )

                print(f"\nChunk ID: {chunk_id}")
                print(f"Text: {chunk_text}")

            evaluation_result = self.evaluator.evaluate(
                retrieved_chunks=retrieved_chunks,
                relevant_chunk_ids=relevant_chunk_ids,
                k=k
            )

            print(
                f"\nEvaluation Result: "
                f"{vars(evaluation_result)}"
            )

            self.collect(
                sample.question,
                evaluation_result
            )

        return self.aggregate()

    @staticmethod
    def get_chunk_text(
        chunks: list[Chunk],
        chunk_id: str
    ) -> str:

        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.chunk_text

        raise KeyError(
            f"Chunk not found: {chunk_id}"
        )

    def collect(
        self,
        question: str,
        evaluation_result: EvaluationResult
    ) -> None:

        self.results.append(
            QueryEvaluationResult(
                question=question,
                evaluation_result=evaluation_result
            )
        )

    def aggregate(self) -> BenchmarkResult:

        total = len(self.results)

        if total == 0:
            raise ValueError(
                "No evaluation results available."
            )

        recall = sum(
            result.evaluation_result.recall
            for result in self.results
        ) / total

        precision = sum(
            result.evaluation_result.precision
            for result in self.results
        ) / total

        hit_rate = sum(
            result.evaluation_result.hit_rate
            for result in self.results
        ) / total

        mrr = sum(
            result.evaluation_result.reciprocal_rank
            for result in self.results
        ) / total

        return BenchmarkResult(
            recall=recall,
            precision=precision,
            hit_rate=hit_rate,
            mrr=mrr
        )