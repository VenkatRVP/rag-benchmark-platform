from retrievers.base_retriever import BaseRetriever
from evaluation.evaluation_dataset import EvaluationDataset
from evaluation.retrieval_evaluator import RetrievalEvaluator
from models.evaluation_result import EvaluationResult
from models.query_evaluation_result import QueryEvaluationResult
from models.benchmark_result import BenchmarkResult


class BenchmarkRunner:

    def __init__(
        self,
        retriever: BaseRetriever,
        evaluation_dataset: EvaluationDataset,
        evaluator: RetrievalEvaluator
    ) -> None:
        self.retriever = retriever
        self.evaluation_dataset = evaluation_dataset
        self.evaluator = evaluator
        self.results: list[QueryEvaluationResult] = []

    def run(self, k: int) -> BenchmarkResult:
        self.results.clear()

        for sample in self.evaluation_dataset.samples:

            retrieved_results = self.retriever.retrieve(
                sample.question,
                k
            )

            retrieved_chunks = [
                result[0]
                for result in retrieved_results
            ]

            retrieved_chunk_ids = [
                chunk.chunk_id
                for chunk in retrieved_chunks
            ]

            relevant_chunk_ids = (
                self.evaluation_dataset.get_relevant_chunk_ids(
                    sample.question
                )
            )

            evaluation_result = self.evaluator.evaluate(
                retrieved_chunks=retrieved_chunks,
                relevant_chunk_ids=relevant_chunk_ids,
                k=k
            )

            self.collect(
                question=sample.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=relevant_chunk_ids,
                evaluation_result=evaluation_result
            )

        return self.aggregate()

    def collect(
        self,
        question: str,
        retrieved_chunk_ids: list[str],
        relevant_chunk_ids: list[str],
        evaluation_result: EvaluationResult
    ) -> None:
        self.results.append(
            QueryEvaluationResult(
                question=question,
                evaluation_result=evaluation_result,
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=relevant_chunk_ids
            )
        )

    def get_results(self) -> list[QueryEvaluationResult]:
        return self.results

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

    def print_query_results(self) -> None:
        for result in self.results:
            print("\n" + "=" * 60)
            print(f"Question: {result.question}")

            print("\nGround Truth:")
            for chunk_id in result.relevant_chunk_ids:
                print(chunk_id)

            print("\nRetrieved Chunks:")
            for rank, chunk_id in enumerate(
                result.retrieved_chunk_ids,
                start=1
            ):
                print(f"Rank {rank}: {chunk_id}")

            print("\nEvaluation:")
            print(vars(result.evaluation_result))

    def print_error_analysis(self) -> None:
        for result in self.results:

            relevant_ids = set(result.relevant_chunk_ids)
            retrieved_ids = set(result.retrieved_chunk_ids)

            relevant_retrieved = (
                retrieved_ids.intersection(relevant_ids)
            )

            missing_relevant = (
                relevant_ids.difference(retrieved_ids)
            )

            false_positives = (
                retrieved_ids.difference(relevant_ids)
            )

            print("\n" + "=" * 60)
            print(f"Question: {result.question}")

            print("\nGround Truth:")
            for chunk_id in result.relevant_chunk_ids:
                print(chunk_id)

            print("\nRetrieved:")
            for rank, chunk_id in enumerate(
                result.retrieved_chunk_ids,
                start=1
            ):
                print(f"Rank {rank}: {chunk_id}")

            print("\nRelevant Retrieved:")
            for chunk_id in relevant_retrieved:
                print(chunk_id)

            print("\nMissing Relevant:")
            for chunk_id in missing_relevant:
                print(chunk_id)

            print("\nFalse Positives:")
            for chunk_id in false_positives:
                print(chunk_id)

            print("\nMetrics:")
            print(vars(result.evaluation_result))