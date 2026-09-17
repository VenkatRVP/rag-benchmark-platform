import pytest

from benchmark.benchmark_runner import BenchmarkRunner
from evaluation.evaluation_dataset import EvaluationDataset
from models.evaluation_sample import EvaluationSample
from evaluation.retrieval_evaluator import RetrievalEvaluator
from models.chunk import Chunk
from retrievers.base_retriever import BaseRetriever

class FakeRetriever(BaseRetriever):

    def index(self, chunks: list[Chunk]) -> None:
        pass

    def retrieve(
    self,
    query: str,
    k: int = 3
) -> list[tuple[Chunk, float]]:

        if query == "Question 1":
            return [
                (create_chunk("A"), 0.1),
                (create_chunk("C"), 0.2),
                (create_chunk("D"), 0.3)
            ]

        return [
            (create_chunk("C"), 0.1),
            (create_chunk("D"), 0.2),
            (create_chunk("E"), 0.3)
        ]

def create_chunk(chunk_id: str) -> Chunk:
    return Chunk(
            document_name="test.pdf",
            page_number=1,
            chunk_text="test",
            chunk_id=chunk_id
        )

def test_benchmark_runner_collects_results():

    samples = [
        EvaluationSample(
            question="Question 1",
            relevant_chunk_ids=["A"]
        ),
        EvaluationSample(
            question="Question 2",
            relevant_chunk_ids=["A"]
        )
    ]

    dataset = EvaluationDataset(samples)
    dataset.loaded = True

    runner = BenchmarkRunner(
        retriever=FakeRetriever(),
        evaluation_dataset=dataset,
        evaluator=RetrievalEvaluator()
    )

    result = runner.run(k=3)

    assert len(runner.results) == 2

def test_benchmark_runner_stores_query_results():

    samples = [
        EvaluationSample(
            question="Question 1",
            relevant_chunk_ids=["A"]
        )
    ]

    dataset = EvaluationDataset(samples)
    dataset.loaded = True

    runner = BenchmarkRunner(
        retriever=FakeRetriever(),
        evaluation_dataset=dataset,
        evaluator=RetrievalEvaluator()
    )

    runner.run(k=3)

    query_result = runner.results[0]

    assert query_result.question == "Question 1"
    assert query_result.evaluation_result.recall == 1.0
    assert query_result.evaluation_result.precision == 1 / 3

def test_benchmark_runner_calculates_average_metrics():

    samples = [
        EvaluationSample(
            question="Question 1",
            relevant_chunk_ids=["A"]
        ),
        EvaluationSample(
            question="Question 2",
            relevant_chunk_ids=["A"]
        )
    ]

    dataset = EvaluationDataset(samples)
    dataset.loaded = True

    runner = BenchmarkRunner(
        retriever=FakeRetriever(),
        evaluation_dataset=dataset,
        evaluator=RetrievalEvaluator()
    )

    result = runner.run(k=3)

    assert result.recall == 0.5
    assert result.precision == 1 / 6
    assert result.hit_rate == 0.5
    assert result.mrr == 0.5