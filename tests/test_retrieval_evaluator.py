import pytest

from evaluation.retrieval_evaluator import RetrievalEvaluator
from models.chunk import Chunk


def create_chunk(chunk_id):
    return Chunk(
        document_name="test.pdf",
        page_number=1,
        chunk_text="test",
        chunk_id=chunk_id
    )


def test_recall_all_relevant_chunks_retrieved():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("B"),
        create_chunk("C")
    ]

    relevant = ["A", "B"]

    result = evaluator.recall_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0


def test_recall_one_relevant_chunk_retrieved():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A", "B"]

    result = evaluator.recall_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.5


def test_recall_no_relevant_chunks_retrieved():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("E")
    ]

    relevant = ["A", "B"]

    result = evaluator.recall_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.0


def test_recall_respects_k():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("A")
    ]

    relevant = ["A"]

    result = evaluator.recall_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=2
    )

    assert result == 0.0


def test_recall_invalid_k():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]
    relevant = ["A"]

    with pytest.raises(ValueError):
        evaluator.recall_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=relevant,
            k=0
        )


def test_recall_empty_relevant_ids():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]

    with pytest.raises(ValueError):
        evaluator.recall_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=[],
            k=3
        )

def test_precision_all_chunks_are_relevant():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("B"),
        create_chunk("C")
    ]

    relevant = ["A", "B", "C"]

    result = evaluator.precision_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0

def test_precision_one_relevant_chunk():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A", "B"]

    result = evaluator.precision_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1 / 3


def test_precision_no_relevant_chunks():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("E")
    ]

    relevant = ["A", "B"]

    result = evaluator.precision_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.0

def test_precision_respects_k():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A"]

    result = evaluator.precision_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=1
    )

    assert result == 1.0

def test_precision_when_fewer_chunks_than_k():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C")
    ]

    relevant = ["A", "B"]

    result = evaluator.precision_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=5
    )

    assert result == 1 / 2

def test_precision_invalid_k():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]
    relevant = ["A"]

    with pytest.raises(ValueError):
        evaluator.precision_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=relevant,
            k=0
        )

def test_precision_empty_relevant_ids():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]

    with pytest.raises(ValueError):
        evaluator.precision_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=[],
            k=3
        )

def test_hit_rate_relevant_chunk_retrieved():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A", "B"]

    result = evaluator.hit_rate_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0

def test_hit_rate_relevant_chunk_retrieved():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A", "B"]

    result = evaluator.hit_rate_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0

def test_hit_rate_respects_k():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("A")
    ]

    relevant = ["A"]

    result = evaluator.hit_rate_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=2
    )

    assert result == 0.0

def test_hit_rate_multiple_relevant_chunks():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("B"),
        create_chunk("C")
    ]

    relevant = ["A", "B"]

    result = evaluator.hit_rate_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0

def test_hit_rate_invalid_k():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]
    relevant = ["A"]

    with pytest.raises(ValueError):
        evaluator.hit_rate_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=relevant,
            k=0
        )

def test_hit_rate_empty_relevant_ids():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]

    with pytest.raises(ValueError):
        evaluator.hit_rate_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=[],
            k=3
        )

def test_hit_rate_empty_retrieved_chunks():
    evaluator = RetrievalEvaluator()

    retrieved = []
    relevant = ["A"]

    result = evaluator.hit_rate_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.0

def test_reciprocal_rank_relevant_chunk_at_rank_one():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("A"),
        create_chunk("C"),
        create_chunk("D")
    ]

    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1.0

def test_reciprocal_rank_relevant_chunk_at_rank_two():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("A"),
        create_chunk("D")
    ]

    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1 / 2

def test_reciprocal_rank_relevant_chunk_at_rank_three():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("A")
    ]

    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1 / 3

def test_reciprocal_rank_relevant_chunk_at_rank_three():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("A")
    ]

    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1 / 3

def test_reciprocal_rank_no_relevant_chunk():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("D"),
        create_chunk("E")
    ]

    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.0

def test_reciprocal_rank_uses_first_relevant_chunk():
    evaluator = RetrievalEvaluator()

    retrieved = [
        create_chunk("C"),
        create_chunk("A"),
        create_chunk("B")
    ]

    relevant = ["A", "B"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 1 / 2

def test_reciprocal_rank_invalid_k():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]
    relevant = ["A"]

    with pytest.raises(ValueError):
        evaluator.reciprocal_rank_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=relevant,
            k=0
        )

def test_reciprocal_rank_empty_relevant_ids():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]

    with pytest.raises(ValueError):
        evaluator.reciprocal_rank_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=[],
            k=3
        )

def test_reciprocal_rank_empty_relevant_ids():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("A")]

    with pytest.raises(ValueError):
        evaluator.reciprocal_rank_at_k(
            retrieved_chunks=retrieved,
            relevant_chunk_ids=[],
            k=3
        )

def test_reciprocal_rank_empty_retrieved_chunks():
    evaluator = RetrievalEvaluator()

    retrieved = []
    relevant = ["A"]

    result = evaluator.reciprocal_rank_at_k(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result == 0.0

def test_evaluate():
    evaluator = RetrievalEvaluator()

    retrieved = [create_chunk("C"),create_chunk("A"),create_chunk("D")]
    relevant = ["A","B"]

    result = evaluator.evaluate(
        retrieved_chunks=retrieved,
        relevant_chunk_ids=relevant,
        k=3
    )

    assert result.recall == 1/2
    assert result.precision == 1/3
    assert result.hit_rate == 1.0
    assert result.reciprocal_rank == 1/2