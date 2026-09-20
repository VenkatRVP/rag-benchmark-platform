import pytest

from models.evaluation_sample import EvaluationSample


def test_valid_sample():
    sample = EvaluationSample(
        question="What is systems engineering?",
        relevant_chunk_ids=["chunk_1", "chunk_2"]
    )

    assert sample.question == "What is systems engineering?"
    assert sample.relevant_chunk_ids == [
        "chunk_1",
        "chunk_2"
    ]


def test_empty_question():
    with pytest.raises(ValueError):
        EvaluationSample(
            question="",
            relevant_chunk_ids=["chunk_1"]
        )


def test_relevant_chunk_ids_must_be_list():
    with pytest.raises(ValueError):
        EvaluationSample(
            question="What is systems engineering?",
            relevant_chunk_ids="chunk_1"
        )


def test_relevant_chunk_ids_cannot_be_empty():
    with pytest.raises(ValueError):
        EvaluationSample(
            question="What is systems engineering?",
            relevant_chunk_ids=[]
        )


def test_chunk_id_must_be_string():
    with pytest.raises(ValueError):
        EvaluationSample(
            question="What is systems engineering?",
            relevant_chunk_ids=["chunk_1", 123]
        )


def test_chunk_id_cannot_be_empty():
    with pytest.raises(ValueError):
        EvaluationSample(
            question="What is systems engineering?",
            relevant_chunk_ids=[""]
        )