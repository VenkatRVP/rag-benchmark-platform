import pytest

from evaluation.evaluation_dataset import EvaluationDataset


def test_valid_dataset_loads():
    dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

    assert isinstance(dataset, EvaluationDataset)


def test_correct_number_of_samples():
    dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

    assert len(dataset.samples) == 10


def test_sample_fields_are_accessible():
    dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

    sample = dataset.samples[0]

    assert sample.question
    assert sample.relevant_chunk_ids


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        EvaluationDataset.load("does_not_exist.json")

def test_get_relevant_chunk_ids():
    dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

    chunk_ids = dataset.get_relevant_chunk_ids(
        "What is systems engineering?"
    )

    assert chunk_ids == [
        "nasa_systems_engineering_handbook_0.pdf_67",
        "nasa_systems_engineering_handbook_0.pdf_69"
    ]


def test_question_not_found():
    dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

    with pytest.raises(KeyError):
        dataset.get_relevant_chunk_ids(
            "This question does not exist"
        )