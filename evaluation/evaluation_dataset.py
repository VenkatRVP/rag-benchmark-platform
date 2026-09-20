import json
from pathlib import Path

from models.evaluation_sample import EvaluationSample


class EvaluationDataset:

    def __init__(
        self,
        samples: list[EvaluationSample]
    ) -> None:
        self.samples = samples
        self.loaded = False

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":

        file_path = Path(path)

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:
                raw_list = json.load(file)

            samples = [
                EvaluationSample(**item)
                for item in raw_list
            ]

            dataset = cls(samples)
            dataset.loaded = True

            return dataset

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Evaluation dataset not found: {file_path}"
            )

    def get_relevant_chunk_ids(
        self,
        question: str
    ) -> list[str]:

        if not self.loaded:
            raise ValueError(
                "Dataset should be loaded"
            )

        for sample in self.samples:
            if sample.question == question:
                return sample.relevant_chunk_ids

        raise KeyError(
            f"Question not found: {question}"
        )