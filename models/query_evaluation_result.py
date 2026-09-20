from models.evaluation_result import EvaluationResult


class QueryEvaluationResult:
    def __init__(
        self,
        question: str,
        retrieved_chunk_ids: list[str],
        relevant_chunk_ids: list[str],
        evaluation_result: EvaluationResult
    ) -> None:
        self.question = question
        self.retrieved_chunk_ids = retrieved_chunk_ids
        self.relevant_chunk_ids = relevant_chunk_ids
        self.evaluation_result = evaluation_result