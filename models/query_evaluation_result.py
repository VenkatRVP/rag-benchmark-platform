from models.evaluation_result import EvaluationResult

class QueryEvaluationResult:
    def __init__(
        self,
        question: str,
        evaluation_result: EvaluationResult
    ) -> None:
        self.question = question
        self.evaluation_result = evaluation_result