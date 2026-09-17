class EvaluationSample:

    def __init__(
        self,
        question: str,
        relevant_chunk_ids: list[str]
    ) -> None:

        if question is None or question == "":
            raise ValueError("Question cannot be empty.")

        if relevant_chunk_ids is None or not relevant_chunk_ids:
            raise ValueError("Relevant Chunk Ids cannot be empty.")

        if not isinstance(relevant_chunk_ids, list):
            raise ValueError(
                "Relevant chunk Ids should be of type list."
            )

        for x in relevant_chunk_ids:
            if x is None or x == "":
                raise ValueError(
                    "Relevant Chunk Id cannot be empty."
                )

            if not isinstance(x, str):
                raise ValueError(
                    "Relevant Chunk Id should be of type string."
                )

        self.question = question
        self.relevant_chunk_ids = relevant_chunk_ids