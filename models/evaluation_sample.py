class EvaluationSample:

    def __init__(
        self,
        question: str,
        relevant_chunk_ids: list[str]
    ) -> None:

        if question is None or question == "":
            raise ValueError(
                "Question cannot be empty."
            )

        if relevant_chunk_ids is None or not relevant_chunk_ids:
            raise ValueError(
                "Relevant Chunk Ids cannot be empty."
            )

        if not isinstance(relevant_chunk_ids, list):
            raise ValueError(
                "Relevant chunk Ids should be of type list."
            )

        for chunk_id in relevant_chunk_ids:
            if chunk_id is None or chunk_id == "":
                raise ValueError(
                    "Relevant Chunk Id cannot be empty."
                )

            if not isinstance(chunk_id, str):
                raise ValueError(
                    "Relevant Chunk Id should be of type string."
                )

        self.question = question
        self.relevant_chunk_ids = relevant_chunk_ids