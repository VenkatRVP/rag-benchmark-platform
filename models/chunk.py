class Chunk:

    def __init__(
        self,
        document_name: str,
        page_number: int,
        chunk_text: str,
        chunk_id: str,
        embedding=None
    ) -> None:
        self.document_name = document_name
        self.page_number = page_number
        self.chunk_text = chunk_text
        self.chunk_id = chunk_id
        self.embedding = embedding