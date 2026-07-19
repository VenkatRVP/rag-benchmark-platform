class Chunk:
    def __init__(self, document_name, page_number, chunk_text, chunk_id, embedding = None):
        self.document_name = document_name
        self.page_number = page_number
        self.chunk_text = chunk_text
        self.chunk_id = chunk_id
        self.embedding = embedding
        