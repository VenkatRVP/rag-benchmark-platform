from chunking import base_chunker
from models.chunk import Chunk
from models.page import Page


class FixedSizeChunker(base_chunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        pages = sorted(
                        pages,
                        key=lambda page: (page.document_name, page.page_number)
                    )
        chunks: list[Chunk] = []
        step_size = self.chunk_size - self.overlap

        current_document = None
        chunk_id = 0

        for page in pages:
            if current_document != page.document_name:
                current_document = page.document_name
                chunk_id = 0

            if not page.text:
                continue

            start = 0
            text_length = len(page.text)

            while start < text_length:
                end = start + self.chunk_size
                chunk_text = page.text[start:end]

                chunks.append(
                    Chunk(
                        page.document_name,
                        page.page_number,
                        chunk_text,
                        chunk_id,
                    )
                )

                chunk_id += 1

                if end >= text_length:
                    break

                start += step_size

        return chunks