from pathlib import Path

from chunking.base_chunker import BaseChunker
from loaders.base_loader import BaseLoader
from models.chunk import Chunk
from models.page import Page
from normalize_text.text_normalizer import normalize


class ChunkPreparation:

    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        data_path: Path
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.data_path = data_path

    def prepare(self) -> list[Chunk]:
        print("Indexing starts..")

        pages = self._normalize_pages(
            self.loader.load(
                file_path=self.data_path
            )
        )

        print(f"Loaded {len(pages)} pages")

        chunks = self.chunker.chunk(pages)

        print(f"Created {len(chunks)} chunks")

        return chunks

    @staticmethod
    def _normalize_pages(
        pages: list[Page]
    ) -> list[Page]:
        for page in pages:
            page.text = normalize(page.text)

        return pages