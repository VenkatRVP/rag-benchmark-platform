from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from preparation.chunk_preparation import ChunkPreparation


def main() -> None:

    loader = PDFLoader()
    chunker = FixedSizeChunker()

    folder_path = Path("data/raw")

    chunk_preparation = ChunkPreparation(
        loader=loader,
        chunker=chunker,
        data_path=folder_path
    )

    chunks = chunk_preparation.prepare()

    with open(
        "data/evaluation_chunks.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for chunk in chunks:
            file.write(
                f"Chunk ID: {chunk.chunk_id}\n"
            )
            file.write(
                f"Page: {chunk.page_number}\n"
            )
            file.write(
                f"Text:\n{chunk.chunk_text}\n"
            )
            file.write(
                "=" * 80 + "\n\n"
            )


if __name__ == "__main__":
    main()