from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from preparation.chunk_preparation import ChunkPreparation


def print_results(retriever_name, query, results):
    print(f"\n{'=' * 60}")
    print(f"{retriever_name}")
    print(f"Query: {query}")
    print(f"{'=' * 60}")

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Score: {score}")
        print(f"Text: {chunk.chunk_text[:300]}")


def main():

    # -------------------------
    # Prepare chunks
    # -------------------------

    loader = PDFLoader()
    chunker = FixedSizeChunker()

    folder_path = Path("data/raw")

    chunk_preparation = ChunkPreparation(
        loader=loader,
        chunker=chunker,
        data_path=folder_path
    )

    chunks = chunk_preparation.prepare()

    with open("data/evaluation_chunks.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"Chunk ID: {chunk.chunk_id}\n")
            f.write(f"Page: {chunk.page_number}\n")
            f.write(f"Text:\n{chunk.chunk_text}\n")
            f.write("=" * 80 + "\n\n")



if __name__ == "__main__":
    main()