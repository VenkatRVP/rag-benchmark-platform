from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from normalize_text.text_normalizer import normalize
from retrievers.bm25_retriever import BM25Retriever


def print_results(
    query: str,
    results
) -> None:

    print(f"\n{'=' * 50}")
    print(f"Query: {query}")
    print(f"{'=' * 50}")

    for rank, (chunk, score) in enumerate(
        results,
        start=1
    ):
        print(f"\nChunk {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"\n{chunk.chunk_text[:500]}")


def main() -> None:

    loader = PDFLoader()
    chunker = FixedSizeChunker()

    data_path = Path("data/raw")

    pages = loader.load(
        file_path=data_path
    )

    for page in pages:
        page.text = normalize(page.text)

    chunks = chunker.chunk(pages)

    print(f"Loaded {len(pages)} pages")
    print(f"Created {len(chunks)} chunks")

    retriever = BM25Retriever()
    retriever.index(chunks)

    queries = [
        "What is systems engineering?",
        "What is the difference between verification and validation?",
        "Who won the FIFA World Cup in 2022?"
    ]

    for query in queries:

        results = retriever.retrieve(
            query=query,
            k=3
        )

        print_results(
            query=query,
            results=results
        )


if __name__ == "__main__":
    main()