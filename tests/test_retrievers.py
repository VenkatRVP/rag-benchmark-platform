from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from vectorstore.faiss_vector_store import FAISSVectorStore
from vectorstore.faiss_factory import FAISSFactory
from preparation.chunk_preparation import ChunkPreparation
from retrievers.retriever_factory import RetrieverFactory


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

    # -------------------------
    # Create embedding/vector store
    # -------------------------

    embedding = SentenceTransformerEmbedding()

    index = FAISSFactory.create_flat_l2(
        embedding.dimension
    )

    vector_store = FAISSVectorStore(index)

    # -------------------------
    # Create factory
    # -------------------------

    factory = RetrieverFactory(
        embedding=embedding,
        vector_store=vector_store
    )

    # -------------------------
    # Create retrievers
    # -------------------------

    dense_retriever = factory.create("dense")
    bm25_retriever = factory.create("bm25")
    hybrid_retriever = factory.create("hybrid")

    # -------------------------
    # Index chunks
    # -------------------------

    dense_retriever.index(chunks)
    bm25_retriever.index(chunks)
    hybrid_retriever.index(chunks)

    # -------------------------
    # Test queries
    # -------------------------

    queries = [
        "What is systems engineering?",
        "What is the difference between verification and validation?",
        "What are verification methods?"
    ]

    # -------------------------
    # Run comparisons
    # -------------------------

    for query in queries:

        dense_results = dense_retriever.retrieve(
            query=query,
            k=3
        )

        bm25_results = bm25_retriever.retrieve(
            query=query,
            k=3
        )

        hybrid_results = hybrid_retriever.retrieve(
            query=query,
            k=3
        )

        print_results(
            "DENSE",
            query,
            dense_results
        )

        print_results(
            "BM25",
            query,
            bm25_results
        )

        print_results(
            "HYBRID",
            query,
            hybrid_results
        )


if __name__ == "__main__":
    main()