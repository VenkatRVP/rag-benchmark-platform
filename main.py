from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from vectorstore.faiss_vector_store import FAISSVectorStore
from vectorstore.faiss_factory import FAISSFactory
from prompt.rag_prompt_builder import RAGPromptBuilder
from llm.ollama_llm import OllamaLLM
from pipeline.rag_pipeline import RAGPipeline
from preparation.chunk_preparation import ChunkPreparation
from retrievers.retriever_factory import RetrieverFactory


def main() -> None:

    loader = PDFLoader()

    chunker = FixedSizeChunker()

    embedding = SentenceTransformerEmbedding()

    index = FAISSFactory.create_flat_l2(
        embedding.dimension
    )

    vector_store = FAISSVectorStore(index)

    prompt_builder = RAGPromptBuilder()

    llm = OllamaLLM()

    folder_path = Path("data/raw")

    chunk_preparation = ChunkPreparation(
        loader=loader,
        chunker=chunker,
        data_path=folder_path
    )

    chunks = chunk_preparation.prepare()

    retriever_factory = RetrieverFactory(
        embedding=embedding,
        vector_store=vector_store
    )

    retriever = retriever_factory.create(
        "hybrid"
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        chunks=chunks,
        logging=False
    )

    pipeline.index()

    print("\nRAG Pipeline is ready!")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        try:
            answer = pipeline.ask(query)
            print(f"\nAssistant: {answer}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()