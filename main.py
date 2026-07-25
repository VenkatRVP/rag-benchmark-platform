from pathlib import Path
import faiss

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from vectorstore.faiss_vector_store import FAISSVectorStore
from vectorstore.faiss_factory import FAISSFactory
from retrievers.dense_retriever import DenseRetriever
from prompt.rag_prompt_builder import RAGPromptBuilder
from llm.ollama_llm import OllamaLLM
from pipeline.rag_pipeline import RAGPipeline

loader = PDFLoader()

chunker = FixedSizeChunker()

embedding = SentenceTransformerEmbedding()

index = FAISSFactory.create_flat_l2(embedding.dimension)

vector_store = FAISSVectorStore(index)

retriever = DenseRetriever(
    embedding=embedding,
    vector_store=vector_store
)

prompt_builder = RAGPromptBuilder()

llm = OllamaLLM()

folder_path = Path("data/raw")

pipeline = RAGPipeline(
    loader=loader,
    chunker=chunker,
    embedding=embedding,
    vector_store=vector_store,
    retriever=retriever,
    prompt_builder=prompt_builder,
    llm=llm,
    data_path=folder_path
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