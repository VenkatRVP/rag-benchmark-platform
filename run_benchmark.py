from pathlib import Path
from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from vectorstore.faiss_vector_store import FAISSVectorStore
from vectorstore.faiss_factory import FAISSFactory
from preparation.chunk_preparation import ChunkPreparation
from retrievers.retriever_factory import RetrieverFactory
from evaluation.evaluation_dataset import EvaluationDataset
from benchmark.benchmark_runner import BenchmarkRunner
from evaluation.retrieval_evaluator import RetrievalEvaluator

loader = PDFLoader()

chunker = FixedSizeChunker()

folder_path = Path("data/raw")

chunk_preparation = ChunkPreparation(
    loader=loader,
    chunker=chunker,
    data_path=folder_path
)

chunks = chunk_preparation.prepare()

embedding = SentenceTransformerEmbedding()

hybrid_index = FAISSFactory.create_flat_l2(embedding.dimension)
hybrid_vector_store = FAISSVectorStore(hybrid_index)

dense_index = FAISSFactory.create_flat_l2(embedding.dimension)
dense_vector_store = FAISSVectorStore(dense_index)

dense_factory = RetrieverFactory(
    embedding=embedding,
    vector_store=dense_vector_store
)

hybrid_factory = RetrieverFactory(
    embedding=embedding,
    vector_store=hybrid_vector_store
)

hybrid_retriever = hybrid_factory.create("hybrid")
hybrid_retriever.index(chunks=chunks)

dense_retriever = dense_factory.create("dense")
dense_retriever.index(chunks=chunks)

bm25_retriever = dense_factory.create("bm25")
bm25_retriever.index(chunks=chunks)

dataset = EvaluationDataset.load(
        "data/evaluation/retrieval_dataset.json"
    )

retrieval_evaluator = RetrievalEvaluator()

hybrid_benchmark_runner = BenchmarkRunner(
    hybrid_retriever,
    dataset,
    retrieval_evaluator,
    chunks
)

dense_benchmark_runner = BenchmarkRunner(
    dense_retriever,
    dataset,
    retrieval_evaluator,
    chunks
)

bm25_benchmark_runner = BenchmarkRunner(
    bm25_retriever,
    dataset,
    retrieval_evaluator,
    chunks
)

print("Dense")
dense_results = dense_benchmark_runner.run(k=3)

print("BM 25")
bm25_results = bm25_benchmark_runner.run(k=3)

print("Hybrid")
hybrid_results = hybrid_benchmark_runner.run(k=3)

print("\nDense Results:")
print(vars(dense_results))

print("\nBM25 Results:")
print(vars(bm25_results))

print("\nHybrid Results:")
print(vars(hybrid_results))