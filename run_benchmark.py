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

hybrid_index = FAISSFactory.create_flat_l2(
    embedding.dimension
)
hybrid_vector_store = FAISSVectorStore(hybrid_index)

dense_index = FAISSFactory.create_flat_l2(
    embedding.dimension
)
dense_vector_store = FAISSVectorStore(dense_index)

dense_factory = RetrieverFactory(
    embedding=embedding,
    vector_store=dense_vector_store
)

hybrid_factory = RetrieverFactory(
    embedding=embedding,
    vector_store=hybrid_vector_store
)

hybrid_retriever = hybrid_factory.create(
    "hybrid",
    candidate_k=10,
    rrf_constant=60
)

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
    retrieval_evaluator
)

dense_benchmark_runner = BenchmarkRunner(
    dense_retriever,
    dataset,
    retrieval_evaluator
)

bm25_benchmark_runner = BenchmarkRunner(
    bm25_retriever,
    dataset,
    retrieval_evaluator
)


dense_results = dense_benchmark_runner.run(k=3)
dense_benchmark_runner.print_error_analysis()

bm25_results = bm25_benchmark_runner.run(k=3)
bm25_benchmark_runner.print_error_analysis()

hybrid_results = hybrid_benchmark_runner.run(k=3)
hybrid_benchmark_runner.print_error_analysis()


print("\n" + "=" * 60)
print("RAG RETRIEVAL BENCHMARK")
print("=" * 60)

print(
    f"\n{'Retriever':<15} "
    f"{'Recall':<10} "
    f"{'Precision':<12} "
    f"{'Hit Rate':<10} "
    f"{'MRR':<10}"
)

print("-" * 60)

print(
    f"{'Dense':<15} "
    f"{dense_results.recall:<10.3f} "
    f"{dense_results.precision:<12.3f} "
    f"{dense_results.hit_rate:<10.3f} "
    f"{dense_results.mrr:<10.3f}"
)

print(
    f"{'BM25':<15} "
    f"{bm25_results.recall:<10.3f} "
    f"{bm25_results.precision:<12.3f} "
    f"{bm25_results.hit_rate:<10.3f} "
    f"{bm25_results.mrr:<10.3f}"
)

print(
    f"{'Hybrid':<15} "
    f"{hybrid_results.recall:<10.3f} "
    f"{hybrid_results.precision:<12.3f} "
    f"{hybrid_results.hit_rate:<10.3f} "
    f"{hybrid_results.mrr:<10.3f}"
)