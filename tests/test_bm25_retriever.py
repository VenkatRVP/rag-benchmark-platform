from pathlib import Path

from loaders.pdf_loader import PDFLoader
from chunking.fixed_size_chunker import FixedSizeChunker
from normalize_text.text_normalizer import normalize

from retrievers.bm25_retriever import BM25Retriever


loader = PDFLoader()
chunker = FixedSizeChunker()

data_path = Path("data/raw")

pages = loader.load(file_path=data_path)

for page in pages:
    page.text = normalize(page.text)

chunks = chunker.chunk(pages)

print(f"Loaded {len(pages)} pages")
print(f"Created {len(chunks)} chunks")

retriever = BM25Retriever()
retriever.index(chunks)

query1 = "What is systems engineering?"
query2 = "What is the difference between verification and validation?"
query3 = "Who won the FIFA World Cup in 2022?"

results1 = retriever.retrieve(
    query=query1,
    k=3
)

results2 = retriever.retrieve(
    query=query2,
    k=3
)

results3 = retriever.retrieve(
    query=query3,
    k=3
)

for i, (chunk, score) in enumerate(results1, start=1):
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page: {chunk.page_number}")
    print(f"\n{chunk.chunk_text[:500]}")

for i, (chunk, score) in enumerate(results2, start=1):
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page: {chunk.page_number}")
    print(f"\n{chunk.chunk_text[:500]}")

for i, (chunk, score) in enumerate(results3, start=1):
    print(f"\n{'=' * 50}")
    print(f"Chunk {i}")
    print(f"Score: {score:.4f}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Page: {chunk.page_number}")
    print(f"\n{chunk.chunk_text[:500]}")