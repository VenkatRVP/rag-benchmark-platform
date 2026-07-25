from loaders.base_loader import BaseLoader
from pathlib import Path
from normalize_text.text_normalizer import normalize
from chunking.base_chunker import BaseChunker
from embeddings.base_embedding import BaseEmbedding
from vectorstore.base_vector_store import BaseVectorStore
from retrievers.base_retriever import BaseRetriever
from prompt.base_prompt_builder import BasePromptBuilder
from llm.base_llm import BaseLLM
from models.page import Page

class RAGPipeline:
    def __init__(
        self,
        loader: BaseLoader,
        chunker: BaseChunker,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        data_path: Path
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding = embedding
        self.vector_store = vector_store
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.data_path = data_path
        self._is_indexed = False

    def index(self):
        print("Indexing starts..")
        pages = self._normalize_pages(self.loader.load(file_path=self.data_path))
        print(f"Loaded {len(pages)} pages")
        chunks = self.chunker.chunk(pages)
        print(f"Created {len(chunks)} chunks")
        chunks = self.embedding.embed(chunks=chunks)
        self.vector_store.add(chunks=chunks)
        print(f"Indexed {len(chunks)} chunks")
        self._is_indexed = True

    def ask(self, query : str)-> str:
        if not self._is_indexed:
            raise RuntimeError(
                "Please call index() before ask()."
            )
        retrieved_chunks = self.retriever.retrieve(query=query, k=3)

        for i, (chunk, score) in enumerate(retrieved_chunks):
            print(f"\nChunk {i+1}")
            print(f"Score: {score}")
            print(chunk.chunk_text)
        prompt = self.prompt_builder.build(query, retrieved_chunks=retrieved_chunks)
        print(f"Calling LLM...")
        return self.llm.generate(prompt)

    @staticmethod
    def _normalize_pages(
        pages: list[Page]
    ) -> list[Page]:
        for page in pages:
            page.text = normalize(page.text)
        return pages