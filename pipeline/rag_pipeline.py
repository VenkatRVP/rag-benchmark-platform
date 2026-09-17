from retrievers.base_retriever import BaseRetriever
from prompt.base_prompt_builder import BasePromptBuilder
from llm.base_llm import BaseLLM
from models.chunk import Chunk

class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        chunks: list[Chunk],
        logging : bool = False):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.chunks = chunks
        self._is_indexed = False
        self.logging = logging

    def index(self):
        self.retriever.index(self.chunks)
        self._is_indexed = True

    def ask(self, query : str)-> str:
        if not self._is_indexed:
            raise RuntimeError(
                "Please call index() before ask()."
            )
        retrieved_chunks = self.retriever.retrieve(query=query, k=3)

        if self.logging:
            for i, (chunk, score) in enumerate(retrieved_chunks):
                print(f"\nChunk {i+1}")
                print(f"Score: {score}")
                print(chunk.chunk_text)

        prompt = self.prompt_builder.build(query, retrieved_chunks=retrieved_chunks)

        if self.logging:
            print(f"Calling LLM...")
        return self.llm.generate(prompt)