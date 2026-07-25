from prompt.base_prompt_builder import BasePromptBuilder
from models.chunk import Chunk

DEFAULT_TEMPLATE = """
You are a helpful AI assistant.

Use ONLY the provided context to answer the user's question.

If the answer cannot be found in the context, reply:
"I don't have enough information in the provided context to answer that question."

Do not make up information or use external knowledge.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
""".strip()

class RAGPromptBuilder(BasePromptBuilder):
    
    def __init__(self, template: str | None = None) -> None:
        self.template = template or DEFAULT_TEMPLATE

    def build(self,user_query : str, retrieved_chunks : list[tuple[Chunk, float]]) -> str :
        context = "\n\n".join(f"Chunk {i+1}:\n{chunk.chunk_text}" for i, (chunk,_) in enumerate(retrieved_chunks))
        return self.template.format(
                context=context,
                question=user_query
            )