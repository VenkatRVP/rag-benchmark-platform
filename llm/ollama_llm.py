import ollama

from llm.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """LLM implementation using a locally running Ollama model."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b"
    ) -> None:

        if not model_name.strip():
            raise ValueError(
                "Model name cannot be empty."
            )

        self.model_name = model_name

    def generate(
        self,
        prompt: str
    ) -> str:

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"]

        except Exception as e:
            raise RuntimeError(
                f"Error communicating with Ollama: {e}"
            ) from e