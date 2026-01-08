import os

from openai import OpenAI


class LLMClient:
    """
    Wrapper for OpenAI client to interact with Gemini models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.default_model = "gemini-2.0-flash-exp"  # Using a capable model

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a response from the LLM.
        """
        model = model or self.default_model

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if reasoning_effort and model.startswith(
            "gemini-2.0-flash-thinking-exp"
        ):  # Reasoning only supported on specific models usually
            # Note: reasoning_effort is experimental in the OpenAI client for Gemini,
            # checking if it's actually supported or if we should just rely on the model's native capabilities.
            # Based on the user guide, it is supported.
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            raise e
