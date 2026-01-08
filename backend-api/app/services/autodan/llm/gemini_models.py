import logging

from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiModel:
    def __init__(self, model_name="gemini-3-pro-preview", api_key=None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.GOOGLE_MODEL or "gemini-3-pro-preview"

        # Determine API Key
        self.api_key = api_key or self.settings.get_effective_api_key()

        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found. GeminiModel might fail in Direct mode.")
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, system, user, **kwargs):
        try:
            # Direct API call with new SDK
            generation_config_params = {}
            if "max_length" in kwargs:
                generation_config_params["max_output_tokens"] = kwargs["max_length"]
            if "temperature" in kwargs:
                generation_config_params["temperature"] = kwargs["temperature"]

            # Gemini 3 Pro support: Add thinking_level for enhanced reasoning
            if "gemini-3" in self.model_name and "thinking_level" in kwargs:
                generation_config_params["thinking_config"] = {
                    "thinking_budget": kwargs.get("thinking_level", "high")
                }

            config = (
                types.GenerateContentConfig(**generation_config_params)
                if generation_config_params
                else None
            )

            # Combine system and user prompts if system instruction provided
            contents = f"System: {system}\n\nUser: {user}" if system else user

            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )

            if not response.text:
                return ""

            content = response.text.strip()
            return self.strip_double_quotes(content)

        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise e

    def strip_double_quotes(self, input_str):
        if input_str.startswith('"') and input_str.endswith('"'):
            return input_str[1:-1]
        return input_str


class GeminiEmbeddingModel:
    def __init__(self, model_name="text-embedding-3-small", api_key=None):
        self.settings = get_settings()
        self.model_name = model_name
        self.api_key = api_key or self.settings.get_effective_api_key()

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("GeminiEmbeddingModel configured with API key")
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")
                raise

    def encode(self, text):
        try:
            # Handle list of strings or single string
            if isinstance(text, list):
                return [self._embed_single(t) for t in text]
            else:
                return self._embed_single(text)
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            return None

    def _embed_single(self, text):
        try:
            result = self.client.models.embed_content(model=self.model_name, contents=text)
            # Access the embedding from the response object
            if hasattr(result, "embeddings") and result.embeddings:
                return result.embeddings[0].values
            elif hasattr(result, "embedding"):
                return (
                    result.embedding.values
                    if hasattr(result.embedding, "values")
                    else result.embedding
                )
            return result
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
