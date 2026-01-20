import logging
import os

from openai import AzureOpenAI, OpenAI

from app.core.config import get_settings


class OpenAIEmbeddingModel:
    def __init__(
        self,
        azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-02-01",
        azure_deployment_name: str | None = None,
        azure_api_key: str | None = None,
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-ada-002",
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = get_settings()
        self.logger = logger

        # Direct Mode
        self.azure = azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name
        self.embedding_model = embedding_model

        if self.azure:
            self.client = AzureOpenAI(
                api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            self.client = OpenAI(
                api_key=openai_api_key
                or self.settings.OPENAI_API_KEY
                or os.getenv("OPENAI_API_KEY"),
            )

    def encode(self, text):
        try:
            single_input = False
            if isinstance(text, str):
                text = [text]
                single_input = True

            # Use modern OpenAI API client for both Azure and OpenAI
            if self.azure:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.azure_deployment_name,
                )
            else:
                response = self.client.embeddings.create(input=text, model=self.embedding_model)

            # Parse response - both Azure and OpenAI use the same format now
            embeddings = [item.embedding for item in response.data]

            if single_input and len(embeddings) == 1:
                return embeddings[0]
            return embeddings
        except Exception as e:
            if self.logger:
                self.logger.error(f"Embedding error: {e}", exc_info=True)
            else:
                pass
            return None
