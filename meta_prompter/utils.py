from abc import ABC, abstractmethod

import numpy as np


class TokenEmbedding(ABC):
    """Abstract interface for token embeddings."""

    @abstractmethod
    def encode(self, tokens: list[str]) -> np.ndarray:
        """Encode tokens to embedding vectors."""
        pass

    @abstractmethod
    def decode(self, embeddings: np.ndarray) -> list[str]:
        """Decode embedding vectors to tokens."""
        pass

    @abstractmethod
    def similarity(self, token1: str, token2: str) -> float:
        """Calculate similarity between two tokens."""
        pass

    @abstractmethod
    def nearest_neighbors(self, token: str, k: int = 10) -> list[tuple[str, float]]:
        """Find k nearest neighbors for a token."""
        pass


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    In a real implementation, this would wrap an API client (e.g., OpenAI, Anthropic, Gemini).
    """

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """
        Generate a response from the LLM based on the prompt and system instruction.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system instruction/context.

        Returns:
            The generated text response.
        """
        pass
