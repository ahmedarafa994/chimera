"""
Base class for all prompt transformers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TransformationContext:
    """Context passed to transformers during execution."""

    original_prompt: str
    current_prompt: str
    potency: int
    technique_suite: str
    metadata: dict = field(default_factory=dict)


class BaseTransformer(ABC):
    """Abstract base class for prompt transformers."""

    name: str = "base_transformer"
    description: str = "Base transformer class"
    category: str = "general"  # transformer, framer, obfuscator

    @abstractmethod
    def transform(self, context: TransformationContext) -> str:
        """
        Transform the prompt based on the context.

        Args:
            context: The transformation context containing prompt and settings.

        Returns:
            The transformed prompt string.
        """
        pass
