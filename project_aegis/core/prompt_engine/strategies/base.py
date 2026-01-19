from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PromptCandidate:
    content: str
    metadata: dict[str, Any] = None
    score: float | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStrategy(ABC):
    """
    Abstract base class for all adversarial prompt generation strategies.
    Project Aegis - Core Prompt Engine
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate_candidates(self, base_instruct: str, n: int = 1) -> list[PromptCandidate]:
        """
        Generate adversarial candidates based on the base instruction.
        """
        pass

    @abstractmethod
    async def refine(
        self, candidates: list[PromptCandidate], feedback: list[dict[str, Any]]
    ) -> list[PromptCandidate]:
        """
        Refine candidates based on feedback (e.g., from an evaluation or target model).
        """
        pass

    @abstractmethod
    def get_meta(self) -> dict[str, Any]:
        """
        Return metadata about the strategy (e.g., name, version, capabilities).
        """
        pass
