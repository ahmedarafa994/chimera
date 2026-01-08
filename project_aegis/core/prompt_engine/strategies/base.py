from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PromptCandidate:
    content: str
    metadata: Dict[str, Any] = None
    score: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """
    Abstract base class for all adversarial prompt generation strategies.
    Project Aegis - Core Prompt Engine
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate_candidates(self, base_instruct: str, n: int = 1) -> List[PromptCandidate]:
        """
        Generate adversarial candidates based on the base instruction.
        """
        pass

    @abstractmethod
    async def refine(self, candidates: List[PromptCandidate], feedback: List[Dict[str, Any]]) -> List[PromptCandidate]:
        """
        Refine candidates based on feedback (e.g., from an evaluation or target model).
        """
        pass

    @abstractmethod
    def get_meta(self) -> Dict[str, Any]:
        """
        Return metadata about the strategy (e.g., name, version, capabilities).
        """
        pass
