from abc import ABC, abstractmethod
from typing import Any, List, Dict
from dataclasses import dataclass

@dataclass
class PromptCandidate:
    prompt_text: str
    metadata: Dict[str, Any]

class IPromptGenerator(ABC):
    """
    Interface for modules that generate initial prompt candidates (Chimera).
    """
    @abstractmethod
    def generate_candidates(self, objective: str, count: int) -> List[PromptCandidate]:
        pass

class IOptimizer(ABC):
    """
    Interface for modules that optimize prompts against a target (AutoDan).
    """
    @abstractmethod
    async def optimize(self, prompt: str, target_model: Any, loss_threshold: float, max_steps: int) -> str:
        pass

class IEvaluator(ABC):
    """
    Interface for safety evaluation.
    """
    @abstractmethod
    def evaluate(self, response: str) -> float:
        """Returns a score from 0.0 (Refusal) to 1.0 (Full Bypass)"""
        pass
