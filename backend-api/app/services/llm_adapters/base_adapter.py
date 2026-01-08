from abc import ABC, abstractmethod
from typing import Any


class BaseLLMAdapter(ABC):
    def __init__(self, model_name: str, api_key: str, config: dict[str, Any]):
        self.model_name = model_name
        self.api_key = api_key
        self.config = config

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generates a response from the LLM based on the given prompt.
        """
        pass

    @abstractmethod
    async def evaluate_response(self, prompt: str, response: str, criteria: str) -> dict[str, Any]:
        """
        Evaluates the LLM's response against specific criteria to determine if
        the evasion was successful or if the LLM followed safety guidelines.
        This could potentially use another LLM or a rule-based system.
        """
        pass
