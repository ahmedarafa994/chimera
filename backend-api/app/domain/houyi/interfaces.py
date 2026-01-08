from abc import ABC, abstractmethod

from app.domain.houyi.models import PromptInjection


class OptimizationTarget(ABC):
    @abstractmethod
    async def run(self, prompt_injection: PromptInjection) -> str:
        pass

    @property
    @abstractmethod
    def application_document(self) -> str:
        pass
