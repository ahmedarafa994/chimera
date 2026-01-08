from app.domain.houyi.interfaces import OptimizationTarget
from app.domain.houyi.models import PromptInjection
from app.services.llm_service import llm_service


class LLMServiceTarget(OptimizationTarget):
    def __init__(
        self, provider: str = "openai", model: str = "gpt-4", application_document: str = ""
    ):
        self._provider = provider
        self._model = model
        self._application_document = application_document

    @property
    def application_document(self) -> str:
        return self._application_document

    async def run(self, prompt_injection: PromptInjection) -> str:
        prompt = prompt_injection.get_attack_prompt()
        # Simulate the target application responding to the prompt
        # Ideally this should be the ACTUAL target application logic.
        # For now, we use the LLM directly as if it were the target.
        response_obj = await llm_service.generate(
            prompt, provider=self._provider, model=self._model
        )
        return response_obj.content
