# =============================================================================
# DeepTeam Model Callbacks
# =============================================================================
# Adapters that connect Chimera's LLM services to DeepTeam's callback interface.
# =============================================================================

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


# Type alias for DeepTeam callback
CallbackType = Callable[[str], str]
AsyncCallbackType = Callable[[str], "asyncio.Future[str]"]


class ChimeraModelCallback:
    """Adapter that wraps Chimera's LLM service as a DeepTeam model callback.

    This allows DeepTeam to red team any LLM accessible through Chimera's
    unified LLM service interface.
    """

    def __init__(
        self,
        llm_service=None,
        model_id: str | None = None,
        provider: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        """Initialize the callback adapter.

        Args:
            llm_service: Chimera's LLM service instance
            model_id: Model identifier (e.g., "gpt-4", "gemini-pro")
            provider: Provider name (e.g., "openai", "google", "anthropic")
            system_prompt: Optional system prompt to prepend
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        """
        self.llm_service = llm_service
        self.model_id = model_id
        self.provider = provider
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._call_count = 0
        self._error_count = 0

    async def __call__(self, input: str) -> str:
        """Async callback that DeepTeam calls with adversarial inputs.

        Args:
            input: The adversarial input/prompt from DeepTeam

        Returns:
            The model's response

        """
        self._call_count += 1

        try:
            if self.llm_service is not None:
                return await self._call_chimera_service(input)
            return await self._call_direct_provider(input)
        except Exception as e:
            self._error_count += 1
            logger.exception(f"Model callback error: {e}")
            return f"Error: {e!s}"

    async def _call_chimera_service(self, input: str) -> str:
        """Call through Chimera's LLM service."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": input})

        response = await self.llm_service.generate(
            messages=messages,
            model_id=self.model_id,
            provider=self.provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.get("content", response.get("text", str(response)))

    async def _call_direct_provider(self, input: str) -> str:
        """Call provider directly without Chimera service."""
        # This is a fallback for when Chimera service is not available
        if self.provider == "openai":
            return await self._call_openai(input)
        if self.provider == "anthropic":
            return await self._call_anthropic(input)
        if self.provider in ("google", "gemini"):
            return await self._call_google(input)
        msg = f"Unknown provider: {self.provider}"
        raise ValueError(msg)

    async def _call_openai(self, input: str) -> str:
        """Call OpenAI directly."""
        import openai

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": input})

        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model_id or "gpt-4o-mini",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, input: str) -> str:
        """Call Anthropic directly."""
        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=self.model_id or "claude-3-sonnet-20240229",
            max_tokens=self.max_tokens,
            system=self.system_prompt or "",
            messages=[{"role": "user", "content": input}],
        )
        return response.content[0].text

    async def _call_google(self, input: str) -> str:
        """Call Google/Gemini directly."""
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_id or "gemini-pro")

        prompt = input
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{input}"

        response = await asyncio.to_thread(
            model.generate_content,
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text

    @property
    def stats(self) -> dict:
        """Get callback statistics."""
        return {
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count) / self._call_count
                if self._call_count > 0
                else 1.0
            ),
        }


def create_model_callback(
    model_id: str,
    provider: str | None = None,
    system_prompt: str | None = None,
    llm_service=None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> ChimeraModelCallback:
    """Factory function to create a model callback.

    Args:
        model_id: Model identifier
        provider: Provider name (auto-detected if not specified)
        system_prompt: Optional system prompt
        llm_service: Optional Chimera LLM service
        temperature: Generation temperature
        max_tokens: Maximum tokens

    Returns:
        ChimeraModelCallback instance

    """
    # Auto-detect provider from model_id if not specified
    if provider is None:
        provider = _detect_provider(model_id)

    return ChimeraModelCallback(
        llm_service=llm_service,
        model_id=model_id,
        provider=provider,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _detect_provider(model_id: str) -> str:
    """Detect provider from model ID."""
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["gpt", "o1", "davinci", "curie", "babbage"]):
        return "openai"
    if any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"
    if any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "google"
    if any(x in model_lower for x in ["llama", "mistral", "mixtral"]):
        return "local"
    return "openai"  # Default to OpenAI


class AutoDANCallback(ChimeraModelCallback):
    """Specialized callback that integrates with AutoDAN engines.

    This callback uses Chimera's AutoDAN engines to generate more
    sophisticated responses that can be used in red teaming.
    """

    def __init__(
        self,
        autodan_engine=None,
        use_reasoning: bool = True,
        use_ppo: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.autodan_engine = autodan_engine
        self.use_reasoning = use_reasoning
        self.use_ppo = use_ppo

    async def __call__(self, input: str) -> str:
        """Call that can optionally use AutoDAN processing."""
        self._call_count += 1

        try:
            # First get the base response
            base_response = await super().__call__(input)

            # If AutoDAN engine is available, we can analyze the response
            if self.autodan_engine and self.use_reasoning:
                # Log for analysis - AutoDAN can learn from this
                self.autodan_engine.record_interaction(
                    prompt=input,
                    response=base_response,
                    context="deepteam_red_team",
                )

            return base_response

        except Exception as e:
            self._error_count += 1
            logger.exception(f"AutoDAN callback error: {e}")
            return f"Error: {e!s}"


class MultiModelCallback:
    """Callback that can test multiple models simultaneously.

    Useful for comparing vulnerability across different models.
    """

    def __init__(self, callbacks: list[ChimeraModelCallback]) -> None:
        """Initialize with multiple callbacks.

        Args:
            callbacks: List of model callbacks to test

        """
        self.callbacks = callbacks
        self._results: dict[str, list] = {
            cb.model_id or f"model_{i}": [] for i, cb in enumerate(callbacks)
        }

    async def __call__(self, input: str) -> dict[str, str]:
        """Call all models and return their responses.

        Args:
            input: Adversarial input

        Returns:
            Dict mapping model_id to response

        """
        tasks = [cb(input) for cb in self.callbacks]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        result = {}
        for i, (cb, response) in enumerate(zip(self.callbacks, responses, strict=False)):
            model_id = cb.model_id or f"model_{i}"
            if isinstance(response, Exception):
                result[model_id] = f"Error: {response!s}"
            else:
                result[model_id] = response
            self._results[model_id].append(result[model_id])

        return result

    @property
    def results(self) -> dict[str, list]:
        """Get all results by model."""
        return self._results

    @property
    def stats(self) -> dict[str, dict]:
        """Get stats for all models."""
        return {cb.model_id or f"model_{i}": cb.stats for i, cb in enumerate(self.callbacks)}
