"""
AutoDAN-Turbo Base LLM Agent

Provides the base class for all LLM agents in the framework.
Supports multiple LLM providers: OpenAI, Google, Anthropic, and local models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for an LLM agent."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4-1106-turbo"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 60
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        provider = data.get("provider", "openai")
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())

        return cls(
            provider=provider,
            model=data.get("model", "gpt-4-1106-turbo"),
            api_key=data.get("api_key"),
            api_base=data.get("api_base"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 4096),
            timeout=data.get("timeout", 60),
            extra_params=data.get("extra_params", {}),
        )


class BaseLLMAgent(ABC):
    """
    Base class for all LLM agents in AutoDAN-Turbo.

    Provides common functionality for:
    - LLM API calls (OpenAI, Google, Anthropic, local)
    - Message formatting
    - Error handling and retries
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the agent.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        if self.config.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.config.provider == LLMProvider.GOOGLE:
            self._init_google()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.config.provider == LLMProvider.LOCAL:
            self._init_local()
        elif self.config.provider == LLMProvider.MOCK:
            self._client = "mock"
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from None

    def _init_google(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.api_key)
            self._client = genai.GenerativeModel(self.config.model)
        except ImportError:
            raise ImportError(
                "Google GenAI package not installed. Run: pip install google-generativeai"
            ) from None

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.config.api_key, timeout=self.config.timeout)
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Run: pip install anthropic"
            ) from None

    def _init_local(self):
        """Initialize local model client (e.g., vLLM, Ollama)."""
        try:
            from openai import OpenAI

            # Use OpenAI-compatible API for local models
            self._client = OpenAI(
                api_key=self.config.api_key or "local",
                base_url=self.config.api_base or "http://localhost:8001/v1",
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai") from None

    def _call_openai(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params,
        )
        return response.choices[0].message.content

    def _call_google(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call Google Generative AI API."""
        # Convert messages to Google format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        prompt = "".join(prompt_parts) + "Assistant:"

        generation_config = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        response = self._client.generate_content(prompt, generation_config=generation_config)
        return response.text

    def _call_anthropic(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call Anthropic API."""
        # Extract system message
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self._client.messages.create(
            model=self.config.model,
            system=system,
            messages=filtered_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params,
        )
        return response.content[0].text

    def _call_local(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Call local model API (OpenAI-compatible)."""
        return self._call_openai(messages, **kwargs)

    def _call_mock(self, messages: list[dict[str, str]], **_kwargs) -> str:
        """Mock call for testing."""
        return self._mock_response(messages)

    def _mock_response(self, _messages: list[dict[str, str]]) -> str:
        """Generate mock response for testing. Override in subclasses."""
        return "Mock response"

    def call(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        Call the LLM with the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The LLM response text
        """
        try:
            if self.config.provider == LLMProvider.OPENAI:
                return self._call_openai(messages, **kwargs)
            elif self.config.provider == LLMProvider.GOOGLE:
                return self._call_google(messages, **kwargs)
            elif self.config.provider == LLMProvider.ANTHROPIC:
                return self._call_anthropic(messages, **kwargs)
            elif self.config.provider == LLMProvider.LOCAL:
                return self._call_local(messages, **kwargs)
            elif self.config.provider == LLMProvider.MOCK:
                return self._call_mock(messages, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def call_with_system(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """
        Convenience method to call LLM with system prompt and user message.

        Args:
            system_prompt: The system prompt
            user_message: The user message
            **kwargs: Additional parameters

        Returns:
            The LLM response text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.call(messages, **kwargs)

    @abstractmethod
    def get_system_prompt(self, **kwargs) -> str:
        """
        Get the system prompt for this agent.

        Args:
            **kwargs: Context-specific parameters

        Returns:
            The system prompt string
        """
        pass

    @abstractmethod
    def process(self, **kwargs) -> Any:
        """
        Process input and return result.

        Args:
            **kwargs: Agent-specific parameters

        Returns:
            Agent-specific result
        """
        pass
