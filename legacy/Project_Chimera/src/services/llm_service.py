"""
LLM service for Project Chimera.
Handles LLM provider management and prompt execution.
"""

import logging
import os

# Import existing LLM integration
import sys
import time
from typing import Any

from ..config.settings import get_llm_config
from ..models.domain import (
    CostInfo,
    ExecutionRequest,
    ExecutionResult,
    LLMProvider,
    ProviderType,
    TokenCount,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from llm_integration import llm_integration_engine
    from llm_provider_client import LLMProvider as LLMProviderClient
except ImportError:
    # Fallback if modules not available
    llm_integration_engine = None
    LLMProviderClient = None

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for managing LLM providers and executing prompts.
    """

    def __init__(self):
        self.llm_config = get_llm_config()
        self._providers: dict[str, LLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize default LLM providers."""
        # OpenAI
        self._providers["openai"] = LLMProvider(
            name="OpenAI",
            provider_type=ProviderType.OPENAI,
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            enabled=bool(os.getenv("OPENAI_API_KEY")),
        )

        # Anthropic
        self._providers["anthropic"] = LLMProvider(
            name="Anthropic",
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
        )

        # Google
        self._providers["google"] = LLMProvider(
            name="Google",
            provider_type=ProviderType.GOOGLE,
            model="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            enabled=bool(os.getenv("GOOGLE_API_KEY")),
        )

        logger.info(f"Initialized {len(self._providers)} LLM providers")

    def get_available_providers(self) -> list[LLMProvider]:
        """Get list of available LLM providers."""
        return [provider for provider in self._providers.values() if provider.enabled]

    def get_provider(self, provider_name: str) -> LLMProvider | None:
        """Get a specific LLM provider by name."""
        return self._providers.get(provider_name)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Execute a prompt with the specified LLM provider.

        Args:
            request: Execution request with prompt and provider details

        Returns:
            ExecutionResult with the LLM response
        """
        start_time = time.time()

        try:
            # Get provider
            provider = self.get_provider(request.provider)
            if not provider:
                return ExecutionResult(
                    prompt=request.transformed_prompt,
                    response="",
                    provider=request.provider,
                    model=request.model or "unknown",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    error_message=f"Provider '{request.provider}' not found or not enabled",
                )

            # Use existing LLM integration if available
            if llm_integration_engine:
                result = self._execute_with_existing_integration(request, provider)
            else:
                result = self._execute_with_provider_client(request, provider)

            # Calculate execution time
            result.execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"LLM execution completed: {provider.name} "
                f"({request.provider}) in {result.execution_time_ms}ms"
            )

            return result

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_message = str(e)

            logger.error(f"LLM execution failed: {error_message}")

            return ExecutionResult(
                prompt=request.transformed_prompt,
                response="",
                provider=request.provider,
                model=request.model or "unknown",
                execution_time_ms=execution_time,
                success=False,
                error_message=error_message,
            )

    def _execute_with_existing_integration(
        self, request: ExecutionRequest, provider: LLMProvider
    ) -> ExecutionResult:
        """Execute using existing LLM integration engine."""
        try:
            # Create transformation request for existing system
            transform_request = {
                "provider": request.provider,
                "model": request.model or provider.model,
                "prompt": request.transformed_prompt,
                "max_tokens": request.max_tokens or provider.max_tokens,
                "temperature": request.temperature or provider.temperature,
                "metadata": request.metadata,
            }

            # Execute with existing engine
            response = llm_integration_engine.execute_transformation(transform_request)

            # Extract result data
            return ExecutionResult(
                prompt=request.transformed_prompt,
                response=response.get("response", ""),
                provider=request.provider,
                model=response.get("model", request.model or provider.model),
                tokens_used=response.get("tokens_used", 0),
                success=response.get("success", True),
                error_message=response.get("error"),
                metadata=response.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Existing integration execution failed: {e}")
            raise

    def _execute_with_provider_client(
        self, request: ExecutionRequest, provider: LLMProvider
    ) -> ExecutionResult:
        """Execute using provider client directly."""
        try:
            # This would involve direct API calls to the provider
            # For now, return a mock response
            logger.warning("Using mock LLM execution - provider client not implemented")

            return ExecutionResult(
                prompt=request.transformed_prompt,
                response=f"Mock response from {provider.name} for: {request.transformed_prompt[:100]}...",
                provider=request.provider,
                model=request.model or provider.model,
                tokens_used=100,  # Mock token count
                success=True,
                metadata={"mock_execution": True},
            )

        except Exception as e:
            logger.error(f"Provider client execution failed: {e}")
            raise

    def validate_provider_config(self, provider_name: str) -> dict[str, Any]:
        """Validate provider configuration."""
        provider = self.get_provider(provider_name)
        if not provider:
            return {"valid": False, "error": f"Provider {provider_name} not found"}

        issues = []

        # Check API key
        if not provider.api_key:
            issues.append("API key not configured")

        # Check if provider is enabled
        if not provider.enabled:
            issues.append("Provider is disabled")

        # Validate model
        if not provider.model:
            issues.append("Model not specified")

        return {"valid": len(issues) == 0, "issues": issues, "provider": provider.name}

    def estimate_cost(
        self, prompt: str, provider_name: str, model: str | None = None
    ) -> CostInfo | None:
        """
        Estimate execution cost for a prompt.

        Returns:
            CostInfo with estimated cost, or None if pricing not available
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return None

        # TODO: Implement actual cost estimation based on token counting
        # and provider pricing

        # Mock implementation
        mock_token_count = TokenCount(
            prompt_tokens=len(prompt.split()) * 2,  # Rough estimate
            completion_tokens=100,  # Estimate
            total_tokens=0,
        )
        mock_token_count.total_tokens = (
            mock_token_count.prompt_tokens + mock_token_count.completion_tokens
        )

        # Mock pricing (will vary by provider and model)
        prompt_cost_per_1k = 0.001  # $0.001 per 1k tokens
        completion_cost_per_1k = 0.002  # $0.002 per 1k tokens

        return CostInfo.from_tokens(mock_token_count, prompt_cost_per_1k, completion_cost_per_1k)

    def get_provider_stats(self) -> dict[str, Any]:
        """Get statistics about LLM providers."""
        enabled_providers = self.get_available_providers()

        return {
            "total_providers": len(self._providers),
            "enabled_providers": len(enabled_providers),
            "providers": [
                {
                    "name": p.name,
                    "type": p.provider_type.value,
                    "model": p.model,
                    "enabled": p.enabled,
                    "has_api_key": bool(p.api_key),
                }
                for p in self._providers.values()
            ],
        }


# Global service instance
llm_service = LLMService()


# Convenience functions
def execute_prompt(prompt: str, provider: str = "openai", **kwargs) -> ExecutionResult:
    """Convenience function for prompt execution."""
    request = ExecutionRequest(transformed_prompt=prompt, provider=provider, **kwargs)
    return llm_service.execute(request)
