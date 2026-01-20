"""Tests for LLMService failover functionality (Story 1.2).

Tests cover:
- Automatic provider failover on circuit breaker open
- Failover chain configuration
- Performance statistics with failover info
- Failover enable/disable
"""

from typing import Never
from unittest.mock import patch

import pytest

from app.core.circuit_breaker import CircuitBreakerOpen
from app.core.unified_errors import ProviderNotAvailableError
from app.domain.interfaces import LLMProvider
from app.domain.models import GenerationConfig, LLMProviderType, PromptRequest, PromptResponse
from app.services.llm_service import LLMService

# =============================================================================
# Fixtures
# =============================================================================


class MockAsyncProvider(LLMProvider):
    """Mock async provider for testing."""

    def __init__(self, name: str, should_fail: bool = False, fail_with_cb: bool = False) -> None:
        self.name = name
        self.should_fail = should_fail
        self.fail_with_cb = fail_with_cb
        self.call_count = 0

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.call_count += 1
        if self.fail_with_cb:
            raise CircuitBreakerOpen(self.name, retry_after=60.0)
        if self.should_fail:
            msg = f"Provider {self.name} failed"
            raise Exception(msg)
        return PromptResponse(
            text=f"Response from {self.name}",
            model_used="test-model",
            provider=self.name,
            usage_metadata={},
            finish_reason="STOP",
            latency_ms=100.0,
        )

    async def check_health(self) -> bool:
        return not self.should_fail


@pytest.fixture
def llm_service():
    """Create a fresh LLMService for testing."""
    return LLMService()


@pytest.fixture
def sample_request():
    """Sample prompt request for testing."""
    return PromptRequest(
        prompt="What is the capital of France?",
        model="test-model",
        config=GenerationConfig(temperature=0.7, max_output_tokens=100),
        provider=LLMProviderType.OPENAI,
    )


# =============================================================================
# Failover Chain Tests
# =============================================================================


class TestFailoverChain:
    """Tests for failover chain configuration."""

    def test_default_failover_chains_defined(self, llm_service) -> None:
        """Default failover chains are defined for all major providers."""
        chains = llm_service._DEFAULT_FAILOVER_CHAIN

        assert "gemini" in chains
        assert "openai" in chains
        assert "anthropic" in chains
        assert "deepseek" in chains

        # Each chain has alternative providers
        assert len(chains["gemini"]) >= 2
        assert len(chains["openai"]) >= 2

    def test_set_custom_failover_chain(self, llm_service) -> None:
        """Custom failover chains can be set."""
        custom_chain = ["provider1", "provider2"]
        llm_service.set_failover_chain("custom", custom_chain)

        assert llm_service._failover_chains["custom"] == custom_chain

    def test_get_failover_providers_filters_unregistered(self, llm_service) -> None:
        """Failover chain filters to only registered providers."""
        # Register only one provider
        provider = MockAsyncProvider("openai")
        llm_service.register_provider("openai", provider)

        # Get failover for gemini (which lists openai)
        failovers = llm_service._get_failover_providers("gemini")

        # Should only include registered provider
        assert "openai" in failovers
        assert "anthropic" not in failovers  # Not registered


# =============================================================================
# Failover Execution Tests
# =============================================================================


class TestFailoverExecution:
    """Tests for failover execution during generation."""

    @pytest.mark.asyncio
    async def test_primary_success_no_failover(self, llm_service, sample_request) -> None:
        """Successful primary provider doesn't trigger failover."""
        primary = MockAsyncProvider("openai")
        backup = MockAsyncProvider("anthropic")

        llm_service.register_provider("openai", primary, is_default=True)
        llm_service.register_provider("anthropic", backup)

        # Mock circuit breaker to not interfere
        with patch.object(llm_service, "_call_with_circuit_breaker") as mock_cb:
            mock_cb.return_value = PromptResponse(
                text="Success from openai",
                model_used="test-model",
                provider="openai",
                usage_metadata={},
                finish_reason="STOP",
                latency_ms=100.0,
            )

            response = await llm_service.generate_text(sample_request)

            assert "openai" in response.provider.lower() or "success" in response.text.lower()
            assert backup.call_count == 0

    @pytest.mark.asyncio
    async def test_failover_on_circuit_breaker_open(self, llm_service, sample_request) -> None:
        """Circuit breaker open triggers failover to backup."""
        primary = MockAsyncProvider("openai", fail_with_cb=True)
        backup = MockAsyncProvider("anthropic")

        llm_service.register_provider("openai", primary, is_default=True)
        llm_service.register_provider("anthropic", backup)

        # Set up the failover chain
        llm_service.set_failover_chain("openai", ["anthropic"])

        # Mock to simulate CB open then successful backup
        call_count = [0]

        async def mock_cb_call(provider_name, func, *args, **kwargs):
            call_count[0] += 1
            if provider_name == "openai":
                msg = "openai"
                raise CircuitBreakerOpen(msg, retry_after=60.0)
            return await func(*args, **kwargs)

        with patch.object(llm_service, "_call_with_circuit_breaker", side_effect=mock_cb_call):
            response = await llm_service.generate_text(sample_request)

            assert "anthropic" in response.provider.lower()
            assert call_count[0] >= 2  # Primary + backup

    @pytest.mark.asyncio
    async def test_failover_all_providers_fail(self, llm_service, sample_request) -> None:
        """All providers failing raises ProviderNotAvailableError."""
        primary = MockAsyncProvider("openai", fail_with_cb=True)
        backup = MockAsyncProvider("anthropic", fail_with_cb=True)

        llm_service.register_provider("openai", primary, is_default=True)
        llm_service.register_provider("anthropic", backup)
        llm_service.set_failover_chain("openai", ["anthropic"])

        async def mock_cb_always_fail(provider_name, func, *args, **kwargs) -> Never:
            raise CircuitBreakerOpen(provider_name, retry_after=60.0)

        with patch.object(
            llm_service,
            "_call_with_circuit_breaker",
            side_effect=mock_cb_always_fail,
        ):
            with pytest.raises(ProviderNotAvailableError) as exc_info:
                await llm_service.generate_text(sample_request)

            assert "failover" in exc_info.value.message.lower()
            assert "tried_providers" in exc_info.value.details

    @pytest.mark.asyncio
    async def test_failover_disabled(self, llm_service, sample_request) -> None:
        """Failover disabled doesn't try backup providers."""
        primary = MockAsyncProvider("openai", fail_with_cb=True)
        backup = MockAsyncProvider("anthropic")

        llm_service.register_provider("openai", primary, is_default=True)
        llm_service.register_provider("anthropic", backup)
        llm_service.set_failover_chain("openai", ["anthropic"])
        llm_service.enable_failover(False)

        async def mock_cb_primary_fail(provider_name, func, *args, **kwargs):
            if provider_name == "openai":
                msg = "openai"
                raise CircuitBreakerOpen(msg, retry_after=60.0)
            return await func(*args, **kwargs)

        with patch.object(
            llm_service,
            "_call_with_circuit_breaker",
            side_effect=mock_cb_primary_fail,
        ):
            with pytest.raises(ProviderNotAvailableError):
                await llm_service.generate_text(sample_request)

            # Backup should not have been called
            assert backup.call_count == 0


# =============================================================================
# Performance Stats Tests
# =============================================================================


class TestPerformanceStats:
    """Tests for performance statistics with failover info."""

    def test_stats_include_failover_config(self, llm_service) -> None:
        """Performance stats include failover configuration."""
        llm_service.register_provider("openai", MockAsyncProvider("openai"))
        llm_service.register_provider("anthropic", MockAsyncProvider("anthropic"))

        stats = llm_service.get_performance_stats()

        assert "failover_enabled" in stats
        assert "failover_chains" in stats
        assert isinstance(stats["failover_chains"], dict)

    def test_enable_failover_updates_state(self, llm_service) -> None:
        """Enable/disable failover updates internal state."""
        assert llm_service._FAILOVER_ENABLED is True

        llm_service.enable_failover(False)
        assert llm_service._FAILOVER_ENABLED is False

        llm_service.enable_failover(True)
        assert llm_service._FAILOVER_ENABLED is True


# =============================================================================
# Fallback Request Tests
# =============================================================================


class TestFallbackRequest:
    """Tests for fallback request creation."""

    def test_create_fallback_request(self, llm_service, sample_request) -> None:
        """Fallback request is created correctly."""
        fallback = llm_service._create_fallback_request(sample_request, "anthropic")

        # Should preserve prompt
        assert fallback.prompt == sample_request.prompt
        # Model should be None (use default)
        assert fallback.model is None
        # Config should be preserved
        assert fallback.config == sample_request.config

    def test_fallback_request_strips_api_key(self, llm_service) -> None:
        """Fallback request doesn't use original API key."""
        original = PromptRequest(
            prompt="test",
            api_key="original-key",
        )

        fallback = llm_service._create_fallback_request(original, "anthropic")

        assert fallback.api_key is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestFailoverIntegration:
    """Integration tests for failover with LLMService."""

    @pytest.mark.asyncio
    async def test_failover_chain_priority_order(self, llm_service, sample_request) -> None:
        """Failover tries providers in chain order."""
        providers = [
            MockAsyncProvider("openai", fail_with_cb=True),
            MockAsyncProvider("anthropic", fail_with_cb=True),
            MockAsyncProvider("gemini", fail_with_cb=False),  # This succeeds
        ]

        llm_service.register_provider("openai", providers[0], is_default=True)
        llm_service.register_provider("anthropic", providers[1])
        llm_service.register_provider("gemini", providers[2])
        llm_service.set_failover_chain("openai", ["anthropic", "gemini"])

        call_order = []

        async def mock_cb_track(provider_name, func, *args, **kwargs):
            call_order.append(provider_name)
            if provider_name in ["openai", "anthropic"]:
                raise CircuitBreakerOpen(provider_name, retry_after=60.0)
            return await func(*args, **kwargs)

        with patch.object(llm_service, "_call_with_circuit_breaker", side_effect=mock_cb_track):
            response = await llm_service.generate_text(sample_request)

            # Should have tried in order
            assert call_order == ["openai", "anthropic", "gemini"]
            assert "gemini" in response.provider.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
