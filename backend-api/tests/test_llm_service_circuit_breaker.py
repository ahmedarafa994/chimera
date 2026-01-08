"""
Tests for LLM Service CircuitBreakerOpen exception handling.

These tests verify that the LLM service properly handles circuit breaker
exceptions and provides meaningful error messages to users.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMServiceCircuitBreakerHandling:
    """Test suite for CircuitBreakerOpen exception handling in LLM service."""

    @pytest.fixture
    def llm_service(self):
        """Create a fresh LLM service instance for testing."""
        from app.services.llm_service import LLMService

        return LLMService()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.generate = AsyncMock()
        return provider

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_converts_to_provider_not_available(
        self, llm_service, mock_provider
    ):
        """Test that CircuitBreakerOpen is converted to ProviderNotAvailableError."""
        from app.core.circuit_breaker import CircuitBreakerOpen
        from app.core.unified_errors import ProviderNotAvailableError
        from app.domain.models import LLMProviderType, PromptRequest

        # Register the mock provider
        llm_service.register_provider("test_provider", mock_provider, is_default=True)

        # Make the circuit breaker wrapper raise CircuitBreakerOpen
        with patch.object(
            llm_service,
            "_call_with_circuit_breaker",
            side_effect=CircuitBreakerOpen("test_provider", 30.0),
        ):
            request = PromptRequest(prompt="Test prompt", provider=LLMProviderType("test_provider"))

            with pytest.raises(ProviderNotAvailableError) as exc_info:
                await llm_service.generate_text(request)

            # Verify error contains retry information
            error = exc_info.value
            assert "test_provider" in str(error)
            assert error.details is not None
            assert "retry_after_seconds" in error.details
            assert error.details["circuit_state"] == "open"

    @pytest.mark.asyncio
    async def test_generate_method_handles_circuit_breaker_open(self, llm_service, mock_provider):
        """Test that the generate() method also handles CircuitBreakerOpen."""
        from app.core.circuit_breaker import CircuitBreakerOpen
        from app.core.unified_errors import ProviderNotAvailableError

        # Register the mock provider
        llm_service.register_provider("gemini", mock_provider, is_default=True)

        # Make the provider raise CircuitBreakerOpen
        mock_provider.generate.side_effect = CircuitBreakerOpen("gemini", 60.0)

        with pytest.raises(ProviderNotAvailableError) as exc_info:
            await llm_service.generate(prompt="Test prompt", provider="gemini")

        error = exc_info.value
        assert "gemini" in str(error)

    @pytest.mark.asyncio
    async def test_circuit_breaker_wrapper_created_on_registration(
        self, llm_service, mock_provider
    ):
        """Test that circuit breaker wrapper is created when provider is registered."""
        # Initially no wrappers
        assert len(llm_service._circuit_breaker_cache) == 0

        # Register provider
        llm_service.register_provider("new_provider", mock_provider)

        # Wrapper should be created
        assert "new_provider" in llm_service._circuit_breaker_cache

    @pytest.mark.asyncio
    async def test_performance_stats_include_circuit_breakers(self, llm_service, mock_provider):
        """Test that performance stats include circuit breaker information."""
        llm_service.register_provider("stats_provider", mock_provider)

        stats = llm_service.get_performance_stats()

        assert "circuit_breakers" in stats
        assert "stats_provider" in stats["circuit_breakers"]


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        from app.core.circuit_breaker import (
            CircuitBreakerRegistry,
            CircuitState,
        )

        # Get a circuit breaker with low threshold for testing
        breaker = CircuitBreakerRegistry.get(
            "test_integration", failure_threshold=2, recovery_timeout=60.0
        )

        # Reset to known state
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

        # Record failures
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Verify can_execute returns False when open
        assert not breaker.can_execute()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test that circuit breaker recovers after timeout."""
        import time

        from app.core.circuit_breaker import CircuitBreakerRegistry, CircuitState

        # Get a circuit breaker with very short recovery timeout
        breaker = CircuitBreakerRegistry.get(
            "test_recovery",
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms
        )

        # Reset and trip the breaker
        breaker.reset()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open on next check
        assert breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_success(self):
        """Test that circuit breaker closes after successful calls in half-open."""
        import time

        from app.core.circuit_breaker import CircuitBreakerRegistry, CircuitState

        breaker = CircuitBreakerRegistry.get(
            "test_close", failure_threshold=1, recovery_timeout=0.1, half_open_max_calls=3
        )

        # Reset and trip the breaker
        breaker.reset()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)
        breaker.can_execute()  # Transition to half-open

        # Record successes (need success_threshold successes)
        breaker.record_success()
        breaker.record_success()

        # Should be closed now
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics collection."""

    def test_circuit_breaker_status_includes_metrics(self):
        """Test that circuit breaker status includes metrics."""
        from app.core.circuit_breaker import CircuitBreakerRegistry

        breaker = CircuitBreakerRegistry.get("test_metrics")
        breaker.reset()

        # Record some activity
        breaker.record_success()
        breaker.record_success()
        breaker.record_failure()

        status = breaker.get_status()

        assert "metrics" in status
        metrics = status["metrics"]
        assert metrics["total_calls"] == 3
        assert metrics["successful_calls"] == 2
        assert metrics["failed_calls"] == 1

    def test_circuit_breaker_registry_get_all_status(self):
        """Test getting status of all circuit breakers."""
        from app.core.circuit_breaker import CircuitBreakerRegistry

        # Create a few breakers
        CircuitBreakerRegistry.get("breaker_a")
        CircuitBreakerRegistry.get("breaker_b")

        all_status = CircuitBreakerRegistry.get_all_status()

        assert "breaker_a" in all_status
        assert "breaker_b" in all_status
        assert "state" in all_status["breaker_a"]
