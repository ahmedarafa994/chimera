"""
Tests for Story 1.2: Direct API Integration

Tests cover:
- LLMProvider interface compliance
- Provider client implementations (Qwen, Cursor, etc.)
- Retry logic with exponential backoff
- Provider failover mechanism
- Rate limit tracking
- Streaming support
- Token counting
"""

from unittest.mock import MagicMock, patch

import pytest

from app.domain.interfaces import LLMProvider
from app.domain.models import (
    GenerationConfig,
    PromptRequest,
    PromptResponse,
)
from app.infrastructure.cursor_client import CursorClient
from app.infrastructure.provider_manager import (
    FailoverConfig,
    ProviderManager,
    ProviderState,
    ProviderStatus,
)
from app.infrastructure.qwen_client import QwenClient
from app.infrastructure.retry_handler import (
    BackoffStrategy,
    RetryableError,
    RetryConfig,
    RetryExhaustedError,
    RetryHandler,
    get_provider_retry_config,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.QWEN_API_KEY = "test-qwen-key"
    settings.CURSOR_API_KEY = "test-cursor-key"
    settings.get_provider_endpoint.return_value = None
    return settings


@pytest.fixture
def sample_request():
    """Sample prompt request for testing."""
    return PromptRequest(
        prompt="What is the capital of France?",
        model="test-model",
        config=GenerationConfig(
            temperature=0.7,
            max_output_tokens=100,
        ),
    )


@pytest.fixture
def sample_response():
    """Sample prompt response for testing."""
    return PromptResponse(
        text="The capital of France is Paris.",
        model_used="test-model",
        provider="test",
        usage_metadata={
            "prompt_token_count": 10,
            "candidates_token_count": 8,
            "total_token_count": 18,
        },
        finish_reason="STOP",
        latency_ms=150.0,
    )


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        self.call_count = 0

    async def generate(self, request: PromptRequest) -> PromptResponse:
        self.call_count += 1
        if self.should_fail:
            raise Exception(f"Provider {self.name} failed")
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


# =============================================================================
# LLMProvider Interface Tests
# =============================================================================


class TestLLMProviderInterface:
    """Tests for LLMProvider interface compliance."""

    def test_provider_interface_has_required_methods(self):
        """Verify LLMProvider has all required abstract methods."""
        assert hasattr(LLMProvider, "generate")
        assert hasattr(LLMProvider, "check_health")
        assert hasattr(LLMProvider, "generate_stream")
        assert hasattr(LLMProvider, "count_tokens")

    def test_mock_provider_implements_interface(self):
        """Mock provider implements LLMProvider correctly."""
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_mock_provider_generate(self, sample_request):
        """Mock provider generate works correctly."""
        provider = MockProvider(name="test")
        response = await provider.generate(sample_request)

        assert response.text == "Response from test"
        assert response.provider == "test"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_provider_health_check(self):
        """Mock provider health check works correctly."""
        healthy_provider = MockProvider(should_fail=False)
        unhealthy_provider = MockProvider(should_fail=True)

        assert await healthy_provider.check_health() is True
        assert await unhealthy_provider.check_health() is False


# =============================================================================
# Retry Handler Tests
# =============================================================================


class TestRetryHandler:
    """Tests for retry logic with exponential backoff."""

    def test_retry_config_defaults(self):
        """RetryConfig has sensible defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.strategy == BackoffStrategy.EXPONENTIAL

    def test_provider_specific_configs(self):
        """Provider-specific retry configs are defined."""
        openai_config = get_provider_retry_config("openai")
        google_config = get_provider_retry_config("google")

        # OpenAI should have more retries
        assert openai_config.max_retries == 5
        assert google_config.max_retries == 3

    def test_calculate_exponential_delay(self):
        """Exponential backoff calculates delays correctly."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        handler = RetryHandler(config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 2.0
        assert handler.calculate_delay(2) == 4.0
        assert handler.calculate_delay(3) == 8.0

    def test_calculate_delay_with_max_cap(self):
        """Delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_multiplier=10.0,
            max_delay=5.0,
            jitter=False,
        )
        handler = RetryHandler(config)

        # 10.0 should be capped to 5.0
        assert handler.calculate_delay(1) == 5.0
        # 100.0 should also be capped to 5.0
        assert handler.calculate_delay(2) == 5.0

    def test_calculate_linear_delay(self):
        """Linear backoff calculates delays correctly."""
        config = RetryConfig(
            initial_delay=2.0,
            strategy=BackoffStrategy.LINEAR,
            jitter=False,
        )
        handler = RetryHandler(config)

        assert handler.calculate_delay(0) == 2.0
        assert handler.calculate_delay(1) == 4.0
        assert handler.calculate_delay(2) == 6.0

    def test_is_retryable_error(self):
        """Retryable errors are identified correctly."""
        handler = RetryHandler(RetryConfig())

        # Retryable errors
        assert handler.is_retryable_error(RetryableError("test"))
        assert handler.is_retryable_error(TimeoutError())
        assert handler.is_retryable_error(ConnectionError())
        assert handler.is_retryable_error(TimeoutError())

        # Rate limit errors (in message)
        assert handler.is_retryable_error(Exception("rate limit exceeded"))
        assert handler.is_retryable_error(Exception("Error 429: Too many requests"))

        # Non-retryable errors
        assert not handler.is_retryable_error(ValueError("invalid input"))

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self):
        """Successful execution doesn't retry."""
        handler = RetryHandler(RetryConfig(max_retries=3))

        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await handler.execute_with_retry(success_func)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_eventual_success(self):
        """Retry succeeds after initial failures."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.01,  # Fast for testing
            jitter=False,
        )
        handler = RetryHandler(config)

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await handler.execute_with_retry(flaky_func)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_exhausted(self):
        """RetryExhaustedError raised when all retries fail."""
        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        async def always_fail():
            raise ConnectionError("Persistent failure")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await handler.execute_with_retry(always_fail)

        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_error, ConnectionError)

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        """Non-retryable errors are not retried."""
        handler = RetryHandler(RetryConfig(max_retries=3))

        call_count = 0

        async def fail_once():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            await handler.execute_with_retry(fail_once)

        # Should only be called once
        assert call_count == 1


# =============================================================================
# Provider Manager Tests
# =============================================================================


class TestProviderManager:
    """Tests for provider manager and failover."""

    def test_register_provider(self):
        """Provider registration works correctly."""
        manager = ProviderManager()
        provider = MockProvider("test")

        manager.register_provider("test", provider, priority=10, is_default=True)

        assert manager.get_provider("test") == provider
        assert manager._default_provider == "test"
        assert "test" in manager._provider_states

    def test_provider_priority_ordering(self):
        """Providers are ordered by priority."""
        manager = ProviderManager()

        manager.register_provider("low", MockProvider("low"), priority=1)
        manager.register_provider("high", MockProvider("high"), priority=100)
        manager.register_provider("med", MockProvider("med"), priority=50)

        # Higher priority first
        assert manager._provider_order[0] == "high"
        assert manager._provider_order[1] == "med"
        assert manager._provider_order[2] == "low"

    def test_get_available_providers(self):
        """Available providers excludes unhealthy ones."""
        manager = ProviderManager()

        manager.register_provider("healthy", MockProvider("healthy"), priority=10)
        manager.register_provider("unhealthy", MockProvider("unhealthy"), priority=20)

        # Mark one as unhealthy
        manager._provider_states["unhealthy"].status = ProviderStatus.UNHEALTHY
        manager._provider_states["healthy"].status = ProviderStatus.HEALTHY

        available = manager.get_available_providers()

        assert "healthy" in available
        assert "unhealthy" not in available

    @pytest.mark.asyncio
    async def test_generate_with_failover_primary_success(self, sample_request):
        """Primary provider success doesn't trigger failover."""
        manager = ProviderManager()

        primary = MockProvider("primary")
        backup = MockProvider("backup")

        manager.register_provider("primary", primary, priority=100, is_default=True)
        manager.register_provider("backup", backup, priority=50)

        # Mark both as healthy
        manager._provider_states["primary"].status = ProviderStatus.HEALTHY
        manager._provider_states["backup"].status = ProviderStatus.HEALTHY

        response = await manager.generate_with_failover(sample_request)

        assert "primary" in response.text
        assert primary.call_count == 1
        assert backup.call_count == 0

    @pytest.mark.asyncio
    async def test_generate_with_failover_to_backup(self, sample_request):
        """Failed primary triggers failover to backup."""
        config = FailoverConfig(
            enabled=True,
            failover_on_error=True,
        )
        manager = ProviderManager(config)

        primary = MockProvider("primary", should_fail=True)
        backup = MockProvider("backup", should_fail=False)

        manager.register_provider("primary", primary, priority=100, is_default=True)
        manager.register_provider("backup", backup, priority=50)

        # Mark both as healthy initially
        manager._provider_states["primary"].status = ProviderStatus.HEALTHY
        manager._provider_states["backup"].status = ProviderStatus.HEALTHY

        # Use shorter retry config for tests
        manager._retry_handlers["primary"] = RetryHandler(
            RetryConfig(max_retries=0)  # No retries for faster test
        )

        response = await manager.generate_with_failover(sample_request)

        assert "backup" in response.text
        assert backup.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_failover_all_fail(self, sample_request):
        """All providers failing raises RetryExhaustedError."""
        config = FailoverConfig(
            enabled=True,
            max_failover_attempts=2,
        )
        manager = ProviderManager(config)

        provider1 = MockProvider("p1", should_fail=True)
        provider2 = MockProvider("p2", should_fail=True)

        manager.register_provider("p1", provider1, priority=100)
        manager.register_provider("p2", provider2, priority=50)

        manager._provider_states["p1"].status = ProviderStatus.HEALTHY
        manager._provider_states["p2"].status = ProviderStatus.HEALTHY

        # No retries for faster tests
        for name in ["p1", "p2"]:
            manager._retry_handlers[name] = RetryHandler(
                RetryConfig(max_retries=0)
            )

        with pytest.raises(RetryExhaustedError):
            await manager.generate_with_failover(sample_request)

    @pytest.mark.asyncio
    async def test_health_check_updates_status(self):
        """Health check updates provider status."""
        manager = ProviderManager()

        healthy = MockProvider("healthy", should_fail=False)
        unhealthy = MockProvider("unhealthy", should_fail=True)

        manager.register_provider("healthy", healthy)
        manager.register_provider("unhealthy", unhealthy)

        status1 = await manager.check_provider_health("healthy")
        status2 = await manager.check_provider_health("unhealthy")

        assert status1 == ProviderStatus.HEALTHY
        assert status2 == ProviderStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_record_success_updates_stats(self):
        """Recording success updates provider statistics."""
        manager = ProviderManager()
        manager.register_provider("test", MockProvider("test"))

        await manager._record_success("test", 150.0)

        state = manager._provider_states["test"]
        assert state.total_requests == 1
        assert state.consecutive_failures == 0
        assert state.average_latency_ms > 0

    @pytest.mark.asyncio
    async def test_record_failure_updates_stats(self):
        """Recording failure updates provider statistics."""
        manager = ProviderManager()
        manager.register_provider("test", MockProvider("test"))

        await manager._record_failure("test", Exception("test"))

        state = manager._provider_states["test"]
        assert state.total_requests == 1
        assert state.total_failures == 1
        assert state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_rate_limit_tracking(self):
        """Rate limited providers are tracked correctly."""
        manager = ProviderManager()
        manager.register_provider("test", MockProvider("test"))

        await manager._record_failure(
            "test",
            Exception("rate limit exceeded"),
            is_rate_limit=True,
        )

        state = manager._provider_states["test"]
        assert state.status == ProviderStatus.RATE_LIMITED
        assert state.rate_limit_reset is not None


# =============================================================================
# Qwen Client Tests
# =============================================================================


class TestQwenClient:
    """Tests for Qwen provider client."""

    def test_client_initialization_without_key(self, mock_settings):
        """Client initializes without API key (disabled)."""
        mock_settings.QWEN_API_KEY = None

        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            client = QwenClient(mock_settings)
            assert client.client is None

    def test_client_initialization_with_key(self, mock_settings):
        """Client initializes with API key."""
        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.qwen_client.AsyncOpenAI") as mock_openai:
                QwenClient(mock_settings)
                mock_openai.assert_called_once()

    def test_get_model_name_default(self, mock_settings):
        """Default model is used when not specified."""
        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.qwen_client.AsyncOpenAI"):
                client = QwenClient(mock_settings)

                request = PromptRequest(prompt="test")
                model = client._get_model_name(request)

                assert model == QwenClient.DEFAULT_MODEL

    def test_get_model_name_from_request(self, mock_settings):
        """Model from request is used when specified."""
        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.qwen_client.AsyncOpenAI"):
                client = QwenClient(mock_settings)

                request = PromptRequest(prompt="test", model="qwen-max")
                model = client._get_model_name(request)

                assert model == "qwen-max"

    def test_validate_api_key(self, mock_settings):
        """API key validation works correctly."""
        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.qwen_client.AsyncOpenAI"):
                client = QwenClient(mock_settings)

                assert client._validate_api_key("sk-valid-key-12345") is True
                assert client._validate_api_key("short") is False
                assert client._validate_api_key("") is False
                assert client._validate_api_key(None) is False

    @pytest.mark.asyncio
    async def test_qwen_implements_interface(self, mock_settings):
        """QwenClient implements LLMProvider interface."""
        with patch("app.infrastructure.qwen_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.qwen_client.AsyncOpenAI"):
                client = QwenClient(mock_settings)

                assert isinstance(client, LLMProvider)
                assert hasattr(client, "generate")
                assert hasattr(client, "check_health")
                assert hasattr(client, "generate_stream")
                assert hasattr(client, "count_tokens")


# =============================================================================
# Cursor Client Tests
# =============================================================================


class TestCursorClient:
    """Tests for Cursor provider client."""

    def test_client_initialization_without_key(self, mock_settings):
        """Client initializes without API key (disabled)."""
        mock_settings.CURSOR_API_KEY = None

        with patch("app.infrastructure.cursor_client.get_settings", return_value=mock_settings):
            client = CursorClient(mock_settings)
            assert client.client is None

    def test_client_initialization_with_key(self, mock_settings):
        """Client initializes with API key."""
        with patch("app.infrastructure.cursor_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.cursor_client.AsyncOpenAI") as mock_openai:
                CursorClient(mock_settings)
                mock_openai.assert_called_once()

    def test_get_model_name_default(self, mock_settings):
        """Default model is used when not specified."""
        with patch("app.infrastructure.cursor_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.cursor_client.AsyncOpenAI"):
                client = CursorClient(mock_settings)

                request = PromptRequest(prompt="test")
                model = client._get_model_name(request)

                assert model == CursorClient.DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_cursor_implements_interface(self, mock_settings):
        """CursorClient implements LLMProvider interface."""
        with patch("app.infrastructure.cursor_client.get_settings", return_value=mock_settings):
            with patch("app.infrastructure.cursor_client.AsyncOpenAI"):
                client = CursorClient(mock_settings)

                assert isinstance(client, LLMProvider)
                assert hasattr(client, "generate")
                assert hasattr(client, "check_health")
                assert hasattr(client, "generate_stream")
                assert hasattr(client, "count_tokens")


# =============================================================================
# Provider State Tests
# =============================================================================


class TestProviderState:
    """Tests for ProviderState data class."""

    def test_default_state(self):
        """Provider state has correct defaults."""
        state = ProviderState(name="test")

        assert state.name == "test"
        assert state.status == ProviderStatus.UNKNOWN
        assert state.priority == 0
        assert state.consecutive_failures == 0
        assert state.total_requests == 0

    def test_provider_status_enum(self):
        """ProviderStatus enum has all expected values."""
        assert ProviderStatus.HEALTHY.value == "healthy"
        assert ProviderStatus.DEGRADED.value == "degraded"
        assert ProviderStatus.UNHEALTHY.value == "unhealthy"
        assert ProviderStatus.RATE_LIMITED.value == "rate_limited"
        assert ProviderStatus.UNKNOWN.value == "unknown"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDirectAPIIntegration:
    """Integration tests for direct API communication."""

    @pytest.mark.asyncio
    async def test_full_failover_chain(self, sample_request):
        """Test complete failover chain from primary to backup."""
        manager = ProviderManager(
            FailoverConfig(
                enabled=True,
                max_failover_attempts=3,
            )
        )

        # Set up providers with different behaviors
        providers = [
            ("p1", MockProvider("p1", should_fail=True), 100),
            ("p2", MockProvider("p2", should_fail=True), 90),
            ("p3", MockProvider("p3", should_fail=False), 80),
        ]

        for name, provider, priority in providers:
            manager.register_provider(name, provider, priority=priority)
            manager._provider_states[name].status = ProviderStatus.HEALTHY
            manager._retry_handlers[name] = RetryHandler(
                RetryConfig(max_retries=0)
            )

        response = await manager.generate_with_failover(sample_request)

        # Should have tried p1, p2, then succeeded with p3
        assert "p3" in response.text
        assert providers[0][1].call_count == 1  # p1 tried once
        assert providers[1][1].call_count == 1  # p2 tried once
        assert providers[2][1].call_count == 1  # p3 succeeded

    @pytest.mark.asyncio
    async def test_retry_then_failover(self, sample_request):
        """Test retry exhaustion followed by failover."""
        manager = ProviderManager(
            FailoverConfig(enabled=True)
        )

        primary = MockProvider("primary", should_fail=True)
        backup = MockProvider("backup", should_fail=False)

        manager.register_provider("primary", primary, priority=100)
        manager.register_provider("backup", backup, priority=50)

        manager._provider_states["primary"].status = ProviderStatus.HEALTHY
        manager._provider_states["backup"].status = ProviderStatus.HEALTHY

        # Allow 1 retry for primary
        manager._retry_handlers["primary"] = RetryHandler(
            RetryConfig(max_retries=1, initial_delay=0.01, jitter=False)
        )

        response = await manager.generate_with_failover(sample_request)

        # Primary should have been called twice (initial + 1 retry)
        assert primary.call_count == 2
        # Then failover to backup
        assert "backup" in response.text

    def test_manager_stats(self):
        """Provider manager returns comprehensive stats."""
        manager = ProviderManager()

        manager.register_provider("p1", MockProvider("p1"), priority=100)
        manager.register_provider("p2", MockProvider("p2"), priority=50)

        stats = manager.get_stats()

        assert "providers" in stats
        assert "p1" in stats["providers"]
        assert "p2" in stats["providers"]
        assert "provider_order" in stats
        assert "failover_enabled" in stats


# =============================================================================
# Backoff Strategy Tests
# =============================================================================


class TestBackoffStrategies:
    """Tests for different backoff strategies."""

    def test_constant_backoff(self):
        """Constant backoff returns same delay."""
        config = RetryConfig(
            initial_delay=5.0,
            strategy=BackoffStrategy.CONSTANT,
            jitter=False,
        )
        handler = RetryHandler(config)

        assert handler.calculate_delay(0) == 5.0
        assert handler.calculate_delay(1) == 5.0
        assert handler.calculate_delay(2) == 5.0

    def test_fibonacci_backoff(self):
        """Fibonacci backoff follows Fibonacci sequence."""
        config = RetryConfig(
            initial_delay=1.0,
            strategy=BackoffStrategy.FIBONACCI,
            jitter=False,
            max_delay=100.0,
        )
        handler = RetryHandler(config)

        assert handler.calculate_delay(0) == 1.0
        assert handler.calculate_delay(1) == 1.0
        assert handler.calculate_delay(2) == 2.0
        assert handler.calculate_delay(3) == 3.0
        assert handler.calculate_delay(4) == 5.0

    def test_jitter_adds_variance(self):
        """Jitter adds variance to delays."""
        config = RetryConfig(
            initial_delay=10.0,
            jitter=True,
            jitter_factor=0.25,
        )
        handler = RetryHandler(config)

        delays = [handler.calculate_delay(0) for _ in range(10)]

        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # All delays should be within jitter range
        assert all(7.5 <= d <= 12.5 for d in delays)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
