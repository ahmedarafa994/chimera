"""
Tests for Proxy Mode Integration (Story 1.3).

Tests cover:
- ProxyClient: HTTP communication, connection pooling, retries
- ProxyProviderAdapter: LLM provider protocol, fallback logic
- ProxyHealthMonitor: Health checks, metrics tracking
- API endpoints: /health/proxy endpoints
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.models import GenerationConfig, PromptRequest, PromptResponse

# =============================================================================
# ProxyClient Tests
# =============================================================================


class TestProxyClient:
    """Tests for the ProxyClient class."""

    @pytest.fixture
    def proxy_client(self):
        """Create a ProxyClient instance for testing."""
        from app.infrastructure.proxy.proxy_client import ProxyClient

        return ProxyClient(
            endpoint="http://localhost:8080",
            timeout=10.0,
            max_retries=3,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, proxy_client):
        """Test ProxyClient initializes correctly."""
        assert proxy_client.endpoint == "http://localhost:8080"
        assert proxy_client._timeout == 10.0
        assert proxy_client._max_retries == 3
        assert not proxy_client.is_healthy

    @pytest.mark.asyncio
    async def test_get_stats(self, proxy_client):
        """Test stats retrieval."""
        stats = proxy_client.get_stats()

        assert "endpoint" in stats
        assert "timeout" in stats
        assert "request_count" in stats
        assert "error_count" in stats
        assert "is_healthy" in stats
        assert stats["request_count"] == 0

    @pytest.mark.asyncio
    async def test_send_request_success(self, proxy_client):
        """Test successful request through proxy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "Generated response",
            "model": "gpt-4",
            "usage": {"total_tokens": 100},
        }

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=AsyncMock(
                post=AsyncMock(return_value=mock_response)
            ),
        ):
            result = await proxy_client.send_request(
                provider="openai",
                model="gpt-4",
                prompt="Hello, world!",
            )

            assert result["text"] == "Generated response"
            assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_send_request_connection_error(self, proxy_client):
        """Test handling of connection errors."""
        from app.infrastructure.proxy.proxy_client import ProxyConnectionError

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=mock_client,
        ), pytest.raises(ProxyConnectionError):
            await proxy_client.send_request(
                provider="openai",
                model="gpt-4",
                prompt="Test",
            )

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, proxy_client):
        """Test handling of timeout errors."""
        from app.infrastructure.proxy.proxy_client import ProxyTimeoutError

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=mock_client,
        ), pytest.raises(ProxyTimeoutError):
            await proxy_client.send_request(
                provider="openai",
                model="gpt-4",
                prompt="Test",
            )

    @pytest.mark.asyncio
    async def test_health_check_success(self, proxy_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=mock_client,
        ):
            status = await proxy_client.check_health()

            assert status.is_healthy
            assert status.error is None
            assert status.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self, proxy_client):
        """Test failed health check."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.is_closed = False

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=mock_client,
        ):
            status = await proxy_client.check_health()

            assert not status.is_healthy
            assert status.error is not None
            assert status.consecutive_failures >= 1

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, proxy_client):
        """Test retry logic on 5xx errors."""
        error_response = MagicMock()
        error_response.status_code = 503
        error_response.json.return_value = {"error": "Service unavailable"}
        error_response.text = "Service unavailable"

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"text": "Success"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[error_response, error_response, success_response]
        )

        with patch.object(
            proxy_client,
            "_get_client",
            return_value=mock_client,
        ), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await proxy_client.send_request(
                provider="openai",
                model="gpt-4",
                prompt="Test",
            )

            assert result["text"] == "Success"

    @pytest.mark.asyncio
    async def test_close_client(self, proxy_client):
        """Test closing the client."""
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        proxy_client._client = mock_client

        await proxy_client.close()

        mock_client.aclose.assert_called_once()
        assert proxy_client._client is None


# =============================================================================
# ProxyProviderAdapter Tests
# =============================================================================


class TestProxyProviderAdapter:
    """Tests for the ProxyProviderAdapter class."""

    @pytest.fixture
    def mock_proxy_client(self):
        """Create a mock ProxyClient."""
        client = MagicMock()
        client.is_healthy = True
        client.check_health = AsyncMock(
            return_value=MagicMock(is_healthy=True)
        )
        client.send_request = AsyncMock(
            return_value={
                "text": "Generated text",
                "model": "gpt-4",
                "usage": {"total_tokens": 50},
            }
        )
        return client

    @pytest.fixture
    def mock_fallback_provider(self):
        """Create a mock fallback provider."""
        provider = MagicMock()
        provider.generate = AsyncMock(
            return_value=PromptResponse(
                text="Fallback response",
                model_used="fallback-model",
                provider="fallback",
                latency_ms=100.0,
            )
        )
        provider.generate_stream = AsyncMock()
        provider.count_tokens = AsyncMock(return_value=10)
        return provider

    @pytest.fixture
    def adapter(self, mock_proxy_client, mock_fallback_provider):
        """Create a ProxyProviderAdapter with mocked dependencies."""
        from app.infrastructure.proxy.proxy_provider_adapter import (
            ProxyProviderAdapter,
        )

        return ProxyProviderAdapter(
            provider_name="openai",
            default_model="gpt-4",
            fallback_provider=mock_fallback_provider,
            proxy_client=mock_proxy_client,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, adapter, mock_proxy_client):
        """Test successful generation through proxy."""
        request = PromptRequest(
            prompt="Hello, world!",
            config=GenerationConfig(temperature=0.7),
        )

        response = await adapter.generate(request)

        assert response.text == "Generated text"
        assert "proxy:openai" in response.provider
        mock_proxy_client.send_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_fallback_on_connection_error(
        self, adapter, mock_proxy_client, mock_fallback_provider
    ):
        """Test fallback to direct mode on connection error."""
        from app.infrastructure.proxy.proxy_client import ProxyConnectionError

        mock_proxy_client.send_request = AsyncMock(
            side_effect=ProxyConnectionError("Connection failed")
        )

        request = PromptRequest(prompt="Test prompt")

        with patch.object(
            adapter, "_should_fallback", return_value=True
        ):
            response = await adapter.generate(request)

            assert response.text == "Fallback response"
            assert "fallback" in response.provider
            mock_fallback_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_health(self, adapter, mock_proxy_client):
        """Test health check through proxy."""
        is_healthy = await adapter.check_health()

        assert is_healthy
        mock_proxy_client.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_tokens_with_fallback(
        self, adapter, mock_fallback_provider
    ):
        """Test token counting using fallback provider."""
        count = await adapter.count_tokens("Test text", "gpt-4")

        assert count == 10
        mock_fallback_provider.count_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_tokens_without_fallback(self, mock_proxy_client):
        """Test token counting estimation without fallback."""
        from app.infrastructure.proxy.proxy_provider_adapter import (
            ProxyProviderAdapter,
        )

        adapter = ProxyProviderAdapter(
            provider_name="openai",
            proxy_client=mock_proxy_client,
            fallback_provider=None,
        )

        count = await adapter.count_tokens("Test text with many words")

        # Should use character-based estimation
        assert count > 0

    def test_get_stats(self, adapter):
        """Test statistics retrieval."""
        stats = adapter.get_stats()

        assert stats["provider"] == "openai"
        assert stats["default_model"] == "gpt-4"
        assert stats["request_count"] == 0
        assert stats["has_fallback"] is True


# =============================================================================
# ProxyHealthMonitor Tests
# =============================================================================


class TestProxyHealthMonitor:
    """Tests for the ProxyHealthMonitor class."""

    @pytest.fixture
    def mock_proxy_client(self):
        """Create a mock ProxyClient."""
        from app.infrastructure.proxy.proxy_client import (
            ProxyConnectionState,
            ProxyHealthStatus,
        )

        client = MagicMock()
        client.is_healthy = True
        client.check_health = AsyncMock(
            return_value=ProxyHealthStatus(
                is_healthy=True,
                latency_ms=50.0,
                last_check=0,
                connection_state=ProxyConnectionState.CONNECTED,
            )
        )
        client.get_stats = MagicMock(return_value={"request_count": 0})
        return client

    @pytest.fixture
    def health_monitor(self, mock_proxy_client):
        """Create a ProxyHealthMonitor with mocked client."""
        from app.infrastructure.proxy.proxy_health import ProxyHealthMonitor

        return ProxyHealthMonitor(
            proxy_client=mock_proxy_client,
            check_interval=1,
            history_size=10,
        )

    @pytest.mark.asyncio
    async def test_check_now(self, health_monitor, mock_proxy_client):
        """Test immediate health check."""
        status = await health_monitor.check_now()

        assert status.is_healthy
        mock_proxy_client.check_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, health_monitor, mock_proxy_client):
        """Test that metrics are tracked correctly."""
        # Perform multiple health checks
        await health_monitor.check_now()
        await health_monitor.check_now()

        metrics = health_monitor.metrics

        assert metrics.total_checks == 2
        assert metrics.successful_checks == 2
        assert metrics.failed_checks == 0
        assert metrics.uptime_percent == 100.0

    @pytest.mark.asyncio
    async def test_history_tracking(self, health_monitor):
        """Test that history is maintained."""
        await health_monitor.check_now()
        await health_monitor.check_now()

        history = health_monitor.history

        assert len(history) == 2
        assert all(r.is_healthy for r in history)

    @pytest.mark.asyncio
    async def test_history_size_limit(self, health_monitor, mock_proxy_client):
        """Test that history respects size limit."""
        # Perform more checks than history size
        for _ in range(15):
            await health_monitor.check_now()

        history = health_monitor.history

        # Should be limited to history_size (10)
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_start_stop(self, health_monitor):
        """Test starting and stopping the monitor."""
        await health_monitor.start()
        assert health_monitor.is_running

        await health_monitor.stop()
        assert not health_monitor.is_running

    @pytest.mark.asyncio
    async def test_get_status(self, health_monitor):
        """Test comprehensive status report."""
        await health_monitor.check_now()

        status = health_monitor.get_status()

        assert "is_healthy" in status
        assert "monitoring_active" in status
        assert "metrics" in status
        assert "client" in status
        assert "recent_checks" in status

    def test_reset_metrics(self, health_monitor):
        """Test resetting metrics."""
        health_monitor._metrics.total_checks = 10
        health_monitor._metrics.successful_checks = 8

        health_monitor.reset_metrics()

        assert health_monitor.metrics.total_checks == 0
        assert health_monitor.metrics.successful_checks == 0


# =============================================================================
# Proxy Health API Endpoint Tests
# =============================================================================


class TestProxyHealthEndpoints:
    """Tests for proxy health API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    @pytest.mark.asyncio
    async def test_get_proxy_health_disabled(self):
        """Test health endpoint when proxy mode is disabled."""
        from app.api.v1.endpoints.proxy_health import get_proxy_health

        with patch(
            "app.api.v1.endpoints.proxy_health.settings"
        ) as mock_settings:
            mock_settings.PROXY_MODE_ENABLED = False
            mock_settings.PROXY_MODE_ENDPOINT = "http://localhost:8080"
            mock_settings.PROXY_MODE_FALLBACK_TO_DIRECT = True

            response = await get_proxy_health()

            assert not response.is_healthy
            assert not response.proxy_enabled
            assert response.connection_state == "disabled"

    @pytest.mark.asyncio
    async def test_get_proxy_health_enabled(self):
        """Test health endpoint when proxy mode is enabled."""
        from app.api.v1.endpoints.proxy_health import get_proxy_health
        from app.infrastructure.proxy.proxy_client import (
            ProxyConnectionState,
            ProxyHealthStatus,
        )

        mock_client = MagicMock()
        mock_client.endpoint = "http://localhost:8080"
        mock_client.check_health = AsyncMock(
            return_value=ProxyHealthStatus(
                is_healthy=True,
                latency_ms=25.0,
                last_check=0,
                connection_state=ProxyConnectionState.CONNECTED,
            )
        )

        with patch(
            "app.api.v1.endpoints.proxy_health.settings"
        ) as mock_settings:
            mock_settings.PROXY_MODE_ENABLED = True
            mock_settings.PROXY_MODE_FALLBACK_TO_DIRECT = True

            with patch(
                "app.api.v1.endpoints.proxy_health.get_proxy_client",
                return_value=mock_client,
            ):
                response = await get_proxy_health()

                assert response.is_healthy
                assert response.proxy_enabled
                assert response.latency_ms == 25.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestProxyModeIntegration:
    """Integration tests for proxy mode."""

    @pytest.mark.asyncio
    async def test_proxy_client_global_instance(self):
        """Test global proxy client instance management."""
        from app.infrastructure.proxy.proxy_client import (
            close_proxy_client,
            get_proxy_client,
        )

        # Get instance
        client1 = get_proxy_client()
        client2 = get_proxy_client()

        # Should return same instance
        assert client1 is client2

        # Close and verify new instance is created
        await close_proxy_client()
        client3 = get_proxy_client()

        # New instance after close
        assert client3 is not client1

        # Cleanup
        await close_proxy_client()

    @pytest.mark.asyncio
    async def test_health_monitor_global_instance(self):
        """Test global health monitor instance management."""
        from app.infrastructure.proxy.proxy_health import (
            get_health_monitor,
            start_health_monitoring,
            stop_health_monitoring,
        )

        # Get instance
        monitor1 = get_health_monitor()
        monitor2 = get_health_monitor()

        assert monitor1 is monitor2

        # Start and stop
        await start_health_monitoring()
        assert monitor1.is_running

        await stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_create_proxy_adapter_factory(self):
        """Test proxy adapter factory function."""
        from app.infrastructure.proxy.proxy_provider_adapter import (
            create_proxy_adapter,
        )

        adapter = create_proxy_adapter(
            provider_name="anthropic",
            default_model="claude-3",
        )

        assert adapter.provider_name == "anthropic"
        assert adapter._default_model == "claude-3"
        assert adapter._fallback_provider is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestProxyErrorHandling:
    """Tests for proxy error handling scenarios."""

    @pytest.mark.asyncio
    async def test_proxy_response_error_400(self):
        """Test handling of 4xx errors from proxy."""
        from app.infrastructure.proxy.proxy_client import (
            ProxyClient,
            ProxyResponseError,
        )

        client = ProxyClient(endpoint="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.text = "Bad request"

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_get_client", return_value=mock_http_client
        ):
            with pytest.raises(ProxyResponseError) as exc_info:
                await client.send_request(
                    provider="openai",
                    model="gpt-4",
                    prompt="Test",
                )

            assert exc_info.value.status_code == 400

        await client.close()

    @pytest.mark.asyncio
    async def test_proxy_rate_limit_error(self):
        """Test handling of rate limit errors (429)."""
        from app.infrastructure.proxy.proxy_client import (
            ProxyClient,
            ProxyResponseError,
        )

        client = ProxyClient(endpoint="http://localhost:8080")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": "Rate limited"}
        mock_response.text = "Rate limited"
        mock_response.headers = {"Retry-After": "60"}

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch.object(
            client, "_get_client", return_value=mock_http_client
        ):
            with pytest.raises(ProxyResponseError) as exc_info:
                await client.send_request(
                    provider="openai",
                    model="gpt-4",
                    prompt="Test",
                )

            assert exc_info.value.status_code == 429
            assert exc_info.value.retry_after == 60.0

        await client.close()

    @pytest.mark.asyncio
    async def test_consecutive_failures_tracking(self):
        """Test tracking of consecutive health check failures."""
        from app.infrastructure.proxy.proxy_client import (
            ProxyConnectionState,
            ProxyHealthStatus,
        )
        from app.infrastructure.proxy.proxy_health import ProxyHealthMonitor

        mock_client = MagicMock()
        failed_status = ProxyHealthStatus(
            is_healthy=False,
            latency_ms=0,
            last_check=0,
            error="Connection refused",
            consecutive_failures=1,
            connection_state=ProxyConnectionState.ERROR,
        )
        mock_client.check_health = AsyncMock(return_value=failed_status)
        mock_client.get_stats = MagicMock(return_value={})

        monitor = ProxyHealthMonitor(
            proxy_client=mock_client,
            check_interval=1,
        )

        # Perform multiple failed checks
        for _ in range(5):
            await monitor.check_now()

        assert monitor.metrics.failed_checks == 5
        assert monitor.metrics.successful_checks == 0
