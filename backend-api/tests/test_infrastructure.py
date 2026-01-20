"""Comprehensive Infrastructure Tests for Chimera Backend.

Tests for:
1. Structured Logging System
2. Logging Middleware
3. Health Check System
4. Health Check Endpoints
5. Router Integration
6. Configuration Updates
"""

import json
import logging
import sys

import pytest

# Add backend-api to path for imports
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])


class TestStructuredLogging:
    """Tests for backend-api/app/core/structured_logging.py."""

    def test_structured_log_formatter_json_output(self) -> None:
        """Verify JSON log formatting with proper field structure."""
        from app.core.structured_logging import StructuredLogFormatter

        formatter = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        log_data = json.loads(output)

        assert "message" in log_data
        assert log_data["message"] == "Test message"
        assert "logger" in log_data
        assert "level" in log_data
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data
        assert "environment" in log_data
        assert "service" in log_data
        assert "version" in log_data

    def test_structured_log_formatter_with_exception(self) -> None:
        """Verify exception info is properly formatted."""
        from app.core.structured_logging import StructuredLogFormatter

        formatter = StructuredLogFormatter()

        try:
            msg = "Test error"
            raise ValueError(msg)
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        log_data = json.loads(output)

        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert "Test error" in log_data["exception"]["message"]

    def test_request_context_variables(self) -> None:
        """Test request tracing with correlation IDs."""
        from app.core.structured_logging import (
            clear_request_context,
            generate_request_id,
            request_id_var,
            session_id_var,
            set_request_context,
            user_id_var,
        )

        request_id = generate_request_id()
        assert request_id is not None
        assert len(request_id) == 36

        set_request_context(request_id="test-123", user_id="user-456", session_id="sess-789")

        assert request_id_var.get() == "test-123"
        assert user_id_var.get() == "user-456"
        assert session_id_var.get() == "sess-789"

        clear_request_context()
        assert request_id_var.get() == ""

    def test_error_tracker_capture_exception(self) -> None:
        """Test Sentry-compatible error tracking integration."""
        from app.core.structured_logging import ErrorTracker

        tracker = ErrorTracker()

        try:
            msg = "Test runtime error"
            raise RuntimeError(msg)
        except RuntimeError as e:
            event_id = tracker.capture_exception(
                exception=e,
                context={"key": "value"},
                tags={"environment": "test"},
                user={"id": "user-123"},
            )

        assert event_id is not None
        assert tracker.get_error_count() == 1

        recent_errors = tracker.get_recent_errors(limit=1)
        assert len(recent_errors) == 1
        assert recent_errors[0]["exception"]["type"] == "RuntimeError"

    def test_log_level_from_environment(self) -> None:
        """Validate log level configuration from environment variables."""
        from app.core.config import settings

        assert settings.LOG_LEVEL.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestLoggingMiddleware:
    """Tests for backend-api/app/middleware/request_logging.py."""

    def test_metrics_middleware_get_metrics(self) -> None:
        """Test metrics retrieval."""
        from fastapi import FastAPI

        from app.middleware.request_logging import MetricsMiddleware

        app = FastAPI()
        middleware = MetricsMiddleware(app)
        metrics = middleware.get_metrics()

        assert "total_requests" in metrics
        assert "total_errors" in metrics
        assert "error_rate" in metrics
        assert "avg_response_time_ms" in metrics
        assert "status_codes" in metrics
        assert "endpoints" in metrics

    def test_metrics_middleware_reset(self) -> None:
        """Test metrics reset functionality."""
        from fastapi import FastAPI

        from app.middleware.request_logging import MetricsMiddleware

        app = FastAPI()
        middleware = MetricsMiddleware(app)
        middleware._request_count = 10
        middleware._error_count = 2
        middleware.reset_metrics()

        metrics = middleware.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0


class TestHealthCheckSystem:
    """Tests for backend-api/app/core/health.py."""

    def test_health_status_enum(self) -> None:
        """Test health status values."""
        from app.core.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_check_result_to_dict(self) -> None:
        """Test HealthCheckResult serialization."""
        from app.core.health import HealthCheckResult, HealthStatus

        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=15.5,
            details={"key": "value"},
        )

        data = result.to_dict()
        assert data["name"] == "test_check"
        assert data["status"] == "healthy"
        assert data["latency_ms"] == 15.5
        assert "timestamp" in data

    def test_health_checker_initialization(self) -> None:
        """Test HealthChecker initializes with default checks."""
        from app.core.health import HealthChecker

        checker = HealthChecker()
        assert "database" in checker._checks
        assert "redis" in checker._checks
        assert "llm_service" in checker._checks
        assert "cache" in checker._checks

    @pytest.mark.asyncio
    async def test_health_checker_run_single_check(self) -> None:
        """Test running a single health check."""
        from app.core.health import HealthChecker, HealthCheckResult, HealthStatus

        checker = HealthChecker()

        async def always_healthy():
            return HealthCheckResult(name="test", status=HealthStatus.HEALTHY)

        checker.register_check("test", always_healthy)
        result = await checker.run_check("test")

        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_liveness_check(self) -> None:
        """Test liveness probe."""
        from app.core.health import HealthChecker, HealthStatus

        checker = HealthChecker()
        result = await checker.liveness_check()

        assert result.name == "liveness"
        assert result.status == HealthStatus.HEALTHY
        assert "uptime_seconds" in result.details


class TestHealthEndpoints:
    """Tests for backend-api/app/api/v1/endpoints/health.py."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.v1.endpoints.health import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_endpoint_returns_status(self, client) -> None:
        """Test /health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data

    def test_health_live_endpoint(self, client) -> None:
        """Test /health/live endpoint for liveness probes."""
        response = client.get("/health/live")
        assert response.status_code in [200, 503]

        data = response.json()
        assert data["name"] == "liveness"

    def test_health_ready_endpoint(self, client) -> None:
        """Test /health/ready endpoint for readiness probes."""
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]

        data = response.json()
        assert data["name"] == "readiness"

    def test_list_health_checks_endpoint(self, client) -> None:
        """Test /health/checks endpoint lists all checks."""
        response = client.get("/health/checks")
        assert response.status_code == 200

        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert "last_results" in data

    def test_individual_health_check_endpoint(self, client) -> None:
        """Test /health/{check_name} endpoint for specific check."""
        response = client.get("/health/cache")
        # May return 200, 404 (unknown check), or 503 (unhealthy)
        assert response.status_code in [200, 404, 503]

        data = response.json()
        assert "name" in data
        assert "status" in data


class TestConfiguration:
    """Tests for backend-api/app/core/config.py."""

    def test_settings_loads_defaults(self) -> None:
        """Validate environment variable loading."""
        from app.core.config import settings

        assert settings.API_V1_STR == "/api/v1"
        assert settings.PROJECT_NAME == "Chimera Backend"
        assert settings.VERSION is not None

    def test_settings_cache_configuration(self) -> None:
        """Test cache configuration defaults."""
        from app.core.config import settings

        assert isinstance(settings.ENABLE_CACHE, bool)
        assert settings.CACHE_MAX_MEMORY_ITEMS > 0
        assert settings.CACHE_DEFAULT_TTL > 0

    def test_settings_redis_configuration(self) -> None:
        """Test Redis configuration."""
        from app.core.config import settings

        assert settings.REDIS_URL is not None
        assert settings.REDIS_CONNECTION_TIMEOUT > 0

    def test_settings_provider_models(self) -> None:
        """Test provider models configuration."""
        from app.core.config import settings

        models = settings.get_provider_models()
        assert "google" in models
        assert "anthropic" in models
        assert "openai" in models

    def test_get_settings_function(self) -> None:
        """Test get_settings returns singleton."""
        from app.core.config import get_settings, settings

        assert get_settings() is settings


class TestCSRFMiddleware:
    """Tests for backend-api/app/core/middleware.py."""

    def test_csrf_middleware_allows_safe_methods(self) -> None:
        """Test CSRF middleware allows GET, HEAD, OPTIONS."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.core.middleware import CSRFMiddleware

        app = FastAPI()
        app.add_middleware(CSRFMiddleware)

        @app.get("/test")
        async def test_get():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200

    def test_csrf_middleware_allows_api_key_auth(self) -> None:
        """Test CSRF middleware allows requests with API key."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.core.middleware import CSRFMiddleware

        app = FastAPI()
        app.add_middleware(CSRFMiddleware)

        @app.post("/test")
        async def test_post():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.post("/test", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
