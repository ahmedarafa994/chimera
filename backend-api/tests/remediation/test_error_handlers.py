"""Unit tests for API Error Handlers."""

import pytest
from fastapi import HTTPException

from app.api.error_handlers import ErrorResponseBuilder, api_error_handler


class TestErrorResponseBuilder:
    """Test suite for ErrorResponseBuilder."""

    def test_bad_request(self):
        """Test bad request response generation."""
        exc = ErrorResponseBuilder.bad_request(message="Invalid input")

        assert exc.status_code == 400
        assert exc.detail["error"] == "BAD_REQUEST"
        assert exc.detail["message"] == "Invalid input"

    def test_bad_request_with_details(self):
        """Test bad request with additional details."""
        exc = ErrorResponseBuilder.bad_request(
            message="Invalid input", field="name", reason="required"
        )

        assert exc.detail["details"]["field"] == "name"
        assert exc.detail["details"]["reason"] == "required"

    def test_not_found(self):
        """Test not found response generation."""
        exc = ErrorResponseBuilder.not_found(resource="Technique", identifier="tech_123")

        assert exc.status_code == 404
        assert exc.detail["error"] == "NOT_FOUND"

    def test_rate_limited(self):
        """Test rate limited response generation."""
        exc = ErrorResponseBuilder.rate_limited(retry_after=60)

        assert exc.status_code == 429
        # Accept either format
        assert "RATE_LIMIT" in exc.detail["error"]
        assert exc.detail["details"]["retry_after_seconds"] == 60

    def test_provider_unavailable(self):
        """Test provider unavailable response generation."""
        exc = ErrorResponseBuilder.provider_unavailable(provider="openai")

        assert exc.status_code == 503
        assert exc.detail["error"] == "PROVIDER_UNAVAILABLE"
        assert "openai" in exc.detail["message"]

    def test_internal_error(self):
        """Test internal error response generation."""
        exc = ErrorResponseBuilder.internal_error()

        assert exc.status_code == 500
        assert exc.detail["error"] == "INTERNAL_ERROR"


class TestApiErrorHandler:
    """Test suite for api_error_handler decorator."""

    @pytest.mark.asyncio
    async def test_passes_through_successful_response(self):
        """Test decorator passes through successful responses."""

        @api_error_handler("test_endpoint")
        async def success_func():
            return {"status": "ok"}

        result = await success_func()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_handles_value_error(self):
        """Test decorator handles ValueError as bad request."""

        @api_error_handler("test_endpoint")
        async def bad_input_func():
            raise ValueError("Invalid input value")

        with pytest.raises(HTTPException) as exc_info:
            await bad_input_func()

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_passes_through_http_exception(self):
        """Test decorator passes through existing HTTPExceptions."""

        @api_error_handler("test_endpoint")
        async def http_error_func():
            raise HTTPException(status_code=403, detail="Forbidden")

        with pytest.raises(HTTPException) as exc_info:
            await http_error_func()

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "Forbidden"

    @pytest.mark.asyncio
    async def test_handles_unexpected_exception(self):
        """Test decorator handles unexpected exceptions as internal error."""

        @api_error_handler("test_endpoint")
        async def unexpected_error_func():
            raise RuntimeError("Unexpected error")

        with pytest.raises(HTTPException) as exc_info:
            await unexpected_error_func()

        assert exc_info.value.status_code == 500
        # Should not expose internal error message
        msg = exc_info.value.detail.get("message", "")
        assert "RuntimeError" not in msg

    @pytest.mark.asyncio
    async def test_includes_request_id_in_response(self):
        """Test decorator includes request ID in error responses."""

        @api_error_handler("test_endpoint")
        async def error_func():
            raise ValueError("Error")

        with pytest.raises(HTTPException) as exc_info:
            await error_func()

        detail = exc_info.value.detail
        assert "request_id" in detail
        assert detail["request_id"].startswith("req_")
