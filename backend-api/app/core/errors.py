import logging
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# CORS headers to include in error responses
def get_cors_headers(request: Request) -> dict[str, str]:
    """Get CORS headers based on the request origin."""
    origin = request.headers.get("origin", "")
    allowed_origins = [
        "http://localhost:3001",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3001",
    ]

    if origin in allowed_origins:
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, X-API-Key, Accept, Origin",
        }
    return {}


class AppError(Exception):
    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class LLMProviderError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=status.HTTP_502_BAD_GATEWAY)


class ProviderNotAvailableError(AppError):
    def __init__(self, provider: str):
        super().__init__(
            f"Provider {provider} is not available", status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class TransformationError(AppError):
    """Base error for transformation failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)
        self.details = details or {}


class InvalidPotencyError(TransformationError):
    """Error for invalid potency levels."""

    pass


class InvalidTechniqueError(TransformationError):
    """Error for invalid technique suites."""

    pass


async def app_exception_handler(request: Request, exc: AppError):
    logger.error(f"AppError: {exc.detail} (Status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request),
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle FastAPI HTTPException without crashing."""
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=get_cors_headers(request),
    )


async def global_exception_handler(request: Request, exc: Exception):
    # Don't log HTTPException as error - it's already handled
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    logger.error(f"Global Exception: {exc!s}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"},
        headers=get_cors_headers(request),
    )
