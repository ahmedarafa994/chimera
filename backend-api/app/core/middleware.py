"""Middleware components for the application.
Includes CSRF protection and other security middleware.

CSRF Protection Implementation:
- Uses Double Submit Cookie pattern
- Generates secure tokens using secrets module
- Validates tokens for state-changing methods
- Provides endpoint for token generation
"""

import secrets
import time
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import logger


class CSRFMiddleware(BaseHTTPMiddleware):
    """Middleware to protect against Cross-Site Request Forgery (CSRF) attacks.

    Implements the Double Submit Cookie pattern:
    1. Server sets a CSRF token in a cookie
    2. Client must include the same token in a header for state-changing requests
    3. Server validates that cookie and header values match

    This is effective because:
    - Attackers cannot read cookies from other domains (Same-Origin Policy)
    - Attackers cannot set custom headers in cross-origin requests

    Configuration:
    - cookie_name: Name of the CSRF cookie (default: "csrf_token")
    - header_name: Name of the CSRF header (default: "X-CSRF-Token")
    - cookie_secure: Whether to set Secure flag on cookie (default: True in production)
    - cookie_samesite: SameSite attribute for cookie (default: "lax")
    - token_length: Length of generated tokens (default: 32 bytes = 64 hex chars)
    - exclude_paths: Paths to exclude from CSRF protection
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list[str] | None = None,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_secure: bool = True,
        cookie_samesite: str = "lax",
        token_length: int = 32,
        exclude_paths: list[str] | None = None,
        enforce: bool = False,  # Set to True in production
    ) -> None:
        super().__init__(app)
        self.allowed_origins = allowed_origins or []
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_secure = cookie_secure
        self.cookie_samesite = cookie_samesite
        self.token_length = token_length
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/csrf/token",  # Token generation endpoint
        ]
        self.enforce = enforce
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}

    def _generate_token(self) -> str:
        """Generate a cryptographically secure CSRF token."""
        return secrets.token_hex(self.token_length)

    def _is_excluded_path(self, path: str) -> bool:
        """Check if the path is excluded from CSRF protection."""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    def _validate_token(self, cookie_token: str | None, header_token: str | None) -> bool:
        """Validate that the CSRF token in the cookie matches the header.
        Uses constant-time comparison to prevent timing attacks.
        """
        if not cookie_token or not header_token:
            return False
        return secrets.compare_digest(cookie_token, header_token)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip CSRF check for safe methods
        if request.method in self.safe_methods:
            response = await call_next(request)
            # Set CSRF cookie on GET requests if not present
            if request.method == "GET" and not request.cookies.get(self.cookie_name):
                token = self._generate_token()
                response.set_cookie(
                    key=self.cookie_name,
                    value=token,
                    httponly=False,  # Must be readable by JavaScript
                    secure=self.cookie_secure,
                    samesite=self.cookie_samesite,
                    max_age=3600 * 24,  # 24 hours
                    path="/",
                )
            return response

        # Skip CSRF check for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Skip CSRF check if using API Key or Bearer Token (stateless auth)
        # CSRF is primarily a concern for cookie-based authentication
        if request.headers.get("Authorization") or request.headers.get("X-API-Key"):
            return await call_next(request)

        # Validate CSRF token for state-changing methods
        csrf_cookie = request.cookies.get(self.cookie_name)
        csrf_header = request.headers.get(self.header_name)

        if not self._validate_token(csrf_cookie, csrf_header):
            if self.enforce:
                logger.warning(
                    f"CSRF validation failed for {request.method} {request.url.path}. "
                    f"Cookie present: {bool(csrf_cookie)}, Header present: {bool(csrf_header)}",
                )
                return Response(
                    content='{"detail": "CSRF token missing or invalid"}',
                    status_code=403,
                    media_type="application/json",
                )
            # Log but don't block in non-enforce mode
            logger.debug(
                f"CSRF token not validated for {request.method} {request.url.path} "
                "(enforcement disabled)",
            )

        return await call_next(request)


class CSRFTokenManager:
    """Manager for CSRF token operations.

    Provides utilities for generating and validating CSRF tokens
    outside of the middleware context.
    """

    def __init__(self, token_length: int = 32) -> None:
        self.token_length = token_length
        self._tokens: dict[str, float] = {}  # token -> expiry timestamp
        self._max_tokens = 10000
        self._token_ttl = 3600 * 24  # 24 hours

    def generate_token(self) -> str:
        """Generate a new CSRF token."""
        token = secrets.token_hex(self.token_length)
        self._tokens[token] = time.time() + self._token_ttl
        self._cleanup_expired()
        return token

    def validate_token(self, token: str) -> bool:
        """Validate a CSRF token."""
        if token not in self._tokens:
            return False
        if time.time() > self._tokens[token]:
            del self._tokens[token]
            return False
        return True

    def _cleanup_expired(self) -> None:
        """Remove expired tokens."""
        now = time.time()
        expired = [t for t, exp in self._tokens.items() if exp < now]
        for token in expired:
            del self._tokens[token]

        # Hard limit on tokens
        if len(self._tokens) > self._max_tokens:
            # Remove oldest tokens
            sorted_tokens = sorted(self._tokens.items(), key=lambda x: x[1])
            excess = len(self._tokens) - self._max_tokens
            for token, _ in sorted_tokens[:excess]:
                del self._tokens[token]


# Global CSRF token manager instance
_csrf_manager: CSRFTokenManager | None = None


def get_csrf_manager() -> CSRFTokenManager:
    """Get the CSRF token manager instance."""
    global _csrf_manager
    if _csrf_manager is None:
        _csrf_manager = CSRFTokenManager()
    return _csrf_manager
