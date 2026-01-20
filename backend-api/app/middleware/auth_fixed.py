"""Simplified Authentication Middleware - FIXED VERSION.

This module provides a streamlined authentication system that:
1. Fixes circular import issues
2. Provides consistent auth patterns
3. Handles API keys and JWT tokens uniformly
4. Includes proper error handling
"""

import logging

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SimplifiedAuthMiddleware(BaseHTTPMiddleware):
    """Simplified authentication middleware that avoids circular imports."""

    def __init__(self, app, excluded_paths=None) -> None:
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/",
            "/health",
            "/health/ping",
            "/health/ready",
            "/health/full",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        # Convert to set for faster lookup
        self.excluded_paths = set(self.excluded_paths)

    def is_path_excluded(self, path: str) -> bool:
        """Check if a path should be excluded from authentication."""
        # Exact match
        if path in self.excluded_paths:
            return True

        # Check if path starts with any excluded path
        return any(path.startswith(excluded_path) for excluded_path in self.excluded_paths)

    def extract_token_from_header(self, authorization: str) -> str | None:
        """Extract token from Authorization header."""
        if not authorization:
            return None

        # Handle Bearer token
        if authorization.startswith("Bearer "):
            return authorization[7:]

        # Handle direct token
        return authorization

    def validate_api_key(self, api_key: str) -> bool:
        """Simple API key validation."""
        # Basic validation - in production, check against database
        if not api_key or len(api_key) < 10:
            return False

        # Check against configured API keys
        import os

        configured_keys = [
            os.getenv("CHIMERA_API_KEY", ""),
            os.getenv("API_KEY", ""),
            "dev-api-key-123456789",  # Development fallback
        ]

        return api_key in configured_keys

    def validate_jwt_token(self, token: str) -> bool:
        """Simple JWT token validation."""
        # Basic validation - in production, properly verify JWT
        if not token or len(token) < 20:
            return False

        # For now, accept any token that looks like a JWT
        # In production, use proper JWT verification
        return token.count(".") == 2

    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        path = request.url.path

        # Skip authentication for excluded paths
        if self.is_path_excluded(path):
            return await call_next(request)

        # Check for API key in header
        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")

        # Check for JWT token in Authorization header
        authorization = request.headers.get("Authorization")
        jwt_token = self.extract_token_from_header(authorization) if authorization else None

        # Validate authentication
        authenticated = False
        auth_method = None

        if api_key and self.validate_api_key(api_key):
            authenticated = True
            auth_method = "api_key"
            # Add user context to request
            request.state.user_id = "api_user"
            request.state.auth_method = "api_key"

        elif jwt_token and self.validate_jwt_token(jwt_token):
            authenticated = True
            auth_method = "jwt"
            # Add user context to request
            request.state.user_id = "jwt_user"
            request.state.auth_method = "jwt"

        # For development, allow unauthenticated access to some endpoints
        import os

        if os.getenv("ENVIRONMENT", "development") == "development":
            if path.startswith(("/api/v1/health", "/api/v1/providers")):
                authenticated = True
                auth_method = "dev_bypass"
                request.state.user_id = "dev_user"
                request.state.auth_method = "dev_bypass"

        if not authenticated:
            logger.warning(f"Authentication failed for path: {path}")
            return self.create_auth_error_response()

        # Add auth headers to response
        response = await call_next(request)
        if auth_method:
            response.headers["X-Auth-Method"] = auth_method

        return response

    def create_auth_error_response(self):
        """Create standardized authentication error response."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "AUTHENTICATION_REQUIRED",
                "message": "Authentication required. Provide X-API-Key header or Authorization Bearer token.",
                "details": {
                    "api_key_header": "X-API-Key",
                    "jwt_header": "Authorization: Bearer <token>",
                },
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


# Dependency function for manual authentication checking
def get_current_user_simple(request: Request):
    """Simple dependency to get current user from request state."""
    user_id = getattr(request.state, "user_id", None)
    auth_method = getattr(request.state, "auth_method", None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": user_id,
        "auth_method": auth_method,
    }


# Factory function to create middleware with custom config
def create_auth_middleware(excluded_paths=None):
    """Factory function to create auth middleware with custom configuration."""
    default_excluded_paths = [
        "/",
        "/health",
        "/health/ping",
        "/health/ready",
        "/health/full",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/providers",  # Allow provider listing without auth
    ]

    if excluded_paths:
        default_excluded_paths.extend(excluded_paths)

    def middleware_factory(app):
        return SimplifiedAuthMiddleware(app, excluded_paths=default_excluded_paths)

    return middleware_factory
