import os
import time

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import logger


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(self, app, excluded_paths: list | None = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/", "/health", "/docs", "/openapi.json", "/redoc"]
        # Ensure paths are normalized
        self.excluded_paths = [path.rstrip("/") for path in self.excluded_paths]
        self.valid_api_keys = self._load_valid_keys()

    def _load_valid_keys(self) -> list:
        """Load valid API keys from environment variables."""
        keys = []

        # Load from environment variable
        env_key = os.getenv("CHIMERA_API_KEY")
        if env_key:
            keys.append(env_key)

        # Load from settings if available
        if hasattr(settings, "CHIMERA_API_KEY") and settings.CHIMERA_API_KEY:
            keys.append(settings.CHIMERA_API_KEY)

        # CRIT-002 FIX: Removed hardcoded dev key - fail-closed in production
        # Log warning if no keys configured
        if not keys:
            if os.getenv("ENVIRONMENT", "development") == "production":
                raise ValueError(
                    "Production environment requires API key configuration (CHIMERA_API_KEY)"
                )
            logger.warning("No API keys configured - authentication disabled for development mode")

        return keys

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths (using prefix matching)
        path = request.url.path.rstrip("/")

        # Check if path matches any excluded path (exact or prefix match)
        for excluded_path in self.excluded_paths:
            if path == excluded_path or path.startswith(excluded_path + "/"):
                return await call_next(request)

        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check API key
        api_key = self._extract_api_key(request)

        if not self._is_valid_api_key(api_key):
            logger.warning(f"Invalid API key attempt from {self._get_client_ip(request)}")
            # Return JSONResponse instead of raising HTTPException to avoid middleware crash
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Process request
        response = await call_next(request)
        return response

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from headers only (HIGH-003 FIX: removed query parameter support)."""
        # Try Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer ") and len(auth_header) > 7:
            return auth_header[7:]

        # Try X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            return api_key_header

        return None

    def _is_valid_api_key(self, api_key: str | None) -> bool:
        """Validate API key using timing-safe comparison (CRIT-002 & HIGH-002 FIX)."""
        if not api_key:
            return False

        # CRIT-002 FIX: Removed weak development mode bypass - fail-closed always
        # If no keys configured, authentication fails (logged in _load_valid_keys)
        if not self.valid_api_keys:
            return False

        # HIGH-002 FIX: Use timing-safe comparison to prevent timing attacks
        import secrets

        return any(secrets.compare_digest(api_key, valid_key) for valid_key in self.valid_api_keys)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP address
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to client IP
        return request.client.host if request.client else "unknown"


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """Middleware for security audit logging."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Log request details for security monitoring
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        method = request.method
        path = request.url.path

        # Log suspicious patterns
        self._check_suspicious_patterns(request, client_ip, user_agent)

        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log access for monitoring
        logger.info(
            f"API_ACCESS: {method} {path} - "
            f"IP: {client_ip} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s - "
            f"UA: {user_agent[:100]}"  # Truncate long user agents
        )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _check_suspicious_patterns(self, request: Request, client_ip: str, user_agent: str):
        """Check for suspicious request patterns."""
        suspicious_patterns = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "burp",
            "owasp",
            "zap",
            "scanner",
            "crawler",
        ]

        user_agent_lower = user_agent.lower()
        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                logger.warning(f"SUSPICIOUS_USER_AGENT: {pattern} detected from {client_ip}")
                break

        # Check for common attack patterns in query parameters
        attack_patterns = [
            "<script",
            "javascript:",
            "onerror=",
            "onload=",
            "union select",
            "drop table",
            "insert into",
            "delete from",
            "exec(",
            "eval(",
            "system(",
        ]

        query_string = str(request.query_params).lower()
        for pattern in attack_patterns:
            if pattern in query_string:
                logger.warning(f"SUSPICIOUS_QUERY_PATTERN: {pattern} detected from {client_ip}")
                break


def get_current_user_id(request: Request) -> str:
    """
    Get the current user ID from the request.
    In this simple API key auth system, we return a static ID or derive it from the key.
    """
    # For now, return a placeholder or hash of the API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"user_{hash(api_key)}"
    return "anonymous_user"


def get_client_info(request: Request) -> dict:
    """Get client information from request."""
    return {
        "ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "timestamp": time.time(),
    }
