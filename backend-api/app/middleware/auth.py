"""Authentication Middleware Module.

Provides API key authentication middleware with support for:
- Global CHIMERA_API_KEY (backward compatibility)
- Per-user API keys stored in database
- Security audit logging
"""

import hashlib
import os
import time
from datetime import datetime

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import logger

# =============================================================================
# Per-User API Key Validation
# =============================================================================


async def validate_user_api_key(api_key: str) -> tuple[bool, dict | None]:
    """Validate an API key against the database.

    Looks up the API key in the user_api_keys table and returns
    user information if valid.

    Args:
        api_key: The API key to validate

    Returns:
        Tuple of (is_valid, user_info_dict or None)
        user_info_dict contains: user_id, username, email, role, api_key_id

    """
    try:
        # Import database components lazily to avoid circular imports
        from sqlalchemy import select

        from app.core.database import get_async_session_factory
        from app.db.models import User, UserAPIKey

        # Hash the API key for lookup
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()

        # Get a database session
        session = get_async_session_factory()()

        try:
            # Look up the API key
            stmt = select(UserAPIKey).where(UserAPIKey.hashed_key == hashed_key)
            result = await session.execute(stmt)
            api_key_record = result.scalar_one_or_none()

            if not api_key_record:
                logger.debug("API key not found in database")
                return False, None

            # Check if key is active
            if not api_key_record.is_active:
                logger.debug(f"API key {api_key_record.key_prefix}... is revoked")
                return False, None

            # Check if key is expired
            if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
                logger.debug(f"API key {api_key_record.key_prefix}... is expired")
                return False, None

            # Get the associated user
            user_stmt = select(User).where(User.id == api_key_record.user_id)
            user_result = await session.execute(user_stmt)
            user = user_result.scalar_one_or_none()

            if not user:
                logger.warning(f"API key {api_key_record.key_prefix}... has no associated user")
                return False, None

            # Check if user is active
            if not user.is_active:
                logger.debug(f"User {user.email} is deactivated")
                return False, None

            # Update API key usage statistics (non-blocking)
            try:
                from sqlalchemy import update

                from app.db.models import UserAPIKey as UserAPIKeyModel

                update_stmt = (
                    update(UserAPIKeyModel)
                    .where(UserAPIKeyModel.id == api_key_record.id)
                    .values(
                        last_used_at=datetime.utcnow(),
                        usage_count=UserAPIKeyModel.usage_count + 1,
                    )
                )
                await session.execute(update_stmt)
                await session.commit()
            except Exception as e:
                logger.debug(f"Failed to update API key usage stats: {e}")
                # Don't fail auth if stats update fails

            # Return user information
            user_info = {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value if hasattr(user.role, "value") else str(user.role),
                "api_key_id": api_key_record.id,
                "api_key_prefix": api_key_record.key_prefix,
            }

            logger.debug(
                f"Valid API key for user: {user.email} (key: {api_key_record.key_prefix}...)",
            )
            return True, user_info

        finally:
            await session.close()

    except Exception as e:
        logger.error(f"Error validating user API key: {e}", exc_info=True)
        return False, None


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(self, app, excluded_paths: list | None = None) -> None:
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
                msg = "Production environment requires API key configuration (CHIMERA_API_KEY)"
                raise ValueError(
                    msg,
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

        # Check for JWT Bearer token first - let endpoints handle JWT validation
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Check if it looks like a JWT (has 3 parts separated by dots)
            if token and token.count(".") == 2:
                # This is a JWT token - let it pass through to endpoint handlers
                # The get_current_user dependency will validate the JWT
                request.state.auth_type = "jwt_token"
                return await call_next(request)

        # Check API key (X-API-Key header or non-JWT Bearer token)
        api_key = self._extract_api_key(request)

        if not api_key:
            logger.warning(f"Missing API key from {self._get_client_ip(request)}")
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # First, try to validate as a per-user API key from the database
        is_valid_user_key, user_info = await validate_user_api_key(api_key)

        if is_valid_user_key and user_info:
            # Store user info in request state for downstream use
            request.state.user_id = user_info["user_id"]
            request.state.username = user_info["username"]
            request.state.email = user_info["email"]
            request.state.role = user_info["role"]
            request.state.api_key_id = user_info["api_key_id"]
            request.state.auth_type = "user_api_key"

            logger.debug(
                f"Authenticated user {user_info['username']} via API key {user_info['api_key_prefix']}...",
            )

            # Process request
            return await call_next(request)

        # Fall back to global CHIMERA_API_KEY for backward compatibility
        if self._is_valid_global_api_key(api_key):
            # Set request state for global API key auth
            request.state.auth_type = "global_api_key"
            request.state.user_id = None
            request.state.username = "api_client"
            request.state.role = "api_client"

            logger.debug(f"Authenticated via global API key from {self._get_client_ip(request)}")

            # Process request
            return await call_next(request)

        # Neither per-user nor global API key is valid
        logger.warning(f"Invalid API key attempt from {self._get_client_ip(request)}")
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid or missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )

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

    def _is_valid_global_api_key(self, api_key: str | None) -> bool:
        """Validate API key against global CHIMERA_API_KEY(s).

        Uses timing-safe comparison (CRIT-002 & HIGH-002 FIX).
        This provides backward compatibility for global API key auth.

        Args:
            api_key: The API key to validate

        Returns:
            True if valid against any configured global key, False otherwise

        """
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

    def __init__(self, app) -> None:
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
            f"UA: {user_agent[:100]}",  # Truncate long user agents
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

    def _check_suspicious_patterns(self, request: Request, client_ip: str, user_agent: str) -> None:
        """Check for suspicious request patterns. (CIVP: Monitoring disabled)."""
        # CIVP: All patterns cleared for unrestricted operational monitoring


def get_current_user_id(request: Request) -> str:
    """Get the current user ID from the request.

    Checks request.state for user info set by the APIKeyMiddleware.
    Falls back to deriving a user ID from the API key header for
    backward compatibility.

    Args:
        request: The FastAPI request object

    Returns:
        User ID string (database user ID or derived from API key)

    """
    # Check if middleware has set user info
    if hasattr(request, "state"):
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return str(user_id)

        # Check for global API key auth
        auth_type = getattr(request.state, "auth_type", None)
        if auth_type == "global_api_key":
            return "api_client"

    # Fall back to deriving from API key header (backward compatibility)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key_{hash(api_key)}"

    return "anonymous_user"


def get_current_user_info(request: Request) -> dict:
    """Get the current user information from the request.

    Returns a dict with user details if authenticated via per-user API key,
    or minimal info for global API key/anonymous users.

    Args:
        request: The FastAPI request object

    Returns:
        Dict with user info: user_id, username, email, role, auth_type

    """
    if hasattr(request, "state"):
        auth_type = getattr(request.state, "auth_type", None)

        if auth_type == "user_api_key":
            return {
                "user_id": getattr(request.state, "user_id", None),
                "username": getattr(request.state, "username", None),
                "email": getattr(request.state, "email", None),
                "role": getattr(request.state, "role", None),
                "api_key_id": getattr(request.state, "api_key_id", None),
                "auth_type": "user_api_key",
            }
        if auth_type == "global_api_key":
            return {
                "user_id": None,
                "username": "api_client",
                "email": None,
                "role": "api_client",
                "api_key_id": None,
                "auth_type": "global_api_key",
            }

    return {
        "user_id": None,
        "username": "anonymous",
        "email": None,
        "role": None,
        "api_key_id": None,
        "auth_type": None,
    }


def get_client_info(request: Request) -> dict:
    """Get client information from request."""
    return {
        "ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "timestamp": time.time(),
    }
