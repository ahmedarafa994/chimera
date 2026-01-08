"""
Security utilities for API authentication
"""

import os

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logging import logger

# HTTP Bearer scheme for API key authentication
security = HTTPBearer(auto_error=False)


def get_api_key(credentials: HTTPAuthorizationCredentials | None = None) -> str:
    """
    Validate and return API key from request
    """
    # If credentials provided via Bearer token
    if credentials:
        api_key = credentials.credentials
    else:
        # Check header as fallback
        # Note: This function signature doesn't have request, but HTTPBearer usually handles it.
        # If we need header fallback, we might need to adjust how this is called or use a different dependency.
        # For now, we assume Bearer token is the primary method.
        api_key = None

    # Check if API key is configured
    configured_keys = []

    # Load from environment variable
    env_key = settings.CHIMERA_API_KEY
    if env_key:
        configured_keys.append(env_key)

    # Also check legacy env var if not in settings
    legacy_env_key = os.getenv("CHIMERA_API_KEY")
    if legacy_env_key and legacy_env_key not in configured_keys:
        configured_keys.append(legacy_env_key)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate key
    if not configured_keys:
        # If no keys configured in server, fail secure
        logger.error("No API keys configured on server. Authentication impossible.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error",
        )

    if api_key in configured_keys:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
