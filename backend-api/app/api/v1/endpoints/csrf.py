"""CSRF Token API Endpoint.

Provides endpoint for generating CSRF tokens for frontend use.
SEC-002: CSRF protection endpoint implementation.
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.core.middleware import get_csrf_manager

router = APIRouter(prefix="/csrf", tags=["security"])


class CSRFTokenResponse(BaseModel):
    """Response containing CSRF token."""

    token: str
    expires_in_seconds: int = 86400  # 24 hours


@router.get("/token", response_model=CSRFTokenResponse)
async def get_csrf_token(response: Response):
    """Generate a new CSRF token.

    This endpoint should be called by the frontend on initial page load
    to obtain a CSRF token for subsequent state-changing requests.

    The token should be included in the X-CSRF-Token header for:
    - POST requests
    - PUT requests
    - DELETE requests
    - PATCH requests

    Note: This endpoint is excluded from CSRF protection itself.
    """
    manager = get_csrf_manager()
    token = manager.generate_token()

    # Also set the token as a cookie for Double Submit Cookie pattern
    response.set_cookie(
        key="csrf_token",
        value=token,
        httponly=False,  # Must be readable by JavaScript
        secure=True,  # Only send over HTTPS in production
        samesite="lax",
        max_age=86400,  # 24 hours
        path="/",
    )

    return CSRFTokenResponse(token=token)


@router.post("/validate")
async def validate_csrf_token(token: str):
    """Validate a CSRF token.

    This endpoint can be used to check if a token is still valid
    before making a state-changing request.
    """
    manager = get_csrf_manager()
    is_valid = manager.validate_token(token)

    return {
        "valid": is_valid,
        "message": "Token is valid" if is_valid else "Token is invalid or expired",
    }
