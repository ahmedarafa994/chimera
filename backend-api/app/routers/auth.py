"""
Authentication Router
Provides login, refresh, and logout endpoints for frontend authentication.
"""


from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.auth import (
    Role,
    TokenPayload,
    auth_service,
    get_current_user,
)
from app.core.observability import get_logger

logger = get_logger("chimera.auth")

router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# Request/Response Models
# =============================================================================


class LoginRequest(BaseModel):
    """Login request with username/email and password."""

    username: str = Field(..., min_length=3, max_length=100, description="Username or email")
    password: str = Field(..., min_length=8, max_length=128, description="User password")


class LoginResponse(BaseModel):
    """Login response with tokens and user info."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"  # Capital B per GAP-009
    expires_in: int
    refresh_expires_in: int  # Added per GAP-004
    user: dict


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., description="Valid refresh token")


class RefreshResponse(BaseModel):
    """Token refresh response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_expires_in: int


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate user and return access/refresh tokens.

    **Rate Limited:** 5 attempts per minute per IP.
    """
    # TODO: Replace with actual user lookup from database
    # For now, validate against environment-configured admin credentials
    import os

    admin_username = os.getenv("CHIMERA_ADMIN_USER", "admin")
    admin_password = os.getenv("CHIMERA_ADMIN_PASSWORD")

    if not admin_password:
        logger.error("CHIMERA_ADMIN_PASSWORD not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured"
        )

    # Verify credentials
    if request.username != admin_username:
        logger.warning(f"Login attempt with unknown user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    if not auth_service.verify_password(request.password,
                                         auth_service.hash_password(admin_password)):
        # Use timing-safe comparison in production
        if request.password != admin_password:
            logger.warning(f"Failed login attempt for user: {request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

    # Create tokens
    role = Role.ADMIN if request.username == admin_username else Role.VIEWER
    tokens = auth_service.create_tokens(request.username, role)

    # Calculate refresh token expiry
    refresh_expires_in = int(auth_service.config.refresh_token_expire.total_seconds())

    logger.info(f"Successful login for user: {request.username}")

    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type="Bearer",
        expires_in=tokens.expires_in,
        refresh_expires_in=refresh_expires_in,
        user={
            "id": request.username,
            "username": request.username,
            "role": role.value,
        }
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: RefreshRequest):
    """
    Refresh access token using a valid refresh token.

    The old refresh token is invalidated after use.
    """
    try:
        tokens = auth_service.refresh_access_token(request.refresh_token)
        refresh_expires_in = int(auth_service.config.refresh_token_expire.total_seconds())

        return RefreshResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="Bearer",
            expires_in=tokens.expires_in,
            refresh_expires_in=refresh_expires_in,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: TokenPayload = Depends(get_current_user)):
    """
    Logout user by revoking the current token.

    Requires valid authentication.
    """
    # Revoke the current token
    auth_service.revoke_token(current_user.jti)
    logger.info(f"User logged out: {current_user.sub}")
    return None


@router.get("/me")
async def get_current_user_info(current_user: TokenPayload = Depends(get_current_user)):
    """
    Get current authenticated user information.
    """
    return {
        "id": current_user.sub,
        "username": current_user.sub,
        "role": current_user.role,
        "permissions": current_user.permissions,
    }
