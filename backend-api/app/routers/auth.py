"""
Authentication Router
Provides login, refresh, and logout endpoints for frontend authentication.

Supports two authentication modes:
1. Database users - Users registered in the database (primary mode)
2. Environment admin - Fallback admin account from env variables (backward compatibility)
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import AuditAction, AuditSeverity, audit_log
from app.core.auth import Role, TokenPayload, auth_service, get_current_user
from app.core.database import get_async_session_factory
from app.core.observability import get_logger
from app.services.user_service import get_user_service

logger = get_logger("chimera.auth")

router = APIRouter(prefix="/auth", tags=["authentication"])


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session() -> AsyncSession:
    """
    Get an async database session for the request.

    Yields a session and ensures proper cleanup.
    """
    print("[AUTH_DEBUG] get_db_session: Creating session factory", flush=True)
    logger.debug("[AUTH_DEBUG] get_db_session: Creating session factory")
    session = get_async_session_factory()()
    print(f"[AUTH_DEBUG] get_db_session: Session created {session}", flush=True)
    logger.debug(f"[AUTH_DEBUG] get_db_session: Session created {session}")
    try:
        print("[AUTH_DEBUG] get_db_session: Yielding session", flush=True)
        logger.debug("[AUTH_DEBUG] get_db_session: Yielding session")
        yield session
    except Exception as e:
        print(f"[AUTH_DEBUG] get_db_session: Error during session yield: {e}", flush=True)
        logger.error(f"[AUTH_DEBUG] get_db_session: Error during session yield: {e}")
        raise
    finally:
        print("[AUTH_DEBUG] get_db_session: Closing session", flush=True)
        logger.debug("[AUTH_DEBUG] get_db_session: Closing session")
        await session.close()
        print("[AUTH_DEBUG] get_db_session: Session closed", flush=True)
        logger.debug("[AUTH_DEBUG] get_db_session: Session closed")


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
    requires_verification: bool = False  # Added for email verification flow


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
async def login(
    request: LoginRequest,
    http_request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Authenticate user and return access/refresh tokens.

    **Authentication Modes**:
    1. Database users (primary) - Registered users in the database
    2. Environment admin (fallback) - For backward compatibility

    **Rate Limited:** 5 attempts per minute per IP.

    **Requirements**:
    - User must have verified email to login
    - User account must be active

    **Audit Logging**:
    - All login attempts are logged for security compliance
    """
    client_ip = http_request.client.host if http_request.client else None
    user_agent = http_request.headers.get("User-Agent", "")

    # Try database authentication first
    user_service = get_user_service(session)
    auth_result = await user_service.authenticate_user(
        identifier=request.username,
        password=request.password,
        update_last_login=True,
    )

    if auth_result.success and auth_result.tokens:
        # Successful database authentication
        user = auth_result.user

        # Log successful authentication
        audit_log(
            action=AuditAction.AUTH_LOGIN,
            user_id=str(user.id),
            resource="/auth/login",
            details={
                "method": "database",
                "email": user.email,
                "username": user.username,
                "role": user.role.value,
            },
            ip_address=client_ip,
            user_agent=user_agent,
            severity=AuditSeverity.INFO,
        )

        logger.info(f"Successful login for user: {user.email} (id={user.id})")

        # Calculate refresh token expiry
        refresh_expires_in = int(auth_service.config.refresh_token_expire.total_seconds())

        return LoginResponse(
            access_token=auth_result.tokens.access_token,
            refresh_token=auth_result.tokens.refresh_token,
            token_type="Bearer",
            expires_in=auth_result.tokens.expires_in,
            refresh_expires_in=refresh_expires_in,
            user={
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "role": user.role.value,
                "is_verified": user.is_verified,
            },
            requires_verification=False,
        )

    # Check if user exists but email not verified
    if auth_result.requires_verification:
        # Log failed login due to unverified email
        audit_log(
            action=AuditAction.AUTH_FAILED,
            user_id=request.username,
            resource="/auth/login",
            details={
                "reason": "email_not_verified",
                "method": "database",
            },
            ip_address=client_ip,
            user_agent=user_agent,
            severity=AuditSeverity.WARNING,
        )

        logger.warning(f"Login failed for {request.username}: email not verified")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Email not verified",
                "message": "Please verify your email before logging in. Check your inbox for the verification link.",
                "requires_verification": True,
            },
        )

    # Database authentication failed - try environment admin fallback
    # This maintains backward compatibility with existing deployments
    env_auth_result = await _authenticate_env_admin(request.username, request.password)

    if env_auth_result:
        # Successful environment admin authentication
        tokens, role = env_auth_result

        # Log successful authentication
        audit_log(
            action=AuditAction.AUTH_LOGIN,
            user_id=request.username,
            resource="/auth/login",
            details={
                "method": "environment",
                "username": request.username,
                "role": role.value,
            },
            ip_address=client_ip,
            user_agent=user_agent,
            severity=AuditSeverity.INFO,
        )

        logger.info(f"Successful login for env admin: {request.username}")

        # Calculate refresh token expiry
        refresh_expires_in = int(auth_service.config.refresh_token_expire.total_seconds())

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
                "is_verified": True,  # Env admin is always verified
            },
            requires_verification=False,
        )

    # All authentication methods failed
    audit_log(
        action=AuditAction.AUTH_FAILED,
        user_id=request.username,
        resource="/auth/login",
        details={
            "reason": auth_result.error or "invalid_credentials",
            "method": "all",
        },
        ip_address=client_ip,
        user_agent=user_agent,
        severity=AuditSeverity.WARNING,
    )

    logger.warning(f"Failed login attempt for user: {request.username}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
    )


async def _authenticate_env_admin(
    username: str,
    password: str,
) -> tuple | None:
    """
    Authenticate against environment-configured admin credentials.

    This is a fallback for backward compatibility with deployments
    that use CHIMERA_ADMIN_USER and CHIMERA_ADMIN_PASSWORD.

    Args:
        username: Username to check
        password: Password to verify

    Returns:
        Tuple of (TokenResponse, Role) if authenticated, None otherwise
    """
    admin_username = os.getenv("CHIMERA_ADMIN_USER", "admin")
    admin_password = os.getenv("CHIMERA_ADMIN_PASSWORD")

    # If no admin password configured, env admin is disabled
    if not admin_password:
        return None

    # Check username match
    if username != admin_username:
        return None

    # Verify password using timing-safe comparison
    import secrets

    if not secrets.compare_digest(password, admin_password):
        return None

    # Create tokens for env admin
    role = Role.ADMIN
    tokens = auth_service.create_tokens(admin_username, role)

    return (tokens, role)


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
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token"
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
