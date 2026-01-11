"""
Authentication API Endpoints

Provides user authentication endpoints:
- POST /register - User registration with email verification
- GET /verify-email/{token} - Email verification
- POST /resend-verification - Resend verification email
- POST /forgot-password - Request password reset
- POST /reset-password - Reset password with token

All endpoints follow security best practices for user authentication.
"""

import logging
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session_factory
from app.core.rate_limit import (
    RateLimitConfig,
    get_rate_limiter,
    get_rate_limit_key,
)
from app.db.models import UserRole
from app.services.email_service import email_service
from app.services.password_validator import validate_password
from app.services.user_service import UserService, get_user_service

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Database Session Dependency
# =============================================================================


async def get_db_session() -> AsyncSession:
    """
    Get an async database session for the request.

    Yields a session and ensures proper cleanup.
    """
    session = get_async_session_factory()()
    try:
        yield session
    finally:
        await session.close()


# =============================================================================
# Request/Response Models
# =============================================================================


class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(
        ...,
        description="User's email address",
        examples=["user@example.com"],
    )
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Username (3-50 characters)",
        examples=["john_doe"],
    )
    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description="Password (minimum 12 characters with mixed case, numbers, and special characters)",
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        import re
        # Allow alphanumeric, underscores, and hyphens
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        # Cannot start with a number
        if v[0].isdigit():
            raise ValueError("Username cannot start with a number")
        return v.lower()  # Normalize to lowercase


class UserRegistrationResponse(BaseModel):
    """Response model for successful registration."""

    success: bool = True
    message: str = "Registration successful. Please check your email to verify your account."
    user_id: int
    email: str
    username: str


class RegistrationErrorResponse(BaseModel):
    """Response model for registration errors."""

    success: bool = False
    errors: list[str]
    password_feedback: Optional[dict] = None


class PasswordStrengthRequest(BaseModel):
    """Request model for password strength check."""

    password: str = Field(..., min_length=1)
    email: Optional[str] = None
    username: Optional[str] = None


class PasswordStrengthResponse(BaseModel):
    """Response model for password strength check."""

    is_valid: bool
    strength: str
    score: int
    errors: list[dict]
    warnings: list[dict]
    suggestions: list[str]


class EmailVerificationResponse(BaseModel):
    """Response model for successful email verification."""

    success: bool = True
    message: str = "Email verified successfully. You can now log in."
    email: str
    username: str


class EmailVerificationErrorResponse(BaseModel):
    """Response model for email verification errors."""

    success: bool = False
    error: str
    token_expired: bool = False


class ResendVerificationRequest(BaseModel):
    """Request model for resending verification email."""

    email: EmailStr = Field(
        ...,
        description="Email address to resend verification to",
        examples=["user@example.com"],
    )


class ResendVerificationResponse(BaseModel):
    """Response model for resend verification."""

    success: bool = True
    message: str = "If this email is registered and unverified, a verification email has been sent."


class ResendVerificationErrorResponse(BaseModel):
    """Response model for resend verification errors."""

    success: bool = False
    error: str
    already_verified: bool = False


class ForgotPasswordRequest(BaseModel):
    """Request model for password reset request (forgot password)."""

    email: EmailStr = Field(
        ...,
        description="Email address to send password reset link to",
        examples=["user@example.com"],
    )


class ForgotPasswordResponse(BaseModel):
    """Response model for forgot password request."""

    success: bool = True
    message: str = "If this email is registered, a password reset link has been sent."


class ResetPasswordRequest(BaseModel):
    """Request model for password reset (using token from email)."""

    token: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="Password reset token from email (64-character hex string)",
    )
    new_password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description="New password (minimum 12 characters with mixed case, numbers, and special characters)",
    )


class ResetPasswordResponse(BaseModel):
    """Response model for successful password reset."""

    success: bool = True
    message: str = "Password has been reset successfully. You can now log in with your new password."


class ResetPasswordErrorResponse(BaseModel):
    """Response model for password reset errors."""

    success: bool = False
    error: str
    token_expired: bool = False
    password_errors: Optional[list[str]] = None


# =============================================================================
# Rate Limiting Configuration for Auth Endpoints
# =============================================================================

# Resend verification: 3 requests per minute per IP to prevent abuse
RESEND_VERIFICATION_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=3,
    burst_size=1,
    cost_weight=1,
)

# Forgot password: 3 requests per minute per IP to prevent enumeration attacks
FORGOT_PASSWORD_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=3,
    burst_size=1,
    cost_weight=1,
)

# Reset password: 5 requests per minute per IP to prevent brute force attacks
RESET_PASSWORD_RATE_LIMIT = RateLimitConfig(
    requests_per_minute=5,
    burst_size=2,
    cost_weight=1,
)


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="""
Register a new user account with email verification.

**Password Requirements**:
- Minimum 12 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)
- At least one special character (!@#$%^&* etc.)
- Cannot contain your email or username
- Cannot be a common password

**Flow**:
1. User submits registration form
2. Password is validated for strength
3. Email and username uniqueness is checked
4. User is created with `is_verified=false`
5. Verification email is sent to user's email address
6. User must click verification link to activate account

**Example Request**:
```json
{
  "email": "user@example.com",
  "username": "john_doe",
  "password": "MySecureP@ssw0rd!"
}
```
    """,
    responses={
        201: {
            "description": "User registered successfully",
            "model": UserRegistrationResponse,
        },
        400: {
            "description": "Invalid registration data (validation errors, weak password, etc.)",
            "model": RegistrationErrorResponse,
        },
        409: {
            "description": "Email or username already exists",
            "model": RegistrationErrorResponse,
        },
    },
    tags=["authentication"],
)
async def register_user(
    request: UserRegistrationRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> UserRegistrationResponse:
    """
    Register a new user with email verification.

    Creates a new user account with:
    - Password strength validation
    - Email/username uniqueness check
    - Email verification token generation
    - Verification email sent asynchronously
    """
    # Validate password strength first (for better error messages)
    password_result = validate_password(
        request.password,
        email=request.email,
        username=request.username,
    )

    if not password_result.is_valid:
        error_messages = [error.message for error in password_result.errors]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "errors": error_messages,
                "password_feedback": password_result.to_dict(),
            },
        )

    # Create user service and register user
    user_service = get_user_service(session)

    result = await user_service.register_user(
        email=request.email,
        username=request.username,
        password=request.password,
        role=UserRole.VIEWER,  # Default role for new users
    )

    if not result.success:
        # Determine error type and appropriate status code
        if any("already" in error.lower() for error in result.errors):
            status_code = status.HTTP_409_CONFLICT
        else:
            status_code = status.HTTP_400_BAD_REQUEST

        raise HTTPException(
            status_code=status_code,
            detail={
                "success": False,
                "errors": result.errors,
            },
        )

    # Send verification email in background
    if result.verification_token and result.user:
        background_tasks.add_task(
            _send_verification_email,
            email=result.user.email,
            username=result.user.username,
            verification_token=result.verification_token,
        )
        logger.info(f"Verification email queued for {result.user.email}")

    return UserRegistrationResponse(
        success=True,
        message="Registration successful. Please check your email to verify your account.",
        user_id=result.user.id,
        email=result.user.email,
        username=result.user.username,
    )


@router.post(
    "/check-password-strength",
    status_code=status.HTTP_200_OK,
    summary="Check password strength",
    description="""
Validate password strength without creating a user.

Useful for real-time password strength feedback during registration.

Returns detailed feedback including:
- Validation errors (must be fixed)
- Warnings (recommendations)
- Suggestions for improvement
- Entropy score (0-100)
- Strength level (very_weak, weak, fair, strong, very_strong)
    """,
    tags=["authentication"],
)
async def check_password_strength(
    request: PasswordStrengthRequest,
) -> PasswordStrengthResponse:
    """
    Check password strength without creating a user.

    Returns detailed feedback for password improvement.
    """
    result = validate_password(
        request.password,
        email=request.email,
        username=request.username,
    )

    return PasswordStrengthResponse(
        is_valid=result.is_valid,
        strength=result.strength.value,
        score=result.score,
        errors=[{"type": e.error_type.value, "message": e.message} for e in result.errors],
        warnings=[{"type": w.error_type.value, "message": w.message} for w in result.warnings],
        suggestions=result.suggestions,
    )


@router.get(
    "/verify-email/{token}",
    status_code=status.HTTP_200_OK,
    summary="Verify user email",
    description="""
Verify a user's email address using the verification token sent via email.

**Token Format**:
- 64-character hex string generated during registration
- Token expires after 24 hours

**Flow**:
1. User clicks verification link in email
2. Frontend extracts token and calls this endpoint
3. If valid, user's email is marked as verified
4. User can now log in

**Security**:
- Token is cleared after successful verification
- Expired tokens are rejected
- Invalid tokens return generic error

**Example URL**:
```
/api/v1/auth/verify-email/abc123def456...
```
    """,
    responses={
        200: {
            "description": "Email verified successfully",
            "model": EmailVerificationResponse,
        },
        400: {
            "description": "Invalid or expired verification token",
            "model": EmailVerificationErrorResponse,
        },
        404: {
            "description": "Token not found",
            "model": EmailVerificationErrorResponse,
        },
    },
    tags=["authentication"],
)
async def verify_email(
    token: str,
    session: AsyncSession = Depends(get_db_session),
) -> EmailVerificationResponse:
    """
    Verify user's email using verification token.

    Validates the token, marks the user as verified,
    and clears the verification token.
    """
    # Validate token format (should be 64 chars hex)
    if not token or len(token) != 64:
        logger.warning(f"Email verification failed: invalid token format")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Invalid verification token format",
                "token_expired": False,
            },
        )

    # Verify using user service
    user_service = get_user_service(session)
    result = await user_service.verify_email(token)

    if not result.success:
        # Determine if token is expired vs invalid
        token_expired = "expired" in (result.error or "").lower()

        logger.info(f"Email verification failed: {result.error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": result.error or "Verification failed",
                "token_expired": token_expired,
            },
        )

    logger.info(f"Email verified successfully: {result.user.email}")

    return EmailVerificationResponse(
        success=True,
        message="Email verified successfully. You can now log in.",
        email=result.user.email,
        username=result.user.username,
    )


@router.post(
    "/resend-verification",
    status_code=status.HTTP_200_OK,
    summary="Resend verification email",
    description="""
Resend the email verification link for users who haven't verified their email yet.

**Security Features**:
- Rate limited to 3 requests per minute per IP address to prevent abuse
- Returns success even if email doesn't exist (prevents email enumeration)
- Only works for unverified users

**Flow**:
1. User requests resend with their email address
2. System checks if email exists and is unverified
3. New verification token is generated (24-hour expiry)
4. Verification email is sent to user's email address
5. Response indicates success (regardless of email existence for security)

**Example Request**:
```json
{
  "email": "user@example.com"
}
```

**Note**: For security reasons, this endpoint will return success even if:
- The email is not registered
- The email is already verified

This prevents attackers from using this endpoint to determine which emails are registered.
    """,
    responses={
        200: {
            "description": "Verification email resent (or would be if applicable)",
            "model": ResendVerificationResponse,
        },
        400: {
            "description": "Email is already verified",
            "model": ResendVerificationErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
        },
    },
    tags=["authentication"],
)
async def resend_verification(
    request_data: ResendVerificationRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ResendVerificationResponse:
    """
    Resend verification email to a user.

    Rate limited to prevent abuse. Generates a new verification
    token and sends a new verification email.
    """
    # Rate limit check - stricter limit for this endpoint
    await _check_resend_rate_limit(request)

    user_service = get_user_service(session)

    # Resend verification email
    success, new_token, error = await user_service.resend_verification_email(
        email=request_data.email
    )

    if not success and error:
        # Only return error if user explicitly already verified
        if "already verified" in error.lower():
            logger.info(f"Resend verification rejected: {request_data.email} already verified")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "success": False,
                    "error": error,
                    "already_verified": True,
                },
            )
        # For other errors, log but return generic success (security)
        logger.warning(f"Resend verification error for {request_data.email}: {error}")

    # Send email in background if we have a new token
    if new_token:
        # We need to get the username for the email template
        user = await user_service.get_user_by_email(request_data.email)
        if user:
            background_tasks.add_task(
                _send_verification_email,
                email=user.email,
                username=user.username,
                verification_token=new_token,
            )
            logger.info(f"Resend verification email queued for {request_data.email}")
    else:
        # Log for security monitoring (but don't reveal to user)
        logger.info(f"Resend verification requested for {request_data.email} (no action taken)")

    # Always return success message for security (prevents email enumeration)
    return ResendVerificationResponse(
        success=True,
        message="If this email is registered and unverified, a verification email has been sent.",
    )


@router.post(
    "/forgot-password",
    status_code=status.HTTP_200_OK,
    summary="Request password reset",
    description="""
Request a password reset link for a registered user.

**Security Features**:
- Rate limited to 3 requests per minute per IP address to prevent abuse
- Returns success regardless of whether the email exists (prevents email enumeration)
- Reset tokens expire after 1 hour
- Only active users can receive password reset emails

**Flow**:
1. User submits their email address
2. System checks if email exists and user is active
3. If valid, a secure reset token is generated (1-hour expiry)
4. Password reset email is sent with reset link
5. Response indicates success (regardless of email existence for security)

**Example Request**:
```json
{
  "email": "user@example.com"
}
```

**Note**: For security reasons, this endpoint will always return success,
even if the email is not registered. This prevents attackers from using
this endpoint to determine which emails are registered.
    """,
    responses={
        200: {
            "description": "Password reset email sent (or would be if applicable)",
            "model": ForgotPasswordResponse,
        },
        429: {
            "description": "Rate limit exceeded",
        },
    },
    tags=["authentication"],
)
async def forgot_password(
    request_data: ForgotPasswordRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
) -> ForgotPasswordResponse:
    """
    Request a password reset link for a registered user.

    Rate limited to prevent abuse. Generates a secure reset token
    and sends a password reset email to the user.

    Security: Always returns success to prevent email enumeration attacks.
    """
    # Rate limit check - stricter limit for this endpoint
    await _check_forgot_password_rate_limit(request)

    user_service = get_user_service(session)

    # Request password reset - this handles all logic including security checks
    result = await user_service.request_password_reset(email=request_data.email)

    # Send email in background if we have a reset token
    if result.reset_token:
        # Get user for email template
        user = await user_service.get_user_by_email(request_data.email)
        if user:
            background_tasks.add_task(
                _send_password_reset_email,
                email=user.email,
                username=user.username,
                reset_token=result.reset_token,
            )
            logger.info(f"Password reset email queued for {request_data.email}")
    else:
        # Log for security monitoring (but don't reveal to user)
        logger.info(f"Password reset requested for {request_data.email} (no action taken)")

    # Always return success message for security (prevents email enumeration)
    return ForgotPasswordResponse(
        success=True,
        message="If this email is registered, a password reset link has been sent.",
    )


@router.post(
    "/reset-password",
    status_code=status.HTTP_200_OK,
    summary="Reset password with token",
    description="""
Reset a user's password using the token received via email.

**Prerequisites**:
- User must have requested a password reset via POST /forgot-password
- User must have received the reset token via email
- Reset token must not be expired (tokens expire after 1 hour)

**Password Requirements**:
- Minimum 12 characters
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)
- At least one special character (!@#$%^&* etc.)
- Cannot contain your email or username
- Cannot be a common password

**Security Features**:
- Rate limited to 5 requests per minute per IP to prevent brute force attacks
- Token is invalidated after successful password reset
- All existing sessions are implicitly invalidated (user must log in again)
- Password change is logged to audit trail

**Example Request**:
```json
{
  "token": "abc123def456...64_character_hex_string",
  "new_password": "MyNewSecureP@ssw0rd!"
}
```

**Flow**:
1. User receives reset email with token
2. User clicks link or copies token to reset password form
3. Frontend calls this endpoint with token and new password
4. If valid, password is updated and user can log in
    """,
    responses={
        200: {
            "description": "Password reset successfully",
            "model": ResetPasswordResponse,
        },
        400: {
            "description": "Invalid request (weak password, invalid token)",
            "model": ResetPasswordErrorResponse,
        },
        429: {
            "description": "Rate limit exceeded",
        },
    },
    tags=["authentication"],
)
async def reset_password(
    request_data: ResetPasswordRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> ResetPasswordResponse:
    """
    Reset user's password using reset token from email.

    Validates the reset token and new password strength,
    then updates the password if valid.

    Security:
    - Rate limited to prevent brute force attacks
    - Token is cleared after successful reset
    - Password change is logged to audit trail
    """
    # Import audit for logging
    from app.core.audit import AuditAction, audit_log

    # Rate limit check - prevent brute force attacks on reset tokens
    await _check_reset_password_rate_limit(request)

    # Validate token format (should be 64 chars hex)
    if not request_data.token or len(request_data.token) != 64:
        logger.warning("Password reset failed: invalid token format")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Invalid reset token format",
                "token_expired": False,
            },
        )

    # Validate new password strength first (better error messages)
    password_result = validate_password(request_data.new_password)
    if not password_result.is_valid:
        error_messages = [error.message for error in password_result.errors]
        logger.info("Password reset failed: weak password")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": "Password does not meet security requirements",
                "token_expired": False,
                "password_errors": error_messages,
            },
        )

    # Reset password using user service
    user_service = get_user_service(session)
    result = await user_service.reset_password(
        token=request_data.token,
        new_password=request_data.new_password,
    )

    if not result.success:
        # Determine if token is expired vs invalid
        is_expired = "expired" in (result.error or "").lower()

        logger.info(f"Password reset failed: {result.error}")

        # Log failed attempt to audit (without user info since token is invalid)
        client_ip = request.client.host if request.client else None
        audit_log(
            action=AuditAction.AUTH_FAILED,
            resource="/api/v1/auth/reset-password",
            details={
                "reason": "password_reset_failed",
                "error": result.error,
                "token_expired": is_expired,
            },
            ip_address=client_ip,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": result.error or "Password reset failed",
                "token_expired": is_expired,
                "password_errors": (
                    [e.message for e in result.validation_result.errors]
                    if result.validation_result and not result.validation_result.is_valid
                    else None
                ),
            },
        )

    # Log successful password reset to audit trail
    client_ip = request.client.host if request.client else None
    audit_log(
        action=AuditAction.USER_MODIFY,
        resource="/api/v1/auth/reset-password",
        details={
            "action": "password_reset",
            "method": "email_token",
        },
        ip_address=client_ip,
    )

    logger.info("Password reset successful")

    return ResetPasswordResponse(
        success=True,
        message="Password has been reset successfully. You can now log in with your new password.",
    )


# =============================================================================
# Rate Limiting Helpers
# =============================================================================


async def _check_forgot_password_rate_limit(request: Request) -> None:
    """
    Check rate limit for forgot password endpoint.

    Stricter limits to prevent abuse and email bombing.
    """
    import asyncio

    limiter = get_rate_limiter()
    # Use IP-based key for forgot password
    client_ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:forgot_password:ip:{client_ip}"

    # Check rate limit - handle both sync and async limiters
    result = limiter.check_rate_limit(key, FORGOT_PASSWORD_RATE_LIMIT)
    if asyncio.iscoroutine(result):
        allowed, info = await result
    else:
        allowed, info = result

    if not allowed:
        logger.warning(f"Forgot password rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before requesting another password reset.",
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 3)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info.get("reset_at", int(time.time()) + 60)),
                "Retry-After": str(info.get("retry_after", 60)),
            },
        )


async def _check_resend_rate_limit(request: Request) -> None:
    """
    Check rate limit for resend verification endpoint.

    Stricter limits to prevent abuse and email bombing.
    """
    import asyncio

    limiter = get_rate_limiter()
    # Use IP-based key for resend verification
    client_ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:resend_verification:ip:{client_ip}"

    # Check rate limit - handle both sync and async limiters
    result = limiter.check_rate_limit(key, RESEND_VERIFICATION_RATE_LIMIT)
    if asyncio.iscoroutine(result):
        allowed, info = await result
    else:
        allowed, info = result

    if not allowed:
        logger.warning(f"Resend verification rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before requesting another verification email.",
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 3)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info.get("reset_at", int(time.time()) + 60)),
                "Retry-After": str(info.get("retry_after", 60)),
            },
        )


async def _check_reset_password_rate_limit(request: Request) -> None:
    """
    Check rate limit for reset password endpoint.

    Limits to prevent brute force attacks on reset tokens.
    """
    import asyncio

    limiter = get_rate_limiter()
    # Use IP-based key for reset password
    client_ip = request.client.host if request.client else "unknown"
    key = f"ratelimit:reset_password:ip:{client_ip}"

    # Check rate limit - handle both sync and async limiters
    result = limiter.check_rate_limit(key, RESET_PASSWORD_RATE_LIMIT)
    if asyncio.iscoroutine(result):
        allowed, info = await result
    else:
        allowed, info = result

    if not allowed:
        logger.warning(f"Reset password rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many password reset attempts. Please wait before trying again.",
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 5)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(info.get("reset_at", int(time.time()) + 60)),
                "Retry-After": str(info.get("retry_after", 60)),
            },
        )


# =============================================================================
# Background Tasks
# =============================================================================


async def _send_verification_email(
    email: str,
    username: str,
    verification_token: str,
) -> None:
    """
    Send verification email to user.

    This runs as a background task to avoid blocking the registration response.
    """
    try:
        result = await email_service.send_verification_email(
            email=email,
            username=username,
            verification_token=verification_token,
        )

        if result.success:
            if result.dev_mode:
                logger.info(f"Verification email logged (dev mode) for {email}")
            else:
                logger.info(f"Verification email sent to {email}")
        else:
            logger.error(f"Failed to send verification email to {email}: {result.error}")
    except Exception as e:
        logger.error(f"Error sending verification email to {email}: {e}", exc_info=True)


async def _send_password_reset_email(
    email: str,
    username: str,
    reset_token: str,
) -> None:
    """
    Send password reset email to user.

    This runs as a background task to avoid blocking the forgot password response.
    """
    try:
        result = await email_service.send_password_reset_email(
            email=email,
            username=username,
            reset_token=reset_token,
        )

        if result.success:
            if result.dev_mode:
                logger.info(f"Password reset email logged (dev mode) for {email}")
            else:
                logger.info(f"Password reset email sent to {email}")
        else:
            logger.error(f"Failed to send password reset email to {email}: {result.error}")
    except Exception as e:
        logger.error(f"Error sending password reset email to {email}: {e}", exc_info=True)
