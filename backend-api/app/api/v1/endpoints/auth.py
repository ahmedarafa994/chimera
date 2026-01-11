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
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session_factory
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
