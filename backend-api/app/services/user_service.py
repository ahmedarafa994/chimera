"""
User Service - Business logic layer for user operations.

This module orchestrates user operations including:
- User registration with password validation
- User authentication (login verification)
- Email verification token management
- Password reset flow management
- Profile updates

Uses the UserRepository for data access and integrates with AuthService
for password hashing and JWT token generation.
"""

import contextlib
import logging
import secrets
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import Role, TokenResponse, auth_service
from app.core.exceptions import ValidationError
from app.db.models import User, UserRole
from app.repositories.user_repository import (
    EmailAlreadyExistsError,
    InvalidTokenError,
    UsernameAlreadyExistsError,
    get_user_repository,
)
from app.services.password_validator import PasswordValidationResult, validate_password

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Service Responses
# =============================================================================


@dataclass
class RegistrationResult:
    """Result of user registration."""

    success: bool
    user: User | None = None
    verification_token: str | None = None
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class AuthenticationResult:
    """Result of user authentication."""

    success: bool
    user: User | None = None
    tokens: TokenResponse | None = None
    error: str | None = None
    requires_verification: bool = False


@dataclass
class PasswordResetResult:
    """Result of password reset operation."""

    success: bool
    reset_token: str | None = None
    error: str | None = None


@dataclass
class EmailVerificationResult:
    """Result of email verification operation."""

    success: bool
    user: User | None = None
    error: str | None = None


@dataclass
class PasswordChangeResult:
    """Result of password change operation."""

    success: bool
    error: str | None = None
    validation_result: PasswordValidationResult | None = None


# =============================================================================
# User Role Mapping
# =============================================================================


def _db_role_to_auth_role(db_role: UserRole) -> Role:
    """
    Map database UserRole to auth module Role.

    The database uses UserRole enum (admin, researcher, viewer) while
    the auth module uses Role enum with additional roles for different contexts.

    Mapping:
    - admin -> ADMIN
    - researcher -> RESEARCHER (can execute operations, manage campaigns)
    - viewer -> VIEWER (read-only)
    """
    role_mapping = {
        UserRole.ADMIN: Role.ADMIN,
        UserRole.RESEARCHER: Role.RESEARCHER,  # Direct mapping to RESEARCHER role
        UserRole.VIEWER: Role.VIEWER,
    }
    return role_mapping.get(db_role, Role.VIEWER)


# =============================================================================
# User Service
# =============================================================================


class UserService:
    """
    Service layer for user operations.

    Orchestrates user management operations including registration,
    authentication, password management, and email verification.

    Uses:
    - UserRepository for database operations
    - AuthService for password hashing and JWT generation
    - PasswordValidator for password strength validation
    """

    # Token configuration
    VERIFICATION_TOKEN_EXPIRY_HOURS = 24
    PASSWORD_RESET_TOKEN_EXPIRY_HOURS = 1

    def __init__(self, session: AsyncSession):
        """
        Initialize user service.

        Args:
            session: SQLAlchemy async session for database operations
        """
        self._session = session
        self._repository = get_user_repository(session)

    # =========================================================================
    # Registration Operations
    # =========================================================================

    async def register_user(
        self,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.VIEWER,
    ) -> RegistrationResult:
        """
        Register a new user with password validation.

        Validates password strength, checks email/username uniqueness,
        creates user with verification token, and returns result.

        Args:
            email: User's email address
            username: User's chosen username
            password: Plain text password (will be validated and hashed)
            role: User role (default: VIEWER)

        Returns:
            RegistrationResult with user and verification token on success
        """
        errors = []

        # Validate password strength
        password_result = validate_password(password, email=email, username=username)
        if not password_result.is_valid:
            for error in password_result.errors:
                errors.append(error.message)
            logger.info(f"Registration failed for {email}: password validation failed")
            return RegistrationResult(success=False, errors=errors)

        # Hash password
        hashed_password = auth_service.hash_password(password)

        # Generate verification token
        verification_token = self._generate_token()

        try:
            # Create user in database
            user = await self._repository.create_user(
                email=email,
                username=username,
                hashed_password=hashed_password,
                role=role,
                is_active=True,
                is_verified=False,
                email_verification_token=verification_token,
            )

            # Set verification token expiry
            await self._repository.set_verification_token(
                user_id=user.id,
                token=verification_token,
                expires_hours=self.VERIFICATION_TOKEN_EXPIRY_HOURS,
            )

            # Commit the transaction
            await self._session.commit()

            logger.info(f"User registered successfully: {email} (id={user.id})")

            return RegistrationResult(
                success=True,
                user=user,
                verification_token=verification_token,
            )

        except EmailAlreadyExistsError:
            logger.info(f"Registration failed: email already exists - {email}")
            return RegistrationResult(
                success=False,
                errors=["Email is already registered"],
            )

        except UsernameAlreadyExistsError:
            logger.info(f"Registration failed: username already taken - {username}")
            return RegistrationResult(
                success=False,
                errors=["Username is already taken"],
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Registration failed with unexpected error: {e}", exc_info=True)
            return RegistrationResult(
                success=False,
                errors=["An unexpected error occurred during registration"],
            )

    # =========================================================================
    # Authentication Operations
    # =========================================================================

    async def authenticate_user(
        self,
        identifier: str,
        password: str,
        update_last_login: bool = True,
    ) -> AuthenticationResult:
        """
        Authenticate a user by email/username and password.

        Verifies credentials against the database and returns JWT tokens
        on successful authentication.

        Args:
            identifier: Email or username
            password: Plain text password
            update_last_login: Whether to update last_login timestamp

        Returns:
            AuthenticationResult with tokens on success, error on failure
        """
        try:
            with open("backend_debug.log", "a") as f:
                f.write(f"DEBUG: authenticate_user called for {identifier}\n")
        except Exception:
            pass

        # Find user by email or username
        user = await self._repository.get_by_email(identifier)
        if not user:
            user = await self._repository.get_by_username(identifier)

        if not user:
            return AuthenticationResult(
                success=False,
                error="Invalid credentials",
            )

        try:
            with open("backend_debug.log", "a") as f:
                f.write(f"DEBUG: User found: {user.username}\n")
        except Exception:
            pass

        # Verify password
        if not auth_service.verify_password(password, user.hashed_password):
            logger.info(f"Authentication failed: invalid password - {identifier}")
            return AuthenticationResult(
                success=False,
                error="Invalid credentials",
            )

        # Check if active
        if not user.is_active:
            logger.info(f"Authentication failed: user inactive - {identifier}")
            return AuthenticationResult(
                success=False,
                error="Account is deactivated. Please contact support.",
            )

        # Check email verification
        if not user.is_verified:
            logger.info(f"Authentication failed: email not verified - {identifier}")
            return AuthenticationResult(
                success=False,
                error="Email not verified. Please check your email for verification link.",
                requires_verification=True,
            )

        try:
            with open("backend_debug.log", "a") as f:
                f.write(f"DEBUG: Password verified. Update last login: {update_last_login}\n")
        except Exception:
            pass

        try:
            # Update last login timestamp
            if update_last_login:
                with open("backend_debug.log", "a") as f:
                    f.write("DEBUG: Calling repository.update_last_login\n")
                await self._repository.update_last_login(user.id)
                with open("backend_debug.log", "a") as f:
                    f.write("DEBUG: Calling session.commit\n")
                await self._session.commit()
                with open("backend_debug.log", "a") as f:
                    f.write("DEBUG: Commit successful\n")

            # Generate JWT tokens
            auth_role = _db_role_to_auth_role(user.role)
            tokens = auth_service.create_tokens(
                user_id=str(user.id),
                role=auth_role,
            )

            logger.info(f"User authenticated successfully: {user.email} (id={user.id})")

            return AuthenticationResult(
                success=True,
                user=user,
                tokens=tokens,
            )

        except Exception as e:
            logger.error(f"Authentication failed with unexpected error: {e}", exc_info=True)
            with open("backend_debug.log", "a") as f:
                f.write(f"DEBUG: Error during auth: {e}\n")
            # Attempt rollback
            with contextlib.suppress(Exception):
                await self._session.rollback()

            return AuthenticationResult(
                success=False,
                error="An unexpected error occurred during authentication",
            )

    async def get_user_by_id(self, user_id: int) -> User | None:
        """
        Get user by ID.

        Args:
            user_id: User's unique identifier

        Returns:
            User object if found, None otherwise
        """
        return await self._repository.get_by_id(user_id)

    async def get_user_by_email(self, email: str) -> User | None:
        """
        Get user by email.

        Args:
            email: User's email address

        Returns:
            User object if found, None otherwise
        """
        return await self._repository.get_by_email(email)

    # =========================================================================
    # Email Verification Operations
    # =========================================================================

    async def verify_email(self, token: str) -> EmailVerificationResult:
        """
        Verify user's email using verification token.

        Args:
            token: Email verification token from email link

        Returns:
            EmailVerificationResult with user on success
        """
        try:
            user = await self._repository.verify_email(token)
            await self._session.commit()

            logger.info(f"Email verified successfully: {user.email} (id={user.id})")

            return EmailVerificationResult(
                success=True,
                user=user,
            )

        except InvalidTokenError:
            logger.info("Email verification failed: invalid or expired token")
            return EmailVerificationResult(
                success=False,
                error="Verification token is invalid or expired",
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Email verification failed: {e}", exc_info=True)
            return EmailVerificationResult(
                success=False,
                error="An unexpected error occurred during email verification",
            )

    async def resend_verification_email(
        self,
        email: str,
    ) -> tuple[bool, str | None, str | None]:
        """
        Generate a new verification token for resending verification email.

        Args:
            email: User's email address

        Returns:
            Tuple of (success, new_token, error_message)
        """
        try:
            user = await self._repository.get_by_email(email)

            if not user:
                # Return success even if user doesn't exist (security best practice)
                logger.info(f"Resend verification: user not found - {email}")
                return (True, None, None)

            if user.is_verified:
                logger.info(f"Resend verification: user already verified - {email}")
                return (False, None, "Email is already verified")

            # Generate new verification token
            new_token = self._generate_token()
            success = await self._repository.resend_verification(
                user_id=user.id,
                new_token=new_token,
                expires_hours=self.VERIFICATION_TOKEN_EXPIRY_HOURS,
            )

            if success:
                await self._session.commit()
                logger.info(f"New verification token generated: {email}")
                return (True, new_token, None)

            return (False, None, "Failed to generate new verification token")

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Resend verification failed: {e}", exc_info=True)
            return (False, None, "An unexpected error occurred")

    # =========================================================================
    # Password Reset Operations
    # =========================================================================

    async def request_password_reset(self, email: str) -> PasswordResetResult:
        """
        Request a password reset for a user.

        Generates a reset token that can be used to reset the password.
        For security, always returns success even if email doesn't exist.

        Args:
            email: User's email address

        Returns:
            PasswordResetResult with reset token on success
        """
        try:
            user = await self._repository.get_by_email(email)

            if not user:
                # Return success without token (security best practice)
                # This prevents email enumeration attacks
                logger.info(f"Password reset requested for non-existent email: {email}")
                return PasswordResetResult(success=True, reset_token=None)

            if not user.is_active:
                # Silently ignore inactive users
                logger.info(f"Password reset requested for inactive user: {email}")
                return PasswordResetResult(success=True, reset_token=None)

            # Generate reset token
            reset_token = self._generate_token()
            success = await self._repository.set_reset_token(
                user_id=user.id,
                token=reset_token,
                expires_hours=self.PASSWORD_RESET_TOKEN_EXPIRY_HOURS,
            )

            if success:
                await self._session.commit()
                logger.info(f"Password reset token generated: {email} (id={user.id})")
                return PasswordResetResult(
                    success=True,
                    reset_token=reset_token,
                )

            return PasswordResetResult(
                success=False,
                error="Failed to generate password reset token",
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Password reset request failed: {e}", exc_info=True)
            return PasswordResetResult(
                success=False,
                error="An unexpected error occurred",
            )

    async def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> PasswordChangeResult:
        """
        Reset user's password using reset token.

        Validates the new password strength and updates the password if valid.

        Args:
            token: Password reset token from email link
            new_password: New plain text password

        Returns:
            PasswordChangeResult with validation details
        """
        try:
            # First, find user by reset token to get their email for validation
            user = await self._repository.get_by_reset_token(token)

            if not user:
                logger.info("Password reset failed: invalid or expired token")
                return PasswordChangeResult(
                    success=False,
                    error="Password reset token is invalid or expired",
                )

            # Validate new password strength
            validation_result = validate_password(
                new_password,
                email=user.email,
                username=user.username,
            )

            if not validation_result.is_valid:
                error_messages = [e.message for e in validation_result.errors]
                logger.info(f"Password reset failed: weak password for user {user.id}")
                return PasswordChangeResult(
                    success=False,
                    error="; ".join(error_messages),
                    validation_result=validation_result,
                )

            # Hash new password
            hashed_password = auth_service.hash_password(new_password)

            # Update password in database
            await self._repository.reset_password(token, hashed_password)
            await self._session.commit()

            logger.info(f"Password reset successful: {user.email} (id={user.id})")

            return PasswordChangeResult(
                success=True,
                validation_result=validation_result,
            )

        except InvalidTokenError:
            logger.info("Password reset failed: invalid or expired token")
            return PasswordChangeResult(
                success=False,
                error="Password reset token is invalid or expired",
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Password reset failed: {e}", exc_info=True)
            return PasswordChangeResult(
                success=False,
                error="An unexpected error occurred during password reset",
            )

    async def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str,
    ) -> PasswordChangeResult:
        """
        Change user's password (requires current password verification).

        Args:
            user_id: User's unique identifier
            current_password: Current plain text password for verification
            new_password: New plain text password

        Returns:
            PasswordChangeResult with validation details
        """
        try:
            user = await self._repository.get_by_id(user_id)

            if not user:
                logger.warning(f"Password change failed: user not found - {user_id}")
                return PasswordChangeResult(
                    success=False,
                    error="User not found",
                )

            # Verify current password
            if not auth_service.verify_password(current_password, user.hashed_password):
                logger.info(f"Password change failed: incorrect current password - user {user_id}")
                return PasswordChangeResult(
                    success=False,
                    error="Current password is incorrect",
                )

            # Validate new password strength
            validation_result = validate_password(
                new_password,
                email=user.email,
                username=user.username,
            )

            if not validation_result.is_valid:
                error_messages = [e.message for e in validation_result.errors]
                logger.info(f"Password change failed: weak password - user {user_id}")
                return PasswordChangeResult(
                    success=False,
                    error="; ".join(error_messages),
                    validation_result=validation_result,
                )

            # Hash new password and update
            hashed_password = auth_service.hash_password(new_password)
            await self._repository.update_password(user_id, hashed_password)
            await self._session.commit()

            logger.info(f"Password changed successfully: {user.email} (id={user_id})")

            return PasswordChangeResult(
                success=True,
                validation_result=validation_result,
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Password change failed: {e}", exc_info=True)
            return PasswordChangeResult(
                success=False,
                error="An unexpected error occurred during password change",
            )

    # =========================================================================
    # Profile Operations
    # =========================================================================

    async def update_profile(
        self,
        user_id: int,
        **kwargs,
    ) -> User | None:
        """
        Update user profile fields.

        Allowed fields: username
        (email changes require re-verification, handled separately)

        Args:
            user_id: User's unique identifier
            **kwargs: Fields to update

        Returns:
            Updated User object or None if not found
        """
        # Filter allowed fields
        allowed_fields = {"username"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not filtered_kwargs:
            logger.debug(f"No valid fields to update for user {user_id}")
            return await self._repository.get_by_id(user_id)

        try:
            user = await self._repository.update_user(user_id, **filtered_kwargs)
            await self._session.commit()

            if user:
                logger.info(f"Profile updated: {user.email} (id={user_id})")

            return user

        except UsernameAlreadyExistsError:
            await self._session.rollback()
            raise ValidationError(
                message="Username is already taken",
                details={"username": kwargs.get("username")},
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Profile update failed: {e}", exc_info=True)
            raise

    # =========================================================================
    # Account Status Operations
    # =========================================================================

    async def deactivate_user(self, user_id: int) -> bool:
        """
        Deactivate a user account.

        Args:
            user_id: User's unique identifier

        Returns:
            True if deactivated, False if user not found
        """
        try:
            success = await self._repository.deactivate_user(user_id)
            if success:
                await self._session.commit()
                logger.info(f"User deactivated: id={user_id}")
            return success

        except Exception as e:
            await self._session.rollback()
            logger.error(f"User deactivation failed: {e}", exc_info=True)
            raise

    async def activate_user(self, user_id: int) -> bool:
        """
        Activate a user account.

        Args:
            user_id: User's unique identifier

        Returns:
            True if activated, False if user not found
        """
        try:
            success = await self._repository.activate_user(user_id)
            if success:
                await self._session.commit()
                logger.info(f"User activated: id={user_id}")
            return success

        except Exception as e:
            await self._session.rollback()
            logger.error(f"User activation failed: {e}", exc_info=True)
            raise

    async def update_user_role(self, user_id: int, role: UserRole) -> bool:
        """
        Update user's role.

        Args:
            user_id: User's unique identifier
            role: New role to assign

        Returns:
            True if role was updated, False if user not found
        """
        try:
            success = await self._repository.update_role(user_id, role)
            if success:
                await self._session.commit()
                logger.info(f"User role updated: id={user_id}, role={role}")
            return success

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Role update failed: {e}", exc_info=True)
            raise

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _generate_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.

        Args:
            length: Length of the token in bytes (hex encoded = 2x length)

        Returns:
            Hex-encoded random token
        """
        return secrets.token_hex(length)

    async def validate_password_strength(
        self,
        password: str,
        email: str | None = None,
        username: str | None = None,
    ) -> PasswordValidationResult:
        """
        Validate password strength without creating a user.

        Useful for real-time password strength feedback during registration.

        Args:
            password: Password to validate
            email: Optional email to check password doesn't contain
            username: Optional username to check password doesn't contain

        Returns:
            PasswordValidationResult with detailed feedback
        """
        return validate_password(password, email=email, username=username)


# =============================================================================
# Factory Function
# =============================================================================


def get_user_service(session: AsyncSession) -> UserService:
    """
    Get user service instance.

    Args:
        session: SQLAlchemy async session

    Returns:
        UserService instance
    """
    return UserService(session)


# =============================================================================
# Dependency Injection for FastAPI
# =============================================================================


async def get_user_service_dependency(session: AsyncSession) -> UserService:
    """
    FastAPI dependency for getting user service.

    Usage:
        @router.post("/register")
        async def register(
            user_service: UserService = Depends(get_user_service_dependency)
        ):
            ...

    Note: This requires a session dependency to be injected.
    Typically used with:
        async def get_user_service_dependency(
            session: AsyncSession = Depends(get_db_session)
        ) -> UserService:
            return get_user_service(session)
    """
    return get_user_service(session)
