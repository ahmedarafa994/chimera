"""User Repository - Database access layer for user operations.

This module provides async CRUD operations for user management using
SQLAlchemy 2.0 patterns. It implements the repository pattern for
clean separation between business logic and data access.
"""

import logging
import secrets
from datetime import datetime, timedelta

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.db.models import User, UserAPIKey, UserRole

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Repository Exceptions
# =============================================================================


class UserNotFoundError(NotFoundError):
    """Raised when a user is not found."""

    error_code = "USER_NOT_FOUND"
    message = "User not found"


class UserAlreadyExistsError(ConflictError):
    """Raised when attempting to create a user that already exists."""

    error_code = "USER_ALREADY_EXISTS"
    message = "User already exists"


class EmailAlreadyExistsError(ConflictError):
    """Raised when email is already registered."""

    error_code = "EMAIL_ALREADY_EXISTS"
    message = "Email is already registered"


class UsernameAlreadyExistsError(ConflictError):
    """Raised when username is already taken."""

    error_code = "USERNAME_ALREADY_EXISTS"
    message = "Username is already taken"


class InvalidTokenError(NotFoundError):
    """Raised when a verification or reset token is invalid."""

    error_code = "INVALID_TOKEN"
    message = "Invalid or expired token"


# =============================================================================
# User Repository
# =============================================================================


class UserRepository:
    """Repository for user database operations.

    Provides async CRUD methods for user management including:
    - User creation with uniqueness enforcement
    - User lookup by email, username, or ID
    - Email verification token management
    - Password reset token management
    - User updates and deactivation

    All operations use SQLAlchemy 2.0 async patterns.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize user repository.

        Args:
            session: SQLAlchemy async session

        """
        self._session = session

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_by_id(self, user_id: int) -> User | None:
        """Get user by ID.

        Args:
            user_id: User's unique identifier

        Returns:
            User if found, None otherwise

        """
        try:
            stmt = select(User).where(User.id == user_id)
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug(f"Found user by id={user_id}: {user.email}")
            else:
                logger.debug(f"No user found with id={user_id}")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by id={user_id}: {e}", exc_info=True)
            raise

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email address.

        Args:
            email: User's email address (case-insensitive)

        Returns:
            User if found, None otherwise

        """
        try:
            # Case-insensitive email lookup
            stmt = select(User).where(func.lower(User.email) == email.lower())
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug(f"Found user by email: {email}")
            else:
                logger.debug(f"No user found with email: {email}")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by email={email}: {e}", exc_info=True)
            raise

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: User's username (case-insensitive)

        Returns:
            User if found, None otherwise

        """
        try:
            # Case-insensitive username lookup
            stmt = select(User).where(func.lower(User.username) == username.lower())
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug(f"Found user by username: {username}")
            else:
                logger.debug(f"No user found with username: {username}")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by username={username}: {e}", exc_info=True)
            raise

    async def get_by_email_or_username(self, identifier: str) -> User | None:
        """Get user by email or username.

        Useful for login where user can provide either identifier.

        Args:
            identifier: Email or username (case-insensitive)

        Returns:
            User if found, None otherwise

        """
        try:
            stmt = select(User).where(
                or_(
                    func.lower(User.email) == identifier.lower(),
                    func.lower(User.username) == identifier.lower(),
                ),
            )
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug(f"Found user by identifier: {identifier}")
            else:
                logger.debug(f"No user found with identifier: {identifier}")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by identifier={identifier}: {e}", exc_info=True)
            raise

    async def get_by_verification_token(self, token: str) -> User | None:
        """Get user by email verification token.

        Args:
            token: Email verification token

        Returns:
            User if found with valid token, None otherwise

        """
        try:
            stmt = select(User).where(
                and_(
                    User.email_verification_token == token,
                    or_(
                        User.email_verification_token_expires.is_(None),
                        User.email_verification_token_expires > datetime.utcnow(),
                    ),
                ),
            )
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug("Found user by verification token")
            else:
                logger.debug("No user found with valid verification token")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by verification token: {e}", exc_info=True)
            raise

    async def get_by_reset_token(self, token: str) -> User | None:
        """Get user by password reset token.

        Args:
            token: Password reset token

        Returns:
            User if found with valid (non-expired) token, None otherwise

        """
        try:
            stmt = select(User).where(
                and_(
                    User.password_reset_token == token,
                    User.password_reset_token_expires > datetime.utcnow(),
                ),
            )
            result = await self._session.execute(stmt)
            user = result.scalar_one_or_none()

            if user:
                logger.debug("Found user by reset token")
            else:
                logger.debug("No user found with valid reset token")

            return user

        except Exception as e:
            logger.error(f"Failed to get user by reset token: {e}", exc_info=True)
            raise

    async def exists_by_email(self, email: str) -> bool:
        """Check if a user exists with the given email.

        Args:
            email: Email address to check

        Returns:
            True if email exists, False otherwise

        """
        try:
            stmt = select(User.id).where(func.lower(User.email) == email.lower())
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none() is not None

        except Exception as e:
            logger.error(f"Failed to check email existence: {e}", exc_info=True)
            raise

    async def exists_by_username(self, username: str) -> bool:
        """Check if a user exists with the given username.

        Args:
            username: Username to check

        Returns:
            True if username exists, False otherwise

        """
        try:
            stmt = select(User.id).where(func.lower(User.username) == username.lower())
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none() is not None

        except Exception as e:
            logger.error(f"Failed to check username existence: {e}", exc_info=True)
            raise

    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        role: UserRole | None = None,
        is_active: bool | None = None,
        is_verified: bool | None = None,
        search: str | None = None,
    ) -> tuple[list[User], int]:
        """List users with pagination and filtering.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            role: Filter by role
            is_active: Filter by active status
            is_verified: Filter by verification status
            search: Search by email or username

        Returns:
            Tuple of (list of users, total count)

        """
        try:
            # Build base query
            stmt = select(User)
            count_stmt = select(func.count(User.id))

            # Apply filters
            conditions = []
            if role is not None:
                conditions.append(User.role == role)
            if is_active is not None:
                conditions.append(User.is_active == is_active)
            if is_verified is not None:
                conditions.append(User.is_verified == is_verified)
            if search:
                search_pattern = f"%{search}%"
                conditions.append(
                    or_(
                        User.email.ilike(search_pattern),
                        User.username.ilike(search_pattern),
                    ),
                )

            if conditions:
                stmt = stmt.where(and_(*conditions))
                count_stmt = count_stmt.where(and_(*conditions))

            # Get total count
            count_result = await self._session.execute(count_stmt)
            total_count = count_result.scalar_one()

            # Apply pagination and ordering
            stmt = stmt.order_by(User.created_at.desc()).limit(limit).offset(offset)
            result = await self._session.execute(stmt)
            users = list(result.scalars().all())

            logger.debug(f"Listed {len(users)} users (total: {total_count})")
            return users, total_count

        except Exception as e:
            logger.error(f"Failed to list users: {e}", exc_info=True)
            raise

    # =========================================================================
    # Create Operations
    # =========================================================================

    async def create_user(
        self,
        email: str,
        username: str,
        hashed_password: str,
        role: UserRole = UserRole.VIEWER,
        is_active: bool = True,
        is_verified: bool = False,
        email_verification_token: str | None = None,
        email_verification_token_expires: datetime | None = None,
    ) -> User:
        """Create a new user.

        Args:
            email: User's email address
            username: User's username
            hashed_password: Pre-hashed password
            role: User's role (default: VIEWER)
            is_active: Whether user is active (default: True)
            is_verified: Whether email is verified (default: False)
            email_verification_token: Optional verification token
            email_verification_token_expires: Optional token expiry

        Returns:
            Created User object

        Raises:
            EmailAlreadyExistsError: If email is already registered
            UsernameAlreadyExistsError: If username is already taken

        """
        try:
            # Create user instance
            user = User(
                email=email.lower(),  # Store email in lowercase
                username=username,
                hashed_password=hashed_password,
                role=role,
                is_active=is_active,
                is_verified=is_verified,
                email_verification_token=email_verification_token,
                email_verification_token_expires=email_verification_token_expires,
            )

            self._session.add(user)
            await self._session.flush()
            await self._session.refresh(user)

            logger.info(f"Created user: {user.email} (id={user.id}, role={user.role})")
            return user

        except IntegrityError as e:
            await self._session.rollback()
            error_str = str(e).lower()
            if "email" in error_str:
                logger.warning(f"Email already exists: {email}")
                raise EmailAlreadyExistsError(
                    message=f"Email '{email}' is already registered",
                    details={"email": email},
                )
            if "username" in error_str:
                logger.warning(f"Username already exists: {username}")
                raise UsernameAlreadyExistsError(
                    message=f"Username '{username}' is already taken",
                    details={"username": username},
                )
            logger.error(f"IntegrityError creating user: {e}", exc_info=True)
            raise UserAlreadyExistsError(
                message="User already exists",
                details={"email": email, "username": username},
            )

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to create user: {e}", exc_info=True)
            raise

    # =========================================================================
    # Update Operations
    # =========================================================================

    async def update_user(
        self,
        user_id: int,
        **kwargs,
    ) -> User | None:
        """Update user fields.

        Args:
            user_id: User's ID
            **kwargs: Fields to update (username, role, is_active, etc.)

        Returns:
            Updated User object or None if not found

        Raises:
            EmailAlreadyExistsError: If updating email to an existing one
            UsernameAlreadyExistsError: If updating username to an existing one

        """
        try:
            # Get existing user
            user = await self.get_by_id(user_id)
            if not user:
                logger.warning(f"User not found for update: id={user_id}")
                return None

            # Update fields
            for key, value in kwargs.items():
                if hasattr(user, key):
                    if key == "email" and value:
                        value = value.lower()
                    setattr(user, key, value)

            # Mark as updated
            user.updated_at = datetime.utcnow()

            await self._session.flush()
            await self._session.refresh(user)

            logger.info(f"Updated user: {user.email} (id={user_id})")
            return user

        except IntegrityError as e:
            await self._session.rollback()
            error_str = str(e).lower()
            if "email" in error_str:
                logger.warning("Email already exists on update")
                raise EmailAlreadyExistsError(
                    message="Email is already registered",
                    details={"user_id": user_id},
                )
            if "username" in error_str:
                logger.warning("Username already exists on update")
                raise UsernameAlreadyExistsError(
                    message="Username is already taken",
                    details={"user_id": user_id},
                )
            logger.error(f"IntegrityError updating user: {e}", exc_info=True)
            raise

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to update user: {e}", exc_info=True)
            raise

    async def update_password(self, user_id: int, hashed_password: str) -> bool:
        """Update user's password.

        Args:
            user_id: User's ID
            hashed_password: New hashed password

        Returns:
            True if password was updated, False if user not found

        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    hashed_password=hashed_password,
                    updated_at=datetime.utcnow(),
                    # Clear any password reset tokens
                    password_reset_token=None,
                    password_reset_token_expires=None,
                )
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Updated password for user id={user_id}")
                return True
            logger.warning(f"User not found for password update: id={user_id}")
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to update password: {e}", exc_info=True)
            raise

    async def update_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp.

        Args:
            user_id: User's ID

        Returns:
            True if updated, False if user not found

        """
        try:
            stmt = update(User).where(User.id == user_id).values(last_login=datetime.utcnow())
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.debug(f"Updated last_login for user id={user_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update last_login: {e}", exc_info=True)
            raise

    # =========================================================================
    # Email Verification Operations
    # =========================================================================

    async def set_verification_token(
        self,
        user_id: int,
        token: str,
        expires_hours: int = 24,
    ) -> bool:
        """Set email verification token for user.

        Args:
            user_id: User's ID
            token: Verification token (should be hashed before storage in production)
            expires_hours: Token validity in hours (default: 24)

        Returns:
            True if token was set, False if user not found

        """
        try:
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    email_verification_token=token,
                    email_verification_token_expires=expires_at,
                    updated_at=datetime.utcnow(),
                )
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Set verification token for user id={user_id}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to set verification token: {e}", exc_info=True)
            raise

    async def verify_email(self, token: str) -> User | None:
        """Verify user's email using verification token.

        Args:
            token: Email verification token

        Returns:
            Verified User object or None if token is invalid/expired

        Raises:
            InvalidTokenError: If token is invalid or expired

        """
        try:
            # Find user by token
            user = await self.get_by_verification_token(token)
            if not user:
                logger.warning("Invalid or expired verification token")
                raise InvalidTokenError(message="Verification token is invalid or expired")

            # Mark email as verified and clear token
            user.is_verified = True
            user.email_verification_token = None
            user.email_verification_token_expires = None
            user.updated_at = datetime.utcnow()

            await self._session.flush()
            await self._session.refresh(user)

            logger.info(f"Email verified for user: {user.email} (id={user.id})")
            return user

        except InvalidTokenError:
            raise
        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to verify email: {e}", exc_info=True)
            raise

    async def resend_verification(
        self,
        user_id: int,
        new_token: str,
        expires_hours: int = 24,
    ) -> bool:
        """Resend verification email by generating new token.

        Only works for unverified users.

        Args:
            user_id: User's ID
            new_token: New verification token
            expires_hours: Token validity in hours

        Returns:
            True if token was regenerated, False if user not found or already verified

        """
        try:
            user = await self.get_by_id(user_id)
            if not user:
                logger.warning(f"User not found for resend verification: id={user_id}")
                return False

            if user.is_verified:
                logger.warning(f"User already verified: id={user_id}")
                return False

            return await self.set_verification_token(user_id, new_token, expires_hours)

        except Exception as e:
            logger.error(f"Failed to resend verification: {e}", exc_info=True)
            raise

    # =========================================================================
    # Password Reset Operations
    # =========================================================================

    async def set_reset_token(
        self,
        user_id: int,
        token: str,
        expires_hours: int = 1,
    ) -> bool:
        """Set password reset token for user.

        Args:
            user_id: User's ID
            token: Reset token (should be hashed before storage in production)
            expires_hours: Token validity in hours (default: 1)

        Returns:
            True if token was set, False if user not found

        """
        try:
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    password_reset_token=token,
                    password_reset_token_expires=expires_at,
                    updated_at=datetime.utcnow(),
                )
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Set reset token for user id={user_id}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to set reset token: {e}", exc_info=True)
            raise

    async def clear_reset_token(self, user_id: int) -> bool:
        """Clear password reset token for user.

        Args:
            user_id: User's ID

        Returns:
            True if token was cleared, False if user not found

        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    password_reset_token=None,
                    password_reset_token_expires=None,
                    updated_at=datetime.utcnow(),
                )
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Cleared reset token for user id={user_id}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to clear reset token: {e}", exc_info=True)
            raise

    async def reset_password(self, token: str, new_hashed_password: str) -> User | None:
        """Reset user's password using reset token.

        Args:
            token: Password reset token
            new_hashed_password: New hashed password

        Returns:
            User object if password was reset, None if token invalid

        Raises:
            InvalidTokenError: If token is invalid or expired

        """
        try:
            # Find user by token
            user = await self.get_by_reset_token(token)
            if not user:
                logger.warning("Invalid or expired reset token")
                raise InvalidTokenError(message="Password reset token is invalid or expired")

            # Update password and clear token
            user.hashed_password = new_hashed_password
            user.password_reset_token = None
            user.password_reset_token_expires = None
            user.updated_at = datetime.utcnow()

            await self._session.flush()
            await self._session.refresh(user)

            logger.info(f"Password reset for user: {user.email} (id={user.id})")
            return user

        except InvalidTokenError:
            raise
        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to reset password: {e}", exc_info=True)
            raise

    # =========================================================================
    # Account Status Operations
    # =========================================================================

    async def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account.

        Args:
            user_id: User's ID

        Returns:
            True if deactivated, False if user not found

        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(is_active=False, updated_at=datetime.utcnow())
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Deactivated user id={user_id}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to deactivate user: {e}", exc_info=True)
            raise

    async def activate_user(self, user_id: int) -> bool:
        """Activate a user account.

        Args:
            user_id: User's ID

        Returns:
            True if activated, False if user not found

        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(is_active=True, updated_at=datetime.utcnow())
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Activated user id={user_id}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to activate user: {e}", exc_info=True)
            raise

    async def update_role(self, user_id: int, role: UserRole) -> bool:
        """Update user's role.

        Args:
            user_id: User's ID
            role: New role

        Returns:
            True if role was updated, False if user not found

        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(role=role, updated_at=datetime.utcnow())
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.info(f"Updated role for user id={user_id} to {role}")
                return True
            return False

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to update role: {e}", exc_info=True)
            raise

    # =========================================================================
    # Delete Operations
    # =========================================================================

    async def delete_user(self, user_id: int) -> bool:
        """Delete a user permanently.

        Note: Consider using deactivate_user instead for audit trail.

        Args:
            user_id: User's ID

        Returns:
            True if deleted, False if user not found

        """
        try:
            user = await self.get_by_id(user_id)
            if not user:
                logger.warning(f"User not found for deletion: id={user_id}")
                return False

            await self._session.delete(user)
            await self._session.flush()

            logger.info(f"Deleted user id={user_id}")
            return True

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to delete user: {e}", exc_info=True)
            raise

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def count_users(
        self,
        role: UserRole | None = None,
        is_active: bool | None = None,
    ) -> int:
        """Count users with optional filtering.

        Args:
            role: Filter by role
            is_active: Filter by active status

        Returns:
            Number of users matching criteria

        """
        try:
            stmt = select(func.count(User.id))

            conditions = []
            if role is not None:
                conditions.append(User.role == role)
            if is_active is not None:
                conditions.append(User.is_active == is_active)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            result = await self._session.execute(stmt)
            return result.scalar_one()

        except Exception as e:
            logger.error(f"Failed to count users: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token.

        Args:
            length: Length of the token in bytes (actual string will be 2x due to hex encoding)

        Returns:
            Hex-encoded random token

        """
        return secrets.token_hex(length)

    # =========================================================================
    # API Key Operations
    # =========================================================================

    async def list_api_keys(self, user_id: int) -> list[UserAPIKey]:
        """List all API keys for a user.

        Args:
            user_id: User's ID

        Returns:
            List of UserAPIKey objects (active and inactive)

        """
        try:
            stmt = (
                select(UserAPIKey)
                .where(UserAPIKey.user_id == user_id)
                .order_by(UserAPIKey.created_at.desc())
            )
            result = await self._session.execute(stmt)
            keys = list(result.scalars().all())

            logger.debug(f"Found {len(keys)} API keys for user id={user_id}")
            return keys

        except Exception as e:
            logger.error(f"Failed to list API keys for user id={user_id}: {e}", exc_info=True)
            raise

    async def count_active_api_keys(self, user_id: int) -> int:
        """Count active API keys for a user.

        Args:
            user_id: User's ID

        Returns:
            Number of active API keys

        """
        try:
            stmt = select(func.count(UserAPIKey.id)).where(
                and_(
                    UserAPIKey.user_id == user_id,
                    UserAPIKey.is_active,
                    or_(
                        UserAPIKey.expires_at.is_(None),
                        UserAPIKey.expires_at > datetime.utcnow(),
                    ),
                ),
            )
            result = await self._session.execute(stmt)
            count = result.scalar_one()

            logger.debug(f"User id={user_id} has {count} active API keys")
            return count

        except Exception as e:
            logger.error(f"Failed to count API keys for user id={user_id}: {e}", exc_info=True)
            raise

    async def get_api_key_by_id(self, key_id: int, user_id: int) -> UserAPIKey | None:
        """Get an API key by ID, ensuring it belongs to the specified user.

        Args:
            key_id: API key ID
            user_id: User's ID (for ownership verification)

        Returns:
            UserAPIKey if found and owned by user, None otherwise

        """
        try:
            stmt = select(UserAPIKey).where(
                and_(
                    UserAPIKey.id == key_id,
                    UserAPIKey.user_id == user_id,
                ),
            )
            result = await self._session.execute(stmt)
            key = result.scalar_one_or_none()

            if key:
                logger.debug(f"Found API key id={key_id} for user id={user_id}")
            else:
                logger.debug(f"API key id={key_id} not found for user id={user_id}")

            return key

        except Exception as e:
            logger.error(f"Failed to get API key id={key_id}: {e}", exc_info=True)
            raise

    async def get_api_key_by_hash(self, hashed_key: str) -> UserAPIKey | None:
        """Get an API key by its hash.

        Used for API key authentication lookup.

        Args:
            hashed_key: The hashed API key value

        Returns:
            UserAPIKey if found, None otherwise

        """
        try:
            stmt = select(UserAPIKey).where(UserAPIKey.hashed_key == hashed_key)
            result = await self._session.execute(stmt)
            key = result.scalar_one_or_none()

            if key:
                logger.debug(f"Found API key by hash: prefix={key.key_prefix}")
            else:
                logger.debug("API key not found by hash")

            return key

        except Exception as e:
            logger.error(f"Failed to get API key by hash: {e}", exc_info=True)
            raise

    async def create_api_key(
        self,
        user_id: int,
        hashed_key: str,
        key_prefix: str,
        name: str | None = None,
        expires_at: datetime | None = None,
    ) -> UserAPIKey:
        """Create a new API key for a user.

        Args:
            user_id: User's ID
            hashed_key: The hashed API key value (for secure storage)
            key_prefix: First 8 characters of the key for identification
            name: Optional friendly name for the key
            expires_at: Optional expiration datetime

        Returns:
            Created UserAPIKey object

        """
        try:
            api_key = UserAPIKey(
                user_id=user_id,
                hashed_key=hashed_key,
                key_prefix=key_prefix,
                name=name,
                expires_at=expires_at,
                is_active=True,
            )

            self._session.add(api_key)
            await self._session.flush()
            await self._session.refresh(api_key)

            logger.info(f"Created API key for user id={user_id}: prefix={key_prefix}, name={name}")
            return api_key

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to create API key for user id={user_id}: {e}", exc_info=True)
            raise

    async def revoke_api_key(self, key_id: int, user_id: int) -> bool:
        """Revoke (deactivate) an API key.

        Args:
            key_id: API key ID
            user_id: User's ID (for ownership verification)

        Returns:
            True if revoked, False if not found or not owned by user

        """
        try:
            # First verify ownership
            key = await self.get_api_key_by_id(key_id, user_id)
            if not key:
                logger.warning(f"API key id={key_id} not found for user id={user_id}")
                return False

            if not key.is_active:
                logger.info(f"API key id={key_id} is already revoked")
                return True  # Already revoked, consider success

            # Revoke the key
            key.is_active = False
            key.revoked_at = datetime.utcnow()

            await self._session.flush()

            logger.info(f"Revoked API key id={key_id} for user id={user_id}")
            return True

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to revoke API key id={key_id}: {e}", exc_info=True)
            raise

    async def delete_api_key(self, key_id: int, user_id: int) -> bool:
        """Permanently delete an API key.

        Args:
            key_id: API key ID
            user_id: User's ID (for ownership verification)

        Returns:
            True if deleted, False if not found

        """
        try:
            key = await self.get_api_key_by_id(key_id, user_id)
            if not key:
                logger.warning(f"API key id={key_id} not found for deletion")
                return False

            await self._session.delete(key)
            await self._session.flush()

            logger.info(f"Deleted API key id={key_id} for user id={user_id}")
            return True

        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to delete API key id={key_id}: {e}", exc_info=True)
            raise

    async def update_api_key_usage(self, key_id: int) -> bool:
        """Update API key usage statistics (last_used_at, usage_count).

        Args:
            key_id: API key ID

        Returns:
            True if updated, False if not found

        """
        try:
            stmt = (
                update(UserAPIKey)
                .where(UserAPIKey.id == key_id)
                .values(
                    last_used_at=datetime.utcnow(),
                    usage_count=UserAPIKey.usage_count + 1,
                )
            )
            result = await self._session.execute(stmt)

            if result.rowcount > 0:
                logger.debug(f"Updated usage for API key id={key_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to update API key usage: {e}", exc_info=True)
            raise


# =============================================================================
# Factory Function
# =============================================================================


def get_user_repository(session: AsyncSession) -> UserRepository:
    """Get user repository instance.

    Args:
        session: SQLAlchemy async session

    Returns:
        UserRepository instance

    """
    return UserRepository(session)
