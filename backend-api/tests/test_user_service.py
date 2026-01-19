"""
Comprehensive Tests for User Service.

This module provides thorough unit testing for the UserService class:
- User registration flow
- User authentication flow
- Email verification flow
- Password reset flow
- Password change flow
- Profile updates
- Account status management

Test categories:
- Success cases: Verify expected behavior for valid inputs
- Failure cases: Verify proper error handling for invalid inputs
- Edge cases: Boundary conditions and unusual scenarios

Markers:
- @pytest.mark.unit: Unit tests (fast, isolated)
- @pytest.mark.asyncio: Async tests
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import Role
from app.db.models import User, UserRole
from app.repositories.user_repository import (
    EmailAlreadyExistsError,
    InvalidTokenError,
    UsernameAlreadyExistsError,
)
from app.services.user_service import UserService, _db_role_to_auth_role

# Test fixtures and configuration
pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock SQLAlchemy async session."""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_repository() -> MagicMock:
    """Create a mock user repository."""
    repo = MagicMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_email = AsyncMock()
    repo.get_by_username = AsyncMock()
    repo.get_by_email_or_username = AsyncMock()
    repo.get_by_reset_token = AsyncMock()
    repo.create_user = AsyncMock()
    repo.update_user = AsyncMock()
    repo.update_password = AsyncMock()
    repo.update_last_login = AsyncMock()
    repo.verify_email = AsyncMock()
    repo.resend_verification = AsyncMock()
    repo.set_verification_token = AsyncMock()
    repo.set_reset_token = AsyncMock()
    repo.reset_password = AsyncMock()
    repo.activate_user = AsyncMock()
    repo.deactivate_user = AsyncMock()
    repo.update_role = AsyncMock()
    return repo


@pytest.fixture
def mock_user() -> MagicMock:
    """Create a mock user object."""
    user = MagicMock(spec=User)
    user.id = 1
    user.email = "testuser@example.com"
    user.username = "testuser"
    user.hashed_password = "$2b$12$hashedpassword"
    user.role = UserRole.VIEWER
    user.is_active = True
    user.is_verified = True
    user.email_verification_token = None
    user.password_reset_token = None
    user.last_login = None
    user.created_at = datetime.utcnow()
    return user


@pytest.fixture
def user_service(mock_session: AsyncMock, mock_repository: MagicMock) -> UserService:
    """Create a UserService with mocked dependencies."""
    with patch("app.services.user_service.get_user_repository", return_value=mock_repository):
        service = UserService(mock_session)
        service._repository = mock_repository
        return service


# =============================================================================
# Role Mapping Tests
# =============================================================================


class TestRoleMapping:
    """Tests for database role to auth role mapping."""

    def test_admin_role_mapping(self):
        """Test admin role maps correctly."""
        assert _db_role_to_auth_role(UserRole.ADMIN) == Role.ADMIN

    def test_researcher_role_mapping(self):
        """Test researcher role maps correctly."""
        assert _db_role_to_auth_role(UserRole.RESEARCHER) == Role.RESEARCHER

    def test_viewer_role_mapping(self):
        """Test viewer role maps correctly."""
        assert _db_role_to_auth_role(UserRole.VIEWER) == Role.VIEWER


# =============================================================================
# Registration Tests
# =============================================================================


class TestRegistration:
    """Tests for user registration."""

    async def test_register_user_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful user registration."""
        mock_repository.create_user.return_value = mock_user
        mock_repository.set_verification_token.return_value = True

        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="SecureP@ssw0rd123!",
        )

        assert result.success is True
        assert result.user is not None
        assert result.verification_token is not None
        assert len(result.errors) == 0

    async def test_register_user_weak_password(self, user_service: UserService):
        """Test registration fails with weak password."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="weak",
        )

        assert result.success is False
        assert len(result.errors) > 0

    async def test_register_user_password_too_short(self, user_service: UserService):
        """Test registration fails with password under minimum length."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="Short1!",  # Less than 12 characters
        )

        assert result.success is False
        assert any("12" in error or "length" in error.lower() for error in result.errors)

    async def test_register_user_password_no_uppercase(self, user_service: UserService):
        """Test registration fails without uppercase letter."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="allowercase123!",
        )

        assert result.success is False
        assert any("uppercase" in error.lower() for error in result.errors)

    async def test_register_user_password_no_lowercase(self, user_service: UserService):
        """Test registration fails without lowercase letter."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="ALLUPPERCASE123!",
        )

        assert result.success is False
        assert any("lowercase" in error.lower() for error in result.errors)

    async def test_register_user_password_no_digit(self, user_service: UserService):
        """Test registration fails without digit."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="NoDigitsHere!@#",
        )

        assert result.success is False
        assert any("digit" in error.lower() or "number" in error.lower() for error in result.errors)

    async def test_register_user_password_no_special(self, user_service: UserService):
        """Test registration fails without special character."""
        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="NoSpecialChars123",
        )

        assert result.success is False
        assert any("special" in error.lower() for error in result.errors)

    async def test_register_user_duplicate_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test registration fails with duplicate email."""
        mock_repository.create_user.side_effect = EmailAlreadyExistsError("Email exists")

        result = await user_service.register_user(
            email="existing@example.com",
            username="newuser",
            password="SecureP@ssw0rd123!",
        )

        assert result.success is False
        assert any("email" in error.lower() for error in result.errors)

    async def test_register_user_duplicate_username(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test registration fails with duplicate username."""
        mock_repository.create_user.side_effect = UsernameAlreadyExistsError("Username exists")

        result = await user_service.register_user(
            email="newuser@example.com",
            username="existinguser",
            password="SecureP@ssw0rd123!",
        )

        assert result.success is False
        assert any("username" in error.lower() for error in result.errors)

    async def test_register_user_with_role(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test registration with specific role."""
        mock_user.role = UserRole.RESEARCHER
        mock_repository.create_user.return_value = mock_user
        mock_repository.set_verification_token.return_value = True

        result = await user_service.register_user(
            email="researcher@example.com",
            username="researcher",
            password="SecureP@ssw0rd123!",
            role=UserRole.RESEARCHER,
        )

        assert result.success is True
        mock_repository.create_user.assert_called_once()
        call_kwargs = mock_repository.create_user.call_args.kwargs
        assert call_kwargs["role"] == UserRole.RESEARCHER


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for user authentication."""

    async def test_authenticate_user_success_with_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful authentication with email."""
        mock_repository.get_by_email_or_username.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True
            mock_auth.create_tokens.return_value = MagicMock(
                access_token="token",
                refresh_token="refresh",
                expires_in=3600,
            )

            result = await user_service.authenticate_user(
                identifier="testuser@example.com",
                password="correct_password",
            )

            assert result.success is True
            assert result.tokens is not None
            assert result.user is not None

    async def test_authenticate_user_success_with_username(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful authentication with username."""
        mock_repository.get_by_email_or_username.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True
            mock_auth.create_tokens.return_value = MagicMock(
                access_token="token",
                refresh_token="refresh",
                expires_in=3600,
            )

            result = await user_service.authenticate_user(
                identifier="testuser",
                password="correct_password",
            )

            assert result.success is True

    async def test_authenticate_user_not_found(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test authentication fails when user not found."""
        mock_repository.get_by_email_or_username.return_value = None

        result = await user_service.authenticate_user(
            identifier="nonexistent@example.com",
            password="any_password",
        )

        assert result.success is False
        assert "invalid" in result.error.lower()

    async def test_authenticate_user_wrong_password(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test authentication fails with wrong password."""
        mock_repository.get_by_email_or_username.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = False

            result = await user_service.authenticate_user(
                identifier="testuser@example.com",
                password="wrong_password",
            )

            assert result.success is False
            assert "invalid" in result.error.lower()

    async def test_authenticate_user_inactive(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test authentication fails for inactive user."""
        mock_user.is_active = False
        mock_repository.get_by_email_or_username.return_value = mock_user

        result = await user_service.authenticate_user(
            identifier="inactive@example.com",
            password="correct_password",
        )

        assert result.success is False
        assert "deactivated" in result.error.lower() or "inactive" in result.error.lower()

    async def test_authenticate_user_unverified_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test authentication requires email verification."""
        mock_user.is_verified = False
        mock_repository.get_by_email_or_username.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True

            result = await user_service.authenticate_user(
                identifier="unverified@example.com",
                password="correct_password",
            )

            assert result.success is False
            assert result.requires_verification is True

    async def test_authenticate_user_updates_last_login(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test that successful authentication updates last_login."""
        mock_repository.get_by_email_or_username.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True
            mock_auth.create_tokens.return_value = MagicMock(
                access_token="token",
                refresh_token="refresh",
                expires_in=3600,
            )

            await user_service.authenticate_user(
                identifier="testuser@example.com",
                password="correct_password",
                update_last_login=True,
            )

            mock_repository.update_last_login.assert_called_once()


# =============================================================================
# Email Verification Tests
# =============================================================================


class TestEmailVerification:
    """Tests for email verification."""

    async def test_verify_email_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful email verification."""
        mock_repository.verify_email.return_value = mock_user

        result = await user_service.verify_email(token="valid_token")

        assert result.success is True
        assert result.user is not None

    async def test_verify_email_invalid_token(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test email verification fails with invalid token."""
        mock_repository.verify_email.side_effect = InvalidTokenError("Invalid token")

        result = await user_service.verify_email(token="invalid_token")

        assert result.success is False
        assert "invalid" in result.error.lower() or "expired" in result.error.lower()

    async def test_resend_verification_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful resend of verification email."""
        mock_user.is_verified = False
        mock_repository.get_by_email.return_value = mock_user
        mock_repository.resend_verification.return_value = True

        success, token, _error = await user_service.resend_verification_email(
            email="unverified@example.com"
        )

        assert success is True
        assert token is not None

    async def test_resend_verification_already_verified(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test resend fails when already verified."""
        mock_user.is_verified = True
        mock_repository.get_by_email.return_value = mock_user

        success, _token, error = await user_service.resend_verification_email(
            email="verified@example.com"
        )

        assert success is False
        assert "already verified" in error.lower()

    async def test_resend_verification_nonexistent_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test resend for non-existent email returns success (security)."""
        mock_repository.get_by_email.return_value = None

        success, token, _error = await user_service.resend_verification_email(
            email="nonexistent@example.com"
        )

        # Returns success to prevent email enumeration
        assert success is True
        assert token is None


# =============================================================================
# Password Reset Tests
# =============================================================================


class TestPasswordReset:
    """Tests for password reset flow."""

    async def test_request_password_reset_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful password reset request."""
        mock_repository.get_by_email.return_value = mock_user
        mock_repository.set_reset_token.return_value = True

        result = await user_service.request_password_reset(email="testuser@example.com")

        assert result.success is True
        assert result.reset_token is not None

    async def test_request_password_reset_nonexistent_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test password reset request for non-existent email."""
        mock_repository.get_by_email.return_value = None

        result = await user_service.request_password_reset(email="nonexistent@example.com")

        # Returns success to prevent email enumeration
        assert result.success is True
        assert result.reset_token is None

    async def test_request_password_reset_inactive_user(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test password reset request for inactive user."""
        mock_user.is_active = False
        mock_repository.get_by_email.return_value = mock_user

        result = await user_service.request_password_reset(email="inactive@example.com")

        # Returns success but no token
        assert result.success is True
        assert result.reset_token is None

    async def test_reset_password_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful password reset."""
        mock_repository.get_by_reset_token.return_value = mock_user
        mock_repository.reset_password.return_value = True

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.hash_password.return_value = "new_hashed_password"

            result = await user_service.reset_password(
                token="valid_token",
                new_password="NewSecureP@ssw0rd123!",
            )

            assert result.success is True

    async def test_reset_password_invalid_token(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test password reset fails with invalid token."""
        mock_repository.get_by_reset_token.return_value = None

        result = await user_service.reset_password(
            token="invalid_token",
            new_password="NewSecureP@ssw0rd123!",
        )

        assert result.success is False
        assert "invalid" in result.error.lower() or "expired" in result.error.lower()

    async def test_reset_password_weak_new_password(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test password reset fails with weak new password."""
        mock_repository.get_by_reset_token.return_value = mock_user

        result = await user_service.reset_password(
            token="valid_token",
            new_password="weak",
        )

        assert result.success is False
        assert result.validation_result is not None
        assert result.validation_result.is_valid is False


# =============================================================================
# Password Change Tests
# =============================================================================


class TestPasswordChange:
    """Tests for password change flow."""

    async def test_change_password_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful password change."""
        mock_repository.get_by_id.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True
            mock_auth.hash_password.return_value = "new_hashed_password"

            result = await user_service.change_password(
                user_id=1,
                current_password="current_password",
                new_password="NewSecureP@ssw0rd123!",
            )

            assert result.success is True

    async def test_change_password_wrong_current(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test password change fails with wrong current password."""
        mock_repository.get_by_id.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = False

            result = await user_service.change_password(
                user_id=1,
                current_password="wrong_password",
                new_password="NewSecureP@ssw0rd123!",
            )

            assert result.success is False
            assert "incorrect" in result.error.lower()

    async def test_change_password_user_not_found(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test password change fails when user not found."""
        mock_repository.get_by_id.return_value = None

        result = await user_service.change_password(
            user_id=999,
            current_password="current_password",
            new_password="NewSecureP@ssw0rd123!",
        )

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_change_password_weak_new_password(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test password change fails with weak new password."""
        mock_repository.get_by_id.return_value = mock_user

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.verify_password.return_value = True

            result = await user_service.change_password(
                user_id=1,
                current_password="current_password",
                new_password="weak",
            )

            assert result.success is False
            assert result.validation_result is not None


# =============================================================================
# Profile Update Tests
# =============================================================================


class TestProfileUpdate:
    """Tests for profile updates."""

    async def test_update_profile_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test successful profile update."""
        mock_user.username = "newusername"
        mock_repository.update_user.return_value = mock_user

        result = await user_service.update_profile(user_id=1, username="newusername")

        assert result is not None
        assert result.username == "newusername"

    async def test_update_profile_duplicate_username(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test profile update fails with duplicate username."""
        from app.core.exceptions import ValidationError

        mock_repository.update_user.side_effect = UsernameAlreadyExistsError("Username taken")

        with pytest.raises(ValidationError) as exc_info:
            await user_service.update_profile(user_id=1, username="existinguser")

        assert "username" in str(exc_info.value).lower()

    async def test_update_profile_ignores_invalid_fields(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test that invalid fields are ignored."""
        mock_repository.get_by_id.return_value = mock_user

        # email is not an allowed field for update_profile
        await user_service.update_profile(
            user_id=1,
            email="newemail@example.com",  # Should be ignored
        )

        # Should not call update_user since email is filtered out
        mock_repository.update_user.assert_not_called()


# =============================================================================
# Account Status Tests
# =============================================================================


class TestAccountStatus:
    """Tests for account status management."""

    async def test_deactivate_user_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test successful user deactivation."""
        mock_repository.deactivate_user.return_value = True

        result = await user_service.deactivate_user(user_id=1)

        assert result is True
        mock_repository.deactivate_user.assert_called_once_with(1)

    async def test_deactivate_user_not_found(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test deactivation of non-existent user."""
        mock_repository.deactivate_user.return_value = False

        result = await user_service.deactivate_user(user_id=999)

        assert result is False

    async def test_activate_user_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test successful user activation."""
        mock_repository.activate_user.return_value = True

        result = await user_service.activate_user(user_id=1)

        assert result is True
        mock_repository.activate_user.assert_called_once_with(1)

    async def test_update_user_role_success(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test successful role update."""
        mock_repository.update_role.return_value = True

        result = await user_service.update_user_role(user_id=1, role=UserRole.RESEARCHER)

        assert result is True
        mock_repository.update_role.assert_called_once_with(1, UserRole.RESEARCHER)


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    async def test_get_user_by_id(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test get user by ID."""
        mock_repository.get_by_id.return_value = mock_user

        result = await user_service.get_user_by_id(user_id=1)

        assert result == mock_user

    async def test_get_user_by_id_not_found(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test get user by ID when not found."""
        mock_repository.get_by_id.return_value = None

        result = await user_service.get_user_by_id(user_id=999)

        assert result is None

    async def test_get_user_by_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
    ):
        """Test get user by email."""
        mock_repository.get_by_email.return_value = mock_user

        result = await user_service.get_user_by_email(email="testuser@example.com")

        assert result == mock_user

    async def test_validate_password_strength(
        self,
        user_service: UserService,
    ):
        """Test password strength validation utility."""
        result = await user_service.validate_password_strength(
            password="SecureP@ssw0rd123!",
        )

        assert result.is_valid is True

    async def test_validate_password_strength_with_context(
        self,
        user_service: UserService,
    ):
        """Test password validation with email/username context."""
        result = await user_service.validate_password_strength(
            password="SecureP@ssw0rd123!",
            email="user@example.com",
            username="testuser",
        )

        assert result.is_valid is True

    def test_generate_token_length(self, user_service: UserService):
        """Test that generated tokens have correct length."""
        # Default length is 32 bytes = 64 hex characters
        token = user_service._generate_token()
        assert len(token) == 64
        assert all(c in "0123456789abcdef" for c in token)

    def test_generate_token_custom_length(self, user_service: UserService):
        """Test token generation with custom length."""
        token = user_service._generate_token(length=16)
        assert len(token) == 32  # 16 bytes = 32 hex characters

    def test_generate_token_uniqueness(self, user_service: UserService):
        """Test that generated tokens are unique."""
        tokens = [user_service._generate_token() for _ in range(100)]
        assert len(set(tokens)) == 100  # All unique


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    async def test_register_user_database_error(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_session: AsyncMock,
    ):
        """Test registration handles database errors gracefully."""
        mock_repository.create_user.side_effect = Exception("Database error")

        result = await user_service.register_user(
            email="newuser@example.com",
            username="newuser",
            password="SecureP@ssw0rd123!",
        )

        assert result.success is False
        assert "unexpected error" in result.errors[0].lower()
        mock_session.rollback.assert_called()

    async def test_authenticate_user_database_error(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ):
        """Test authentication handles database errors gracefully."""
        mock_repository.get_by_email_or_username.side_effect = Exception("Database error")

        result = await user_service.authenticate_user(
            identifier="testuser@example.com",
            password="password",
        )

        assert result.success is False
        assert "unexpected error" in result.error.lower()

    async def test_verify_email_database_error(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_session: AsyncMock,
    ):
        """Test email verification handles database errors gracefully."""
        mock_repository.verify_email.side_effect = Exception("Database error")

        result = await user_service.verify_email(token="valid_token")

        assert result.success is False
        assert "unexpected error" in result.error.lower()
        mock_session.rollback.assert_called()

    async def test_reset_password_database_error(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
        mock_user: MagicMock,
        mock_session: AsyncMock,
    ):
        """Test password reset handles database errors gracefully."""
        mock_repository.get_by_reset_token.return_value = mock_user
        mock_repository.reset_password.side_effect = Exception("Database error")

        with patch("app.services.user_service.auth_service") as mock_auth:
            mock_auth.hash_password.return_value = "hashed"

            result = await user_service.reset_password(
                token="valid_token",
                new_password="NewSecureP@ssw0rd123!",
            )

            assert result.success is False
            assert "unexpected error" in result.error.lower()
            mock_session.rollback.assert_called()
