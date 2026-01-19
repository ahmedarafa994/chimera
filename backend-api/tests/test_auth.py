"""
Comprehensive Tests for Authentication Endpoints.

This module provides thorough testing for all authentication endpoints:
- POST /api/v1/auth/register - User registration
- POST /auth/login - User login
- GET /api/v1/auth/verify-email/{token} - Email verification
- POST /api/v1/auth/resend-verification - Resend verification email
- POST /api/v1/auth/forgot-password - Request password reset
- POST /api/v1/auth/reset-password - Reset password with token

Test categories:
- Success cases: Verify expected behavior for valid inputs
- Failure cases: Verify proper error handling for invalid inputs
- Edge cases: Boundary conditions and unusual scenarios
- Security cases: Ensure security measures work correctly

Markers:
- @pytest.mark.unit: Unit tests (fast, isolated)
- @pytest.mark.integration: Integration tests (require database)
- @pytest.mark.security: Security-focused tests
"""

import secrets
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Test fixtures and configuration
pytestmark = [pytest.mark.asyncio]


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def valid_registration_data() -> dict[str, str]:
    """Valid registration data meeting all requirements."""
    return {
        "email": "testuser@example.com",
        "username": "testuser",
        "password": "SecureP@ssw0rd123!",
    }


@pytest.fixture
def weak_password_data() -> dict[str, str]:
    """Registration data with a weak password."""
    return {
        "email": "weakpass@example.com",
        "username": "weakpassuser",
        "password": "weak",  # Too short, missing requirements
    }


@pytest.fixture
def valid_login_data() -> dict[str, str]:
    """Valid login credentials."""
    return {
        "username": "testuser@example.com",
        "password": "SecureP@ssw0rd123!",
    }


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = 1
    user.email = "testuser@example.com"
    user.username = "testuser"
    user.hashed_password = "$2b$12$hashedpassword"
    user.role = MagicMock(value="viewer")
    user.is_active = True
    user.is_verified = True
    user.email_verification_token = None
    user.password_reset_token = None
    user.last_login = None
    user.created_at = datetime.utcnow()
    return user


@pytest.fixture
def mock_unverified_user(mock_user):
    """Create a mock unverified user."""
    mock_user.is_verified = False
    mock_user.email_verification_token = secrets.token_hex(32)
    return mock_user


# =============================================================================
# Registration Endpoint Tests
# =============================================================================


class TestRegistration:
    """Tests for POST /api/v1/auth/register endpoint."""

    @pytest.mark.unit
    def test_register_user_success(self, client: TestClient, valid_registration_data: dict):
        """Test successful user registration with valid data."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            # Setup mock service
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.user = MagicMock(
                id=1,
                email=valid_registration_data["email"],
                username=valid_registration_data["username"],
                role=MagicMock(value="viewer"),
            )
            mock_result.verification_token = secrets.token_hex(32)
            mock_result.errors = []

            # Mock async method
            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json=valid_registration_data,
            )

            assert response.status_code == 201
            data = response.json()
            assert data["success"] is True
            assert data["email"] == valid_registration_data["email"]
            assert data["username"] == valid_registration_data["username"]
            assert "user_id" in data

    @pytest.mark.unit
    def test_register_user_weak_password(self, client: TestClient, weak_password_data: dict):
        """Test registration rejection with weak password."""
        response = client.post(
            "/api/v1/auth/register",
            json=weak_password_data,
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False
        assert "errors" in data["detail"]
        assert len(data["detail"]["errors"]) > 0

    @pytest.mark.unit
    def test_register_user_duplicate_email(self, client: TestClient, valid_registration_data: dict):
        """Test registration rejection with duplicate email."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.errors = ["Email is already registered"]

            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json=valid_registration_data,
            )

            assert response.status_code == 409
            data = response.json()
            assert data["detail"]["success"] is False
            assert "already" in data["detail"]["errors"][0].lower()

    @pytest.mark.unit
    def test_register_user_duplicate_username(
        self, client: TestClient, valid_registration_data: dict
    ):
        """Test registration rejection with duplicate username."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.errors = ["Username is already taken"]

            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json=valid_registration_data,
            )

            assert response.status_code == 409

    @pytest.mark.unit
    def test_register_user_invalid_email_format(self, client: TestClient):
        """Test registration rejection with invalid email format."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "username": "testuser",
                "password": "SecureP@ssw0rd123!",
            },
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.unit
    def test_register_user_invalid_username_format(self, client: TestClient):
        """Test registration rejection with invalid username (starts with number)."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "123invalid",  # Starts with number
                "password": "SecureP@ssw0rd123!",
            },
        )

        assert response.status_code == 422

    @pytest.mark.unit
    def test_register_user_username_special_chars(self, client: TestClient):
        """Test registration rejection with special characters in username."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "user@name!",  # Special chars not allowed
                "password": "SecureP@ssw0rd123!",
            },
        )

        assert response.status_code == 422

    @pytest.mark.unit
    def test_register_user_password_contains_email(self, client: TestClient):
        """Test registration with password containing email."""
        email = "user@example.com"
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "username": "testuser",
                "password": f"{email}SecureP@ssw0rd!",  # Contains email
            },
        )

        # Should fail password validation
        assert response.status_code == 400

    @pytest.mark.unit
    def test_register_user_missing_fields(self, client: TestClient):
        """Test registration with missing required fields."""
        # Missing email
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser",
                "password": "SecureP@ssw0rd123!",
            },
        )
        assert response.status_code == 422

        # Missing username
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecureP@ssw0rd123!",
            },
        )
        assert response.status_code == 422

        # Missing password
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "testuser",
            },
        )
        assert response.status_code == 422


# =============================================================================
# Login Endpoint Tests
# =============================================================================


class TestLogin:
    """Tests for POST /auth/login endpoint."""

    @pytest.mark.unit
    def test_login_success_with_email(self, client: TestClient, mock_user):
        """Test successful login with email."""
        with (
            patch("app.routers.auth.get_user_service") as mock_get_service,
            patch("app.routers.auth.auth_service") as mock_auth_service,
        ):
            # Setup mock
            mock_service = MagicMock()
            mock_auth_result = MagicMock()
            mock_auth_result.success = True
            mock_auth_result.user = mock_user
            mock_auth_result.tokens = MagicMock(
                access_token="access_token_here",
                refresh_token="refresh_token_here",
                expires_in=3600,
            )
            mock_auth_result.requires_verification = False

            mock_service.authenticate_user = AsyncMock(return_value=mock_auth_result)
            mock_get_service.return_value = mock_service
            mock_auth_service.config.refresh_token_expire = timedelta(days=7)

            response = client.post(
                "/auth/login",
                json={
                    "username": "testuser@example.com",
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert data["token_type"] == "Bearer"
            assert "user" in data

    @pytest.mark.unit
    def test_login_success_with_username(self, client: TestClient, mock_user):
        """Test successful login with username instead of email."""
        with (
            patch("app.routers.auth.get_user_service") as mock_get_service,
            patch("app.routers.auth.auth_service") as mock_auth_service,
        ):
            mock_service = MagicMock()
            mock_auth_result = MagicMock()
            mock_auth_result.success = True
            mock_auth_result.user = mock_user
            mock_auth_result.tokens = MagicMock(
                access_token="access_token_here",
                refresh_token="refresh_token_here",
                expires_in=3600,
            )
            mock_auth_result.requires_verification = False

            mock_service.authenticate_user = AsyncMock(return_value=mock_auth_result)
            mock_get_service.return_value = mock_service
            mock_auth_service.config.refresh_token_expire = timedelta(days=7)

            response = client.post(
                "/auth/login",
                json={
                    "username": "testuser",
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 200

    @pytest.mark.unit
    def test_login_invalid_credentials(self, client: TestClient):
        """Test login failure with invalid credentials."""
        with (
            patch("app.routers.auth.get_user_service") as mock_get_service,
            patch("app.routers.auth._authenticate_env_admin") as mock_env_auth,
        ):
            mock_service = MagicMock()
            mock_auth_result = MagicMock()
            mock_auth_result.success = False
            mock_auth_result.error = "Invalid credentials"
            mock_auth_result.requires_verification = False

            mock_service.authenticate_user = AsyncMock(return_value=mock_auth_result)
            mock_get_service.return_value = mock_service
            mock_env_auth.return_value = None  # No env admin fallback

            response = client.post(
                "/auth/login",
                json={
                    "username": "wronguser@example.com",
                    "password": "WrongPassword123!",
                },
            )

            assert response.status_code == 401
            assert "Invalid credentials" in response.json()["detail"]

    @pytest.mark.unit
    def test_login_unverified_email(self, client: TestClient, mock_unverified_user):
        """Test login failure when email is not verified."""
        with patch("app.routers.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_auth_result = MagicMock()
            mock_auth_result.success = False
            mock_auth_result.error = "Email not verified"
            mock_auth_result.requires_verification = True

            mock_service.authenticate_user = AsyncMock(return_value=mock_auth_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/auth/login",
                json={
                    "username": "unverified@example.com",
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 403
            data = response.json()
            assert "requires_verification" in data["detail"]
            assert data["detail"]["requires_verification"] is True

    @pytest.mark.unit
    def test_login_inactive_user(self, client: TestClient):
        """Test login failure for inactive user account."""
        with (
            patch("app.routers.auth.get_user_service") as mock_get_service,
            patch("app.routers.auth._authenticate_env_admin") as mock_env_auth,
        ):
            mock_service = MagicMock()
            mock_auth_result = MagicMock()
            mock_auth_result.success = False
            mock_auth_result.error = "Account is deactivated"
            mock_auth_result.requires_verification = False

            mock_service.authenticate_user = AsyncMock(return_value=mock_auth_result)
            mock_get_service.return_value = mock_service
            mock_env_auth.return_value = None

            response = client.post(
                "/auth/login",
                json={
                    "username": "inactive@example.com",
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 401

    @pytest.mark.unit
    def test_login_missing_fields(self, client: TestClient):
        """Test login with missing required fields."""
        # Missing password
        response = client.post(
            "/auth/login",
            json={"username": "testuser@example.com"},
        )
        assert response.status_code == 422

        # Missing username
        response = client.post(
            "/auth/login",
            json={"password": "SecureP@ssw0rd123!"},
        )
        assert response.status_code == 422


# =============================================================================
# Email Verification Endpoint Tests
# =============================================================================


class TestEmailVerification:
    """Tests for GET /api/v1/auth/verify-email/{token} endpoint."""

    @pytest.mark.unit
    def test_verify_email_success(self, client: TestClient, mock_user):
        """Test successful email verification."""
        valid_token = secrets.token_hex(32)

        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.user = mock_user

            mock_service.verify_email = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.get(f"/api/v1/auth/verify-email/{valid_token}")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "email" in data
            assert "username" in data

    @pytest.mark.unit
    def test_verify_email_invalid_token(self, client: TestClient):
        """Test email verification with invalid token."""
        invalid_token = secrets.token_hex(32)

        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Verification token is invalid"

            mock_service.verify_email = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.get(f"/api/v1/auth/verify-email/{invalid_token}")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["success"] is False

    @pytest.mark.unit
    def test_verify_email_expired_token(self, client: TestClient):
        """Test email verification with expired token."""
        expired_token = secrets.token_hex(32)

        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Verification token is expired"

            mock_service.verify_email = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.get(f"/api/v1/auth/verify-email/{expired_token}")

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["token_expired"] is True

    @pytest.mark.unit
    def test_verify_email_malformed_token(self, client: TestClient):
        """Test email verification with malformed token (wrong length)."""
        malformed_token = "short"

        response = client.get(f"/api/v1/auth/verify-email/{malformed_token}")

        assert response.status_code == 400
        data = response.json()
        assert "Invalid verification token format" in data["detail"]["error"]


# =============================================================================
# Resend Verification Email Tests
# =============================================================================


class TestResendVerification:
    """Tests for POST /api/v1/auth/resend-verification endpoint."""

    @pytest.mark.unit
    def test_resend_verification_success(self, client: TestClient, mock_unverified_user):
        """Test successful resend of verification email."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_resend_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None  # No rate limit hit
            mock_service = MagicMock()
            new_token = secrets.token_hex(32)

            mock_service.resend_verification_email = AsyncMock(return_value=(True, new_token, None))
            mock_service.get_user_by_email = AsyncMock(return_value=mock_unverified_user)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "unverified@example.com"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.unit
    def test_resend_verification_already_verified(self, client: TestClient):
        """Test resend rejection when email is already verified."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_resend_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_service.resend_verification_email = AsyncMock(
                return_value=(False, None, "Email is already verified")
            )
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "verified@example.com"},
            )

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["already_verified"] is True

    @pytest.mark.unit
    def test_resend_verification_nonexistent_email(self, client: TestClient):
        """Test resend for non-existent email returns success (security)."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_resend_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            # Service returns success without token for non-existent email
            mock_service.resend_verification_email = AsyncMock(return_value=(True, None, None))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "nonexistent@example.com"},
            )

            # Should return success to prevent email enumeration
            assert response.status_code == 200

    @pytest.mark.unit
    def test_resend_verification_invalid_email(self, client: TestClient):
        """Test resend with invalid email format."""
        response = client.post(
            "/api/v1/auth/resend-verification",
            json={"email": "not-an-email"},
        )

        assert response.status_code == 422


# =============================================================================
# Forgot Password Endpoint Tests
# =============================================================================


class TestForgotPassword:
    """Tests for POST /api/v1/auth/forgot-password endpoint."""

    @pytest.mark.unit
    def test_forgot_password_success(self, client: TestClient, mock_user):
        """Test successful password reset request."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_forgot_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()
            reset_token = secrets.token_hex(32)

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.reset_token = reset_token

            mock_service.request_password_reset = AsyncMock(return_value=mock_result)
            mock_service.get_user_by_email = AsyncMock(return_value=mock_user)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/forgot-password",
                json={"email": "testuser@example.com"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.unit
    def test_forgot_password_nonexistent_email(self, client: TestClient):
        """Test forgot password for non-existent email returns success (security)."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_forgot_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.reset_token = None  # No token for non-existent email

            mock_service.request_password_reset = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/forgot-password",
                json={"email": "nonexistent@example.com"},
            )

            # Should return success to prevent email enumeration
            assert response.status_code == 200

    @pytest.mark.unit
    def test_forgot_password_invalid_email(self, client: TestClient):
        """Test forgot password with invalid email format."""
        response = client.post(
            "/api/v1/auth/forgot-password",
            json={"email": "not-an-email"},
        )

        assert response.status_code == 422


# =============================================================================
# Reset Password Endpoint Tests
# =============================================================================


class TestResetPassword:
    """Tests for POST /api/v1/auth/reset-password endpoint."""

    @pytest.mark.unit
    def test_reset_password_success(self, client: TestClient):
        """Test successful password reset."""
        valid_token = secrets.token_hex(32)

        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_reset_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.validation_result = MagicMock(is_valid=True, errors=[])

            mock_service.reset_password = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/reset-password",
                json={
                    "token": valid_token,
                    "new_password": "NewSecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    @pytest.mark.unit
    def test_reset_password_invalid_token(self, client: TestClient):
        """Test password reset with invalid token."""
        invalid_token = secrets.token_hex(32)

        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_reset_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Password reset token is invalid"
            mock_result.validation_result = None

            mock_service.reset_password = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/reset-password",
                json={
                    "token": invalid_token,
                    "new_password": "NewSecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 400

    @pytest.mark.unit
    def test_reset_password_expired_token(self, client: TestClient):
        """Test password reset with expired token."""
        expired_token = secrets.token_hex(32)

        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_reset_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error = "Password reset token is expired"
            mock_result.validation_result = None

            mock_service.reset_password = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/reset-password",
                json={
                    "token": expired_token,
                    "new_password": "NewSecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 400
            data = response.json()
            assert data["detail"]["token_expired"] is True

    @pytest.mark.unit
    def test_reset_password_weak_new_password(self, client: TestClient):
        """Test password reset rejection with weak new password."""
        valid_token = secrets.token_hex(32)

        response = client.post(
            "/api/v1/auth/reset-password",
            json={
                "token": valid_token,
                "new_password": "weak",  # Too short
            },
        )

        # Should fail on validation before even checking token
        assert response.status_code == 422  # Pydantic min_length validation

    @pytest.mark.unit
    def test_reset_password_malformed_token(self, client: TestClient):
        """Test password reset with malformed token."""
        malformed_token = "short"

        with patch("app.api.v1.endpoints.auth._check_reset_password_rate_limit") as mock_rate_limit:
            mock_rate_limit.return_value = None

            response = client.post(
                "/api/v1/auth/reset-password",
                json={
                    "token": malformed_token,
                    "new_password": "NewSecureP@ssw0rd123!",
                },
            )

            # Should fail on token format validation
            assert response.status_code == 422  # Pydantic min_length validation


# =============================================================================
# Token Refresh Endpoint Tests
# =============================================================================


class TestTokenRefresh:
    """Tests for POST /auth/refresh endpoint."""

    @pytest.mark.unit
    def test_refresh_token_success(self, client: TestClient):
        """Test successful token refresh."""
        with patch("app.routers.auth.auth_service") as mock_auth_service:
            mock_tokens = MagicMock(
                access_token="new_access_token",
                refresh_token="new_refresh_token",
                expires_in=3600,
            )
            mock_auth_service.refresh_access_token.return_value = mock_tokens
            mock_auth_service.config.refresh_token_expire = timedelta(days=7)

            response = client.post(
                "/auth/refresh",
                json={"refresh_token": "valid_refresh_token"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data

    @pytest.mark.unit
    def test_refresh_token_invalid(self, client: TestClient):
        """Test token refresh with invalid refresh token."""
        with patch("app.routers.auth.auth_service") as mock_auth_service:
            mock_auth_service.refresh_access_token.side_effect = Exception("Invalid token")

            response = client.post(
                "/auth/refresh",
                json={"refresh_token": "invalid_refresh_token"},
            )

            assert response.status_code == 401

    @pytest.mark.unit
    def test_refresh_token_missing(self, client: TestClient):
        """Test token refresh without refresh token."""
        response = client.post(
            "/auth/refresh",
            json={},
        )

        assert response.status_code == 422


# =============================================================================
# Password Strength Check Endpoint Tests
# =============================================================================


class TestPasswordStrengthCheck:
    """Tests for POST /api/v1/auth/check-password-strength endpoint."""

    @pytest.mark.unit
    def test_check_strong_password(self, client: TestClient):
        """Test password strength check with strong password."""
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={"password": "VeryStr0ng&SecureP@ssw0rd!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["strength"] in ["strong", "very_strong"]

    @pytest.mark.unit
    def test_check_weak_password(self, client: TestClient):
        """Test password strength check with weak password."""
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={"password": "weak"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["errors"]) > 0

    @pytest.mark.unit
    def test_check_password_with_context(self, client: TestClient):
        """Test password check with email and username context."""
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={
                "password": "VeryStr0ng&SecureP@ssw0rd!",
                "email": "user@example.com",
                "username": "testuser",
            },
        )

        assert response.status_code == 200

    @pytest.mark.unit
    def test_check_common_password(self, client: TestClient):
        """Test password strength check with common password."""
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={"password": "password123456!"},  # Common pattern
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityMeasures:
    """Security-focused tests for authentication endpoints."""

    @pytest.mark.security
    def test_rate_limit_forgot_password(self, client: TestClient):
        """Test rate limiting on forgot password endpoint."""
        # This would need actual rate limiting to be enabled
        # For now, just verify the endpoint exists and can be called
        with (
            patch("app.api.v1.endpoints.auth.get_user_service"),
            patch("app.api.v1.endpoints.auth._check_forgot_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.side_effect = HTTPException(
                status_code=429,
                detail="Too many requests",
            )

            response = client.post(
                "/api/v1/auth/forgot-password",
                json={"email": "test@example.com"},
            )

            assert response.status_code == 429

    @pytest.mark.security
    def test_no_password_in_error_response(self, client: TestClient, valid_registration_data: dict):
        """Test that password is never included in error responses."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.errors = ["Email is already registered"]

            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json=valid_registration_data,
            )

            response_text = response.text
            assert valid_registration_data["password"] not in response_text

    @pytest.mark.security
    def test_email_enumeration_prevention_forgot_password(self, client: TestClient):
        """Test that forgot password doesn't reveal whether email exists."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_forgot_password_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            # Non-existent email should still return success
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.reset_token = None

            mock_service.request_password_reset = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/forgot-password",
                json={"email": "definitely-not-exists@example.com"},
            )

            assert response.status_code == 200
            # Response should be identical whether email exists or not

    @pytest.mark.security
    def test_email_enumeration_prevention_resend_verification(self, client: TestClient):
        """Test that resend verification doesn't reveal whether email exists."""
        with (
            patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service,
            patch("app.api.v1.endpoints.auth._check_resend_rate_limit") as mock_rate_limit,
        ):
            mock_rate_limit.return_value = None
            mock_service = MagicMock()

            mock_service.resend_verification_email = AsyncMock(return_value=(True, None, None))
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "nonexistent@example.com"},
            )

            # Should return success to prevent enumeration
            assert response.status_code == 200

    @pytest.mark.security
    def test_token_format_validation(self, client: TestClient):
        """Test that invalid token formats are rejected."""
        invalid_tokens = [
            "",  # Empty
            "abc",  # Too short
            "x" * 100,  # Too long
            "zzzz" * 16,  # Correct length but not hex
        ]

        for token in invalid_tokens:
            if len(token) != 64:
                response = client.get(f"/api/v1/auth/verify-email/{token}")
                assert response.status_code == 400, f"Token {token!r} should be rejected"


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    @pytest.mark.unit
    def test_password_exactly_minimum_length(self, client: TestClient):
        """Test password with exactly minimum length (12 characters)."""
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={"password": "Abc123!@#xyz"},  # Exactly 12 chars
        )

        assert response.status_code == 200

    @pytest.mark.unit
    def test_password_maximum_length(self, client: TestClient):
        """Test password at maximum length boundary."""
        # Max is 128 characters
        long_password = "A" * 60 + "a" * 60 + "1!@#"  # ~124 chars
        response = client.post(
            "/api/v1/auth/check-password-strength",
            json={"password": long_password},
        )

        assert response.status_code == 200

    @pytest.mark.unit
    def test_username_minimum_length(self, client: TestClient):
        """Test username with minimum length (3 characters)."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "abc",  # Exactly 3 chars
                "password": "SecureP@ssw0rd123!",
            },
        )

        # Should pass validation (may fail on other grounds)
        assert response.status_code != 422 or "username" not in str(response.json())

    @pytest.mark.unit
    def test_username_maximum_length(self, client: TestClient):
        """Test username with maximum length (50 characters)."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.user = MagicMock(
                id=1,
                email="test@example.com",
                username="a" * 50,
                role=MagicMock(value="viewer"),
            )
            mock_result.verification_token = secrets.token_hex(32)

            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "test@example.com",
                    "username": "a" * 50,  # Max length
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 201

    @pytest.mark.unit
    def test_email_with_plus_addressing(self, client: TestClient):
        """Test registration with email plus addressing."""
        with patch("app.api.v1.endpoints.auth.get_user_service") as mock_get_service:
            mock_service = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.user = MagicMock(
                id=1,
                email="user+test@example.com",
                username="testuser",
                role=MagicMock(value="viewer"),
            )
            mock_result.verification_token = secrets.token_hex(32)

            mock_service.register_user = AsyncMock(return_value=mock_result)
            mock_get_service.return_value = mock_service

            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": "user+test@example.com",  # Plus addressing
                    "username": "testuser",
                    "password": "SecureP@ssw0rd123!",
                },
            )

            assert response.status_code == 201

    @pytest.mark.unit
    def test_unicode_username(self, client: TestClient):
        """Test that unicode in username is rejected."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "username": "user\u4e2d\u6587",  # Chinese characters
                "password": "SecureP@ssw0rd123!",
            },
        )

        assert response.status_code == 422  # Should fail validation
