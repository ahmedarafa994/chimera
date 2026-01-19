"""
Authentication and RBAC Module
Production-grade authentication with JWT and Role-Based Access Control
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

# Redis for production token revocation persistence
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Role and Permission Definitions
# =============================================================================


class Role(str, Enum):
    """User roles with hierarchical permissions"""

    ADMIN = "admin"  # Full system access
    RESEARCHER = "researcher"  # Create/edit campaigns, execute jailbreaks (no user management)
    OPERATOR = "operator"  # Execute and manage operations
    DEVELOPER = "developer"  # API access for development
    VIEWER = "viewer"  # Read-only access
    API_CLIENT = "api_client"  # Programmatic API access


class Permission(str, Enum):
    """Granular permissions for access control"""

    # Read permissions
    READ_PROMPTS = "read:prompts"
    READ_TECHNIQUES = "read:techniques"
    READ_PROVIDERS = "read:providers"
    READ_METRICS = "read:metrics"
    READ_LOGS = "read:logs"

    # Write permissions
    WRITE_PROMPTS = "write:prompts"
    WRITE_TECHNIQUES = "write:techniques"
    WRITE_PROVIDERS = "write:providers"

    # Execute permissions
    EXECUTE_TRANSFORM = "execute:transform"
    EXECUTE_ENHANCE = "execute:enhance"
    EXECUTE_JAILBREAK = "execute:jailbreak"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_AUDIT = "admin:audit"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, list[Permission]] = {
    Role.ADMIN: list(Permission),  # All permissions
    Role.RESEARCHER: [
        # Researcher can create/edit campaigns but NOT manage users or system
        # Read permissions (like Viewer)
        Permission.READ_PROMPTS,
        Permission.READ_TECHNIQUES,
        Permission.READ_PROVIDERS,
        Permission.READ_METRICS,
        Permission.READ_LOGS,
        # Write permissions for campaign creation/editing
        Permission.WRITE_PROMPTS,
        Permission.WRITE_TECHNIQUES,
        # Execute permissions for jailbreak research
        Permission.EXECUTE_TRANSFORM,
        Permission.EXECUTE_ENHANCE,
        Permission.EXECUTE_JAILBREAK,
        # NOTE: No ADMIN_USERS, ADMIN_SYSTEM, or ADMIN_AUDIT permissions
        # NOTE: No WRITE_PROVIDERS (system configuration)
    ],
    Role.OPERATOR: [
        Permission.READ_PROMPTS,
        Permission.READ_TECHNIQUES,
        Permission.READ_PROVIDERS,
        Permission.READ_METRICS,
        Permission.WRITE_PROMPTS,
        Permission.WRITE_TECHNIQUES,
        Permission.EXECUTE_TRANSFORM,
        Permission.EXECUTE_ENHANCE,
        Permission.EXECUTE_JAILBREAK,
    ],
    Role.DEVELOPER: [
        Permission.READ_PROMPTS,
        Permission.READ_TECHNIQUES,
        Permission.READ_PROVIDERS,
        Permission.EXECUTE_TRANSFORM,
        Permission.EXECUTE_ENHANCE,
    ],
    Role.VIEWER: [
        Permission.READ_PROMPTS,
        Permission.READ_TECHNIQUES,
        Permission.READ_PROVIDERS,
        Permission.READ_METRICS,
    ],
    Role.API_CLIENT: [
        Permission.READ_PROMPTS,
        Permission.READ_TECHNIQUES,
        Permission.READ_PROVIDERS,
        Permission.EXECUTE_TRANSFORM,
        Permission.EXECUTE_ENHANCE,
        Permission.EXECUTE_JAILBREAK,
    ],
}


# =============================================================================
# Pydantic Models
# =============================================================================


class TokenPayload(BaseModel):
    """JWT token payload"""

    sub: str  # User ID
    role: str
    permissions: list[str]
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation
    type: str  # "access" or "refresh"


class TokenResponse(BaseModel):
    """Token response model - OAuth 2.0 compliant."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"  # Capital B per RFC 6750
    expires_in: int
    refresh_expires_in: int = 604800  # 7 days default

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_expires_in": 604800,
            }
        }


class UserCredentials(BaseModel):
    """User login credentials"""

    username: str
    password: str


# =============================================================================
# Security Configuration
# =============================================================================

# Password hashing - use argon2 (recommended by OWASP) with bcrypt fallback
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthConfig:
    """Authentication configuration"""

    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET")
        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(64)
            logger.warning("JWT_SECRET not set - using generated key (not for production!)")

        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire = timedelta(hours=int(os.getenv("JWT_EXPIRATION_HOURS", "1")))
        self.refresh_token_expire = timedelta(days=7)


auth_config = AuthConfig()


# =============================================================================
# Authentication Service
# =============================================================================


class AuthService:
    """
    Authentication service with JWT and API key support.

    Token revocation is persisted to Redis in production for durability
    across server restarts. Falls back to in-memory storage in development.
    """

    def __init__(self, config: AuthConfig = None):
        self.config = config or auth_config
        self._revoked_tokens: set = set()  # Fallback for development
        self._redis_client = None
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis connection for token revocation persistence."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not installed - using in-memory token revocation")
            return

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("REDIS_URL not set - using in-memory token revocation (OK for development)")
            return

        try:
            self._redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self._redis_client.ping()
            logger.info("Redis connected for token revocation persistence")
        except Exception as e:
            logger.warning(f"Redis connection failed, falling back to in-memory: {e}")
            self._redis_client = None

    # -------------------------------------------------------------------------
    # Password Operations
    # -------------------------------------------------------------------------

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash"""
        return pwd_context.verify(plain_password, hashed_password)

    # -------------------------------------------------------------------------
    # API Key Operations
    # -------------------------------------------------------------------------

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify an API key using timing-safe comparison.

        Args:
            api_key: The API key to verify

        Returns:
            True if valid, False otherwise
        """
        valid_key = os.getenv("CHIMERA_API_KEY")
        if not valid_key:
            logger.error("CHIMERA_API_KEY not configured")
            return False

        # Timing-safe comparison to prevent timing attacks
        return secrets.compare_digest(api_key, valid_key)

    def generate_api_key(self) -> str:
        """Generate a new secure API key"""
        return secrets.token_urlsafe(32)

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for database lookup.

        Uses SHA-256 to match the hashing used when storing user API keys.

        Args:
            api_key: The plain API key to hash

        Returns:
            SHA-256 hex digest of the API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # JWT Operations
    # -------------------------------------------------------------------------

    def create_access_token(
        self, user_id: str, role: Role, additional_claims: dict[str, Any] | None = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: The user's unique identifier
            role: The user's role
            additional_claims: Additional claims to include

        Returns:
            Encoded JWT token
        """
        now = datetime.utcnow()
        permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]

        payload = {
            "sub": user_id,
            "role": role.value,
            "permissions": permissions,
            "exp": now + self.config.access_token_expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16),
            "type": "access",
        }

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def create_refresh_token(self, user_id: str, role: Role) -> str:
        """Create a JWT refresh token"""
        now = datetime.utcnow()

        payload = {
            "sub": user_id,
            "role": role.value,
            "exp": now + self.config.refresh_token_expire,
            "iat": now,
            "jti": secrets.token_urlsafe(16),
            "type": "refresh",
        }

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def create_tokens(self, user_id: str, role: Role) -> TokenResponse:
        """Create both access and refresh tokens"""
        access_token = self.create_access_token(user_id, role)
        refresh_token = self.create_refresh_token(user_id, role)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="Bearer",  # Capital B
            expires_in=int(self.config.access_token_expire.total_seconds()),
            refresh_expires_in=int(self.config.refresh_token_expire.total_seconds()),
        )

    def decode_token(self, token: str) -> TokenPayload:
        """
        Decode and validate a JWT token.

        Args:
            token: The JWT token to decode

        Returns:
            TokenPayload with decoded claims

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])

            # Check if token is revoked
            if self._is_token_revoked(payload.get("jti")):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked"
                )

            return TokenPayload(**payload)

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e!s}"
            )

    def _is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked (Redis or in-memory)."""
        if not jti:
            return False

        # Check Redis first (production)
        if self._redis_client:
            try:
                return self._redis_client.exists(f"revoked_token:{jti}") > 0
            except Exception as e:
                logger.error(f"Redis check failed, falling back to memory: {e}")

        # Fallback to in-memory check
        return jti in self._revoked_tokens

    def revoke_token(self, jti: str, expires_in_seconds: int = 86400):
        """
        Revoke a token by its JTI.

        Args:
            jti: The JWT ID to revoke
            expires_in_seconds: How long to keep the revocation (default: 24h)
        """
        # Store in Redis with automatic expiration (production)
        if self._redis_client:
            try:
                self._redis_client.setex(f"revoked_token:{jti}", expires_in_seconds, "1")
                logger.info(
                    f"Token {jti[:8]}... revoked in Redis (expires in {expires_in_seconds}s)"
                )
                return
            except Exception as e:
                logger.error(f"Redis revocation failed, using memory: {e}")

        # Fallback to in-memory (development)
        self._revoked_tokens.add(jti)
        logger.info(f"Token {jti[:8]}... revoked in memory (not persistent)")
        # Note: In-memory revocations are lost on restart

    def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """Create a new access token using a refresh token"""
        payload = self.decode_token(refresh_token)

        if payload.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token type"
            )

        role = Role(payload.role)
        return self.create_tokens(payload.sub, role)


# Global auth service instance
auth_service = AuthService()


# =============================================================================
# Dependency Injection Functions
# =============================================================================


async def get_api_key(api_key: str = Security(api_key_header)) -> str | None:
    """Extract API key from header"""
    return api_key


async def get_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> str | None:
    """Extract JWT token from Authorization header"""
    if credentials:
        return credentials.credentials
    return None


def _is_jwt_format(token: str) -> bool:
    """Check if a token looks like a JWT (has 3 dot-separated segments)."""
    if not token:
        return False
    parts = token.split(".")
    return len(parts) == 3


def _get_role_from_user_role(user_role: str) -> Role:
    """
    Map a user role string to a Role enum.

    Args:
        user_role: Role string from database (admin, researcher, viewer)

    Returns:
        Role enum value
    """
    role_mapping = {
        "admin": Role.ADMIN,
        "researcher": Role.RESEARCHER,
        "viewer": Role.VIEWER,
    }
    return role_mapping.get(user_role.lower(), Role.API_CLIENT)


async def get_current_user(
    request: Request = None,
    api_key: str = Depends(get_api_key),
    token: str = Depends(get_token),
) -> TokenPayload:
    """
    Get current authenticated user from API key or JWT token.

    Supports multiple authentication methods in order of priority:
    1. JWT Bearer token - for full user authentication
    2. Per-user API key (validated by middleware) - for programmatic access
    3. Global API key (X-API-Key header) - for backward compatibility

    The middleware (APIKeyMiddleware) validates API keys and stores user info
    in request.state for per-user API keys. This function checks that first.
    """
    # Import Request lazily to avoid circular imports in dependency injection

    # Try JWT token first (only if it looks like a JWT)
    if token and _is_jwt_format(token):
        return auth_service.decode_token(token)

    # Check if middleware has set user info for per-user API key authentication
    if request and hasattr(request, "state"):
        auth_type = getattr(request.state, "auth_type", None)

        if auth_type == "user_api_key":
            # Per-user API key was validated by middleware
            user_id = getattr(request.state, "user_id", None)
            getattr(request.state, "username", "unknown")
            role_str = getattr(request.state, "role", "viewer")

            if user_id:
                # Map the database role to our Role enum
                role = _get_role_from_user_role(role_str)
                permissions = [p.value for p in ROLE_PERMISSIONS.get(role, [])]

                return TokenPayload(
                    sub=str(user_id),
                    role=role.value,
                    permissions=permissions,
                    exp=datetime.utcnow() + timedelta(hours=1),
                    iat=datetime.utcnow(),
                    jti=f"user_api_key_{user_id}",
                    type="user_api_key",
                )

        elif auth_type == "global_api_key":
            # Global API key was validated by middleware
            return TokenPayload(
                sub="api_client",
                role=Role.API_CLIENT.value,
                permissions=[p.value for p in ROLE_PERMISSIONS[Role.API_CLIENT]],
                exp=datetime.utcnow() + timedelta(hours=1),
                iat=datetime.utcnow(),
                jti="global_api_key_auth",
                type="api_key",
            )

    # Check if the Bearer token is actually an API key (for backward compatibility)
    if token and not _is_jwt_format(token) and auth_service.verify_api_key(token):
        # Create a synthetic token payload for API key auth via Bearer
        return TokenPayload(
            sub="api_client",
            role=Role.API_CLIENT.value,
            permissions=[p.value for p in ROLE_PERMISSIONS[Role.API_CLIENT]],
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            jti="api_key_auth",
            type="api_key",
        )

    # Fall back to X-API-Key header (for backward compatibility)
    if api_key and auth_service.verify_api_key(api_key):
        # Create a synthetic token payload for API key auth
        return TokenPayload(
            sub="api_client",
            role=Role.API_CLIENT.value,
            permissions=[p.value for p in ROLE_PERMISSIONS[Role.API_CLIENT]],
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            jti="api_key_auth",
            type="api_key",
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: Permission):
    """
    Dependency to require a specific permission.

    Usage:
        @app.get("/admin/users")
        async def get_users(user: TokenPayload = Depends(require_permission(Permission.ADMIN_USERS))):
            ...
    """

    async def permission_checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if permission.value not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value} required",
            )
        return user

    return permission_checker


def require_role(role: Role):
    """
    Dependency to require a specific role.

    Usage:
        @app.get("/admin/system")
        async def admin_only(user: TokenPayload = Depends(require_role(Role.ADMIN))):
            ...
    """

    async def role_checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role != role.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail=f"Role denied: {role.value} required"
            )
        return user

    return role_checker


def require_any_role(roles: list[Role]):
    """
    Dependency to require any of the specified roles.
    """

    async def role_checker(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if user.role not in [r.value for r in roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: required roles: {[r.value for r in roles]}",
            )
        return user

    return role_checker


# =============================================================================
# Audit Logging
# =============================================================================


class AuditLogger:
    """Simple audit logger for security events"""

    def __init__(self):
        self.logger = logging.getLogger("audit")

    def log_authentication(
        self, user_id: str, success: bool, method: str, ip_address: str | None = None
    ):
        """Log authentication attempt"""
        self.logger.info(f"AUTH: user={user_id} success={success} method={method} ip={ip_address}")

    def log_authorization(self, user_id: str, resource: str, action: str, allowed: bool):
        """Log authorization decision"""
        self.logger.info(
            f"AUTHZ: user={user_id} resource={resource} action={action} allowed={allowed}"
        )

    def log_api_access(self, user_id: str, endpoint: str, method: str, status_code: int):
        """Log API access"""
        self.logger.info(
            f"API: user={user_id} endpoint={endpoint} method={method} status={status_code}"
        )


audit_logger = AuditLogger()
