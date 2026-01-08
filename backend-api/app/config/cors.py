"""
CORS Configuration
Secure cross-origin request handling
"""

from pydantic_settings import BaseSettings


class CORSSettings(BaseSettings):
    """CORS configuration settings"""

    # Allowed origins
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]

    # Allow credentials
    CORS_ALLOW_CREDENTIALS: bool = True

    # Allowed methods
    CORS_ALLOW_METHODS: list[str] = [
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "HEAD",
    ]

    # Allowed headers
    CORS_ALLOW_HEADERS: list[str] = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Request-ID",
        "X-Tenant-ID",
        "X-API-Key",
        "X-Idempotency-Key",
        "Cache-Control",
    ]

    # Exposed headers
    CORS_EXPOSE_HEADERS: list[str] = [
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ]

    # Max age for preflight cache
    CORS_MAX_AGE: int = 600  # 10 minutes

    class Config:
        env_prefix = ""
        case_sensitive = True


cors_settings = CORSSettings()


def get_cors_config() -> dict:
    """Get CORS middleware configuration"""
    return {
        "allow_origins": cors_settings.CORS_ORIGINS,
        "allow_credentials": cors_settings.CORS_ALLOW_CREDENTIALS,
        "allow_methods": cors_settings.CORS_ALLOW_METHODS,
        "allow_headers": cors_settings.CORS_ALLOW_HEADERS,
        "expose_headers": cors_settings.CORS_EXPOSE_HEADERS,
        "max_age": cors_settings.CORS_MAX_AGE,
    }
