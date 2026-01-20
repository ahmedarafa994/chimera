"""CORS Configuration Module
Secure Cross-Origin Resource Sharing configuration for production.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def get_allowed_origins() -> list[str]:
    """Get allowed CORS origins from environment.

    In production, this should be explicitly configured.
    """
    origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3001,http://localhost:8080")

    origins = [origin.strip() for origin in origins_str.split(",") if origin.strip()]

    # Validate origins - no wildcards in production
    environment = os.getenv("ENVIRONMENT", "development")
    if environment == "production":
        for origin in origins:
            if origin == "*":
                msg = (
                    "Wildcard CORS origin not allowed in production. "
                    "Set ALLOWED_ORIGINS to specific domains."
                )
                raise ValueError(
                    msg,
                )

    return origins


def configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware with secure settings.

    Security considerations:
    - Explicit origin list (no wildcards in production)
    - Explicit HTTP methods
    - Explicit headers
    - Credentials support with origin restrictions
    """
    allowed_origins = get_allowed_origins()

    # Allowed HTTP methods - be explicit
    allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]

    # Allowed headers - be explicit about what's needed
    allowed_headers = [
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-ID",
        "Accept",
        "Accept-Language",
        "Content-Language",
    ]

    # Headers to expose to the client
    expose_headers = [
        "X-Request-ID",
        "X-Response-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        expose_headers=expose_headers,
        max_age=600,  # Cache preflight requests for 10 minutes
    )


class CORSConfig:
    """CORS configuration class for more granular control."""

    def __init__(self) -> None:
        self.allowed_origins = get_allowed_origins()
        self.allow_credentials = True
        self.allow_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = [
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
        ]
        self.expose_headers = [
            "X-Request-ID",
            "X-Response-Time",
        ]
        self.max_age = 600

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is in the allowed list."""
        return origin in self.allowed_origins

    def add_origin(self, origin: str) -> None:
        """Add an origin to the allowed list (runtime)."""
        if origin not in self.allowed_origins:
            self.allowed_origins.append(origin)

    def remove_origin(self, origin: str) -> None:
        """Remove an origin from the allowed list (runtime)."""
        if origin in self.allowed_origins:
            self.allowed_origins.remove(origin)


# Global CORS config instance
cors_config = CORSConfig()
