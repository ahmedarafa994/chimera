"""
Production Security Environment Configuration
Secure environment variable management and validation
"""

import os
import secrets


class SecurityEnvironment:
    """Manages security-related environment configuration"""

    @staticmethod
    def get_allowed_origins() -> list[str]:
        """Get allowed CORS origins from environment or use secure defaults"""
        origins_str = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3001,http://localhost:8080,http://127.0.0.1:3001,http://127.0.0.1:8080",
        )
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]

    @staticmethod
    def get_api_key() -> str:
        """Get secure API key from environment or generate one"""
        api_key = os.getenv("CHIMERA_API_KEY")
        if not api_key:
            api_key = secrets.token_urlsafe(32)
            print(f"SECURITY WARNING: Generated new API key: {api_key}")
            print("Set CHIMERA_API_KEY environment variable in production!")
        return api_key

    @staticmethod
    def get_rate_limit() -> str:
        """Get rate limit configuration"""
        return os.getenv("RATE_LIMIT_PER_MINUTE", "60")

    @staticmethod
    def is_debug_mode() -> bool:
        """Check if debug mode is enabled (should be False in production)"""
        return os.getenv("DEBUG", "false").lower() == "true"

    @staticmethod
    def validate_production_environment() -> list[str]:
        """Validate production environment security settings"""
        warnings = []

        # Check for development defaults in production
        if os.getenv("ENVIRONMENT") == "production":
            if SecurityEnvironment.is_debug_mode():
                warnings.append("DEBUG mode is enabled in production")

            if not os.getenv("CHIMERA_API_KEY"):
                warnings.append("No CHIMERA_API_KEY set in production")

            allowed_origins = SecurityEnvironment.get_allowed_origins()
            localhost_origins = [
                origin
                for origin in allowed_origins
                if "localhost" in origin or "127.0.0.1" in origin
            ]
            if localhost_origins:
                warnings.append(f"Localhost origins found in production: {localhost_origins}")

        return warnings
