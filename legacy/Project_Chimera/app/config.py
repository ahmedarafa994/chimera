import os
import secrets

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Config:
    """Base configuration."""

    # API Authentication - Use environment variable or generate a secure random key
    API_KEY = os.getenv("CHIMERA_API_KEY") or secrets.token_urlsafe(32)

    # Validate API key strength
    if len(API_KEY) < 16:
        raise ValueError("CHIMERA_API_KEY must be at least 16 characters long for security")

    # Database
    # Using absolute path for SQLite to avoid issues with current working directory
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'chimera_logs.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Application settings
    DEBUG = False
    TESTING = False

    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or secrets.token_hex(32)


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False

    # Production must have API key set via environment
    if not os.getenv("CHIMERA_API_KEY"):
        raise ValueError("CHIMERA_API_KEY must be set in production environment")


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


# Dictionary to map environment names to config classes
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
