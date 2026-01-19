"""
Unified configuration management system for Project Chimera.
Provides environment-based configuration with validation and secure secret management.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    url: str = "sqlite:///chimera_logs.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    api_key_required: bool = True
    default_api_key: str = "chimera_default_key_change_in_production"
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3001", "http://localhost:8080"]
    )


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    default_provider: str = "openai"
    timeout_seconds: int = 30
    max_retries: int = 3
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CacheConfig:
    """Cache configuration settings."""

    enabled: bool = True
    type: str = "simple"  # simple, redis, memcached
    ttl_seconds: int = 300
    redis_url: str | None = None


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""

    enabled: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000
    memory_monitoring: bool = True


class Settings:
    """
    Unified settings manager with environment variable support and validation.
    """

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or self._find_config_file()
        self._config_data = {}
        self._load_configuration()

    def _find_config_file(self) -> str | None:
        """Find configuration file in standard locations."""
        possible_locations = [
            "config.yaml",
            "config.yml",
            "config.json",
            ".chimera/config.yaml",
            os.path.expanduser("~/.chimera/config.yaml"),
        ]

        for location in possible_locations:
            if os.path.exists(location):
                return location
        return None

    def _load_configuration(self):
        """Load configuration from file and environment variables."""
        # Load from file if exists
        if self.config_file:
            try:
                with open(self.config_file) as f:
                    if self.config_file.endswith((".yml", ".yaml")):
                        import yaml

                        self._config_data = yaml.safe_load(f) or {}
                    else:
                        self._config_data = json.load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_file}: {e}")
                self._config_data = {}

        # Override with environment variables
        self._load_env_variables()

    def _load_env_variables(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "CHIMERA_API_KEY": ("security", "default_api_key"),
            "CHIMERA_DEBUG": ("app", "debug"),
            "CHIMERA_DATABASE_URL": ("database", "url"),
            "CHIMERA_CACHE_ENABLED": ("cache", "enabled"),
            "CHIMERA_RATE_LIMIT": ("security", "rate_limit_per_minute"),
            "CHIMERA_DEFAULT_PROVIDER": ("llm", "default_provider"),
            "CHIMERA_CORS_ORIGINS": ("security", "cors_origins"),
            "CHIMERA_PERFORMANCE_MONITORING": ("performance", "enabled"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_data:
                    self._config_data[section] = {}

                # Convert string values to appropriate types
                if key in ["enabled", "debug"]:
                    self._config_data[section][key] = value.lower() in ("true", "1", "yes")
                elif key in ["rate_limit_per_minute", "timeout_seconds", "max_retries"]:
                    try:
                        self._config_data[section][key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                elif key == "cors_origins":
                    self._config_data[section][key] = [
                        origin.strip() for origin in value.split(",")
                    ]
                else:
                    self._config_data[section][key] = value

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config_data.get("database", {})
        return DatabaseConfig(
            url=db_config.get(
                "url", os.getenv("CHIMERA_DATABASE_URL", "sqlite:///chimera_logs.db")
            ),
            echo=db_config.get("echo", False),
            pool_size=db_config.get("pool_size", 5),
            max_overflow=db_config.get("max_overflow", 10),
        )

    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config = self._config_data.get("security", {})
        return SecurityConfig(
            api_key_required=security_config.get("api_key_required", True),
            default_api_key=security_config.get(
                "default_api_key",
                os.getenv("CHIMERA_API_KEY", "chimera_default_key_change_in_production"),
            ),
            rate_limit_enabled=security_config.get("rate_limit_enabled", True),
            rate_limit_per_minute=security_config.get("rate_limit_per_minute", 60),
            cors_origins=security_config.get(
                "cors_origins", ["http://localhost:3001", "http://localhost:8080"]
            ),
        )

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        llm_config = self._config_data.get("llm", {})
        return LLMConfig(
            default_provider=llm_config.get("default_provider", "openai"),
            timeout_seconds=llm_config.get("timeout_seconds", 30),
            max_retries=llm_config.get("max_retries", 3),
            providers=llm_config.get("providers", {}),
        )

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        cache_config = self._config_data.get("cache", {})
        return CacheConfig(
            enabled=cache_config.get("enabled", True),
            type=cache_config.get("type", "simple"),
            ttl_seconds=cache_config.get("ttl_seconds", 300),
            redis_url=cache_config.get("redis_url"),
        )

    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_config = self._config_data.get("performance", {})
        return PerformanceConfig(
            enabled=perf_config.get("enabled", True),
            log_slow_queries=perf_config.get("log_slow_queries", True),
            slow_query_threshold_ms=perf_config.get("slow_query_threshold_ms", 1000),
            memory_monitoring=perf_config.get("memory_monitoring", True),
        )

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value by section and key."""
        return self._config_data.get(section, {}).get(key, default)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        features = self._config_data.get("features", {})
        return features.get(feature_name, False)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate database URL
        db_config = self.get_database_config()
        if not db_config.url:
            errors.append("Database URL is required")

        # Validate API key if required
        security_config = self.get_security_config()
        if security_config.api_key_required and not security_config.default_api_key:
            errors.append("API key is required when API key authentication is enabled")

        # Validate LLM provider
        llm_config = self.get_llm_config()
        if llm_config.default_provider not in ["openai", "anthropic", "google", "local"]:
            errors.append(f"Invalid default LLM provider: {llm_config.default_provider}")

        return errors


# Global settings instance
settings = Settings()


# Convenience functions
def get_database_config() -> DatabaseConfig:
    return settings.get_database_config()


def get_security_config() -> SecurityConfig:
    return settings.get_security_config()


def get_llm_config() -> LLMConfig:
    return settings.get_llm_config()


def get_cache_config() -> CacheConfig:
    return settings.get_cache_config()


def get_performance_config() -> PerformanceConfig:
    return settings.get_performance_config()
