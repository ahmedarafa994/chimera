"""
Configuration Loader
Provides utilities for loading and merging configuration from multiple sources.
"""

import json
import os
from pathlib import Path
from typing import Any

import yaml

from app.core.logging import logger


class ConfigurationLoader:
    """
    Loads configuration from multiple sources with priority:
    1. Environment variables (highest priority)
    2. .env file
    3. JSON/YAML configuration files
    4. Default values (lowest priority)
    """

    def __init__(self):
        self._config_cache: dict[str, Any] = {}
        self._watchers: list[callable] = []

    def load_from_env(self) -> dict[str, str]:
        """Load all relevant configuration from environment variables."""
        env_config = {}

        # Define all configuration keys we care about
        config_keys = [
            # Direct Provider Endpoints
            "DIRECT_OPENAI_BASE_URL",
            "DIRECT_ANTHROPIC_BASE_URL",
            "DIRECT_GOOGLE_BASE_URL",
            "DIRECT_QWEN_BASE_URL",
            "DIRECT_DEEPSEEK_BASE_URL",
            "DIRECT_CURSOR_BASE_URL",
            "GEMINI_DIRECT_BASE_URL",
            "GEMINI_OPENAI_COMPAT_URL",
            "GEMINI_DIRECT_API_KEY",
            # API Keys
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "QWEN_API_KEY",
            "DEEPSEEK_API_KEY",
            "CHIMERA_API_KEY",
            # Validation
            "ENABLE_ENDPOINT_VALIDATION",
        ]

        for key in config_keys:
            value = os.getenv(key)
            if value is not None:
                env_config[key] = value

        return env_config

    def load_from_file(self, path: str | Path) -> dict[str, Any]:
        """
        Load configuration from a JSON or YAML file.
        Supports environment variable interpolation using ${VAR_NAME} syntax.
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return {}

        try:
            with open(path, encoding="utf-8") as f:
                if path.suffix in (".yaml", ".yml"):
                    config = yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    config = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {path.suffix}")
                    return {}

            # Interpolate environment variables
            return self._interpolate_env_vars(config)

        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            return {}

    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Recursively interpolate environment variables in configuration.
        Supports ${VAR_NAME} and ${VAR_NAME:default} syntax.
        """
        if isinstance(config, str):
            return self._interpolate_string(config)
        elif isinstance(config, dict):
            return {k: self._interpolate_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(item) for item in config]
        return config

    def _interpolate_string(self, value: str) -> str:
        """Interpolate environment variables in a string value."""
        import re

        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replace, value)

    def merge_configurations(self, *configs: dict[str, Any]) -> dict[str, Any]:
        """
        Merge multiple configurations with later configs taking precedence.
        Performs deep merge for nested dictionaries.
        """
        result = {}

        for config in configs:
            result = self._deep_merge(result, config)

        return result

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_complete_configuration(
        self, config_files: list[str | Path] | None = None, defaults: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Load complete configuration from all sources.

        Priority (highest to lowest):
        1. Environment variables
        2. Config files (in order provided)
        3. Default values
        """
        configs_to_merge = []

        # Start with defaults
        if defaults:
            configs_to_merge.append(defaults)

        # Load from config files
        if config_files:
            for file_path in config_files:
                file_config = self.load_from_file(file_path)
                if file_config:
                    configs_to_merge.append(file_config)

        # Load from environment (highest priority)
        env_config = self.load_from_env()
        configs_to_merge.append(env_config)

        # Merge all configurations
        return self.merge_configurations(*configs_to_merge)

    def validate_configuration(self, config: dict[str, Any]) -> list[str]:
        """
        Validate configuration and return list of validation errors.
        """
        from app.core.config_validator import ConfigValidator

        validator = ConfigValidator()
        errors = []

        # Validate URL formats
        url_keys = [
            "DIRECT_OPENAI_BASE_URL",
            "DIRECT_ANTHROPIC_BASE_URL",
            "DIRECT_GOOGLE_BASE_URL",
            "DIRECT_QWEN_BASE_URL",
            "DIRECT_DEEPSEEK_BASE_URL",
            "DIRECT_CURSOR_BASE_URL",
            "GEMINI_DIRECT_BASE_URL",
            "GEMINI_OPENAI_COMPAT_URL",
        ]

        for key in url_keys:
            if key in config and not validator.validate_url(config[key]):
                errors.append(f"Invalid URL format for {key}: {config[key]}")

        # Validate API keys if present
        api_key_keys = [
            "GOOGLE_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "QWEN_API_KEY",
            "DEEPSEEK_API_KEY",
            "GEMINI_DIRECT_API_KEY",
        ]

        for key in api_key_keys:
            if config.get(key) and not validator.validate_api_key(config[key]):
                errors.append(f"Invalid API key format for {key}")

        return errors

    def watch_configuration_changes(self, callback: callable) -> None:
        """Register a callback to be notified of configuration changes."""
        self._watchers.append(callback)

    def notify_configuration_change(self) -> None:
        """Notify all watchers of configuration changes."""
        for callback in self._watchers:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error notifying configuration watcher: {e}")


# Singleton instance
_loader_instance: ConfigurationLoader | None = None


def get_configuration_loader() -> ConfigurationLoader:
    """Get the configuration loader singleton."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = ConfigurationLoader()
    return _loader_instance
