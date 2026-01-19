# =============================================================================
# Chimera - Enhanced Configuration Management
# =============================================================================
# Extends the base config.py with advanced features for Story 1.1:
# - Hot-reload capability
# - Configuration validation with connectivity checks
# - Enhanced proxy mode configuration
# - API key encryption integration
# =============================================================================

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from .config import API_KEY_NAME_MAP, APIConnectionMode, settings
from .encryption import EncryptionError, decrypt_api_key, is_encrypted

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration errors"""

    pass


class ValidationError(ConfigurationError):
    """Raised when configuration validation fails"""

    pass


class ProviderConfigValidator:
    """Validates provider configurations including connectivity checks"""

    # API key format patterns for validation (Task 3.1)
    API_KEY_PATTERNS = {
        "openai": r"^sk-[a-zA-Z0-9]{32,}$",
        "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{30,}$",
        "google": r"^[a-zA-Z0-9_\-]{32,}$",
        "deepseek": r"^sk-[a-zA-Z0-9]{32,}$",
        "qwen": r"^[a-zA-Z0-9_\-]{16,}$",
    }

    # Provider base URLs for validation (Task 4.2)
    DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/api/v1",
        "cursor": "https://api.openai.com/v1",  # Cursor uses OpenAI-compatible API
    }

    # Default models per provider (Task 4.1)
    DEFAULT_MODELS = {
        "google": ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
        "openai": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": [
            "claude-sonnet-4-5",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
            "claude-3-opus",
        ],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "qwen": ["qwen-turbo", "qwen-plus", "qwen-max"],
        "cursor": ["gpt-4o", "gpt-4", "gpt-4-turbo"],
    }

    async def validate_api_key_format(self, provider: str, api_key: str) -> tuple[bool, str]:
        """
        Validate API key format for a provider.

        Args:
            provider: Provider name
            api_key: API key to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not api_key:
            return False, f"API key is required for {provider}"

        # Decrypt key if encrypted for validation
        try:
            actual_key = decrypt_api_key(api_key) if is_encrypted(api_key) else api_key
        except EncryptionError as e:
            return False, f"Failed to decrypt API key for {provider}: {e}"

        pattern = self.API_KEY_PATTERNS.get(provider.lower())
        if not pattern:
            # For unknown providers, just check minimum length
            if len(actual_key) < 10:
                return False, f"API key for {provider} must be at least 10 characters"
            return True, ""

        import re

        if not re.match(pattern, actual_key):
            return False, (
                f"API key format invalid for {provider}. "
                f"Expected pattern: {self._get_pattern_description(provider)}"
            )

        return True, ""

    def _get_pattern_description(self, provider: str) -> str:
        """Get human-readable description of API key pattern"""
        descriptions = {
            "openai": "sk-... (starts with 'sk-' followed by 32+ characters)",
            "anthropic": "sk-ant-... (starts with 'sk-ant-' followed by 30+ characters)",
            "google": "32+ alphanumeric characters with underscores/hyphens",
            "deepseek": "sk-... (starts with 'sk-' followed by 32+ characters)",
            "qwen": "16+ alphanumeric characters with underscores/hyphens",
        }
        return descriptions.get(provider.lower(), "Valid API key format")

    async def validate_connectivity(
        self, provider: str, api_key: str, base_url: str
    ) -> tuple[bool, str]:
        """
        Test connectivity to a provider API.

        Args:
            provider: Provider name
            api_key: API key (decrypted if needed)
            base_url: Base URL for the provider

        Returns:
            Tuple of (is_connected, error_message)
        """
        try:
            # Decrypt key if encrypted
            actual_key = decrypt_api_key(api_key) if is_encrypted(api_key) else api_key

            # Define test endpoints and headers for each provider
            test_configs = {
                "openai": {
                    "endpoint": f"{base_url.rstrip('/')}/models",
                    "headers": {"Authorization": f"Bearer {actual_key}"},
                    "method": "GET",
                },
                "anthropic": {
                    "endpoint": f"{base_url.rstrip('/')}/messages",
                    "headers": {
                        "Authorization": f"Bearer {actual_key}",
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    "method": "HEAD",  # Just check if endpoint is accessible
                },
                "google": {
                    "endpoint": f"{base_url.rstrip('/')}/models",
                    "headers": {},
                    "method": "GET",
                    "params": {"key": actual_key},
                },
                "deepseek": {
                    "endpoint": f"{base_url.rstrip('/')}/models",
                    "headers": {"Authorization": f"Bearer {actual_key}"},
                    "method": "GET",
                },
            }

            config = test_configs.get(provider.lower())
            if not config:
                # For unknown providers, just test basic HTTP connectivity
                config = {
                    "endpoint": base_url,
                    "headers": {"Authorization": f"Bearer {actual_key}"},
                    "method": "GET",
                }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.request(
                    method=config["method"],
                    url=config["endpoint"],
                    headers=config["headers"],
                    params=config.get("params", {}),
                ) as response:
                    # Accept various success codes
                    if response.status in [200, 201, 204, 401, 403]:
                        # 401/403 means the endpoint exists but auth failed
                        # This is still a valid connectivity test
                        return True, ""
                    else:
                        return False, f"HTTP {response.status}: {await response.text()}"

        except TimeoutError:
            return False, f"Connection timeout to {provider} API at {base_url}"
        except aiohttp.ClientError as e:
            return False, f"Connection error to {provider}: {e}"
        except Exception as e:
            return False, f"Unexpected error testing {provider} connectivity: {e}"

    async def validate_provider_config(
        self,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        test_connectivity: bool = True,
    ) -> dict[str, Any]:
        """
        Comprehensive validation of a provider configuration.

        Args:
            provider: Provider name
            api_key: API key to validate
            base_url: Base URL to validate
            test_connectivity: Whether to test API connectivity

        Returns:
            Dictionary with validation results
        """
        result = {
            "provider": provider,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
        }

        # Validate API key format
        if api_key:
            key_valid, key_error = await self.validate_api_key_format(provider, api_key)
            if not key_valid:
                result["is_valid"] = False
                result["errors"].append(key_error)
                result["recommendations"].append(
                    f"Ensure your {provider} API key follows the correct format. "
                    f"Check your {provider} dashboard for the correct API key."
                )

        # Validate base URL
        if base_url:
            if not base_url.startswith(("http://", "https://")):
                result["is_valid"] = False
                result["errors"].append("Base URL must start with http:// or https://")
                result["recommendations"].append("Use HTTPS URLs for production environments")
        else:
            # Use default URL
            base_url = self.DEFAULT_BASE_URLS.get(provider.lower())
            if base_url:
                result["warnings"].append(f"Using default base URL: {base_url}")

        # Test connectivity if requested and we have valid credentials
        if test_connectivity and api_key and base_url and result["is_valid"]:
            connected, conn_error = await self.validate_connectivity(provider, api_key, base_url)
            if not connected:
                result["is_valid"] = False
                result["errors"].append(f"Connectivity test failed: {conn_error}")
                result["recommendations"].extend(
                    [
                        f"Check your {provider} API key is active and has proper permissions",
                        f"Verify the base URL {base_url} is correct",
                        "Check your network connectivity and firewall settings",
                        f"Visit {provider}'s documentation for troubleshooting",
                    ]
                )

        return result


class EnhancedConfigManager:
    """
    Enhanced configuration manager with hot-reload and validation capabilities.
    Implements Story 1.1 requirements for provider configuration management.
    """

    def __init__(self):
        self.validator = ProviderConfigValidator()
        self._last_reload = None
        self._config_hash = None
        self._reload_callbacks: list[callable] = []

    def add_reload_callback(self, callback: callable):
        """Add a callback to be called when configuration is reloaded"""
        self._reload_callbacks.append(callback)

    def remove_reload_callback(self, callback: callable):
        """Remove a reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)

    async def reload_config(self) -> dict[str, Any]:
        """
        Hot-reload configuration without application restart.

        Returns:
            Dictionary with reload results
        """
        try:
            logger.info("Starting configuration hot-reload")
            start_time = datetime.now()

            # Re-read environment variables by creating a new Settings instance
            from . import config as config_module

            # Store old settings for comparison
            old_providers = self._get_configured_providers()

            # Reload settings (this re-reads environment variables)
            config_module.settings = config_module.Settings()
            config_module.config = config_module.settings

            # Get new provider configuration
            new_providers = self._get_configured_providers()

            # Calculate changes
            changes = self._calculate_config_changes(old_providers, new_providers)

            # Validate new configuration
            validation_results = await self._validate_all_providers(new_providers)

            # Execute reload callbacks (e.g., to re-register providers)
            callback_results = []
            for callback in self._reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                    callback_results.append({"callback": callback.__name__, "status": "success"})
                except Exception as e:
                    logger.error(f"Error in reload callback {callback.__name__}: {e}")
                    callback_results.append(
                        {"callback": callback.__name__, "status": "error", "error": str(e)}
                    )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self._last_reload = datetime.now()

            result = {
                "status": "success",
                "reloaded_at": self._last_reload.isoformat(),
                "elapsed_ms": elapsed,
                "changes": changes,
                "validation": validation_results,
                "callbacks": callback_results,
            }

            logger.info(f"Configuration hot-reload completed in {elapsed:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Error during configuration hot-reload: {e}")
            return {"status": "error", "error": str(e), "reloaded_at": datetime.now().isoformat()}

    def _get_configured_providers(self) -> dict[str, dict[str, Any]]:
        """Get current provider configuration"""
        providers = {}

        for provider_name, env_var in API_KEY_NAME_MAP.items():
            api_key = getattr(settings, env_var, None)
            if api_key:
                providers[provider_name] = {
                    "api_key": api_key,
                    "base_url": settings.get_provider_endpoint(provider_name),
                    "models": settings.get_provider_models().get(provider_name, []),
                    "connection_mode": settings.get_connection_mode(),
                }

        return providers

    def _calculate_config_changes(
        self, old_providers: dict[str, Any], new_providers: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate changes between old and new provider configurations"""
        changes = {"added": [], "removed": [], "modified": []}

        # Find added providers
        for provider in new_providers:
            if provider not in old_providers:
                changes["added"].append(provider)

        # Find removed providers
        for provider in old_providers:
            if provider not in new_providers:
                changes["removed"].append(provider)

        # Find modified providers
        for provider in new_providers:
            if provider in old_providers:
                if new_providers[provider] != old_providers[provider]:
                    changes["modified"].append(provider)

        return changes

    async def _validate_all_providers(self, providers: dict[str, Any]) -> dict[str, Any]:
        """Validate all provider configurations"""
        validation_results = {
            "valid_providers": [],
            "invalid_providers": [],
            "total_providers": len(providers),
            "validation_errors": [],
        }

        for provider_name, config in providers.items():
            result = await self.validator.validate_provider_config(
                provider=provider_name,
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
                test_connectivity=True,
            )

            if result["is_valid"]:
                validation_results["valid_providers"].append(provider_name)
            else:
                validation_results["invalid_providers"].append(provider_name)
                validation_results["validation_errors"].extend(result["errors"])

        return validation_results

    async def validate_proxy_mode_config(self) -> dict[str, Any]:
        """Validate proxy mode configuration (AIClient-2-API Server)"""
        result = {"is_valid": True, "errors": [], "warnings": [], "recommendations": []}

        if settings.API_CONNECTION_MODE == APIConnectionMode.PROXY:
            proxy_url = settings.PROXY_MODE_ENDPOINT

            # Test proxy connectivity
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(f"{proxy_url}/health") as response:
                        if response.status != 200:
                            result["is_valid"] = False
                            result["errors"].append(
                                f"Proxy server at {proxy_url} returned status {response.status}"
                            )
                            result["recommendations"].extend(
                                [
                                    "Ensure AIClient-2-API Server is running on localhost:8080",
                                    "Check if the health endpoint is available",
                                    "Consider fallback to direct mode if proxy is unavailable",
                                ]
                            )
            except Exception as e:
                result["is_valid"] = False
                result["errors"].append(f"Cannot connect to proxy server at {proxy_url}: {e}")
                result["recommendations"].extend(
                    [
                        "Start AIClient-2-API Server on localhost:8080",
                        "Check network connectivity",
                        "Verify proxy configuration in environment variables",
                    ]
                )

        return result

    async def get_provider_config_summary(self) -> dict[str, Any]:
        """Get a summary of the current provider configuration"""
        providers = self._get_configured_providers()

        summary = {
            "total_providers": len(providers),
            "configured_providers": list(providers.keys()),
            "connection_mode": settings.get_connection_mode(),
            "proxy_config": {
                "enabled": settings.API_CONNECTION_MODE == APIConnectionMode.PROXY,
                "endpoint": (
                    settings.PROXY_MODE_ENDPOINT
                    if hasattr(settings, "PROXY_MODE_ENDPOINT")
                    else None
                ),
                "health_check": (
                    settings.PROXY_MODE_HEALTH_CHECK
                    if hasattr(settings, "PROXY_MODE_HEALTH_CHECK")
                    else None
                ),
            },
            "encryption_status": {},
        }

        # Check encryption status for each provider
        for provider_name, config in providers.items():
            api_key = config.get("api_key")
            summary["encryption_status"][provider_name] = (
                is_encrypted(api_key) if api_key else False
            )

        return summary


# Global configuration manager instance
config_manager = EnhancedConfigManager()
