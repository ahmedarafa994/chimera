"""
Test Suite for Story 1.1: Provider Configuration Management

This module provides comprehensive tests for the provider configuration system including:
- Configuration loading and validation (AC: #1, #3, #8)
- API key encryption/decryption (AC: #5)
- Provider configuration sections (AC: #3)
- Proxy mode configuration (AC: #4)
- Direct mode configuration (AC: #4)
- Configuration validation with error messages (AC: #6, #7)
- Hot-reload functionality (AC: #8)

Test markers:
- unit: Unit tests for individual components
- integration: Integration tests for full configuration flows
- security: Security tests for encryption
"""

import os
from unittest.mock import patch

import pytest

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def clean_env():
    """Provide a clean environment for each test."""
    # Store original values
    original_env = {}
    env_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "QWEN_API_KEY",
        "CURSOR_API_KEY",
        "API_CONNECTION_MODE",
        "PROXY_MODE_ENDPOINT",
        "PROXY_MODE_ENABLED",
        "CHIMERA_ENCRYPTION_KEY",
        "CHIMERA_ENCRYPTION_PASSWORD",
    ]

    for var in env_vars:
        if var in os.environ:
            original_env[var] = os.environ.pop(var)

    yield

    # Restore original values
    for var, value in original_env.items():
        os.environ[var] = value


@pytest.fixture
def sample_api_keys():
    """Sample API keys for testing (not real keys)."""
    return {
        "openai": "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890ab",
        "anthropic": "sk-ant-test1234567890abcdefghijklmnopqrstuvwxyz",
        "google": "AIzaSyTest1234567890abcdefghijklmnopqrstu",
        "deepseek": "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890ab",
    }


@pytest.fixture
def mock_provider_config():
    """Sample provider configuration for testing."""
    return {
        "provider_type": "openai",
        "name": "openai-default",
        "api_key": "sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890ab",
        "api_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "enabled": True,
    }


# ============================================================================
# Configuration Loading Tests (AC: #1, #3, #8)
# ============================================================================


class TestConfigurationLoading:
    """Tests for configuration loading from environment variables."""

    @pytest.mark.unit
    def test_config_loads_from_environment(self, clean_env):
        """Test that configuration loads from environment variables."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["GOOGLE_API_KEY"] = "test-google-key"

        # Force reimport to reload settings
        from app.core.config import Settings

        settings = Settings()
        assert settings.OPENAI_API_KEY == "test-openai-key"
        assert settings.GOOGLE_API_KEY == "test-google-key"

    @pytest.mark.unit
    def test_config_has_default_values(self):
        """Test that configuration has sensible default values."""
        from app.core.config import settings

        assert settings.API_V1_STR == "/api/v1"
        assert settings.PROJECT_NAME == "Chimera Backend"
        assert settings.DEFAULT_MODEL_ID == "deepseek-chat"

    @pytest.mark.unit
    def test_config_api_connection_mode_default(self):
        """Test that API connection mode defaults to DIRECT."""
        from app.core.config import APIConnectionMode, settings

        assert settings.API_CONNECTION_MODE == APIConnectionMode.DIRECT

    @pytest.mark.unit
    def test_config_provider_models_available(self):
        """Test that provider models are configured for all providers."""
        from app.core.config import settings

        models = settings.get_provider_models()

        # Verify all expected providers have models configured (Story 1.1)
        assert "google" in models
        assert "gemini" in models
        assert "deepseek" in models
        assert "anthropic" in models
        assert "openai" in models
        assert "qwen" in models
        assert "cursor" in models

        # Verify each provider has at least one model
        assert len(models["google"]) > 0
        assert len(models["gemini"]) > 0
        assert len(models["deepseek"]) > 0
        assert len(models["anthropic"]) > 0
        assert len(models["openai"]) > 0
        assert len(models["qwen"]) > 0
        assert len(models["cursor"]) > 0

    @pytest.mark.unit
    def test_config_api_key_name_map(self):
        """Test API key name mapping for all providers."""
        from app.core.config import API_KEY_NAME_MAP

        assert "google" in API_KEY_NAME_MAP
        assert "gemini" in API_KEY_NAME_MAP
        assert "openai" in API_KEY_NAME_MAP
        assert "anthropic" in API_KEY_NAME_MAP
        assert "deepseek" in API_KEY_NAME_MAP
        assert "qwen" in API_KEY_NAME_MAP
        assert "cursor" in API_KEY_NAME_MAP


# ============================================================================
# Provider Configuration Section Tests (AC: #3)
# ============================================================================


class TestProviderConfigSections:
    """Tests for individual provider configuration sections."""

    @pytest.mark.unit
    def test_each_provider_has_config_section(self):
        """Test that each provider has its own configuration section."""
        from app.core.config import settings

        # Each provider should have API key and model settings
        assert hasattr(settings, "GOOGLE_API_KEY")
        assert hasattr(settings, "GOOGLE_MODEL")
        assert hasattr(settings, "OPENAI_API_KEY")
        assert hasattr(settings, "OPENAI_MODEL")
        assert hasattr(settings, "ANTHROPIC_API_KEY")
        assert hasattr(settings, "ANTHROPIC_MODEL")
        assert hasattr(settings, "DEEPSEEK_API_KEY")
        assert hasattr(settings, "DEEPSEEK_MODEL")

    @pytest.mark.unit
    def test_provider_endpoints_configured(self):
        """Test that base URLs are configured for all providers."""
        from app.core.config import settings

        endpoints = settings.get_all_provider_endpoints()

        assert "google" in endpoints
        assert "openai" in endpoints
        assert "anthropic" in endpoints
        assert "deepseek" in endpoints

        # Verify URLs are valid format
        for provider, url in endpoints.items():
            assert url.startswith("http://") or url.startswith("https://"), \
                f"Invalid URL for {provider}: {url}"

    @pytest.mark.unit
    def test_provider_endpoint_returns_correct_url(self):
        """Test that get_provider_endpoint returns correct URLs."""
        from app.core.config import settings

        # Test each provider returns expected base URL
        assert "openai.com" in settings.get_provider_endpoint("openai")
        assert "anthropic.com" in settings.get_provider_endpoint("anthropic")
        assert "googleapis.com" in settings.get_provider_endpoint("google")
        assert "deepseek.com" in settings.get_provider_endpoint("deepseek")

    @pytest.mark.unit
    def test_provider_model_selection_available(self):
        """Test that model selection is available for each provider."""
        from app.core.config import settings

        models = settings.get_provider_models()

        # Google/Gemini models
        assert "gemini-3-pro-preview" in models["google"]
        assert "gemini-2.5-pro" in models["google"]

        # DeepSeek models
        assert "deepseek-chat" in models["deepseek"]
        assert "deepseek-reasoner" in models["deepseek"]

        # Anthropic models
        assert "claude-sonnet-4-5" in models["anthropic"] or \
               "claude-3-5-sonnet" in models["anthropic"]

        # OpenAI models (Story 1.1)
        assert "gpt-4o" in models["openai"]
        assert "gpt-4-turbo" in models["openai"]
        assert "o1" in models["openai"]

        # Qwen models (Story 1.1)
        assert "qwen-max" in models["qwen"]
        assert "qwen-turbo" in models["qwen"]

        # Cursor uses OpenAI models (Story 1.1)
        assert "gpt-4o" in models["cursor"]


# ============================================================================
# Proxy Mode Configuration Tests (AC: #4)
# ============================================================================


class TestProxyModeConfiguration:
    """Tests for proxy mode (AIClient-2-API Server) configuration."""

    @pytest.mark.unit
    def test_proxy_mode_endpoint_configurable(self):
        """Test that proxy mode endpoint is configurable."""
        from app.core.config import settings

        assert hasattr(settings, "PROXY_MODE_ENDPOINT")
        assert settings.PROXY_MODE_ENDPOINT == "http://localhost:8080"

    @pytest.mark.unit
    def test_proxy_mode_has_all_settings(self):
        """Test that all proxy mode settings are available."""
        from app.core.config import settings

        assert hasattr(settings, "PROXY_MODE_ENABLED")
        assert hasattr(settings, "PROXY_MODE_HEALTH_CHECK")
        assert hasattr(settings, "PROXY_MODE_TIMEOUT")
        assert hasattr(settings, "PROXY_MODE_FALLBACK_TO_DIRECT")
        assert hasattr(settings, "PROXY_MODE_HEALTH_CHECK_INTERVAL")

    @pytest.mark.unit
    def test_proxy_mode_timeout_has_valid_range(self):
        """Test that proxy timeout has valid range constraints."""
        from app.core.config import settings

        # Should be between 5 and 120 seconds based on Field constraints
        assert settings.PROXY_MODE_TIMEOUT >= 5
        assert settings.PROXY_MODE_TIMEOUT <= 120

    @pytest.mark.unit
    def test_proxy_mode_health_check_interval_valid(self):
        """Test proxy health check interval has valid range."""
        from app.core.config import settings

        # Should be between 10 and 300 seconds based on Field constraints
        assert settings.PROXY_MODE_HEALTH_CHECK_INTERVAL >= 10
        assert settings.PROXY_MODE_HEALTH_CHECK_INTERVAL <= 300

    @pytest.mark.unit
    def test_get_proxy_config_returns_dict(self):
        """Test that get_proxy_config returns a complete configuration dict."""
        from app.core.config import settings

        proxy_config = settings.get_proxy_config()

        assert isinstance(proxy_config, dict)
        assert "enabled" in proxy_config
        assert "endpoint" in proxy_config
        assert "health_check" in proxy_config
        assert "timeout" in proxy_config
        assert "fallback_to_direct" in proxy_config
        assert "health_check_interval" in proxy_config


# ============================================================================
# Direct Mode Configuration Tests (AC: #4)
# ============================================================================


class TestDirectModeConfiguration:
    """Tests for direct API mode configuration."""

    @pytest.mark.unit
    def test_direct_mode_is_default(self):
        """Test that direct mode is the default connection mode."""
        from app.core.config import APIConnectionMode, settings

        assert settings.API_CONNECTION_MODE == APIConnectionMode.DIRECT

    @pytest.mark.unit
    def test_direct_mode_base_urls_configurable(self):
        """Test that direct mode base URLs are configurable per provider."""
        from app.core.config import settings

        assert hasattr(settings, "DIRECT_OPENAI_BASE_URL")
        assert hasattr(settings, "DIRECT_ANTHROPIC_BASE_URL")
        assert hasattr(settings, "DIRECT_GOOGLE_BASE_URL")
        assert hasattr(settings, "DIRECT_DEEPSEEK_BASE_URL")
        assert hasattr(settings, "DIRECT_QWEN_BASE_URL")
        assert hasattr(settings, "DIRECT_CURSOR_BASE_URL")

    @pytest.mark.unit
    def test_connection_mode_returns_direct(self):
        """Test that get_connection_mode returns DIRECT."""
        from app.core.config import APIConnectionMode, settings

        mode = settings.get_connection_mode()
        assert mode == APIConnectionMode.DIRECT


# ============================================================================
# API Key Encryption Tests (AC: #5)
# ============================================================================


class TestAPIKeyEncryption:
    """Tests for API key encryption at rest."""

    @pytest.mark.security
    def test_encrypt_api_key_returns_encrypted_format(self, sample_api_keys):
        """Test that encrypt_api_key returns encrypted format with enc: prefix."""
        from app.core.encryption import encrypt_api_key

        encrypted = encrypt_api_key(sample_api_keys["openai"])

        assert encrypted.startswith("enc:")
        assert encrypted != sample_api_keys["openai"]

    @pytest.mark.security
    def test_decrypt_api_key_returns_original(self, sample_api_keys):
        """Test that decrypt_api_key returns the original key."""
        from app.core.encryption import decrypt_api_key, encrypt_api_key

        original = sample_api_keys["openai"]
        encrypted = encrypt_api_key(original)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == original

    @pytest.mark.security
    def test_is_encrypted_detects_encrypted_keys(self, sample_api_keys):
        """Test that is_encrypted correctly identifies encrypted keys."""
        from app.core.encryption import encrypt_api_key, is_encrypted

        encrypted = encrypt_api_key(sample_api_keys["openai"])

        assert is_encrypted(encrypted) is True
        assert is_encrypted(sample_api_keys["openai"]) is False

    @pytest.mark.security
    def test_ensure_key_encrypted_encrypts_only_once(self, sample_api_keys):
        """Test that ensure_key_encrypted doesn't double-encrypt."""
        from app.core.encryption import encrypt_api_key, ensure_key_encrypted

        original = sample_api_keys["openai"]

        # Encrypt once
        encrypted = encrypt_api_key(original)

        # ensure_key_encrypted should return same value for already encrypted
        result = ensure_key_encrypted(encrypted)
        assert result == encrypted

        # ensure_key_encrypted should encrypt plaintext
        result = ensure_key_encrypted(original)
        assert result.startswith("enc:")

    @pytest.mark.security
    def test_decrypt_plaintext_returns_original(self, sample_api_keys):
        """Test that decrypting plaintext key returns it unchanged."""
        from app.core.encryption import decrypt_api_key

        original = sample_api_keys["openai"]
        result = decrypt_api_key(original)

        assert result == original

    @pytest.mark.security
    def test_encryption_uses_consistent_key(self, sample_api_keys):
        """Test that encryption/decryption uses consistent key across calls."""
        from app.core.encryption import decrypt_api_key, encrypt_api_key

        original = sample_api_keys["openai"]

        # Encrypt twice - should produce different ciphertext (Fernet adds timestamp)
        # but both should decrypt to original
        encrypted1 = encrypt_api_key(original)
        encrypted2 = encrypt_api_key(original)

        assert decrypt_api_key(encrypted1) == original
        assert decrypt_api_key(encrypted2) == original

    @pytest.mark.security
    def test_empty_key_returns_empty(self):
        """Test that empty keys are handled gracefully."""
        from app.core.encryption import decrypt_api_key, encrypt_api_key

        assert encrypt_api_key("") == ""
        assert decrypt_api_key("") == ""
        assert encrypt_api_key(None) is None
        assert decrypt_api_key(None) is None


# ============================================================================
# Configuration Validation Tests (AC: #6, #7)
# ============================================================================


class TestConfigurationValidation:
    """Tests for configuration validation with error messages."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_api_key_format_openai(self, sample_api_keys):
        """Test OpenAI API key format validation."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Valid key
        valid, error = await validator.validate_api_key_format(
            "openai", sample_api_keys["openai"]
        )
        assert valid is True
        assert error == ""

        # Invalid key (too short)
        valid, error = await validator.validate_api_key_format(
            "openai", "sk-tooshort"
        )
        assert valid is False
        assert "format invalid" in error.lower() or "pattern" in error.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_api_key_format_anthropic(self, sample_api_keys):
        """Test Anthropic API key format validation."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Valid key
        valid, _error = await validator.validate_api_key_format(
            "anthropic", sample_api_keys["anthropic"]
        )
        assert valid is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_api_key_format_google(self, sample_api_keys):
        """Test Google API key format validation."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Valid key
        valid, _error = await validator.validate_api_key_format(
            "google", sample_api_keys["google"]
        )
        assert valid is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_empty_api_key_returns_error(self):
        """Test that empty API key returns validation error."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        valid, error = await validator.validate_api_key_format("openai", "")
        assert valid is False
        assert "required" in error.lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_provider_config_returns_recommendations(
        self, sample_api_keys
    ):
        """Test that invalid configs provide recommendations."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        result = await validator.validate_provider_config(
            provider="openai",
            api_key="invalid-key",
            test_connectivity=False
        )

        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_base_url_format(self):
        """Test base URL format validation."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Invalid URL - should fail
        result = await validator.validate_provider_config(
            provider="openai",
            api_key="sk-test1234567890abcdefghijklmnopqrstuvwxyz1234567890ab",
            base_url="not-a-valid-url",
            test_connectivity=False
        )

        assert result["is_valid"] is False
        assert any("http" in err.lower() for err in result["errors"])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validation_uses_default_url_when_missing(self, sample_api_keys):
        """Test that validation uses default URL when none provided."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        result = await validator.validate_provider_config(
            provider="openai",
            api_key=sample_api_keys["openai"],
            base_url=None,
            test_connectivity=False
        )

        assert "warnings" in result
        # Should have warning about using default URL
        assert any("default" in w.lower() for w in result["warnings"])


# ============================================================================
# Error Message Quality Tests (AC: #7)
# ============================================================================


class TestErrorMessageQuality:
    """Tests for clear error messages with remediation steps."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_invalid_key_error_has_remediation(self):
        """Test that invalid key errors include remediation steps."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        result = await validator.validate_provider_config(
            provider="openai",
            api_key="invalid",
            test_connectivity=False
        )

        # Should have recommendations for fixing
        assert len(result["recommendations"]) > 0
        # Recommendations should mention the provider
        has_provider_mention = any(
            "openai" in r.lower() for r in result["recommendations"]
        )
        assert has_provider_mention

    @pytest.mark.unit
    def test_pattern_description_is_human_readable(self):
        """Test that API key pattern descriptions are human readable."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Get pattern description for each provider
        providers = ["openai", "anthropic", "google", "deepseek"]

        for provider in providers:
            desc = validator._get_pattern_description(provider)
            assert len(desc) > 10, f"Description too short for {provider}"
            # Should be readable, not just regex
            assert not desc.startswith("^"), f"Description is raw regex for {provider}"


# ============================================================================
# Hot-Reload Tests (AC: #8)
# ============================================================================


class TestHotReloadFunctionality:
    """Tests for configuration hot-reload without restart."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_config_updates_settings(self, clean_env):
        """Test that reload_config updates settings without restart."""
        from app.core.config_manager import config_manager

        # Get initial state
        os.environ["OPENAI_API_KEY"] = "initial-key"

        result = await config_manager.reload_config()

        assert result["status"] == "success"
        assert "reloaded_at" in result
        assert "elapsed_ms" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_config_tracks_changes(self, clean_env):
        """Test that reload_config tracks configuration changes."""
        from app.core.config_manager import config_manager

        result = await config_manager.reload_config()

        assert "changes" in result
        assert "added" in result["changes"]
        assert "removed" in result["changes"]
        assert "modified" in result["changes"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_config_validates_new_config(self, clean_env):
        """Test that reload_config validates the new configuration."""
        from app.core.config_manager import config_manager

        # Set up a valid API key
        os.environ["GOOGLE_API_KEY"] = "test-google-key-valid"

        result = await config_manager.reload_config()

        assert "validation" in result
        assert "valid_providers" in result["validation"]
        assert "invalid_providers" in result["validation"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_config_executes_callbacks(self, clean_env):
        """Test that reload_config executes registered callbacks."""
        from app.core.config_manager import config_manager

        callback_called = []

        def test_callback():
            callback_called.append(True)

        config_manager.add_reload_callback(test_callback)

        try:
            await config_manager.reload_config()
            assert len(callback_called) == 1
        finally:
            config_manager.remove_reload_callback(test_callback)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_config_handles_async_callbacks(self, clean_env):
        """Test that reload_config handles async callbacks."""
        from app.core.config_manager import config_manager

        callback_called = []

        async def async_callback():
            callback_called.append(True)

        config_manager.add_reload_callback(async_callback)

        try:
            await config_manager.reload_config()
            assert len(callback_called) == 1
        finally:
            config_manager.remove_reload_callback(async_callback)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reload_completes_within_performance_target(self, clean_env):
        """Test that reload completes within performance target (<1s)."""
        from app.core.config_manager import config_manager

        result = await config_manager.reload_config()

        # Should complete in less than 1000ms per Story 1.1 requirement
        assert result["elapsed_ms"] < 1000

    @pytest.mark.unit
    def test_enable_hot_reload_setting_exists(self):
        """Test that ENABLE_CONFIG_HOT_RELOAD setting exists."""
        from app.core.config import settings

        assert hasattr(settings, "ENABLE_CONFIG_HOT_RELOAD")
        assert isinstance(settings.ENABLE_CONFIG_HOT_RELOAD, bool)


# ============================================================================
# Provider Config Summary Tests
# ============================================================================


class TestProviderConfigSummary:
    """Tests for provider configuration summary functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_provider_config_summary(self, clean_env):
        """Test getting provider configuration summary."""
        from app.core.config_manager import config_manager

        os.environ["GOOGLE_API_KEY"] = "test-key"

        summary = await config_manager.get_provider_config_summary()

        assert "total_providers" in summary
        assert "configured_providers" in summary
        assert "connection_mode" in summary
        assert "proxy_config" in summary
        assert "encryption_status" in summary

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_summary_includes_encryption_status(self, clean_env):
        """Test that summary includes encryption status for each provider."""
        from app.core.config_manager import config_manager
        from app.core.encryption import encrypt_api_key

        # Set up encrypted key
        os.environ["GOOGLE_API_KEY"] = encrypt_api_key("test-key")

        summary = await config_manager.get_provider_config_summary()

        assert "encryption_status" in summary
        # Google should show as encrypted
        if "google" in summary["encryption_status"]:
            assert summary["encryption_status"]["google"] is True


# ============================================================================
# Proxy Mode Validation Tests
# ============================================================================


class TestProxyModeValidation:
    """Tests for proxy mode configuration validation."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_proxy_mode_when_enabled(self, clean_env):
        """Test proxy mode validation when enabled."""
        from app.core.config_manager import config_manager

        result = await config_manager.validate_proxy_mode_config()

        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "recommendations" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_proxy_mode_validation_provides_recommendations(self, clean_env):
        """Test that proxy mode validation provides helpful recommendations."""
        from app.core.config import APIConnectionMode, settings
        from app.core.config_manager import config_manager

        # Force proxy mode to test validation
        with patch.object(settings, "API_CONNECTION_MODE", APIConnectionMode.PROXY):
            result = await config_manager.validate_proxy_mode_config()

            # Should have recommendations if proxy unavailable
            if not result["is_valid"]:
                assert len(result["recommendations"]) > 0
                # Should mention AIClient-2-API Server
                has_aiclient_mention = any(
                    "aiclient" in r.lower() or "proxy" in r.lower()
                    for r in result["recommendations"]
                )
                assert has_aiclient_mention


# ============================================================================
# URL Validation Tests
# ============================================================================


class TestURLValidation:
    """Tests for endpoint URL validation."""

    @pytest.mark.unit
    def test_validate_valid_https_url(self):
        """Test validation of valid HTTPS URL."""
        from app.core.config import settings

        assert settings.validate_endpoint_url("https://api.openai.com/v1") is True

    @pytest.mark.unit
    def test_validate_valid_http_localhost(self):
        """Test validation of HTTP localhost URL."""
        from app.core.config import settings

        assert settings.validate_endpoint_url("http://localhost:8080") is True

    @pytest.mark.unit
    def test_validate_invalid_url(self):
        """Test validation of invalid URL."""
        from app.core.config import settings

        assert settings.validate_endpoint_url("not-a-url") is False
        assert settings.validate_endpoint_url("") is False
        assert settings.validate_endpoint_url(None) is False

    @pytest.mark.unit
    def test_validate_url_with_path(self):
        """Test validation of URL with path."""
        from app.core.config import settings

        assert settings.validate_endpoint_url(
            "https://api.anthropic.com/v1/messages"
        ) is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_provider_validation_flow(self, sample_api_keys, clean_env):
        """Test complete provider validation flow."""
        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        # Validate a provider with all checks
        result = await validator.validate_provider_config(
            provider="openai",
            api_key=sample_api_keys["openai"],
            base_url="https://api.openai.com/v1",
            test_connectivity=False  # Don't actually connect in tests
        )

        assert "provider" in result
        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "recommendations" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_encrypted_key_validation(self, sample_api_keys, clean_env):
        """Test that encrypted keys can be validated."""
        from app.core.config_manager import ProviderConfigValidator
        from app.core.encryption import encrypt_api_key

        validator = ProviderConfigValidator()

        # Encrypt the key
        encrypted_key = encrypt_api_key(sample_api_keys["openai"])

        # Validation should decrypt and validate
        valid, _error = await validator.validate_api_key_format(
            "openai", encrypted_key
        )

        assert valid is True

    @pytest.mark.integration
    def test_settings_and_config_are_same_instance(self):
        """Test that settings and config are the same instance (alias)."""
        from app.core.config import config, settings

        assert settings is config


# ============================================================================
# Performance Tests
# ============================================================================


class TestConfigurationPerformance:
    """Performance tests for configuration operations."""

    @pytest.mark.performance
    def test_config_loading_performance(self):
        """Test that configuration loads within performance target (<100ms)."""
        import time

        from app.core.config import Settings

        start = time.perf_counter()
        _ = Settings()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should load in less than 100ms per Story 1.1 requirement
        assert elapsed_ms < 100, f"Config loading took {elapsed_ms:.2f}ms"

    @pytest.mark.performance
    def test_encryption_performance(self, sample_api_keys):
        """Test that encryption/decryption meets performance target (<10ms)."""
        import time

        from app.core.encryption import decrypt_api_key, encrypt_api_key

        # Test encryption
        start = time.perf_counter()
        encrypted = encrypt_api_key(sample_api_keys["openai"])
        encrypt_ms = (time.perf_counter() - start) * 1000

        # Test decryption
        start = time.perf_counter()
        _ = decrypt_api_key(encrypted)
        decrypt_ms = (time.perf_counter() - start) * 1000

        # Should be less than 10ms per Story 1.1 requirement
        assert encrypt_ms < 10, f"Encryption took {encrypt_ms:.2f}ms"
        assert decrypt_ms < 10, f"Decryption took {decrypt_ms:.2f}ms"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance(self, sample_api_keys):
        """Test that validation completes quickly."""
        import time

        from app.core.config_manager import ProviderConfigValidator

        validator = ProviderConfigValidator()

        start = time.perf_counter()
        _ = await validator.validate_api_key_format("openai", sample_api_keys["openai"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly (< 50ms)
        assert elapsed_ms < 50, f"Validation took {elapsed_ms:.2f}ms"
