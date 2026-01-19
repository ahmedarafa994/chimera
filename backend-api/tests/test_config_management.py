# =============================================================================
# Test Suite for Configuration Management System
# =============================================================================
# Comprehensive tests for Story 1.1: Provider Configuration Management
# Tests encryption, validation, hot-reload, and proxy mode functionality
# =============================================================================

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import APIConnectionMode, Settings
from app.core.config_manager import EnhancedConfigManager, ProviderConfigValidator
from app.core.encryption import EncryptionError, decrypt_api_key, encrypt_api_key, is_encrypted


class TestAPIKeyEncryption:
    """Test API key encryption functionality"""

    def test_encrypt_api_key(self):
        """Test API key encryption works correctly"""
        api_key = "sk-test1234567890abcdef"
        encrypted = encrypt_api_key(api_key)

        assert encrypted.startswith("enc:")
        assert encrypted != api_key
        assert len(encrypted) > len(api_key)

    def test_decrypt_api_key(self):
        """Test API key decryption works correctly"""
        api_key = "sk-test1234567890abcdef"
        encrypted = encrypt_api_key(api_key)
        decrypted = decrypt_api_key(encrypted)

        assert decrypted == api_key

    def test_is_encrypted_detection(self):
        """Test detection of encrypted vs plaintext keys"""
        plaintext = "sk-test1234567890abcdef"
        encrypted = encrypt_api_key(plaintext)

        assert not is_encrypted(plaintext)
        assert is_encrypted(encrypted)

    def test_decrypt_plaintext_key_returns_as_is(self):
        """Test that decrypting plaintext key returns it unchanged"""
        plaintext = "sk-test1234567890abcdef"
        result = decrypt_api_key(plaintext)

        assert result == plaintext

    def test_empty_key_handling(self):
        """Test handling of empty or None keys"""
        assert encrypt_api_key("") == ""
        assert encrypt_api_key(None) is None
        assert decrypt_api_key("") == ""
        assert decrypt_api_key(None) is None

    def test_encryption_error_handling(self):
        """Test error handling for invalid encrypted keys"""
        invalid_encrypted = "enc:invalid_base64_data"

        with pytest.raises(EncryptionError):
            decrypt_api_key(invalid_encrypted)


class TestProviderConfigValidator:
    """Test provider configuration validation"""

    def setup_method(self):
        """Setup test environment"""
        self.validator = ProviderConfigValidator()

    @pytest.mark.asyncio
    async def test_validate_openai_api_key_format(self):
        """Test OpenAI API key format validation"""
        # Valid OpenAI key
        valid_key = "sk-1234567890abcdef1234567890abcdef"
        is_valid, error = await self.validator.validate_api_key_format("openai", valid_key)
        assert is_valid
        assert error == ""

        # Invalid format
        invalid_key = "invalid_key_format"
        is_valid, error = await self.validator.validate_api_key_format("openai", invalid_key)
        assert not is_valid
        assert "Expected pattern" in error  # Updated assertion to match actual error message

    @pytest.mark.asyncio
    async def test_validate_anthropic_api_key_format(self):
        """Test Anthropic API key format validation"""
        # Valid Anthropic key
        valid_key = "sk-ant-api03-abcdef1234567890123456789012345678901234567890"
        is_valid, error = await self.validator.validate_api_key_format("anthropic", valid_key)
        assert is_valid

        # Invalid format
        invalid_key = "sk-wrong-format"
        is_valid, _error = await self.validator.validate_api_key_format("anthropic", invalid_key)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_validate_encrypted_api_key(self):
        """Test validation of encrypted API keys"""
        # Encrypt a valid key
        plaintext_key = "sk-1234567890abcdef1234567890abcdef"
        encrypted_key = encrypt_api_key(plaintext_key)

        is_valid, error = await self.validator.validate_api_key_format("openai", encrypted_key)
        assert is_valid
        assert error == ""

    @pytest.mark.asyncio
    async def test_validate_empty_api_key(self):
        """Test validation of empty API key"""
        is_valid, error = await self.validator.validate_api_key_format("openai", "")
        assert not is_valid
        assert "API key is required" in error

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_validate_connectivity_success(self, mock_session):
        """Test successful connectivity validation"""
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = AsyncMock()
        mock_session_instance.request = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        is_connected, error = await self.validator.validate_connectivity(
            "openai", "sk-test1234567890abcdef", "https://api.openai.com/v1"
        )

        assert is_connected
        assert error == ""

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_validate_connectivity_failure(self, mock_session):
        """Test connectivity validation failure"""
        # Mock failed HTTP response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session_instance = AsyncMock()
        mock_session_instance.request = AsyncMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance

        is_connected, error = await self.validator.validate_connectivity(
            "openai", "sk-test1234567890abcdef", "https://api.openai.com/v1"
        )

        assert not is_connected
        assert "HTTP 500" in error

    @pytest.mark.asyncio
    async def test_validate_provider_config_complete(self):
        """Test complete provider configuration validation"""
        with patch.object(self.validator, "validate_connectivity", return_value=(True, "")):
            result = await self.validator.validate_provider_config(
                provider="openai",
                api_key="sk-1234567890abcdef1234567890abcdef",
                base_url="https://api.openai.com/v1",
                test_connectivity=True,
            )

            assert result["is_valid"]
            assert result["provider"] == "openai"
            assert len(result["errors"]) == 0


class TestEnhancedSettings:
    """Test enhanced Settings class functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.settings = Settings()

    def test_get_decrypted_api_key(self):
        """Test getting decrypted API key"""
        # Set an encrypted key
        test_key = "sk-test1234567890abcdef"
        encrypted_key = encrypt_api_key(test_key)

        with patch.object(self.settings, "OPENAI_API_KEY", encrypted_key):
            decrypted = self.settings.get_decrypted_api_key("openai")
            assert decrypted == test_key

    def test_set_encrypted_api_key(self):
        """Test setting encrypted API key"""
        test_key = "sk-test1234567890abcdef"

        # Mock encryption setting
        with patch.object(self.settings, "ENCRYPT_API_KEYS_AT_REST", True):
            success = self.settings.set_encrypted_api_key("openai", test_key)
            assert success

            # Verify key was encrypted
            stored_key = self.settings.OPENAI_API_KEY
            assert is_encrypted(stored_key)

    def test_validate_api_key_format(self):
        """Test API key format validation in settings"""
        # Valid key
        valid_key = "sk-1234567890abcdef1234567890abcdef"
        is_valid, error = self.settings.validate_api_key_format("openai", valid_key)
        assert is_valid
        assert error == ""

        # Invalid key
        invalid_key = "invalid"
        is_valid, error = self.settings.validate_api_key_format("openai", invalid_key)
        assert not is_valid
        assert error != ""

    def test_get_provider_config_dict(self):
        """Test getting provider configuration dictionary"""
        with patch.object(self.settings, "OPENAI_API_KEY", "sk-test123"):
            config_dict = self.settings.get_provider_config_dict()

            assert "openai" in config_dict
            assert config_dict["openai"]["name"] == "openai"
            assert config_dict["openai"]["api_key"] == "sk-test123"

    def test_get_connection_mode(self):
        """Test getting connection mode"""
        assert self.settings.get_connection_mode() == APIConnectionMode.DIRECT

    def test_get_proxy_config(self):
        """Test getting proxy configuration"""
        proxy_config = self.settings.get_proxy_config()

        assert "enabled" in proxy_config
        assert "endpoint" in proxy_config
        assert "health_check" in proxy_config
        assert "timeout" in proxy_config

    @pytest.mark.asyncio
    async def test_reload_configuration(self):
        """Test configuration hot-reload functionality"""
        # Mock environment changes
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-new-key-123"}):
            result = await self.settings.reload_configuration()

            assert result["status"] == "success"
            assert "changes" in result
            assert "reloaded_at" in result


class TestEnhancedConfigManager:
    """Test enhanced configuration manager"""

    def setup_method(self):
        """Setup test environment"""
        self.config_manager = EnhancedConfigManager()

    @pytest.mark.asyncio
    async def test_reload_config(self):
        """Test configuration reload"""
        result = await self.config_manager.reload_config()

        assert "status" in result
        assert "elapsed_ms" in result
        assert "changes" in result

    @pytest.mark.asyncio
    async def test_validate_proxy_mode_config_disabled(self):
        """Test proxy mode validation when disabled"""
        with patch("app.core.config.settings.API_CONNECTION_MODE", APIConnectionMode.DIRECT):
            result = await self.config_manager.validate_proxy_mode_config()

            # Should be valid when proxy mode is disabled
            assert result["is_valid"]

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_validate_proxy_mode_config_enabled_success(self, mock_session):
        """Test proxy mode validation when enabled and proxy is reachable"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response

        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value = mock_response
        mock_session_instance.__aenter__.return_value = mock_session_instance
        mock_session.return_value = mock_session_instance

        with patch("app.core.config.settings.API_CONNECTION_MODE", APIConnectionMode.PROXY):
            result = await self.config_manager.validate_proxy_mode_config()

            assert result["is_valid"]

    @pytest.mark.asyncio
    async def test_get_provider_config_summary(self):
        """Test getting provider configuration summary"""
        summary = await self.config_manager.get_provider_config_summary()

        assert "total_providers" in summary
        assert "configured_providers" in summary
        assert "connection_mode" in summary
        assert "proxy_config" in summary

    def test_add_remove_reload_callback(self):
        """Test adding and removing reload callbacks"""
        callback_called = False

        def test_callback():
            nonlocal callback_called
            callback_called = True

        # Add callback
        self.config_manager.add_reload_callback(test_callback)
        assert test_callback in self.config_manager._reload_callbacks

        # Remove callback
        self.config_manager.remove_reload_callback(test_callback)
        assert test_callback not in self.config_manager._reload_callbacks


class TestConfigurationIntegration:
    """Integration tests for configuration system"""

    @pytest.mark.asyncio
    async def test_end_to_end_provider_configuration(self):
        """Test complete provider configuration flow"""
        settings = Settings()
        validator = ProviderConfigValidator()

        # Test setting up a new provider
        api_key = "sk-test1234567890abcdef1234567890abcdef"

        # Validate key format
        is_valid, _error = await validator.validate_api_key_format("openai", api_key)
        assert is_valid

        # Set encrypted key
        success = settings.set_encrypted_api_key("openai", api_key)
        assert success

        # Verify we can get the decrypted key back
        decrypted = settings.get_decrypted_api_key("openai")
        assert decrypted == api_key

        # Test configuration dict
        config_dict = settings.get_provider_config_dict()
        assert "openai" in config_dict

    @pytest.mark.asyncio
    async def test_configuration_validation_with_errors(self):
        """Test configuration validation with various error scenarios"""
        validator = ProviderConfigValidator()

        # Test invalid API key format
        result = await validator.validate_provider_config(
            provider="openai",
            api_key="invalid_key",
            base_url="https://api.openai.com/v1",
            test_connectivity=False,  # Skip connectivity to focus on format
        )

        assert not result["is_valid"]
        assert len(result["errors"]) > 0
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_proxy_mode_configuration(self):
        """Test proxy mode configuration end-to-end"""
        settings = Settings()

        # Test proxy config retrieval
        proxy_config = settings.get_proxy_config()
        assert proxy_config["endpoint"] == "http://localhost:8080"

        # Test proxy mode validation
        config_manager = EnhancedConfigManager()
        result = await config_manager.validate_proxy_mode_config()

        # Should be valid when proxy mode is disabled (default)
        assert result["is_valid"]


class TestConfigurationEdgeCases:
    """Test edge cases and error scenarios"""

    def test_unknown_provider_handling(self):
        """Test handling of unknown provider names"""
        settings = Settings()

        # Should return None for unknown provider
        result = settings.get_decrypted_api_key("unknown_provider")
        assert result is None

        # Should fail to set key for unknown provider
        success = settings.set_encrypted_api_key("unknown_provider", "test_key")
        assert not success

    @pytest.mark.asyncio
    async def test_validation_with_network_errors(self):
        """Test validation behavior with network connectivity issues"""
        validator = ProviderConfigValidator()

        # Test with unreachable URL
        result = await validator.validate_provider_config(
            provider="openai",
            api_key="sk-1234567890abcdef1234567890abcdef",
            base_url="https://unreachable-url-12345.com",
            test_connectivity=True,
        )

        assert not result["is_valid"]
        assert any("Connection" in error for error in result["errors"])

    def test_encryption_with_malformed_keys(self):
        """Test encryption with malformed keys"""
        # Test with malformed encrypted key
        malformed = "enc:not_valid_base64!@#"

        with pytest.raises(EncryptionError):
            decrypt_api_key(malformed)

    @pytest.mark.asyncio
    async def test_hot_reload_with_no_changes(self):
        """Test hot-reload when no configuration changes occurred"""
        config_manager = EnhancedConfigManager()

        # First reload
        result1 = await config_manager.reload_config()

        # Second reload immediately (no changes)
        result2 = await config_manager.reload_config()

        # Both should succeed
        assert result1["status"] == "success"
        assert result2["status"] == "success"

        # Changes should be minimal or empty
        assert isinstance(result2["changes"], dict)


# Pytest configuration
pytestmark = [
    pytest.mark.unit,  # Mark all tests as unit tests
    pytest.mark.asyncio,  # Allow async tests
]
