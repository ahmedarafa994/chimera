#!/usr/bin/env python3
# =============================================================================
# Configuration System Demo and Test Script
# =============================================================================
# Demonstrates Story 1.1 implementation features:
# - API key encryption/decryption
# - Hot-reload functionality
# - Configuration validation
# - Provider management
# - Proxy mode configuration
# =============================================================================

import asyncio
import os

from app.core.config import Settings
from app.core.config_manager import EnhancedConfigManager
from app.core.encryption import decrypt_api_key, encrypt_api_key


async def demo_api_key_encryption() -> None:
    """Demo API key encryption functionality."""
    # Test with a sample API key
    test_key = "sk-test1234567890abcdef1234567890abcdef"

    # Encrypt the key
    encrypted_key = encrypt_api_key(test_key)

    # Decrypt the key
    decrypt_api_key(encrypted_key)


async def demo_configuration_validation() -> None:
    """Demo configuration validation functionality."""
    config_manager = EnhancedConfigManager()

    # Test valid OpenAI key
    await config_manager.validator.validate_provider_config(
        provider="openai",
        api_key="sk-1234567890abcdef1234567890abcdef",
        base_url="https://api.openai.com/v1",
        test_connectivity=False,  # Skip connectivity for demo
    )

    # Test invalid key format
    await config_manager.validator.validate_provider_config(
        provider="openai",
        api_key="invalid_key_format",
        base_url="https://api.openai.com/v1",
        test_connectivity=False,
    )


async def demo_provider_configuration() -> None:
    """Demo provider configuration management."""
    settings = Settings()

    # Display current provider configuration
    provider_config = settings.get_provider_config_dict()
    for provider in provider_config:
        pass

    # Test API key format validation
    providers_to_test = ["openai", "anthropic", "google", "deepseek"]
    for provider in providers_to_test:
        test_key = f"test-key-for-{provider}"
        _is_valid, _error = settings.validate_api_key_format(provider, test_key)


async def demo_proxy_configuration() -> None:
    """Demo proxy mode configuration."""
    settings = Settings()

    # Display proxy configuration
    proxy_config = settings.get_proxy_config()
    for _key, _value in proxy_config.items():
        pass

    # Test proxy mode validation
    config_manager = EnhancedConfigManager()
    result = await config_manager.validate_proxy_mode_config()
    if result["errors"]:
        pass
    if result["warnings"]:
        pass


async def demo_hot_reload() -> None:
    """Demo hot-reload functionality."""
    Settings()
    config_manager = EnhancedConfigManager()

    # Get current configuration
    await config_manager.get_provider_config_summary()

    # Simulate environment variable change
    original_key = os.environ.get("OPENAI_API_KEY", "")

    # Set a new key in environment
    test_key = "sk-demo1234567890abcdef1234567890abcdef"
    os.environ["OPENAI_API_KEY"] = test_key

    # Test hot-reload
    await config_manager.reload_config()

    # Restore original key
    if original_key:
        os.environ["OPENAI_API_KEY"] = original_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)


async def demo_model_configurations() -> None:
    """Demo provider model configurations."""
    settings = Settings()

    # Display available models for each provider
    provider_models = settings.get_provider_models()
    for models in provider_models.values():
        for _model in models[:5]:  # Show first 5 models
            pass
        if len(models) > 5:
            pass

    # Display provider endpoints
    endpoints = settings.get_all_provider_endpoints()
    for _provider, _endpoint in endpoints.items():
        pass


async def demo_configuration_integration() -> None:
    """Demo end-to-end configuration integration."""
    settings = Settings()
    config_manager = EnhancedConfigManager()

    # Test complete workflow: set key -> encrypt -> validate -> reload

    # 1. Set API key with encryption
    test_key = "sk-integration1234567890abcdef1234567890"
    settings.set_encrypted_api_key("openai", test_key)

    # 2. Validate configuration
    await config_manager.validator.validate_provider_config(
        provider="openai",
        api_key=settings.get_decrypted_api_key("openai"),
        base_url=settings.get_provider_endpoint("openai"),
        test_connectivity=False,
    )

    # 3. Test provider config summary
    await config_manager.get_provider_config_summary()

    # 4. Test hot-reload
    await config_manager.reload_config()


async def run_comprehensive_demo() -> None:
    """Run comprehensive demonstration of all Story 1.1 features."""
    try:
        await demo_api_key_encryption()
        await demo_configuration_validation()
        await demo_provider_configuration()
        await demo_proxy_configuration()
        await demo_model_configurations()
        await demo_hot_reload()
        await demo_configuration_integration()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set up minimal environment for demo
    os.environ.setdefault("LOG_LEVEL", "WARNING")  # Reduce log noise

    # Run the demo
    asyncio.run(run_comprehensive_demo())
