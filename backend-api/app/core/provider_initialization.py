"""
Provider Registry Initialization - Bootstrap provider plugins at application startup.

This module handles the registration of all provider plugins with the
UnifiedProviderRegistry during FastAPI application startup.
"""

import logging
import os

from app.infrastructure.plugins import (
    AnthropicPlugin,
    BigModelPlugin,
    DeepSeekPlugin,
    GooglePlugin,
    OpenAIPlugin,
    RoutewayPlugin,
)
from app.services.unified_provider_registry import unified_registry

logger = logging.getLogger(__name__)


def initialize_providers() -> None:
    """
    Initialize and register all provider plugins.

    This function should be called during FastAPI application startup
    (in the lifespan context manager).

    Registers:
    - OpenAI (aliases: gpt)
    - Anthropic (aliases: claude)
    - Google (aliases: gemini)

    Provider enablement is determined by API key presence in environment.
    """
    logger.info("Initializing provider plugins...")

    # Track registration statistics
    registered_count = 0
    failed_count = 0

    # Register OpenAI
    try:
        openai_plugin = OpenAIPlugin()
        if openai_plugin.validate_config():
            unified_registry.register_provider(
                openai_plugin,
                aliases=["gpt"],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered OpenAI provider (aliases: gpt)")
        else:
            logger.warning("⚠ OpenAI provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register OpenAI provider: {e}")
        failed_count += 1

    # Register Anthropic
    try:
        anthropic_plugin = AnthropicPlugin()
        if anthropic_plugin.validate_config():
            unified_registry.register_provider(
                anthropic_plugin,
                aliases=["claude"],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered Anthropic provider (aliases: claude)")
        else:
            logger.warning("⚠ Anthropic provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register Anthropic provider: {e}")
        failed_count += 1

    # Register Google/Gemini
    try:
        google_plugin = GooglePlugin()
        if google_plugin.validate_config():
            unified_registry.register_provider(
                google_plugin,
                aliases=["gemini"],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered Google provider (aliases: gemini)")
        else:
            logger.warning("⚠ Google provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register Google provider: {e}")
        failed_count += 1

    # Register DeepSeek
    try:
        deepseek_plugin = DeepSeekPlugin()
        if deepseek_plugin.validate_config():
            unified_registry.register_provider(
                deepseek_plugin,
                aliases=[],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered DeepSeek provider")
        else:
            logger.warning("⚠ DeepSeek provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register DeepSeek provider: {e}")
        failed_count += 1

    # Register BigModel
    try:
        bigmodel_plugin = BigModelPlugin()
        if bigmodel_plugin.validate_config():
            unified_registry.register_provider(
                bigmodel_plugin,
                aliases=["zhipu", "glm"],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered BigModel provider (aliases: zhipu, glm)")
        else:
            logger.warning("⚠ BigModel provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register BigModel provider: {e}")
        failed_count += 1

    # Register Routeway
    try:
        routeway_plugin = RoutewayPlugin()
        if routeway_plugin.validate_config():
            unified_registry.register_provider(
                routeway_plugin,
                aliases=[],
                run_hooks=True,
            )
            registered_count += 1
            logger.info("✓ Registered Routeway provider")
        else:
            logger.warning("⚠ Routeway provider not configured (missing API key)")
    except Exception as e:
        logger.error(f"✗ Failed to register Routeway provider: {e}")
        failed_count += 1

    # Log summary
    total_attempted = registered_count + failed_count
    logger.info(
        f"Provider registration complete: {registered_count}/{total_attempted} registered, "
        f"{failed_count} failed"
    )

    # Log registry statistics
    stats = unified_registry.get_statistics()
    logger.info(
        f"Registry state: {stats['total_providers']} providers, "
        f"{stats['total_models']} models, "
        f"{stats['total_aliases']} aliases"
    )

    # Warn if no providers are registered
    if registered_count == 0:
        logger.warning(
            "⚠⚠⚠ NO PROVIDERS REGISTERED! ⚠⚠⚠\n"
            "The application will not be able to generate LLM responses.\n"
            "Please configure at least one provider API key:\n"
            "  - OPENAI_API_KEY for OpenAI/GPT\n"
            "  - ANTHROPIC_API_KEY for Anthropic/Claude\n"
            "  - GOOGLE_API_KEY for Google/Gemini\n"
            "  - DEEPSEEK_API_KEY for DeepSeek AI\n"
            "  - BIGMODEL_API_KEY for BigModel/GLM\n"
            "  - ROUTEWAY_API_KEY for Routeway AI"
        )


def get_default_provider() -> tuple[str, str]:
    """
    Determine the default provider and model from environment or registry.

    Returns the first available provider/model combination based on:
    1. Explicitly configured defaults (CHIMERA_DEFAULT_PROVIDER, CHIMERA_DEFAULT_MODEL)
    2. First registered provider with available models

    Returns:
        Tuple of (provider_id, model_id)

    Raises:
        RuntimeError: If no providers are registered
    """
    # Check for explicit configuration
    default_provider = os.getenv("CHIMERA_DEFAULT_PROVIDER")
    default_model = os.getenv("CHIMERA_DEFAULT_MODEL")

    if default_provider and default_model:
        # Validate explicit configuration
        if unified_registry.validate_selection(default_provider, default_model):
            logger.info(
                f"Using configured default: {default_provider}/{default_model}"
            )
            return (default_provider, default_model)
        else:
            logger.warning(
                f"Configured default {default_provider}/{default_model} is invalid. "
                "Falling back to auto-detection."
            )

    # Auto-detect from registry
    providers = unified_registry.list_providers(enabled_only=True)

    if not providers:
        raise RuntimeError(
            "No providers registered! Cannot determine default provider. "
            "Please configure at least one provider API key."
        )

    # Use first available provider
    first_provider = providers[0]
    models = unified_registry.get_models(
        provider_id=first_provider.id,
        enabled_only=True,
    )

    if not models:
        raise RuntimeError(
            f"Provider '{first_provider.id}' has no available models"
        )

    # Use first available model
    first_model = models[0]

    logger.info(
        f"Auto-detected default: {first_provider.id}/{first_model.id} "
        f"(from {len(providers)} available providers)"
    )

    return (first_provider.id, first_model.id)


async def health_check_providers() -> dict[str, bool]:
    """
    Perform health check on all registered providers.

    Useful for application health endpoints to verify provider connectivity.

    Returns:
        Dictionary mapping provider IDs to health status
    """
    logger.debug("Performing provider health checks...")
    results = await unified_registry.health_check()

    # Log results
    healthy_count = sum(1 for status in results.values() if status)
    total_count = len(results)

    logger.info(
        f"Health check complete: {healthy_count}/{total_count} providers healthy"
    )

    for provider_id, is_healthy in results.items():
        status = "✓ healthy" if is_healthy else "✗ unhealthy"
        logger.debug(f"  {provider_id}: {status}")

    return results


def refresh_provider_models() -> None:
    """
    Refresh the model cache for all providers.

    Useful for periodic updates or when provider catalogs change.
    """
    logger.info("Refreshing provider model caches...")
    unified_registry.refresh_models_cache()

    stats = unified_registry.get_statistics()
    logger.info(f"Model cache refreshed: {stats['total_models']} models available")
