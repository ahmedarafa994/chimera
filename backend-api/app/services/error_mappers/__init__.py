"""
Error Mappers Package for Provider-Specific Error Handling

This package contains error mappers for each AI provider, enabling
provider-specific error handling while maintaining a unified interface.

Each mapper implements the BaseErrorMapper protocol and provides:
- Provider-specific error code mapping
- Custom error messages
- Provider-specific retry strategies
"""

from app.services.error_mappers.openai_error_mapper import OpenAIErrorMapper
from app.services.error_mappers.anthropic_error_mapper import AnthropicErrorMapper
from app.services.error_mappers.google_error_mapper import GoogleErrorMapper
from app.services.error_mappers.deepseek_error_mapper import DeepSeekErrorMapper
from app.services.error_mappers.azure_error_mapper import AzureErrorMapper
from app.services.error_mappers.base_mapper import BaseErrorMapper

__all__ = [
    "BaseErrorMapper",
    "OpenAIErrorMapper",
    "AnthropicErrorMapper",
    "GoogleErrorMapper",
    "DeepSeekErrorMapper",
    "AzureErrorMapper",
]


def register_all_error_mappers():
    """Register all error mappers with the global error handler."""
    from app.services.provider_error_handler import get_error_handler

    handler = get_error_handler()

    # Register each mapper
    mappers = [
        ("openai", OpenAIErrorMapper()),
        ("anthropic", AnthropicErrorMapper()),
        ("google", GoogleErrorMapper()),
        ("gemini", GoogleErrorMapper()),  # Alias
        ("deepseek", DeepSeekErrorMapper()),
        ("azure", AzureErrorMapper()),
        ("azure-openai", AzureErrorMapper()),  # Alias
    ]

    for provider, mapper in mappers:
        handler.register_mapper(provider, mapper)

    return len(mappers)
