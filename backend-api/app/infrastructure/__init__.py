"""Infrastructure Layer.

Story 1.2: Direct API Integration
Contains all LLM provider client implementations and supporting infrastructure.

Providers:
- GeminiClient: Google Gemini/PaLM API
- OpenAIClient: OpenAI GPT models
- AnthropicClient: Anthropic Claude models
- DeepSeekClient: DeepSeek models
- QwenClient: Alibaba Qwen (通义千问) models
- CursorClient: Cursor AI models

Infrastructure:
- RetryHandler: Centralized retry logic with exponential backoff
- ProviderManager: Provider failover and health management
- RedisRateLimiter: Distributed rate limiting
"""

from app.infrastructure.anthropic_client import AnthropicClient
from app.infrastructure.cursor_client import CursorClient
from app.infrastructure.deepseek_client import DeepSeekClient
from app.infrastructure.gemini_client import GeminiClient
from app.infrastructure.openai_client import OpenAIClient
from app.infrastructure.provider_manager import (
    FailoverConfig,
    ProviderManager,
    ProviderState,
    ProviderStatus,
    get_provider_manager,
    initialize_provider_manager,
)
from app.infrastructure.qwen_client import QwenClient
from app.infrastructure.redis_rate_limiter import (
    RedisRateLimiter,
    get_rate_limiter,
    shutdown_rate_limiter,
)
from app.infrastructure.retry_handler import (
    BackoffStrategy,
    RetryableError,
    RetryConfig,
    RetryExhaustedError,
    RetryHandler,
    get_provider_retry_config,
    get_retry_handler,
    with_retry,
)

__all__ = [
    "AnthropicClient",
    "BackoffStrategy",
    "CursorClient",
    "DeepSeekClient",
    "FailoverConfig",
    # Provider Clients
    "GeminiClient",
    "OpenAIClient",
    # Provider Manager
    "ProviderManager",
    "ProviderState",
    "ProviderStatus",
    "QwenClient",
    # Rate Limiter
    "RedisRateLimiter",
    "RetryConfig",
    "RetryExhaustedError",
    # Retry Handler
    "RetryHandler",
    "RetryableError",
    "get_provider_manager",
    "get_provider_retry_config",
    "get_rate_limiter",
    "get_retry_handler",
    "initialize_provider_manager",
    "shutdown_rate_limiter",
    "with_retry",
]
