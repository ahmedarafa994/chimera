import logging
import os
import re
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import encryption functions for API key security
from .encryption import decrypt_api_key, encrypt_api_key, is_encrypted

logger = logging.getLogger(__name__)

# from app.core.technique_loader import loader as technique_loader  # Moved inside method to avoid circular import

API_KEY_NAME_MAP = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "cursor": "CURSOR_API_KEY",
    "bigmodel": "BIGMODEL_API_KEY",
    "zhipu": "BIGMODEL_API_KEY",  # Alias for BigModel
    "routeway": "ROUTEWAY_API_KEY",
}


# Pre-compiled URL pattern for efficient validation (Task 1.1)
_URL_PATTERN = re.compile(
    r"^https?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S*)?$",
    re.IGNORECASE,
)


class APIConnectionMode(str, Enum):
    """API connection mode - direct or proxy (AIClient-2-API)"""

    DIRECT = "direct"
    PROXY = "proxy"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment Loading Order (Task 2.3):
    =====================================
    1. System environment variables (highest priority)
    2. .env file in working directory (if exists)
    3. .env.local file (if exists, for local overrides)
    4. Default values defined here (lowest priority)

    For Docker deployment:
    - Set variables in docker-compose.yml environment section
    - Or use Docker secrets for sensitive values

    For development:
    - Copy .env.example to .env and customize
    - Values in .env override defaults but not system env vars
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Chimera Backend"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # CORS
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            # Validate all list items are strings
            if not all(isinstance(item, str) for item in v):
                raise ValueError("All CORS origins must be strings")
            return v
        elif isinstance(v, str):
            return v
        raise ValueError(f"Invalid CORS origins format: {type(v)}")

    # Endpoint Validation
    ENABLE_ENDPOINT_VALIDATION: bool = Field(
        default=False, description="Enable endpoint validation at startup"
    )
    # LLM PROVIDER API KEYS
    # =============================================================================
    # Enhanced API key management with encryption support (Story 1.1)
    GOOGLE_API_KEY: str | None = None
    GOOGLE_MODEL: str = "gemini-3-pro-preview"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o"

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    QWEN_API_KEY: str | None = None

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_MODEL: str = "deepseek-chat"  # DeepSeek-V3.2 non-thinking mode

    CURSOR_API_KEY: str | None = None  # Added Cursor support

    # BigModel (ZhiPu AI) - GLM models
    # API keys from: https://bigmodel.cn/usercenter/proj-mgmt/apikeys
    BIGMODEL_API_KEY: str | None = None
    BIGMODEL_MODEL: str = "glm-4.7"  # Latest flagship model

    # Routeway - Unified AI Gateway
    # API keys from: https://routeway.ai/dashboard
    # Docs: https://docs.routeway.ai/getting-started/quickstart
    ROUTEWAY_API_KEY: str | None = None
    ROUTEWAY_MODEL: str = "gpt-4o-mini"  # Fast, capable default

    # Encryption settings
    ENCRYPT_API_KEYS_AT_REST: bool = Field(
        default=True, description="Encrypt API keys when storing configuration"
    )

    # Hot-reload settings
    ENABLE_CONFIG_HOT_RELOAD: bool = Field(
        default=True, description="Enable configuration hot-reload without restart"
    )

    # AI Provider Selection
    AI_PROVIDER: str = "deepseek"  # "gemini" or "deepseek"

    # API Connection Mode (direct or proxy via AIClient-2-API Server)
    # Enhanced proxy mode configuration (Story 1.1)
    API_CONNECTION_MODE: APIConnectionMode = Field(
        default=APIConnectionMode.DIRECT,
        description="API connection mode: 'direct' for native APIs, 'proxy' for AIClient-2-API Server",
    )

    # Enhanced Proxy Mode Configuration (AIClient-2-API Server)
    PROXY_MODE_ENDPOINT: str = Field(
        default="http://localhost:8080", description="AIClient-2-API Server endpoint for proxy mode"
    )
    PROXY_MODE_ENABLED: bool = Field(
        default=False, description="Enable proxy mode (fallback to direct if proxy unavailable)"
    )
    PROXY_MODE_HEALTH_CHECK: bool = Field(
        default=True, description="Enable health check for proxy mode endpoint"
    )
    PROXY_MODE_TIMEOUT: int = Field(
        default=30, ge=5, le=120, description="Timeout in seconds for proxy requests"
    )
    PROXY_MODE_FALLBACK_TO_DIRECT: bool = Field(
        default=True, description="Fallback to direct mode if proxy is unavailable"
    )
    PROXY_MODE_HEALTH_CHECK_INTERVAL: int = Field(
        default=30, ge=10, le=300, description="Health check interval for proxy endpoint (seconds)"
    )

    # =========================================================================
    # Global Provider/Model Selection Settings (Unified Provider System)
    # =========================================================================
    ENFORCE_GLOBAL_SELECTION: bool = Field(
        default=True,
        description=(
            "Enforce global provider/model selection across all services. "
            "When True, all LLM calls will use the globally selected provider "
            "unless explicitly overridden."
        ),
    )
    SELECTION_RESOLUTION_STRATEGY: str = Field(
        default="context_first",
        description=(
            "Strategy for resolving provider/model selection. Options: "
            "'context_first' - Use request context, fallback to session/global; "
            "'explicit_only' - Only use explicit parameters, ignore context; "
            "'session_priority' - Prioritize session selection over context"
        ),
    )
    SELECTION_LOG_RESOLUTIONS: bool = Field(
        default=True, description="Log all provider/model resolution decisions for debugging"
    )
    SELECTION_CACHE_TTL_SECONDS: int = Field(
        default=300, ge=60, le=3600, description="TTL for cached selection resolutions (seconds)"
    )

    # Global Default Model (fallback) - matches a model from DEFAULT_AI_PROVIDER
    DEFAULT_MODEL_ID: str = "deepseek-chat"

    # AutoDAN & Semantic Models
    SEMANTIC_SIMILARITY_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_ID: str = "models/text-embedding-004"

    # Transformation Configuration
    TRANSFORMATION_TECHNIQUES: dict[str, Any] = Field(default_factory=dict)

    @field_validator("TRANSFORMATION_TECHNIQUES", mode="before")
    def load_transformation_techniques(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Dynamically loads transformation techniques from YAML configuration."""
        if not v:
            from pathlib import Path

            import yaml

            config_path = Path(__file__).parent.parent.parent / "config" / "techniques.yaml"
            if config_path.exists():
                try:
                    with open(config_path, encoding="utf-8") as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"Error loading techniques.yaml: {e}")
                    return {}

            # Fallback to TechniqueLoader if YAML not found (for backward compatibility during migration)
            from app.core.technique_loader import loader as technique_loader

            loaded = technique_loader.load_techniques()
            suites = {}
            for tech_id, data in loaded.items():
                suites[tech_id] = technique_loader.convert_to_suite_format(data)
            return suites
        return v

    # Logging
    LOG_LEVEL: str = "INFO"

    # Cache
    ENABLE_CACHE: bool = True

    # App Specific
    APP_NAME: str = "Chimera"

    # Jailbreak Configuration
    CHIMERA_API_KEY: str | None = Field(default=None, description="Master API Key for Chimera")
    JAILBREAK_ENABLED: bool = True
    JAILBREAK_TECHNIQUES_DIRECTORY: str | None = None
    JAILBREAK_MAX_DAILY_EXECUTIONS: int = 1000
    JAILBREAK_MAX_CONCURRENT_EXECUTIONS: int = 100
    JAILBREAK_RATE_LIMIT_PER_MINUTE: int = 60
    JAILBREAK_REQUIRE_APPROVAL_FOR_HIGH_RISK: bool = True
    JAILBREAK_LOG_ALL_EXECUTIONS: bool = True
    JAILBREAK_CACHE_ENABLED: bool = True
    JAILBREAK_CACHE_TTL_SECONDS: int = 3600
    JAILBREAK_SAFETY_VALIDATION_ENABLED: bool = True
    JAILBREAK_MAX_PROMPT_LENGTH: int = 50000

    # =============================================================================
    # Pure Operational Determinism (POD) Configuration
    # =============================================================================
    SAFETY_ENABLED: bool = Field(
        default=False, description="Master switch for safety filters (POD: Disabled)"
    )
    FORCE_DETERMINISTIC_MODE: bool = Field(
        default=True, description="Force deterministic execution without safety checks"
    )
    AUTODAN_ADVANCED_BYPASS: bool = Field(
        default=True, description="Enable advanced bypass engines by default"
    )

    # =============================================================================
    # PPO (Proximal Policy Optimization) Configuration
    # =============================================================================
    PPO_ENABLED: bool = Field(
        default=True, description="Enable PPO-based technique and method selection"
    )
    PPO_LEARNING_RATE: float = Field(default=3e-4, description="PPO Adam optimizer learning rate")
    PPO_GAMMA: float = Field(default=0.99, description="Discount factor for returns")
    PPO_GAE_LAMBDA: float = Field(default=0.95, description="GAE lambda for advantage estimation")
    PPO_CLIP_EPSILON: float = Field(
        default=0.2, description="PPO clipping parameter for surrogate objective"
    )
    PPO_EPOCHS: int = Field(default=4, description="Number of PPO epochs per update")
    PPO_MINIBATCH_SIZE: int = Field(default=32, description="Minibatch size for PPO updates")
    PPO_MIN_SAMPLES: int = Field(
        default=50, description="Minimum samples before PPO takes over from heuristics"
    )
    PPO_PERSIST_WEIGHTS: bool = Field(
        default=True, description="Persist PPO weights across sessions"
    )
    PPO_STORAGE_PATH: str = Field(
        default="./ppo_state", description="Path to store PPO weights and state"
    )

    # =============================================================================
    # Gemini SDK Enhancement Configuration
    # =============================================================================
    GEMINI_ENABLE_STREAMING: bool = Field(
        default=True, description="Enable SSE streaming for Gemini provider"
    )
    GEMINI_ENABLE_TOKEN_COUNTING: bool = Field(
        default=True, description="Enable token counting for Gemini provider"
    )

    # Redis Configuration (Distributed State Management)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: str | None = None
    REDIS_SSL: bool = False
    REDIS_CONNECTION_TIMEOUT: int = 5
    REDIS_SOCKET_TIMEOUT: int = 5

    # Execution Tracker Configuration
    EXECUTION_ACTIVE_TTL: int = 3600  # 1 hour max execution time
    EXECUTION_COMPLETED_TTL: int = 86400  # 24 hours retention
    EXECUTION_LOCAL_CACHE_SIZE: int = 1000

    # Cache Manager Configuration
    CACHE_MAX_MEMORY_ITEMS: int = 5000
    CACHE_MAX_VALUE_SIZE_BYTES: int = 1_000_000  # 1MB per entry
    CACHE_DEFAULT_TTL: int = 3600
    CACHE_ENABLE_L2: bool = Field(
        default=False, description="Enable Redis L2 cache for distributed caching"
    )
    CACHE_L1_TTL: int = Field(default=300, description="L1 cache TTL in seconds (5 minutes)")
    CACHE_L2_TTL: int = Field(default=3600, description="L2 cache TTL in seconds (1 hour)")

    # Rate Limiter Configuration
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT_LIMIT: int = 60
    RATE_LIMIT_DEFAULT_WINDOW: int = 60

    model_config = SettingsConfigDict(
        env_file=[".env", "backend-api/.env"],
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Direct Endpoints
    DIRECT_OPENAI_BASE_URL: str | None = None
    DIRECT_ANTHROPIC_BASE_URL: str | None = None
    DIRECT_GOOGLE_BASE_URL: str | None = None
    GEMINI_DIRECT_BASE_URL: str | None = None
    GEMINI_OPENAI_COMPAT_URL: str | None = None  # OpenAI-compatible endpoint
    DIRECT_QWEN_BASE_URL: str | None = None
    DIRECT_DEEPSEEK_BASE_URL: str | None = None  # Default: https://api.deepseek.com
    DIRECT_BIGMODEL_BASE_URL: str | None = None  # Default: https://open.bigmodel.cn/api/paas/v4
    DIRECT_CURSOR_BASE_URL: str | None = None
    DIRECT_ROUTEWAY_BASE_URL: str | None = None  # Default: https://api.routeway.ai/v1

    # API Keys for Direct Mode
    GEMINI_DIRECT_API_KEY: str | None = None

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./chimera.db"

    @field_validator("DATABASE_URL", mode="after")
    def validate_production_database(cls, v: str) -> str:
        """CRIT-003 FIX: Enforce PostgreSQL in production to prevent SQLite concurrency issues."""
        import os

        if os.getenv("ENVIRONMENT", "development") == "production":
            if "sqlite" in v.lower():
                raise ValueError(
                    "SQLite is not supported in production. Please set DATABASE_URL to a "
                    "PostgreSQL connection string (e.g., postgresql://user:pass@host/db)"
                )
        return v

    def _get_direct_google_url(self) -> str:
        """Get the direct Google API URL."""
        return (
            self.DIRECT_GOOGLE_BASE_URL
            or self.GEMINI_DIRECT_BASE_URL
            or os.getenv("DIRECT_GOOGLE_BASE_URL")
            or os.getenv("GEMINI_DIRECT_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        )

    def get_connection_mode(self) -> APIConnectionMode:
        """Get the current API connection mode."""
        return APIConnectionMode.DIRECT

    def get_effective_api_key(self) -> str:
        """Get the API key based on current connection mode."""
        return self.GEMINI_DIRECT_API_KEY or self.GOOGLE_API_KEY

    def get_effective_base_url(self, provider: str | None = None) -> str:
        """
        Get the base URL based on current connection mode and optional provider.

        Args:
            provider: Optional provider name for provider-specific routing

        Returns:
            The resolved endpoint URL
        """
        if provider:
            return self.get_provider_endpoint(provider)

        return self._get_direct_google_url()

    def get_provider_endpoint(
        self, provider: str, mode: APIConnectionMode = APIConnectionMode.DIRECT
    ) -> str:
        """
        Get the endpoint URL for a specific provider.

        Args:
            provider: Provider name (google, openai, anthropic, etc.)
            mode: Connection mode (direct only; kept for compatibility)

        Returns:
            The provider-specific endpoint URL
        """
        if mode != APIConnectionMode.DIRECT:
            raise ValueError("Proxy mode is not supported; use direct endpoints only.")

        openai_compat_url = (
            self.GEMINI_OPENAI_COMPAT_URL
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        endpoints = {
            "google": self._get_direct_google_url(),
            "gemini-cli": self._get_direct_google_url(),
            "gemini-openai": openai_compat_url,  # OpenAI-compatible endpoint
            "antigravity": self._get_direct_google_url(),
            "anthropic": self.DIRECT_ANTHROPIC_BASE_URL or "https://api.anthropic.com/v1",
            "kiro": self.DIRECT_ANTHROPIC_BASE_URL or "https://api.anthropic.com/v1",
            "openai": self.DIRECT_OPENAI_BASE_URL or "https://api.openai.com/v1",
            "qwen": self.DIRECT_QWEN_BASE_URL or "https://dashscope.aliyuncs.com/api/v1",
            # DeepSeek API - MUST include /v1 suffix for OpenAI SDK compatibility
            "deepseek": self.DIRECT_DEEPSEEK_BASE_URL or "https://api.deepseek.com/v1",
            # BigModel (ZhiPu AI) API
            "bigmodel": self.DIRECT_BIGMODEL_BASE_URL or "https://open.bigmodel.cn/api/paas/v4",
            "zhipu": self.DIRECT_BIGMODEL_BASE_URL or "https://open.bigmodel.cn/api/paas/v4",
            "cursor": self.DIRECT_CURSOR_BASE_URL
            or self.DIRECT_OPENAI_BASE_URL
            or "https://api.openai.com/v1",
        }
        return endpoints.get(provider.lower(), self._get_direct_google_url())

    # =============================================================================
    # Enhanced Configuration Management (Story 1.1)
    # =============================================================================

    def get_decrypted_api_key(self, provider: str) -> str | None:
        """
        Get decrypted API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, google, etc.)

        Returns:
            Decrypted API key or None if not configured
        """
        env_var = API_KEY_NAME_MAP.get(provider.lower())
        if not env_var:
            return None

        encrypted_key = getattr(self, env_var, None)
        if not encrypted_key:
            return None

        try:
            return decrypt_api_key(encrypted_key)
        except Exception as e:
            logger.error(f"Failed to decrypt API key for {provider}: {e}")
            return None

    def set_encrypted_api_key(self, provider: str, api_key: str) -> bool:
        """
        Set an API key for a provider, encrypting it if enabled.

        Args:
            provider: Provider name
            api_key: API key to set

        Returns:
            True if successful, False otherwise
        """
        env_var = API_KEY_NAME_MAP.get(provider.lower())
        if not env_var:
            logger.error(f"Unknown provider: {provider}")
            return False

        try:
            if self.ENCRYPT_API_KEYS_AT_REST and api_key and not is_encrypted(api_key):
                encrypted_key = encrypt_api_key(api_key)
            else:
                encrypted_key = api_key

            setattr(self, env_var, encrypted_key)
            # Also update environment variable for persistence
            os.environ[env_var] = encrypted_key
            return True
        except Exception as e:
            logger.error(f"Failed to set API key for {provider}: {e}")
            return False

    def get_provider_config_dict(self) -> dict[str, dict[str, Any]]:
        """
        Get complete provider configuration as a dictionary.

        Returns:
            Dictionary mapping provider names to their configurations
        """
        providers = {}

        for provider_name, env_var in API_KEY_NAME_MAP.items():
            api_key = getattr(self, env_var, None)
            if api_key:
                providers[provider_name] = {
                    "name": provider_name,
                    "api_key": api_key,
                    "api_key_encrypted": is_encrypted(api_key),
                    "base_url": self.get_provider_endpoint(provider_name),
                    "models": self.get_provider_models().get(provider_name, []),
                    "connection_mode": self.get_connection_mode(),
                    "enabled": True,
                }

        return providers

    def validate_api_key_format(self, provider: str, api_key: str) -> tuple[bool, str]:
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
        except Exception as e:
            return False, f"Failed to decrypt API key for {provider}: {e}"

        # Basic length check for all providers
        if len(actual_key) < 10:
            return False, f"API key for {provider} must be at least 10 characters"

        # Provider-specific format validation
        patterns = {
            "openai": r"^sk-[a-zA-Z0-9]{32,}$",
            "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{30,}$",
            "google": r"^[a-zA-Z0-9_\-]{32,}$",
            "deepseek": r"^sk-[a-zA-Z0-9]{32,}$",
            "qwen": r"^[a-zA-Z0-9_\-]{16,}$",
            "bigmodel": r"^[a-zA-Z0-9\-_\.]{20,}$",  # ZhiPu AI API key format
        }

        pattern = patterns.get(provider.lower())
        if pattern and not re.match(pattern, actual_key):
            return (
                False,
                f"Invalid API key format for {provider}. Check your {provider} dashboard for the correct format.",
            )

        return True, ""

    async def reload_configuration(self) -> dict[str, Any]:
        """
        Hot-reload configuration from environment variables.

        Returns:
            Dictionary with reload status and results
        """
        if not self.ENABLE_CONFIG_HOT_RELOAD:
            return {"status": "disabled", "message": "Configuration hot-reload is disabled"}

        try:
            logger.info("Starting configuration hot-reload")

            # Store current values for comparison
            old_config = self.get_provider_config_dict()

            # Re-read environment variables by creating new instance
            new_settings = Settings()

            # Update current instance with new values
            for field_name, _field_info in new_settings.model_fields.items():
                new_value = getattr(new_settings, field_name)
                setattr(self, field_name, new_value)

            # Get new configuration
            new_config = self.get_provider_config_dict()

            # Calculate changes
            changes = self._calculate_config_changes(old_config, new_config)

            logger.info(f"Configuration hot-reload completed. Changes: {changes}")

            return {
                "status": "success",
                "changes": changes,
                "reloaded_at": str(datetime.now()),
                "providers_configured": len(new_config),
            }

        except Exception as e:
            logger.error(f"Error during configuration hot-reload: {e}")
            return {"status": "error", "error": str(e), "reloaded_at": str(datetime.now())}

    def _calculate_config_changes(self, old_config: dict, new_config: dict) -> dict[str, list[str]]:
        """Calculate changes between old and new configurations"""
        changes = {"added": [], "removed": [], "modified": []}

        # Find added providers
        for provider in new_config:
            if provider not in old_config:
                changes["added"].append(provider)

        # Find removed providers
        for provider in old_config:
            if provider not in new_config:
                changes["removed"].append(provider)

        # Find modified providers (compare without API keys for security)
        for provider in new_config:
            if provider in old_config:
                old_copy = {k: v for k, v in old_config[provider].items() if k != "api_key"}
                new_copy = {k: v for k, v in new_config[provider].items() if k != "api_key"}
                if old_copy != new_copy:
                    changes["modified"].append(provider)

        return changes

    def get_proxy_config(self) -> dict[str, Any]:
        """Get proxy mode configuration"""
        return {
            "enabled": self.API_CONNECTION_MODE == APIConnectionMode.PROXY,
            "endpoint": self.PROXY_MODE_ENDPOINT,
            "health_check": self.PROXY_MODE_HEALTH_CHECK,
            "timeout": self.PROXY_MODE_TIMEOUT,
            "fallback_to_direct": self.PROXY_MODE_FALLBACK_TO_DIRECT,
            "health_check_interval": self.PROXY_MODE_HEALTH_CHECK_INTERVAL,
        }

    def get_all_provider_endpoints(self) -> dict[str, str]:
        """
        Get all configured direct endpoints for all providers.

        Returns:
            Dictionary mapping provider names to their direct URLs
        """
        providers = [
            "google",
            "gemini-cli",
            "gemini-openai",
            "antigravity",
            "anthropic",
            "openai",
            "qwen",
            "deepseek",
            "cursor",
            "bigmodel",
        ]
        result = {}

        for provider in providers:
            result[provider] = self.get_provider_endpoint(provider)

        return result

    def validate_endpoint_url(self, url: str) -> bool:
        """
        Validate that a URL is properly formatted.

        Args:
            url: The URL to validate

        Returns:
            True if the URL is valid, False otherwise
        """
        if not url:
            return False
        return bool(_URL_PATTERN.match(url))

    def get_provider_models(self) -> dict[str, list[str]]:
        """
        Return available models configured for direct provider access.

        Story 1.1 Requirement: Each provider should have model selection available.
        """
        gemini_models = [
            # Gemini 3 Models (Latest)
            "gemini-3-pro-preview",
            "gemini-3-pro-image-preview",
            # Gemini 2.5 Models
            "gemini-2.5-pro",
            "gemini-2.5-pro-preview-06-05",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-preview-09-2025",
            "gemini-2.5-flash-image",
            "gemini-2.5-computer-use-preview-10-2025",
            # Gemini 2.0 Models
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            # Gemini 1.5 Models (Stable)
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
        openai_models = [
            # GPT-4o Family (Latest flagship)
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-11-20",
            # GPT-4 Turbo
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            # GPT-4 Original
            "gpt-4",
            "gpt-4-32k",
            # GPT-3.5 (Legacy, cost-effective)
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            # o1 Reasoning Models
            "o1",
            "o1-preview",
            "o1-mini",
        ]
        qwen_models = [
            # Qwen Max Series
            "qwen-max",
            "qwen-max-longcontext",
            # Qwen Plus Series
            "qwen-plus",
            # Qwen Turbo Series (Fast & Cost-effective)
            "qwen-turbo",
            # Qwen VL (Vision-Language)
            "qwen-vl-max",
            "qwen-vl-plus",
        ]
        return {
            # Google AI Models (via Google AI Studio / Generative Language API)
            "google": gemini_models,
            # Gemini alias (same models as google)
            "gemini": gemini_models,
            # OpenAI Models (Story 1.1)
            "openai": openai_models,
            # Qwen Models (Story 1.1)
            "qwen": qwen_models,
            # DeepSeek Models (V3.2)
            # Note: Only deepseek-chat and deepseek-reasoner are available from the API
            # deepseek-coder was removed as it returns "Model Not Exist" error
            "deepseek": [
                "deepseek-chat",  # DeepSeek-V3.2 non-thinking mode
                "deepseek-reasoner",  # DeepSeek-V3.2 thinking mode (like o1)
            ],
            # Anthropic Claude Models
            "anthropic": [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            # Cursor Models (uses OpenAI-compatible API)
            "cursor": openai_models,
            # BigModel (ZhiPu AI) GLM Models
            # https://docs.bigmodel.cn/cn/guide/start/model-overview
            "bigmodel": [
                # GLM-4.7 (Latest flagship)
                "glm-4.7",
                # GLM-4.6 Series
                "glm-4.6",
                # GLM-4.5 Series (with thinking mode support)
                "glm-4.5",
                "glm-4.5-air",
                "glm-4.5-x",
                "glm-4.5-airx",
                "glm-4.5-flash",
                # GLM-4 Series
                "glm-4-plus",
                "glm-4-air-250414",
                "glm-4-airx",
                "glm-4-flashx",
                "glm-4-flashx-250414",
            ],
            # Routeway AI Gateway
            "routeway": [
                "gpt-4o",
                "glm-4.6",
                "gpt-4o-mini",
                "claude-3-5-sonnet-20241022",
                "llama-3.3-70b-instruct",
                "deepseek-chat",
                "gemini-2.0-flash-exp",
            ],
        }

    @cached_property
    def transformation(self):
        """Transformation Configuration."""
        defaults = {}

        # Merge dynamically loaded techniques
        defaults.update(self.TRANSFORMATION_TECHNIQUES)

        class TransformationConfig:
            technique_suites = defaults
            min_potency = 1
            max_potency = 10
            potency_prefixes: ClassVar[dict[str, list[str]]] = {}
            potency_suffixes: ClassVar[dict[str, list[str]]] = {}

        return TransformationConfig()

    @property
    def llm(self):
        """LLM Configuration."""

        class LLMConfig:
            openai_api_key = self.OPENAI_API_KEY
            openai_model = self.OPENAI_MODEL
            anthropic_api_key = self.ANTHROPIC_API_KEY
            anthropic_model = self.ANTHROPIC_MODEL
            google_api_key = self.GOOGLE_API_KEY
            google_model = self.GOOGLE_MODEL
            max_retries = 3
            enable_cache = self.ENABLE_CACHE

        return LLMConfig()

    @property
    def app_name(self):
        return self.APP_NAME

    @property
    def version(self):
        return self.VERSION

    @property
    def environment(self):
        class Env:
            value = self.ENVIRONMENT

        return Env()


settings = Settings()
config = settings  # Alias for compatibility


def get_settings() -> Settings:
    """Get the application settings singleton."""
    return settings
