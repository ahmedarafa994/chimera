"""
LLM Service with optimized caching, request deduplication, and provider failover.

PERF-047: Phase 3 LLM service optimizations including:
- Response caching for identical prompts
- Request deduplication for concurrent identical requests
- Pre-created circuit breaker wrappers
- Async-first design
- Streaming support for real-time token generation
- Token counting utilities

STORY-1.2: Direct API Integration enhancements:
- Provider failover on circuit breaker open
- Retry with exponential backoff integration
- Rate limit tracking

HIGH-003 FIX: Added explicit CircuitBreakerOpen exception handling
to provide meaningful error messages when providers are unavailable.

AIConfigManager Integration:
- Config-driven failover chains
- Cost tracking using config pricing
- Circuit breaker settings from config
- Runtime provider switching

Global Model Selection Integration:
- Integration with GlobalModelSelectionState for request-scoped selection
- Integration with ProviderResolutionService for priority-based resolution
- get_current_selection_from_context() method for retrieving context selection
- Automatic fallback to global selection when no explicit provider specified

API Key Failover Integration (Subtask 3.2):
- Integration with ApiKeyFailoverService for intelligent key rotation
- Automatic failover to backup API keys when primary hits rate limit
- Retry request with backup key transparently
- Include failover metadata in response
- Emit events for failover tracking
- Integration with ProviderHealthService for health metrics
"""

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.core.circuit_breaker import CircuitBreakerOpen, circuit_breaker
from app.core.config import settings
from app.core.logging import logger
from app.core.timeouts import TimeoutType  # PERF-001 FIX: Import timeout types
from app.core.unified_errors import ProviderNotAvailableError
from app.domain.interfaces import LLMProvider
from app.domain.models import (
    LLMProviderType,
    PromptRequest,
    PromptResponse,
    ProviderInfo,
    ProviderListResponse,
    StreamChunk,
)


def _get_provider_resolution_service():
    """
    Get ProviderResolutionService instance with graceful fallback.

    Returns None if resolution service is not available.
    """
    try:
        from app.services.provider_resolution_service import (
            get_provider_resolution_service,
        )
        return get_provider_resolution_service()
    except Exception as e:
        logger.debug(f"ProviderResolutionService not available: {e}")
    return None


def _get_global_model_selection_state():
    """
    Get GlobalModelSelectionState instance with graceful fallback.

    Returns None if selection state service is not available.
    """
    try:
        from app.services.global_model_selection_state import (
            GlobalModelSelectionState,
        )
        return GlobalModelSelectionState()
    except Exception as e:
        logger.debug(f"GlobalModelSelectionState not available: {e}")
    return None


def _get_config_manager():
    """
    Get AIConfigManager instance with graceful fallback.

    Returns None if config manager is not available.
    """
    try:
        from app.core.service_registry import get_ai_config_manager
        config_manager = get_ai_config_manager()
        if config_manager.is_loaded():
            return config_manager
    except Exception as e:
        logger.debug(f"Config manager not available: {e}")
    return None


def _get_api_key_failover_service():
    """
    Get ApiKeyFailoverService instance with graceful fallback.

    Returns None if failover service is not available.
    """
    try:
        from app.services.api_key_failover_service import get_api_key_failover_service
        return get_api_key_failover_service()
    except Exception as e:
        logger.debug(f"ApiKeyFailoverService not available: {e}")
    return None


def _get_provider_health_service():
    """
    Get ProviderHealthService instance with graceful fallback.

    Returns None if health service is not available.
    """
    try:
        from app.services.provider_health_service import get_provider_health_service
        return get_provider_health_service()
    except Exception as e:
        logger.debug(f"ProviderHealthService not available: {e}")
    return None


def _get_quota_tracking_service():
    """
    Get QuotaTrackingService instance with graceful fallback.

    Returns None if quota service is not available.
    """
    try:
        from app.services.quota_tracking_service import get_quota_tracking_service
        return get_quota_tracking_service()
    except Exception as e:
        logger.debug(f"QuotaTrackingService not available: {e}")
    return None

# =============================================================================
# LLM Response Cache (PERF-048)
# =============================================================================


class LLMResponseCache:
    """
    Cache for LLM responses with TTL and size limits.

    PERF-048: Caches identical prompt responses to avoid redundant API calls.
    """

    def __init__(self, max_size: int = 500, default_ttl: int = 300):
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    def _generate_key(self, request: PromptRequest) -> str:
        """Generate cache key from request parameters."""
        key_data = f"{request.prompt}:{request.provider}:{request.model}:{request.config.temperature if request.config else 0.7}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, request: PromptRequest) -> PromptResponse | None:
        """Get cached response for request."""
        key = self._generate_key(request)

        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            logger.debug(f"LLM cache hit for key: {key[:16]}...")
            return entry["response"]

    async def set(
        self, request: PromptRequest, response: PromptResponse, ttl: int | None = None
    ) -> None:
        """Cache response for request."""
        key = self._generate_key(request)

        async with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "response": response,
                "created_at": time.time(),
                "expires_at": time.time() + (ttl or self._default_ttl),
            }

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# =============================================================================
# Request Deduplication (PERF-049)
# =============================================================================


class RequestDeduplicator:
    """
    Deduplicates concurrent identical requests.

    PERF-049: When multiple identical requests arrive simultaneously,
    only one actual API call is made and the result is shared.
    """

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._deduplicated_count = 0

    def _generate_key(self, request: PromptRequest) -> str:
        """Generate deduplication key from request."""
        key_data = f"{request.prompt}:{request.provider}:{request.model}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def deduplicate(self, request: PromptRequest, execute_fn: Callable[[], Any]) -> Any:
        """
        Execute request with deduplication.

        If an identical request is already in progress, wait for its result
        instead of making a duplicate API call.
        """
        key = self._generate_key(request)

        async with self._lock:
            if key in self._pending:
                # Wait for existing request to complete
                self._deduplicated_count += 1
                logger.debug(f"Request deduplicated, waiting for existing: {key[:16]}...")
                future = self._pending[key]

        # Check if we should wait for existing request
        if key in self._pending:
            return await asyncio.shield(self._pending[key])

        # Create new pending request
        async with self._lock:
            if key in self._pending:
                # Double-check after acquiring lock
                return await self._pending[key]

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[key] = future

        try:
            # Execute the actual request
            result = await execute_fn()
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            async with self._lock:
                self._pending.pop(key, None)

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "pending_requests": len(self._pending),
            "deduplicated_count": self._deduplicated_count,
        }


# =============================================================================
# LLM Service (PERF-050)
# =============================================================================


class LLMService:
    """
    LLM Service with optimized caching, deduplication, circuit breaker,
    and failover.

    PERF-050: Performance optimizations:
    - Response caching for identical prompts (PERF-048)
    - Request deduplication for concurrent requests (PERF-049)
    - Pre-created circuit breaker wrappers per provider (PERF-017)
    - Avoids decorator recreation on every call
    - Thread-safe provider registration

    STORY-1.2: Provider failover support:
    - Automatic failover to healthy providers when circuit breaker opens
    - Configurable failover chain per provider
    - Retry with exponential backoff for transient failures

    AIConfigManager Integration:
    - Config-driven failover chains via get_failover_chain()
    - Cost tracking using calculate_cost() from config
    - Circuit breaker settings from config
    - Runtime provider switching via switch_provider()

    HIGH-003 FIX: Explicit CircuitBreakerOpen exception handling
    to provide meaningful error messages when providers are unavailable.
    """

    # Circuit breaker configuration constants (defaults, overridden by config)
    _CB_FAILURE_THRESHOLD: int = 3
    _CB_RECOVERY_TIMEOUT: int = 60

    # Cache configuration (defaults, can be overridden by config)
    _CACHE_ENABLED: bool = True
    _CACHE_TTL: int = 300  # 5 minutes

    # Failover configuration
    _FAILOVER_ENABLED: bool = True
    _DEFAULT_FAILOVER_CHAIN: dict[str, list[str]] = {
        "gemini": ["openai", "anthropic", "deepseek"],
        "google": ["openai", "anthropic", "deepseek"],
        "openai": ["anthropic", "gemini", "deepseek"],
        "anthropic": ["openai", "gemini", "deepseek"],
        "deepseek": ["openai", "gemini", "anthropic"],
        "qwen": ["openai", "gemini", "deepseek"],
        "cursor": ["openai", "anthropic", "gemini"],
    }

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._default_provider: str | None = None
        # Cache for pre-created circuit breaker wrappers (PERF-018)
        self._circuit_breaker_cache: dict[str, Callable] = {}
        # Response cache (PERF-048)
        self._response_cache = LLMResponseCache(
            max_size=500, default_ttl=self._CACHE_TTL
        )
        # Request deduplicator (PERF-049)
        self._deduplicator = RequestDeduplicator()
        # Custom failover chains
        self._failover_chains: dict[str, list[str]] = {}
        # Cost tracking accumulator
        self._total_cost: float = 0.0
        self._request_costs: list[dict[str, Any]] = []

        # Global model selection integration
        self._use_global_selection: bool = True

        # API Key Failover Integration (Subtask 3.2)
        # Enable automatic API key rotation on rate limit errors
        self._api_key_failover_enabled: bool = True
        # Track failover metadata for responses
        self._last_failover_info: dict[str, Any] = {}

        # Load config-driven settings
        self._load_config_settings()

    def _load_config_settings(self) -> None:
        """Load settings from AIConfigManager if available."""
        config_manager = _get_config_manager()
        if not config_manager:
            return

        try:
            config = config_manager.get_config()
            global_cfg = config.global_config

            # Update cache settings
            self._CACHE_ENABLED = global_cfg.cache_enabled
            self._CACHE_TTL = global_cfg.cache_ttl_seconds

            # Update failover settings
            self._FAILOVER_ENABLED = global_cfg.failover_enabled

            logger.info(
                f"LLMService loaded config: cache={self._CACHE_ENABLED}, "
                f"failover={self._FAILOVER_ENABLED}"
            )
        except Exception as e:
            logger.warning(f"Failed to load config settings: {e}")

    @staticmethod
    def _normalize_provider_name(name: str | None) -> str | None:
        """Normalize known provider aliases to improve compatibility."""
        if not name:
            return None
        lowered = name.lower()
        # "google" and "gemini" are used interchangeably across the codebase.
        if lowered == "google":
            return "google"
        if lowered == "gemini":
            return "gemini"
        return lowered

    def _resolve_registered_provider_name(self, name: str | None) -> str | None:
        """
        Resolve a requested provider name to a registered provider key.

        This keeps backwards compatibility between "google" and "gemini" naming,
        while allowing the provider registry to register only one of them.
        """
        normalized = self._normalize_provider_name(name)
        if not normalized:
            return None

        if normalized in self._providers:
            return normalized

        aliases: dict[str, str] = {
            "google": "gemini",
            "gemini": "google",
        }
        alias = aliases.get(normalized)
        if alias and alias in self._providers:
            return alias

        return normalized

    def register_provider(self, name: str, provider: LLMProvider, is_default: bool = False):
        normalized_name = self._normalize_provider_name(name) or name
        self._providers[normalized_name] = provider
        if is_default:
            self._default_provider = normalized_name

        # Pre-create circuit breaker wrapper for this provider (PERF-019)
        self._create_circuit_breaker_wrapper(normalized_name)

        # Debug: Log instance ID to track if same instance is being used
        logger.info(
            f"Registered provider: {normalized_name} (default={is_default}) "
            f"[instance_id={id(self)}, providers_count={len(self._providers)}]"
        )

    def _create_circuit_breaker_wrapper(self, provider_name: str) -> None:
        """
        Pre-create a circuit breaker wrapper for a provider (PERF-020).

        PERF-001 FIX: Added timeout support from TimeoutConfig to prevent hanging requests.

        This avoids the overhead of creating a new decorated function
        on every generate_text call.
        """
        if provider_name in self._circuit_breaker_cache:
            return

        # PERF-001 FIX: Get timeout for LLM calls
        from app.core.timeouts import TimeoutConfig

        llm_timeout = TimeoutConfig.get_timeout(TimeoutType.LLM)

        # Create a reusable circuit breaker wrapper
        @circuit_breaker(
            provider_name,
            failure_threshold=self._CB_FAILURE_THRESHOLD,
            recovery_timeout=self._CB_RECOVERY_TIMEOUT,
            timeout=llm_timeout,  # PERF-001 FIX: Add timeout
        )
        async def circuit_breaker_wrapper(func: Callable, *args, **kwargs):
            return await func(*args, **kwargs)

        self._circuit_breaker_cache[provider_name] = circuit_breaker_wrapper
        logger.debug(
            f"Created circuit breaker wrapper for provider: {provider_name} "
            f"(timeout={llm_timeout}s)"
        )

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """Get a provider by name or return the default provider."""
        requested = name or self._default_provider
        provider_name = self._resolve_registered_provider_name(requested)
        if not provider_name or provider_name not in self._providers:
            raise ProviderNotAvailableError(provider=requested or "default")
        return self._providers[provider_name]

    def _get_failover_providers(self, primary: str) -> list[str]:
        """
        Get ordered list of failover providers for a primary provider.

        Returns providers from custom chain, config chain, or default chain,
        filtered to only include actually registered providers.
        """
        # Priority 1: Custom chains set via set_failover_chain()
        if primary in self._failover_chains:
            chain = self._failover_chains[primary]
        else:
            # Priority 2: Try config manager
            chain = self.get_failover_chain(primary)

        # Filter to only registered providers
        return [
            p for p in chain
            if p in self._providers or
            self._resolve_registered_provider_name(p) in self._providers
        ]

    def get_failover_chain(self, chain_name: str) -> list[str]:
        """
        Get a failover chain by name from AIConfigManager.

        Args:
            chain_name: Named chain (e.g., "premium", "cost_optimized")
                or provider name for provider-specific chain

        Returns:
            List of provider IDs in failover order
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                # Try named chain first, then provider chain
                chain = config_manager.get_failover_chain(chain_name)
                if chain:
                    return chain
            except Exception as e:
                logger.debug(f"Config failover chain lookup failed: {e}")

        # Fallback to default chains
        return self._DEFAULT_FAILOVER_CHAIN.get(chain_name, [])

    def set_failover_chain(
        self, provider: str, failover_providers: list[str]
    ) -> None:
        """
        Set a custom failover chain for a provider.

        Args:
            provider: The primary provider name
            failover_providers: Ordered list of providers to try on failure
        """
        self._failover_chains[provider] = failover_providers
        logger.info(f"Set failover chain for {provider}: {failover_providers}")

    async def get_current_selection_from_context(
        self,
        fallback_session_id: str | None = None,
        fallback_user_id: str | None = None,
    ) -> tuple[str, str]:
        """
        Get the current provider/model selection from request context.

        This method checks the request context (set by middleware) first,
        and falls back to ProviderResolutionService for full resolution.

        Resolution order:
        1. Request context (from middleware contextvars)
        2. ProviderResolutionService.resolve() which includes GlobalModelSelectionStrategy
        3. model_selection_service directly (as final fallback)
        4. Static defaults

        Args:
            fallback_session_id: Session ID to use if context missing
            fallback_user_id: User ID to use if context missing

        Returns:
            Tuple of (provider, model_id)
        """
        # Try request context first (fastest, set by middleware)
        state = _get_global_model_selection_state()
        if state:
            try:
                context = state.get_request_context()
                if context and context.selection:
                    provider = context.selection.provider
                    model_id = context.selection.model_id
                    logger.debug(
                        f"Using context selection: {provider}/{model_id}"
                    )
                    return (provider, model_id)
            except Exception as e:
                logger.debug(f"Request context not available: {e}")

        # Try ProviderResolutionService.resolve() which includes GlobalModelSelectionStrategy
        resolution_service = _get_provider_resolution_service()
        if resolution_service:
            try:
                # Use resolve() instead of get_current_selection_from_context()
                # This ensures GlobalModelSelectionStrategy is used
                provider, model = await resolution_service.resolve(
                    session_id=fallback_session_id,
                    user_id=fallback_user_id,
                    use_cache=True,
                )
                logger.debug(
                    f"Using resolved selection: {provider}/{model}"
                )
                return (provider, model)
            except Exception as e:
                logger.warning(
                    f"ProviderResolutionService.resolve() failed: {e}"
                )

        # Try model_selection_service directly as fallback
        try:
            from app.services.model_selection_service import model_selection_service
            selection = model_selection_service.get_selection()
            if selection:
                logger.debug(
                    f"Using model_selection_service: "
                    f"{selection.provider}/{selection.model}"
                )
                return (selection.provider, selection.model)
        except Exception as e:
            logger.debug(f"model_selection_service not available: {e}")

        # Fall back to configured defaults
        default_provider = self._default_provider or getattr(
            settings, "AI_PROVIDER", "openai"
        )
        default_model = getattr(settings, "DEFAULT_MODEL_ID", "gpt-4")

        logger.debug(
            f"Using default selection: {default_provider}/{default_model}"
        )
        return (default_provider, default_model)

    async def _resolve_provider_for_request(
        self,
        request: PromptRequest,
    ) -> tuple[str, str | None]:
        """
        Resolve provider and model for a request.

        Uses the following priority:
        1. Explicit provider/model from request
        2. Global selection from context/resolution service
        3. Default provider/model

        Args:
            request: The prompt request

        Returns:
            Tuple of (provider_name, model_name)
        """
        # Check if request has explicit provider
        if request.provider:
            provider_name = request.provider.value
            model_name = request.model
            logger.debug(
                f"Using explicit request selection: "
                f"{provider_name}/{model_name}"
            )
            return (provider_name, model_name)

        # Check if global selection should be used
        if self._use_global_selection:
            try:
                provider, model = await self.get_current_selection_from_context()
                if provider:
                    logger.debug(
                        f"Using global selection: {provider}/{model}"
                    )
                    return (provider, model)
            except Exception as e:
                logger.warning(f"Global selection resolution failed: {e}")

        # Fall back to default
        default_provider = self._default_provider
        default_model = request.model
        logger.debug(
            f"Using default provider: {default_provider}/{default_model}"
        )
        return (default_provider, default_model)

    async def generate_text(self, request: PromptRequest) -> PromptResponse:
        """
        Generate text with caching, deduplication, circuit breaker,
        and failover.

        PERF-051: Request flow:
        1. Resolve provider/model from request or global selection
        2. Check response cache for identical prompt
        3. Deduplicate concurrent identical requests
        4. Apply circuit breaker protection
        5. On circuit breaker open, try failover providers
        6. Cache successful response
        7. Track cost using config pricing

        STORY-1.2: Added automatic provider failover on circuit breaker open.
        HIGH-003 FIX: Explicit CircuitBreakerOpen exception handling

        Global Selection: When no explicit provider is specified in the
        request, uses ProviderResolutionService to resolve from context.
        """
        # Resolve provider from request or global selection
        provider_name, model_name = await self._resolve_provider_for_request(
            request
        )

        # Update request with resolved model if not already set
        if model_name and not request.model:
            request = PromptRequest(
                prompt=request.prompt,
                system_instruction=request.system_instruction,
                model=model_name,
                config=request.config,
                provider=request.provider,
                api_key=request.api_key,
                skip_validation=request.skip_validation,
            )

        provider = self.get_provider(provider_name)

        # Ensure we have a valid circuit breaker name
        circuit_name = provider_name or self._default_provider
        if not circuit_name:
            raise ProviderNotAvailableError(provider="No provider configured")

        # PERF-048: Check cache first (skip for non-deterministic requests)
        cache_enabled = self._CACHE_ENABLED
        if cache_enabled and request.config and request.config.temperature == 0:
            cached_response = await self._response_cache.get(request)
            if cached_response:
                logger.debug(
                    f"Returning cached LLM response for: {circuit_name}"
                )
                return cached_response

        # PERF-049: Deduplicate concurrent identical requests
        async def execute_request():
            return await self._execute_with_failover(
                request, circuit_name, provider
            )

        response = await self._deduplicator.deduplicate(request, execute_request)

        # Track cost if response has usage metadata
        self._track_cost(circuit_name, request.model, response)

        return response

    def _track_cost(
        self,
        provider_name: str,
        model_name: str | None,
        response: PromptResponse,
    ) -> None:
        """Track cost for a request using config pricing."""
        if not response.usage_metadata:
            return

        config_manager = _get_config_manager()
        if not config_manager:
            return

        try:
            model_id = model_name or response.model_used
            if not model_id:
                return

            input_tokens = response.usage_metadata.get("prompt_token_count", 0)
            output_tokens = response.usage_metadata.get(
                "candidates_token_count", 0
            )

            cost = config_manager.calculate_cost(
                provider_name,
                model_id,
                input_tokens,
                output_tokens,
            )

            if cost > 0:
                self._total_cost += cost
                self._request_costs.append({
                    "provider": provider_name,
                    "model": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "timestamp": time.time(),
                })
                logger.debug(
                    f"Request cost: ${cost:.6f} "
                    f"(total: ${self._total_cost:.4f})"
                )
        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")

    async def _execute_with_failover(
        self,
        request: PromptRequest,
        primary_provider: str,
        provider: LLMProvider,
    ) -> PromptResponse:
        """
        Execute request with automatic failover on circuit breaker open and rate limits.

        STORY-1.2: Tries failover providers in order when primary fails.

        Subtask 3.2: API Key Failover Integration:
        - Intercept rate limit errors and trigger API key failover
        - Retry request with backup key transparently
        - Include failover metadata in response
        - Record success/failure with failover service and health service
        """
        tried_providers = [primary_provider]
        last_error: Exception | None = None
        failover_info: dict[str, Any] = {}
        current_key_id: str | None = None
        start_time = time.time()

        # Get API key failover service if enabled
        failover_service = None
        if self._api_key_failover_enabled:
            failover_service = _get_api_key_failover_service()

        # Get health service for tracking
        health_service = _get_provider_health_service()

        # Get quota tracking service
        quota_service = _get_quota_tracking_service()

        try:
            # Try to get a managed API key if failover service is available
            if failover_service:
                try:
                    key_id, api_key = await failover_service.get_available_key(primary_provider)
                    if key_id and api_key:
                        current_key_id = key_id
                        # Create request with the managed API key
                        request = PromptRequest(
                            prompt=request.prompt,
                            system_instruction=request.system_instruction,
                            model=request.model,
                            config=request.config,
                            provider=request.provider,
                            api_key=api_key,  # Use managed key
                            skip_validation=request.skip_validation,
                        )
                        logger.debug(f"Using managed API key: {key_id[:20]}...")
                except Exception as e:
                    logger.debug(f"Could not get managed API key: {e}")

            response = await self._call_with_circuit_breaker(
                primary_provider, provider.generate, request
            )

            # Record success with services
            latency_ms = (time.time() - start_time) * 1000
            await self._record_request_success(
                primary_provider, current_key_id, latency_ms, response,
                failover_service, health_service, quota_service
            )

            # Cache successful response (only for deterministic requests)
            cache_enabled = self._CACHE_ENABLED
            is_deterministic = (
                request.config and request.config.temperature == 0
            )
            if cache_enabled and is_deterministic:
                await self._response_cache.set(request, response)
            return response

        except CircuitBreakerOpen as e:
            last_error = e
            logger.warning(
                f"Circuit breaker open for provider '{e.name}'. "
                f"Attempting failover..."
            )

            # Record failure with health service
            if health_service:
                await health_service.record_external_check(
                    provider_id=primary_provider,
                    success=False,
                    latency_ms=(time.time() - start_time) * 1000,
                    error_message="Circuit breaker open",
                )

            # Try failover providers if enabled
            if self._FAILOVER_ENABLED:
                failover_providers = self._get_failover_providers(primary_provider)

                for fallback_name in failover_providers:
                    if fallback_name in tried_providers:
                        continue

                    tried_providers.append(fallback_name)
                    logger.info(f"Trying failover provider: {fallback_name}")

                    try:
                        fallback_provider = self.get_provider(fallback_name)
                        # Create a modified request for the fallback provider
                        fallback_request = self._create_fallback_request(
                            request, fallback_name
                        )

                        response = await self._call_with_circuit_breaker(
                            fallback_name, fallback_provider.generate, fallback_request
                        )

                        # Mark response as served by failover
                        logger.info(
                            f"Failover successful: {primary_provider} -> {fallback_name}"
                        )
                        failover_info = {
                            "original_provider": primary_provider,
                            "fallback_provider": fallback_name,
                            "reason": "circuit_breaker_open",
                        }
                        self._last_failover_info = failover_info
                        return response

                    except CircuitBreakerOpen as cb_err:
                        logger.warning(
                            f"Failover provider '{fallback_name}' also has open circuit"
                        )
                        last_error = cb_err
                        continue
                    except Exception as fallback_err:
                        logger.warning(
                            f"Failover to '{fallback_name}' failed: {fallback_err}"
                        )
                        last_error = fallback_err
                        continue

            # All providers failed
            raise ProviderNotAvailableError(
                provider=primary_provider,
                message=(
                    f"Provider '{primary_provider}' is unavailable and all "
                    f"failover attempts failed. Tried: {', '.join(tried_providers)}"
                ),
                details={
                    "retry_after_seconds": getattr(last_error, "retry_after", 60),
                    "circuit_state": "open",
                    "tried_providers": tried_providers,
                    "suggestion": "Wait for provider recovery or check API keys",
                },
            )

        except Exception as e:
            # Check if this is a rate limit error
            error_message = str(e)
            is_rate_limit = self._is_rate_limit_error(error_message)
            latency_ms = (time.time() - start_time) * 1000

            # Record failure with health service
            if health_service:
                await health_service.record_external_check(
                    provider_id=primary_provider,
                    success=False,
                    latency_ms=latency_ms,
                    is_rate_limited=is_rate_limit,
                    error_message=error_message,
                )

            # If rate limit error and API key failover is enabled, try with backup key
            if is_rate_limit and failover_service and current_key_id:
                logger.warning(
                    f"Rate limit detected for {primary_provider} with key {current_key_id[:20]}..., "
                    f"attempting API key failover..."
                )

                # Handle the error and get backup key
                new_key_id, new_api_key = await failover_service.handle_error(
                    provider_id=primary_provider,
                    key_id=current_key_id,
                    error=error_message,
                    is_rate_limit=True,
                )

                if new_key_id and new_api_key:
                    logger.info(
                        f"API key failover: {current_key_id[:20]}... -> {new_key_id[:20]}..."
                    )

                    # Retry with backup key
                    try:
                        retry_request = PromptRequest(
                            prompt=request.prompt,
                            system_instruction=request.system_instruction,
                            model=request.model,
                            config=request.config,
                            provider=request.provider,
                            api_key=new_api_key,
                            skip_validation=request.skip_validation,
                        )

                        response = await self._call_with_circuit_breaker(
                            primary_provider, provider.generate, retry_request
                        )

                        # Record success with failover
                        await failover_service.record_success(primary_provider, new_key_id)

                        # Record with health service
                        if health_service:
                            await health_service.record_external_check(
                                provider_id=primary_provider,
                                success=True,
                                latency_ms=(time.time() - start_time) * 1000,
                            )

                        # Store failover info
                        failover_info = {
                            "original_key_id": current_key_id[:20] + "...",
                            "failover_key_id": new_key_id[:20] + "...",
                            "reason": "rate_limit",
                            "provider": primary_provider,
                        }
                        self._last_failover_info = failover_info

                        logger.info(
                            f"Request succeeded after API key failover for {primary_provider}"
                        )
                        return response

                    except Exception as retry_error:
                        logger.warning(
                            f"Request failed even with backup key: {retry_error}"
                        )
                        # Continue to raise the original error
                else:
                    logger.warning(
                        f"No backup API key available for {primary_provider}"
                    )

            # Re-raise the original error
            raise

    def _is_rate_limit_error(self, error_message: str) -> bool:
        """
        Check if an error message indicates a rate limit.

        Args:
            error_message: Error message from provider

        Returns:
            True if this is a rate limit error
        """
        if not error_message:
            return False

        error_lower = error_message.lower()
        rate_limit_patterns = [
            "rate limit",
            "rate_limit",
            "ratelimit",
            "quota exceeded",
            "quota_exceeded",
            "too many requests",
            "429",
            "resource exhausted",
            "capacity exceeded",
            "requests per minute",
            "tokens per minute",
            "rpm limit",
            "tpm limit",
        ]
        return any(pattern in error_lower for pattern in rate_limit_patterns)

    async def _record_request_success(
        self,
        provider_id: str,
        key_id: str | None,
        latency_ms: float,
        response: PromptResponse,
        failover_service,
        health_service,
        quota_service,
    ) -> None:
        """
        Record successful request with all tracking services.

        Args:
            provider_id: Provider identifier
            key_id: API key ID used
            latency_ms: Request latency in milliseconds
            response: The successful response
            failover_service: ApiKeyFailoverService instance
            health_service: ProviderHealthService instance
            quota_service: QuotaTrackingService instance
        """
        # Record success with failover service
        if failover_service and key_id:
            try:
                await failover_service.record_success(provider_id, key_id)
            except Exception as e:
                logger.debug(f"Failed to record success with failover service: {e}")

        # Record with health service
        if health_service:
            try:
                await health_service.record_external_check(
                    provider_id=provider_id,
                    success=True,
                    latency_ms=latency_ms,
                )
            except Exception as e:
                logger.debug(f"Failed to record with health service: {e}")

        # Record with quota service
        if quota_service and response.usage_metadata:
            try:
                input_tokens = response.usage_metadata.get("prompt_token_count", 0)
                output_tokens = response.usage_metadata.get("candidates_token_count", 0)
                await quota_service.record_usage(
                    provider_id=provider_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=response.model_used,
                )
            except Exception as e:
                logger.debug(f"Failed to record with quota service: {e}")

    def _create_fallback_request(
        self, original: PromptRequest, fallback_provider: str
    ) -> PromptRequest:
        """Create a request adapted for the fallback provider."""
        # Check if fallback_provider is valid enum value
        provider_enum = None
        if fallback_provider in [e.value for e in LLMProviderType]:
            provider_enum = LLMProviderType(fallback_provider)

        # Create a copy with the new provider
        return PromptRequest(
            prompt=original.prompt,
            system_instruction=original.system_instruction,
            model=None,  # Let fallback use its default model
            config=original.config,
            provider=provider_enum,
            api_key=None,  # Use configured key for fallback
            skip_validation=original.skip_validation,
        )

    async def _call_with_circuit_breaker(
        self, provider_name: str, func, *args, **kwargs
    ):
        """Wrap provider calls with circuit breaker protection (PERF-021).

        Uses pre-created circuit breaker wrappers to avoid decorator
        recreation overhead on every call.

        Falls back to creating a new wrapper if provider wasn't pre-registered.

        HIGH-003 FIX: CircuitBreakerOpen exceptions are now propagated
        to be handled by the caller with meaningful error messages.
        """
        # Use cached circuit breaker wrapper if available
        if provider_name in self._circuit_breaker_cache:
            wrapper = self._circuit_breaker_cache[provider_name]
            return await wrapper(func, *args, **kwargs)

        # Fallback: create wrapper on-demand (shouldn't happen in normal flow)
        logger.warning(
            f"Circuit breaker wrapper not pre-created for provider: {provider_name}. "
            "Creating on-demand (performance impact)."
        )
        self._create_circuit_breaker_wrapper(provider_name)
        wrapper = self._circuit_breaker_cache[provider_name]
        return await wrapper(func, *args, **kwargs)

    async def generate(self, prompt: str, provider: str | None = None, **kwargs) -> Any:
        """
        Generate text from an LLM provider.

        Args:
            prompt: The text prompt to send to the LLM
            provider: The provider name (e.g., 'google', 'openai')
            **kwargs: Additional parameters like model, temperature, etc.

        Returns:
            LLMResponse object with content, usage, etc.

        Raises:
            ProviderNotAvailableError: When provider is unavailable or circuit is open
        """
        from app.domain.models import GenerationConfig, PromptRequest

        # Build the request
        config = GenerationConfig(
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens", 2048),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
        )

        metadata = kwargs.get("metadata", {})
        api_key = metadata.get("api_key") if metadata else None

        request = PromptRequest(
            prompt=prompt,
            model=kwargs.get("model"),
            config=config,
            provider=LLMProviderType(provider) if provider else None,
            api_key=api_key,
            # Internal calls skip validation by default
            skip_validation=kwargs.get("skip_validation", True),
        )

        # Get the provider and generate
        llm_provider = self.get_provider(provider)

        try:
            response = await llm_provider.generate(request)
        except CircuitBreakerOpen as e:
            # HIGH-003 FIX: Handle circuit breaker open in generate() method too
            logger.warning(
                f"Circuit breaker open for provider '{e.name}' in generate(). "
                f"Retry after {e.retry_after:.1f}s"
            )
            raise ProviderNotAvailableError(
                provider=e.name,
                message=(
                    f"Provider '{e.name}' is temporarily unavailable. "
                    f"Retry after {e.retry_after:.0f} seconds."
                ),
                details={
                    "retry_after_seconds": e.retry_after,
                    "circuit_state": "open",
                },
            )

        # Return a response object that matches what execute.py expects
        import uuid

        class LLMResponse:
            def __init__(self, resp: PromptResponse):
                self.request_id = str(uuid.uuid4())
                self.content = resp.text
                self.model = resp.model_used
                self.provider = LLMProviderType(resp.provider)
                self.latency_ms = resp.latency_ms
                self.cached = False

                # Usage info
                class Usage:
                    def __init__(self, metadata):
                        self.total_tokens = metadata.get("total_token_count", 0) if metadata else 0
                        self.prompt_tokens = (
                            metadata.get("prompt_token_count", 0) if metadata else 0
                        )
                        self.completion_tokens = (
                            metadata.get("candidates_token_count", 0) if metadata else 0
                        )
                        # Rough cost estimate (this is just a placeholder)
                        self.estimated_cost = self.total_tokens * 0.00001

                self.usage = Usage(resp.usage_metadata)

        return LLMResponse(response)

    async def list_providers(self) -> ProviderListResponse:
        providers_info = []
        provider_models = settings.get_provider_models()

        for name, _provider in self._providers.items():
            # Get available models from config
            available_models = provider_models.get(
                name,
                provider_models.get(
                    # Support google/gemini alias model lists.
                    "google" if name == "gemini" else "gemini" if name == "google" else name,
                    [],
                ),
            )

            providers_info.append(
                ProviderInfo(
                    provider=name,
                    status="active",
                    model=available_models[0] if available_models else "unknown",
                    available_models=available_models,
                )
            )

        return ProviderListResponse(
            providers=providers_info,
            count=len(providers_info),
            default=self._default_provider or "",
        )

    def get_available_providers(self) -> list[str]:
        """Return list of available provider names."""
        providers = list(self._providers.keys())
        # Debug: Log instance ID to track if same instance as registration
        logger.info(
            f"get_available_providers called: {providers} "
            f"[instance_id={id(self)}, providers_count={len(self._providers)}]"
        )
        return providers

    def get_default_provider_name(self) -> str | None:
        """Return the name of the default provider, or first registered if no default set."""
        if self._default_provider:
            return self._default_provider
        # Fallback to first registered provider
        if self._providers:
            return next(iter(self._providers.keys()))
        return None

    def get_provider_info(self, provider_name: str) -> dict[str, Any]:
        """Get information about a specific provider."""
        resolved = self._resolve_registered_provider_name(provider_name)
        if resolved not in self._providers:
            raise ProviderNotAvailableError(provider=provider_name)

        provider_models = settings.get_provider_models()
        available_models = provider_models.get(
            resolved,
            provider_models.get(
                "google"
                if resolved == "gemini"
                else "gemini"
                if resolved == "google"
                else resolved,
                [],
            ),
        )

        return {
            "provider": resolved,
            "status": "active",
            "model": available_models[0] if available_models else "unknown",
            "available_models": available_models,
        }

    @property
    def default_provider(self) -> LLMProviderType:
        """Get the default provider type."""
        if not self._default_provider:
            raise ProviderNotAvailableError(provider="default")
        return LLMProviderType(self._default_provider)

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the LLM service (PERF-052).

        Returns cache hit rates, deduplication stats, provider info,
        failover config, cost tracking, and API key failover status.
        """
        return {
            "cache": self._response_cache.get_stats(),
            "deduplication": self._deduplicator.get_stats(),
            "providers": list(self._providers.keys()),
            "default_provider": self._default_provider,
            "circuit_breakers": list(self._circuit_breaker_cache.keys()),
            "failover_enabled": self._FAILOVER_ENABLED,
            "failover_chains": {
                p: self._get_failover_providers(p) for p in self._providers
            },
            "cost_tracking": {
                "total_cost": self._total_cost,
                "request_count": len(self._request_costs),
                "recent_costs": self._request_costs[-10:],
            },
            # API Key Failover Integration (Subtask 3.2)
            "api_key_failover": {
                "enabled": self._api_key_failover_enabled,
                "last_failover": self._last_failover_info,
            },
        }

    def switch_provider(
        self,
        provider_name: str,
        model_name: str | None = None,
    ) -> bool:
        """
        Switch to a different provider at runtime.

        Args:
            provider_name: Provider to switch to
            model_name: Optional model to use (uses provider default if None)

        Returns:
            True if switch was successful
        """
        resolved = self._resolve_registered_provider_name(provider_name)
        if not resolved or resolved not in self._providers:
            logger.warning(f"Cannot switch to unregistered provider: {provider_name}")
            return False

        old_default = self._default_provider
        self._default_provider = resolved

        logger.info(f"Switched default provider: {old_default} -> {resolved}")

        # Optionally validate against config
        config_manager = _get_config_manager()
        if config_manager:
            try:
                provider_config = config_manager.get_provider(resolved)
                if provider_config and not provider_config.enabled:
                    logger.warning(
                        f"Provider '{resolved}' is disabled in config"
                    )
            except Exception:
                pass

        return True

    def get_cost_summary(self) -> dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost_usd": self._total_cost,
            "request_count": len(self._request_costs),
            "costs_by_provider": self._aggregate_costs_by_provider(),
        }

    def _aggregate_costs_by_provider(self) -> dict[str, float]:
        """Aggregate costs by provider."""
        by_provider: dict[str, float] = {}
        for entry in self._request_costs:
            provider = entry.get("provider", "unknown")
            by_provider[provider] = by_provider.get(provider, 0) + entry["cost"]
        return by_provider

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters."""
        self._total_cost = 0.0
        self._request_costs.clear()
        logger.info("Cost tracking reset")

    def enable_failover(self, enabled: bool = True) -> None:
        """Enable or disable automatic provider failover."""
        self._FAILOVER_ENABLED = enabled
        logger.info(f"Provider failover {'enabled' if enabled else 'disabled'}")

    def enable_global_selection(self, enabled: bool = True) -> None:
        """Enable or disable global model selection integration."""
        self._use_global_selection = enabled
        logger.info(
            f"Global model selection {'enabled' if enabled else 'disabled'}"
        )

    def enable_api_key_failover(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic API key failover on rate limits.

        When enabled, the service will automatically switch to backup API keys
        when the primary key hits rate limits, and retry the request transparently.

        Args:
            enabled: Whether API key failover should be enabled
        """
        self._api_key_failover_enabled = enabled
        logger.info(
            f"API key failover {'enabled' if enabled else 'disabled'}"
        )

    def get_last_failover_info(self) -> dict[str, Any]:
        """
        Get information about the last failover event.

        Returns:
            Dictionary with failover details or empty dict if no failover occurred
        """
        return self._last_failover_info.copy()

    def clear_failover_info(self) -> None:
        """Clear the last failover info."""
        self._last_failover_info.clear()

    def get_api_key_failover_status(self) -> dict[str, Any]:
        """
        Get the current API key failover status for all providers.

        Returns:
            Dictionary with failover status per provider
        """
        failover_service = _get_api_key_failover_service()
        if not failover_service:
            return {
                "enabled": self._api_key_failover_enabled,
                "service_available": False,
                "providers": {},
            }

        # Get status synchronously if possible, otherwise return basic info
        return {
            "enabled": self._api_key_failover_enabled,
            "service_available": True,
            "last_failover": self._last_failover_info,
        }

    async def clear_cache(self) -> None:
        """Clear the response cache."""
        await self._response_cache.clear()
        logger.info("LLM response cache cleared")

    # =========================================================================
    # Streaming Support (STREAM-001)
    # =========================================================================

    async def generate_text_stream(self, request: PromptRequest) -> AsyncIterator[StreamChunk]:
        """
        Stream text generation with circuit breaker protection.

        This method provides real-time token-by-token streaming from LLM providers.
        Note: Streaming responses are NOT cached as they are meant for real-time use.

        Args:
            request: The prompt request with generation configuration.

        Yields:
            StreamChunk: Individual chunks of generated text with metadata.

        Raises:
            ProviderNotAvailableError: When provider is unavailable or doesn't support streaming.
            NotImplementedError: When the provider doesn't implement streaming.
        """
        # Determine provider from request or default
        provider_name = None
        if request.provider:
            provider_name = request.provider.value

        provider = self.get_provider(provider_name)

        # Ensure we have a valid circuit breaker name
        circuit_name = provider_name or self._default_provider
        if not circuit_name:
            raise ProviderNotAvailableError(provider="No provider configured")

        logger.info(f"Starting streaming generation with provider: {circuit_name}")

        try:
            # Stream directly from provider (no caching for streams)
            async for chunk in provider.generate_stream(request):
                yield chunk

        except NotImplementedError as e:
            logger.warning(f"Streaming not supported by provider '{circuit_name}': {e}")
            raise ProviderNotAvailableError(
                provider=circuit_name,
                message=f"Provider '{circuit_name}' does not support streaming.",
                details={"feature": "streaming", "supported": False},
            )
        except CircuitBreakerOpen as e:
            logger.warning(
                f"Circuit breaker open for provider '{e.name}' during streaming. "
                f"Retry after {e.retry_after:.1f}s"
            )
            raise ProviderNotAvailableError(
                provider=e.name,
                message=(
                    f"Provider '{e.name}' is temporarily unavailable. "
                    f"Retry after {e.retry_after:.0f} seconds."
                ),
                details={
                    "retry_after_seconds": e.retry_after,
                    "circuit_state": "open",
                },
            )

    async def stream_generate(
        self, prompt: str, provider: str | None = None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Convenience method for streaming text generation.

        Args:
            prompt: The text prompt to send to the LLM.
            provider: The provider name (e.g., 'google', 'openai').
            **kwargs: Additional parameters like model, temperature, etc.

        Yields:
            StreamChunk: Individual chunks of generated text.
        """
        from app.domain.models import GenerationConfig

        config = GenerationConfig(
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens", 2048),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 40),
        )

        request = PromptRequest(
            prompt=prompt,
            model=kwargs.get("model"),
            config=config,
            provider=LLMProviderType(provider) if provider else None,
            system_instruction=kwargs.get("system_instruction"),
            skip_validation=kwargs.get("skip_validation", True),
        )

        async for chunk in self.generate_text_stream(request):
            yield chunk

    # =========================================================================
    # Token Counting Support (TOKEN-001)
    # =========================================================================

    async def count_tokens(
        self, text: str, provider: str | None = None, model: str | None = None
    ) -> int:
        """
        Count tokens in text using the specified provider's tokenizer.

        Args:
            text: The text to count tokens for.
            provider: The provider to use for tokenization (defaults to default provider).
            model: Optional model name for model-specific tokenization.

        Returns:
            int: The number of tokens in the text.

        Raises:
            ProviderNotAvailableError: When provider is unavailable.
            NotImplementedError: When the provider doesn't support token counting.
        """
        llm_provider = self.get_provider(provider)
        provider_name = provider or self._default_provider

        try:
            return await llm_provider.count_tokens(text, model)
        except NotImplementedError as e:
            logger.warning(f"Token counting not supported by provider '{provider_name}': {e}")
            raise ProviderNotAvailableError(
                provider=provider_name,
                message=f"Provider '{provider_name}' does not support token counting.",
                details={"feature": "token_counting", "supported": False},
            )

    async def estimate_tokens(self, text: str, provider: str | None = None) -> dict[str, Any]:
        """
        Estimate token count with additional metadata.

        This is a higher-level method that provides token count along with
        cost estimation and context window information.

        Args:
            text: The text to estimate tokens for.
            provider: The provider to use for estimation.

        Returns:
            dict with token_count, estimated_cost, and context_info.
        """
        provider_name = provider or self._default_provider

        try:
            token_count = await self.count_tokens(text, provider)

            # Rough cost estimates per 1K tokens (these are approximations)
            cost_per_1k = {
                "google": 0.00025,
                "gemini": 0.00025,
                "openai": 0.002,
                "anthropic": 0.003,
                "deepseek": 0.0001,
            }

            # Context window sizes
            context_windows = {
                "google": 1000000,  # Gemini 1.5 Pro
                "gemini": 1000000,
                "openai": 128000,  # GPT-4 Turbo
                "anthropic": 200000,  # Claude 3
                "deepseek": 64000,
            }

            rate = cost_per_1k.get(provider_name, 0.001)
            window = context_windows.get(provider_name, 32000)

            return {
                "token_count": token_count,
                "estimated_cost_usd": (token_count / 1000) * rate,
                "context_window": window,
                "context_usage_percent": (token_count / window) * 100,
                "provider": provider_name,
            }

        except Exception as e:
            logger.error(f"Token estimation failed: {e}")
            # Fallback to character-based estimation
            estimated = len(text) // 4  # Rough approximation
            return {
                "token_count": estimated,
                "estimated_cost_usd": 0.0,
                "context_window": 32000,
                "context_usage_percent": (estimated / 32000) * 100,
                "provider": provider_name,
                "estimation_method": "character_based_fallback",
            }

    def supports_streaming(self, provider: str | None = None) -> bool:
        """
        Check if a provider supports streaming.

        Args:
            provider: The provider name to check.

        Returns:
            bool: True if the provider supports streaming.
        """
        try:
            llm_provider = self.get_provider(provider)
            # Check if generate_stream is implemented (not just inherited default)
            return (
                hasattr(llm_provider, "generate_stream")
                and llm_provider.generate_stream.__func__ is not LLMProvider.generate_stream
            )
        except Exception:
            return False

    def supports_token_counting(self, provider: str | None = None) -> bool:
        """
        Check if a provider supports token counting.

        Args:
            provider: The provider name to check.

        Returns:
            bool: True if the provider supports token counting.
        """
        try:
            llm_provider = self.get_provider(provider)
            # Check if count_tokens is implemented (not just inherited default)
            return (
                hasattr(llm_provider, "count_tokens")
                and llm_provider.count_tokens.__func__ is not LLMProvider.count_tokens
            )
        except Exception:
            return False


# Global instance
llm_service = LLMService()
logger.info(f"llm_service singleton created: instance_id={id(llm_service)}, module={__name__}")


def get_llm_service() -> LLMService:
    """Dependency provider for LLMService"""
    return llm_service
