"""Model Router Service for Multi-Provider AI Model Selection.

This service handles dynamic routing of LLM requests to the correct provider
(Google Gemini or DeepSeek) based on user session selection. It integrates
with the SessionService for state management and LLMService for provider access.

AIConfigManager Integration:
- Config-driven model routing decisions
- Session-based routing with config priorities
- Named failover chains from config
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from app.core.config import settings
from app.core.logging import logger
from app.domain.models import GenerationConfig, LLMProviderType, PromptRequest, PromptResponse


def _get_config_manager():
    """Get AIConfigManager instance with graceful fallback.

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


@dataclass
class ProviderHealth:
    """Health status for a provider."""

    provider: str
    is_healthy: bool
    last_check: datetime
    latency_ms: float | None = None
    error_message: str | None = None
    consecutive_failures: int = 0
    # Config-driven circuit breaker settings
    failure_threshold: int = 3
    recovery_timeout: int = 60


@dataclass
class ModelSelectionEvent:
    """Event emitted when model selection changes."""

    session_id: str
    provider: str
    model: str
    previous_provider: str | None
    previous_model: str | None
    timestamp: datetime


class ModelRouterService:
    """Routes LLM requests to the correct provider based on user selection.

    Features:
    - Dynamic provider routing (Gemini/DeepSeek)
    - Session-based model selection
    - Health monitoring with circuit breaker integration
    - Fallback handling when primary provider fails
    - Event broadcasting for real-time sync

    AIConfigManager Integration:
    - Config-driven model routing decisions
    - Config-driven failover chains
    - Priority-based provider selection
    """

    # Supported providers for this feature (extended from config)
    SUPPORTED_PROVIDERS: ClassVar[set[str]] = {
        "gemini",
        "google",
        "deepseek",
        "openai",
        "anthropic",
    }

    # Provider aliases (normalize to canonical names for display)
    PROVIDER_ALIASES: ClassVar[dict[str, str]] = {
        "google": "gemini",
        "gemini": "gemini",
        "deepseek": "deepseek",
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "gpt": "openai",
    }

    # Map back to session service provider names
    SESSION_PROVIDER_MAP: ClassVar[dict[str, str]] = {
        "gemini": "google",
        "google": "google",
        "deepseek": "deepseek",
        "openai": "openai",
        "anthropic": "anthropic",
    }

    def __init__(self) -> None:
        self._provider_health: dict[str, ProviderHealth] = {}
        self._selection_listeners: set[Callable[[ModelSelectionEvent], Any]] = set()
        self._initialized = False
        # Named failover chains from config
        self._failover_chains: dict[str, list[tuple[str, str]]] = {}

    async def initialize(self) -> None:
        """Initialize the router service."""
        if self._initialized:
            return

        # Load config-driven settings
        self._load_config_settings()

        # Initialize health status for supported providers
        providers_to_init = self._get_enabled_providers()
        for provider in providers_to_init:
            health = self._create_provider_health(provider)
            self._provider_health[provider] = health

        self._initialized = True
        logger.info(
            f"ModelRouterService initialized with providers: {', '.join(providers_to_init)}",
        )

    def _load_config_settings(self) -> None:
        """Load settings from AIConfigManager."""
        config_manager = _get_config_manager()
        if not config_manager:
            return

        try:
            config = config_manager.get_config()

            # Load aliases from config
            if config.aliases:
                for alias, target in config.aliases.items():
                    self.PROVIDER_ALIASES[alias] = target

            # Load failover chains
            self._failover_chains = {}
            for chain_name, chain_def in config.failover_chains.items():
                self._failover_chains[chain_name] = chain_def.chain

            logger.info(f"Loaded {len(self._failover_chains)} failover chains from config")
        except Exception as e:
            logger.warning(f"Failed to load config settings: {e}")

    def _get_enabled_providers(self) -> list[str]:
        """Get list of enabled providers from config or defaults."""
        config_manager = _get_config_manager()
        if config_manager:
            try:
                config = config_manager.get_config()
                enabled = [pid for pid, p in config.providers.items() if p.enabled]
                if enabled:
                    return enabled
            except Exception as e:
                logger.debug(f"Failed to get enabled providers: {e}")

        # Default fallback
        return ["gemini", "deepseek"]

    def _create_provider_health(self, provider: str) -> ProviderHealth:
        """Create ProviderHealth with config-driven settings."""
        config_manager = _get_config_manager()
        failure_threshold = 3
        recovery_timeout = 60

        if config_manager:
            try:
                provider_config = config_manager.get_provider(provider)
                if provider_config:
                    cb = provider_config.circuit_breaker
                    failure_threshold = cb.failure_threshold
                    recovery_timeout = cb.recovery_timeout_seconds
            except Exception:
                pass

        return ProviderHealth(
            provider=provider,
            is_healthy=True,
            last_check=datetime.utcnow(),
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    def _normalize_provider(self, provider: str) -> str:
        """Normalize provider name to canonical form.

        Uses config aliases if available.
        """
        config_manager = _get_config_manager()
        if config_manager:
            try:
                resolved = config_manager.resolve_provider_alias(provider)
                if resolved:
                    return resolved
            except Exception:
                pass

        return self.PROVIDER_ALIASES.get(provider.lower(), provider.lower())

    def _get_llm_service(self):
        """Lazy import to avoid circular dependencies."""
        from app.services.llm_service import llm_service

        return llm_service

    def _get_session_service(self):
        """Lazy import to avoid circular dependencies."""
        from app.services.session_service import session_service

        return session_service

    async def route_request(
        self,
        request: PromptRequest,
        session_id: str | None = None,
    ) -> PromptResponse:
        """Route a request to the appropriate provider based on session selection.

        Args:
            request: The prompt request to process
            session_id: Optional session ID for user-specific routing

        Returns:
            PromptResponse from the selected provider

        """
        llm_service = self._get_llm_service()
        session_service = self._get_session_service()

        # Determine provider and model
        provider_name = None
        model_name = None

        # Priority 1: Request-level override
        if request.provider:
            provider_name = self._normalize_provider(request.provider.value)
            model_name = request.model

        # Priority 2: Session-based selection
        elif session_id:
            session_provider, session_model = session_service.get_session_model(session_id)
            provider_name = self._normalize_provider(session_provider)
            model_name = session_model

        # Priority 3: Default provider
        if not provider_name:
            default_provider, default_model = session_service.get_default_model()
            provider_name = self._normalize_provider(default_provider)
            model_name = model_name or default_model

        # Validate provider is supported and healthy
        if provider_name not in self.SUPPORTED_PROVIDERS:
            logger.warning(f"Unsupported provider {provider_name}, falling back to gemini")
            provider_name = "gemini"

        # Check provider health and potentially fallback
        if not await self._is_provider_healthy(provider_name):
            fallback = await self._get_fallback_provider(provider_name)
            if fallback:
                logger.warning(f"Provider {provider_name} unhealthy, falling back to {fallback}")
                provider_name = fallback

        # Build the final request with resolved provider/model
        final_request = PromptRequest(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            config=request.config or GenerationConfig(),
            model=model_name,
            provider=(
                LLMProviderType(provider_name)
                if provider_name in [e.value for e in LLMProviderType]
                else None
            ),
            api_key=request.api_key,
        )

        logger.info(f"Routing request to provider={provider_name}, model={model_name}")

        # Execute with circuit breaker protection
        try:
            response = await self._execute_with_tracking(
                provider_name,
                llm_service.generate_text,
                final_request,
            )

            # Update health on success
            await self._update_provider_health(provider_name, True)

            return response

        except Exception as e:
            # Update health on failure
            await self._update_provider_health(provider_name, False, str(e))

            # Try fallback if available
            fallback = await self._get_fallback_provider(provider_name)
            if fallback and fallback != provider_name:
                logger.warning(f"Retrying with fallback provider: {fallback}")
                final_request.provider = LLMProviderType(fallback)
                return await llm_service.generate_text(final_request)

            raise

    async def _execute_with_tracking(
        self,
        provider_name: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a function with latency tracking."""
        start_time = datetime.utcnow()

        try:
            result = await func(*args, **kwargs)

            # Track latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            if provider_name in self._provider_health:
                self._provider_health[provider_name].latency_ms = latency_ms

            return result

        except Exception:
            raise

    async def _is_provider_healthy(self, provider: str) -> bool:
        """Check if a provider is healthy.

        Uses config-driven failure threshold.
        """
        health = self._provider_health.get(provider)
        if not health:
            return True  # Assume healthy if no data

        # Use config-driven threshold
        threshold = health.failure_threshold
        return health.consecutive_failures < threshold

    async def _update_provider_health(
        self,
        provider: str,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Update provider health status."""
        if provider not in self._provider_health:
            self._provider_health[provider] = ProviderHealth(
                provider=provider,
                is_healthy=success,
                last_check=datetime.utcnow(),
            )

        health = self._provider_health[provider]
        health.last_check = datetime.utcnow()
        health.is_healthy = success

        if success:
            health.consecutive_failures = 0
            health.error_message = None
        else:
            health.consecutive_failures += 1
            health.error_message = error_message

    async def _get_fallback_provider(self, failed_provider: str) -> str | None:
        """Get a fallback provider when the primary fails.

        Uses config-driven failover chains if available.
        """
        # Try config-driven failover first
        config_manager = _get_config_manager()
        if config_manager:
            try:
                # Get default failover chain
                chain = config_manager.get_failover_chain("cost_optimized")
                if chain:
                    for provider, _model in chain:
                        if provider != failed_provider and await self._is_provider_healthy(
                            provider,
                        ):
                            return provider
            except Exception as e:
                logger.debug(f"Config failover lookup failed: {e}")

        # Static fallback order
        fallback_order = {
            "gemini": "deepseek",
            "deepseek": "gemini",
            "openai": "anthropic",
            "anthropic": "openai",
        }

        fallback = fallback_order.get(failed_provider)
        if fallback and await self._is_provider_healthy(fallback):
            return fallback

        return None

    def get_failover_chain(self, chain_name: str) -> list[tuple[str, str]] | None:
        """Get a named failover chain from config.

        Args:
            chain_name: Name of the chain (e.g., "premium", "cost_optimized")

        Returns:
            List of (provider, model) tuples or None if not found

        """
        # Check cached chains first
        if chain_name in self._failover_chains:
            return self._failover_chains[chain_name]

        # Try loading from config manager
        config_manager = _get_config_manager()
        if config_manager:
            try:
                chain = config_manager.get_failover_chain(chain_name)
                if chain:
                    self._failover_chains[chain_name] = chain
                    return chain
            except Exception as e:
                logger.debug(f"Failed to get failover chain: {e}")

        return None

    async def route_with_failover_chain(
        self,
        request: PromptRequest,
        chain_name: str,
        session_id: str | None = None,
    ) -> PromptResponse:
        """Route a request using a named failover chain.

        Args:
            request: The prompt request
            chain_name: Name of the failover chain to use
            session_id: Optional session ID

        Returns:
            PromptResponse from first successful provider

        """
        llm_service = self._get_llm_service()
        chain = self.get_failover_chain(chain_name)

        if not chain:
            logger.warning(f"Failover chain '{chain_name}' not found, using default routing")
            return await self.route_request(request, session_id)

        last_error = None
        for provider_name, model_name in chain:
            if not await self._is_provider_healthy(provider_name):
                logger.debug(
                    f"Skipping unhealthy provider {provider_name} in chain {chain_name}",
                )
                continue

            try:
                # Build request for this provider
                final_request = PromptRequest(
                    prompt=request.prompt,
                    system_instruction=request.system_instruction,
                    config=request.config or GenerationConfig(),
                    model=model_name,
                    provider=(
                        LLMProviderType(provider_name)
                        if provider_name in [e.value for e in LLMProviderType]
                        else None
                    ),
                    api_key=request.api_key,
                )

                logger.info(f"Routing via chain '{chain_name}' to {provider_name}/{model_name}")

                response = await self._execute_with_tracking(
                    provider_name,
                    llm_service.generate_text,
                    final_request,
                )
                await self._update_provider_health(provider_name, True)
                return response

            except Exception as e:
                last_error = e
                await self._update_provider_health(provider_name, False, str(e))
                logger.warning(f"Provider {provider_name} in chain '{chain_name}' failed: {e}")
                continue

        # All providers in chain failed
        msg = f"All providers in chain '{chain_name}' failed. Last error: {last_error}"
        raise RuntimeError(
            msg,
        )

    async def select_model(
        self,
        session_id: str,
        provider: str,
        model: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Select a model for a session and broadcast the change.

        Args:
            session_id: The session to update
            provider: Provider identifier (gemini/deepseek)
            model: Model identifier

        Returns:
            Tuple of (success, message, session_info)

        """
        session_service = self._get_session_service()

        # Normalize provider name for display
        normalized_provider = self._normalize_provider(provider)

        # Validate provider is supported
        if normalized_provider not in self.SUPPORTED_PROVIDERS:
            return (False, f"Provider '{provider}' not supported. Use 'gemini' or 'deepseek'.", {})

        # Map to session service provider name (e.g., "gemini" -> "google")
        session_provider = self.SESSION_PROVIDER_MAP.get(normalized_provider, normalized_provider)

        # Ensure session exists - create if it doesn't
        existing_session = session_service.get_session(session_id)
        if not existing_session:
            # Create a new session with the requested provider/model
            logger.info(f"Session {session_id} not found, creating new session")
            new_session = session_service.create_session(provider=session_provider, model=model)
            # Update the session_id to match the requested one if possible
            # or use the newly created session
            if new_session:
                return (
                    True,
                    "Session created with selected model",
                    {
                        "session_id": new_session.session_id,
                        "provider": normalized_provider,
                        "model": new_session.selected_model,
                    },
                )

        # Get current selection for event
        current_provider, current_model = session_service.get_session_model(session_id)

        # Update session with the session-compatible provider name
        success, message, session = session_service.update_session_model(
            session_id=session_id,
            provider=session_provider,
            model=model,
        )

        if success and session:
            # Broadcast selection change event
            event = ModelSelectionEvent(
                session_id=session_id,
                provider=normalized_provider,
                model=model,
                previous_provider=current_provider,
                previous_model=current_model,
                timestamp=datetime.utcnow(),
            )
            await self._broadcast_selection_change(event)

            return (
                True,
                message,
                {
                    "session_id": session_id,
                    "provider": session.selected_provider,
                    "model": session.selected_model,
                },
            )

        return (success, message, {})

    def subscribe_to_selection_changes(
        self,
        callback: Callable[[ModelSelectionEvent], Any],
    ) -> Callable[[], None]:
        """Subscribe to model selection change events.

        Args:
            callback: Function to call when selection changes

        Returns:
            Unsubscribe function

        """
        self._selection_listeners.add(callback)

        def unsubscribe() -> None:
            self._selection_listeners.discard(callback)

        return unsubscribe

    async def _broadcast_selection_change(self, event: ModelSelectionEvent) -> None:
        """Broadcast a selection change to all listeners."""
        for listener in self._selection_listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in selection change listener: {e}")

    def get_provider_health(self) -> list[dict[str, Any]]:
        """Get health status for all providers."""
        return [
            {
                "provider": health.provider,
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat(),
                "latency_ms": health.latency_ms,
                "error_message": health.error_message,
                "consecutive_failures": health.consecutive_failures,
            }
            for health in self._provider_health.values()
        ]

    def get_supported_providers(self) -> list[dict[str, Any]]:
        """Get list of supported providers with their models.

        Uses config-driven provider list with fallback to settings.
        """
        # Try config-driven providers first
        config_manager = _get_config_manager()
        if config_manager:
            try:
                return self._get_config_providers(config_manager)
            except Exception as e:
                logger.debug(f"Config provider lookup failed: {e}")

        # Fallback to settings-based providers
        return self._get_settings_providers()

    def _get_config_providers(self, config_manager) -> list[dict[str, Any]]:
        """Get providers from config manager."""
        config = config_manager.get_config()
        result = []

        for provider_id, provider_cfg in config.providers.items():
            if not provider_cfg.enabled:
                continue

            # Get models for this provider
            models = list(provider_cfg.models.keys())
            default_model = (
                provider_cfg.get_default_model().model_id
                if provider_cfg.get_default_model()
                else (models[0] if models else None)
            )

            health = self._provider_health.get(provider_id)

            result.append(
                {
                    "provider": provider_id,
                    "display_name": provider_cfg.name,
                    "models": models,
                    "default_model": default_model,
                    "is_healthy": health.is_healthy if health else True,
                    "status": ("active" if (health and health.is_healthy) else "unknown"),
                    "priority": provider_cfg.priority,
                    "capabilities": {
                        "streaming": (provider_cfg.capabilities.supports_streaming),
                        "vision": provider_cfg.capabilities.supports_vision,
                        "function_calling": (provider_cfg.capabilities.supports_function_calling),
                    },
                },
            )

        # Sort by priority
        result.sort(key=lambda p: p.get("priority", 999))
        return result

    def _get_settings_providers(self) -> list[dict[str, Any]]:
        """Get providers from settings (fallback)."""
        provider_models = settings.get_provider_models()

        result = []
        # Include all commonly used providers, not just gemini/deepseek
        for provider in ["gemini", "deepseek", "routeway", "anthropic", "openai"]:
            # Map gemini to google for model lookup
            lookup_key = "google" if provider == "gemini" else provider
            models = provider_models.get(lookup_key, [])

            health = self._provider_health.get(provider)

            display_name_map = {
                "gemini": "Google Gemini",
                "deepseek": "DeepSeek",
                "routeway": "Routeway Gateway",
                "anthropic": "Anthropic",
                "openai": "OpenAI",
            }
            display_name = display_name_map.get(provider, provider.title())

            result.append(
                {
                    "provider": provider,
                    "display_name": display_name,
                    "models": models,
                    "default_model": models[0] if models else None,
                    "is_healthy": health.is_healthy if health else True,
                    "status": ("active" if (health and health.is_healthy) else "unknown"),
                },
            )

        return result

    def get_available_failover_chains(self) -> dict[str, list[str]]:
        """Get available failover chain names and their provider sequences.

        Returns dict mapping chain name to list of provider names.
        """
        config_manager = _get_config_manager()
        if not config_manager:
            return {}

        try:
            config = config_manager.get_config()
            result = {}
            for chain_name, chain_def in config.failover_chains.items():
                result[chain_name] = [provider for provider, model in chain_def.chain]
            return result
        except Exception as e:
            logger.debug(f"Failed to get failover chains: {e}")
            return {}

    def get_config_priority_order(self) -> list[str]:
        """Get providers ordered by config priority.

        Returns list of provider IDs ordered by priority (lowest first).
        """
        config_manager = _get_config_manager()
        if not config_manager:
            return []

        try:
            config = config_manager.get_config()
            providers = [(pid, p.priority) for pid, p in config.providers.items() if p.enabled]
            providers.sort(key=lambda x: x[1])
            return [pid for pid, _ in providers]
        except Exception as e:
            logger.debug(f"Failed to get priority order: {e}")
            return []


# Global singleton instance
model_router_service = ModelRouterService()


def get_model_router_service() -> ModelRouterService:
    """Get the model router service singleton."""
    return model_router_service
