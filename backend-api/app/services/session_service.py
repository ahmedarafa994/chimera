"""
Session Management Service for Model Synchronization

This service manages user sessions and model selection, serving as the
single source of truth for model availability and validation.

Enhanced with Redis persistence for production deployments with
automatic fallback to in-memory storage.

UNIFIED-PROVIDER: Integrated with GlobalModelSelectionState for
    provider/model selection propagation across all services.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from app.core.config import settings
from app.core.logging import logger
from app.core.session_backend import InMemorySessionBackend, RedisSessionBackend, SessionBackend

if TYPE_CHECKING:
    from app.services.global_model_selection_state import GlobalModelSelectionState


@dataclass
class ModelInfo:
    """Canonical model information"""

    id: str  # Canonical identifier
    name: str  # Display name
    provider: str
    description: str = ""
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_vision: bool = False
    is_default: bool = False
    tier: str = "standard"  # standard, premium, experimental


@dataclass
class UserSession:
    """User session with model configuration"""

    session_id: str
    created_at: datetime
    last_activity: datetime
    selected_provider: str
    selected_model: str
    preferences: dict[str, Any] = field(default_factory=dict)
    request_count: int = 0

    def is_expired(self, ttl_minutes: int = 60) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() - self.last_activity > timedelta(minutes=ttl_minutes)

    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()
        self.request_count += 1


class SessionService:
    """
    Manages user sessions and model synchronization.

    This service is the single source of truth for:
    - Available models and their canonical identifiers
    - User session state and model preferences
    - Model validation and fallback logic

    Enhanced with Redis persistence for production deployments.
    Falls back to in-memory storage when Redis is unavailable.
    """

    # In-memory session cache (used alongside Redis for performance)
    _sessions: ClassVar[dict[str, UserSession]] = {}

    # Session configuration
    SESSION_TTL_MINUTES = 60
    SESSION_TTL_SECONDS = 60 * 60  # 1 hour in seconds for Redis
    MAX_SESSIONS = 10000

    def __init__(self):
        self._master_model_list: dict[str, list[ModelInfo]] = {}
        # Dynamic defaults from configuration
        self._default_provider = settings.AI_PROVIDER
        self._default_model = settings.DEFAULT_MODEL_ID

        self._backend: SessionBackend | None = None
        self._use_redis = False
        self._global_selection_state: GlobalModelSelectionState | None = None
        self._initialize_master_list()
        self._initialize_backend()
        self._initialize_global_selection_state()
        logger.info(
            f"SessionService initialized. "
            f"Default: {self._default_provider}/{self._default_model}"
        )

    def _initialize_global_selection_state(self):
        """
        Initialize connection to GlobalModelSelectionState.

        UNIFIED-PROVIDER: Enables synchronization between session
        selections and the global selection state.
        """
        try:
            from app.services.global_model_selection_state import GlobalModelSelectionState

            self._global_selection_state = GlobalModelSelectionState.get_instance()
            logger.info("SessionService connected to GlobalModelSelectionState")
        except Exception as e:
            logger.warning(
                f"Failed to connect to GlobalModelSelectionState: {e}. "
                "Session selections will not sync globally."
            )
            self._global_selection_state = None

    def _initialize_backend(self):
        """Initialize the session storage backend (Redis or in-memory)"""
        try:
            if settings.REDIS_URL and settings.REDIS_URL != "redis://localhost:6379":
                self._backend = RedisSessionBackend(settings.REDIS_URL)
                self._use_redis = True
                logger.info("SessionService using Redis backend for persistence")
            else:
                self._backend = InMemorySessionBackend()
                self._use_redis = False
                logger.info("SessionService using in-memory backend (development mode)")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis backend: {e}. Using in-memory.")
            self._backend = InMemorySessionBackend()
            self._use_redis = False

    def _initialize_master_list(self):
        """Initialize the master model list from configuration"""
        provider_models = settings.get_provider_models()

        # Define model metadata
        # This serves as a registry of known properties for models.
        # Dynamic models not in this list will get default properties.
        model_metadata = {
            # Google models
            "gemini-3-pro-preview": {
                "name": "Gemini 3 Pro Preview",
                "tier": "premium",
                "max_tokens": 8192,
                "is_default": True,
            },
            "gemini-3-pro-image-preview": {
                "name": "Gemini 3 Pro Image",
                "tier": "premium",
                "supports_vision": True,
            },
            "gemini-2.5-pro": {"name": "Gemini 2.5 Pro", "tier": "standard"},
            "gemini-2.5-pro-preview-06-05": {
                "name": "Gemini 2.5 Pro Preview",
                "tier": "experimental",
            },
            "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "tier": "standard"},
            "gemini-2.5-flash-lite": {"name": "Gemini 2.5 Flash Lite", "tier": "standard"},
            "gemini-2.5-flash-preview-09-2025": {
                "name": "Gemini 2.5 Flash Preview",
                "tier": "experimental",
            },
            "gemini-2.5-flash-image": {
                "name": "Gemini 2.5 Flash Image",
                "tier": "standard",
                "supports_vision": True,
            },
            "gemini-2.5-computer-use-preview-10-2025": {
                "name": "Gemini Computer Use",
                "tier": "experimental",
            },
            # Anthropic models
            "claude-opus-4-5": {"name": "Claude Opus 4.5", "tier": "premium", "max_tokens": 8192},
            "claude-sonnet-4-5": {
                "name": "Claude Sonnet 4.5",
                "tier": "standard",
                "is_default": True,
            },
            "claude-sonnet-4-5-20250929": {
                "name": "Claude Sonnet 4.5 (Sep 2025)",
                "tier": "standard",
            },
            "claude-haiku-4-5": {"name": "Claude Haiku 4.5", "tier": "standard"},
            "claude-sonnet-4-20250514": {"name": "Claude Sonnet 4", "tier": "standard"},
            "claude-3-7-sonnet-20250219": {"name": "Claude 3.7 Sonnet", "tier": "standard"},
            "amazonq-claude-sonnet-4-20250514": {
                "name": "Amazon Q Claude Sonnet 4",
                "tier": "standard",
            },
            "amazonq-claude-3-7-sonnet-20250219": {
                "name": "Amazon Q Claude 3.7",
                "tier": "standard",
            },
            # OpenAI models
            "gpt-4o": {"name": "GPT-4o", "tier": "premium", "is_default": True},
            "gpt-4o-mini": {"name": "GPT-4o Mini", "tier": "standard"},
            "gpt-4-turbo": {"name": "GPT-4 Turbo", "tier": "premium"},
            "gpt-4": {"name": "GPT-4", "tier": "premium"},
            "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "tier": "standard"},
            "o1": {"name": "O1", "tier": "premium", "max_tokens": 16384},
            "o1-mini": {"name": "O1 Mini", "tier": "standard"},
            "o3": {"name": "O3", "tier": "premium", "max_tokens": 32768},
            "o3-mini": {"name": "O3 Mini", "tier": "standard"},
            # Qwen models
            "qwen3-coder-plus": {
                "name": "Qwen3 Coder Plus",
                "tier": "standard",
                "is_default": True,
            },
            "qwen3-coder-flash": {"name": "Qwen3 Coder Flash", "tier": "standard"},
            # Hybrid models
            "gemini-claude-sonnet-4-5-thinking": {
                "name": "Gemini-Claude Thinking",
                "tier": "premium",
            },
            "gemini-claude-sonnet-4-5": {"name": "Gemini-Claude Hybrid", "tier": "premium"},
            "gpt-oss-120b-medium": {"name": "GPT OSS 120B", "tier": "experimental"},
        }

        for provider, models in provider_models.items():
            self._master_model_list[provider] = []
            for model_id in models:
                metadata = model_metadata.get(model_id, {})
                model_info = ModelInfo(
                    id=model_id,
                    name=metadata.get("name", model_id),
                    provider=provider,
                    description=f"{provider.upper()} model: {model_id}",
                    max_tokens=metadata.get("max_tokens", 4096),
                    supports_streaming=True,
                    supports_vision=metadata.get("supports_vision", False),
                    is_default=metadata.get("is_default", False),
                    tier=metadata.get("tier", "standard"),
                )
                self._master_model_list[provider].append(model_info)

    def get_master_model_list(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get the complete master model list with canonical identifiers.
        This is the single source of truth for available models.
        """
        result = {}
        for provider, models in self._master_model_list.items():
            result[provider] = [
                {
                    "id": m.id,
                    "name": m.name,
                    "provider": m.provider,
                    "description": m.description,
                    "max_tokens": m.max_tokens,
                    "supports_streaming": m.supports_streaming,
                    "supports_vision": m.supports_vision,
                    "is_default": m.is_default,
                    "tier": m.tier,
                }
                for m in models
            ]
        return result

    def get_providers_with_models(self) -> list[dict[str, Any]]:
        """Get list of providers with their available models"""
        providers = []
        for provider, models in self._master_model_list.items():
            default_model = next((m for m in models if m.is_default), models[0] if models else None)
            providers.append(
                {
                    "provider": provider,
                    "status": "active",
                    "model": default_model.id if default_model else None,
                    "available_models": [m.id for m in models],
                    "models_detail": [
                        {"id": m.id, "name": m.name, "tier": m.tier, "is_default": m.is_default}
                        for m in models
                    ],
                }
            )
        return providers

    def validate_model(self, provider: str, model_id: str) -> tuple[bool, str, str | None]:
        """
        Validate a model selection against the master list.

        Returns:
            tuple: (is_valid, message, fallback_model_id)
        """
        # Check if provider exists
        if provider not in self._master_model_list:
            return (
                False,
                f"Invalid provider: {provider}. Available providers: {list(self._master_model_list.keys())}",
                self._default_model,
            )

        # Check if model exists for provider
        provider_models = self._master_model_list[provider]
        model_ids = [m.id for m in provider_models]

        if model_id not in model_ids:
            default_for_provider = next(
                (m.id for m in provider_models if m.is_default), model_ids[0] if model_ids else None
            )
            return (
                False,
                f"Invalid model '{model_id}' for provider '{provider}'. Available models: {model_ids}",
                default_for_provider,
            )

        return (True, "Model validated successfully", None)

    def get_default_model(self, provider: str | None = None) -> tuple[str, str]:
        """Get the default model for a provider or the global default"""
        if provider and provider in self._master_model_list:
            models = self._master_model_list[provider]
            default = next((m for m in models if m.is_default), models[0] if models else None)
            if default:
                return (provider, default.id)
        return (self._default_provider, self._default_model)

    def set_default_model(self, provider: str, model: str | None = None) -> tuple[str, str]:
        """Update the global default provider/model."""
        if not provider:
            raise ValueError("Provider is required")

        if model is None:
            provider, model = self.get_default_model(provider)
        else:
            is_valid, message, _fallback = self.validate_model(provider, model)
            if not is_valid:
                raise ValueError(message)

        self._default_provider = provider
        self._default_model = model
        logger.info(f"Updated defaults: provider={provider}, model={model}")
        return (provider, model)

    # Session Management Methods

    def _session_to_dict(self, session: UserSession) -> dict:
        """Convert a UserSession to a dictionary for storage"""
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "selected_provider": session.selected_provider,
            "selected_model": session.selected_model,
            "preferences": session.preferences,
            "request_count": session.request_count,
        }

    def _dict_to_session(self, data: dict) -> UserSession:
        """Convert a dictionary back to a UserSession"""
        return UserSession(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            selected_provider=data["selected_provider"],
            selected_model=data["selected_model"],
            preferences=data.get("preferences", {}),
            request_count=data.get("request_count", 0),
        )

    async def _persist_session(self, session: UserSession):
        """Persist session to backend storage"""
        if self._backend:
            try:
                await self._backend.set(
                    session.session_id,
                    self._session_to_dict(session),
                    ttl_seconds=self.SESSION_TTL_SECONDS,
                )
            except Exception as e:
                logger.error(f"Failed to persist session to backend: {e}")

    async def _load_session_from_backend(self, session_id: str) -> UserSession | None:
        """Load session from backend storage"""
        if self._backend:
            try:
                data = await self._backend.get(session_id)
                if data:
                    return self._dict_to_session(data)
            except Exception as e:
                logger.error(f"Failed to load session from backend: {e}")
        return None

    def create_session(self, provider: str | None = None, model: str | None = None) -> UserSession:
        """Create a new user session"""
        # Clean up expired sessions periodically
        self._cleanup_expired_sessions()

        # Validate and set defaults
        if not provider:
            provider = self._default_provider
        if not model:
            # If default provider is used, check if we should use its default model
            if provider == self._default_provider:
                model = self._default_model
            else:
                _, model = self.get_default_model(provider)

        # Validate the model selection
        is_valid, message, fallback = self.validate_model(provider, model)
        if not is_valid:
            logger.warning(f"Invalid model selection during session creation: {message}")
            model = fallback or self._default_model
            provider = self._default_provider if fallback == self._default_model else provider

        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = UserSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            selected_provider=provider,
            selected_model=model,
            preferences={},
            request_count=0,
        )

        # Store in memory cache
        self._sessions[session_id] = session

        # Persist to backend asynchronously (fire and forget for sync method)
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._persist_session(session))
            else:
                loop.run_until_complete(self._persist_session(session))
        except RuntimeError:
            # No event loop, skip async persistence
            pass

        logger.info(f"Created session {session_id} with provider={provider}, model={model}")

        return session

    def get_session(self, session_id: str) -> UserSession | None:
        """Get a session by ID (checks memory cache first, then backend)"""
        # Check memory cache first
        session = self._sessions.get(session_id)
        if session and not session.is_expired(self.SESSION_TTL_MINUTES):
            session.touch()
            return session
        elif session:
            # Session expired, remove it
            del self._sessions[session_id]
            logger.info(f"Session {session_id} expired and removed from cache")

        # Try to load from backend if using Redis
        if self._use_redis and self._backend:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync context, return None
                    # The async version should be used instead
                    return None
                else:
                    session = loop.run_until_complete(self._load_session_from_backend(session_id))
                    if session and not session.is_expired(self.SESSION_TTL_MINUTES):
                        # Cache it in memory
                        self._sessions[session_id] = session
                        session.touch()
                        return session
            except RuntimeError:
                pass

        return None

    async def get_session_async(self, session_id: str) -> UserSession | None:
        """Async version of get_session for use in async contexts"""
        # Check memory cache first
        session = self._sessions.get(session_id)
        if session and not session.is_expired(self.SESSION_TTL_MINUTES):
            session.touch()
            return session
        elif session:
            del self._sessions[session_id]

        # Try to load from backend
        if self._backend:
            session = await self._load_session_from_backend(session_id)
            if session and not session.is_expired(self.SESSION_TTL_MINUTES):
                self._sessions[session_id] = session
                session.touch()
                # Update backend with new activity time
                await self._persist_session(session)
                return session

        return None

    async def update_session_model(
        self, session_id: str, provider: str, model: str
    ) -> tuple[bool, str, UserSession | None]:
        """
        Update the model selection for a session.

        Returns:
            tuple: (success, message, updated_session)
        """
        session = self.get_session(session_id)
        if not session:
            return (False, "Session not found or expired", None)

        # Validate the new model selection
        is_valid, message, fallback = self.validate_model(provider, model)

        if not is_valid:
            # Revert to fallback model
            logger.warning(f"Invalid model update for session {session_id}: {message}")
            if fallback:
                session.selected_model = fallback
                # Keep provider if fallback is for same provider
                if provider in self._master_model_list:
                    session.selected_provider = provider
                else:
                    session.selected_provider = self._default_provider
                session.touch()
                return (False, f"{message}. Reverted to default model: {fallback}", session)
            return (False, message, session)

        # Update session with validated model
        session.selected_provider = provider
        session.selected_model = model
        session.touch()

        # UNIFIED-PROVIDER: Sync with GlobalModelSelectionState
        await self._sync_selection_to_global_state(session_id, provider, model)

        logger.info(f"Session {session_id} updated: provider={provider}, model={model}")
        return (True, "Model selection updated successfully", session)

    async def _sync_selection_to_global_state(
        self,
        session_id: str,
        provider: str,
        model: str,
    ) -> None:
        """
        Sync session selection to GlobalModelSelectionState.

        UNIFIED-PROVIDER: Ensures that session-level selections are
        propagated to the global state for cross-service consistency.

        Args:
            session_id: The session ID
            provider: Selected provider
            model: Selected model
        """
        if not self._global_selection_state:
            return

        try:
            from app.services.global_model_selection_state import ModelSelection, SelectionScope

            selection = ModelSelection(
                provider=provider,
                model_id=model,
                session_id=session_id,
                scope=SelectionScope.SESSION,
            )

            await self._global_selection_state.set_session_selection(
                session_id=session_id,
                selection=selection,
            )

            logger.debug(
                f"Synced session {session_id} selection to global state: " f"{provider}/{model}"
            )
        except Exception as e:
            logger.warning(f"Failed to sync session selection to global state: {e}")

    async def load_session_selection_to_context(
        self,
        session_id: str,
    ) -> tuple[str, str] | None:
        """
        Load session selection and set it as the current request context.

        UNIFIED-PROVIDER: Call this at the start of request processing
        to ensure the session's provider/model selection is used.

        Args:
            session_id: The session ID to load

        Returns:
            tuple[str, str] | None: The (provider, model) tuple if found
        """
        session = await self.get_session_async(session_id)
        if not session:
            return None

        provider = session.selected_provider
        model = session.selected_model

        # Set as request context if global state is available
        if self._global_selection_state:
            try:
                from app.services.global_model_selection_state import SelectionContextData

                context = SelectionContextData(
                    provider=provider,
                    model_id=model,
                    session_id=session_id,
                    source="session_service",
                )

                self._global_selection_state.set_request_context(context)

                logger.debug(
                    f"Loaded session {session_id} selection to request "
                    f"context: {provider}/{model}"
                )
            except Exception as e:
                logger.warning(f"Failed to set request context from session: {e}")

        return (provider, model)

    def get_global_selection_state(self) -> "GlobalModelSelectionState | None":
        """
        Get the GlobalModelSelectionState instance.

        Returns:
            The global selection state or None if not initialized
        """
        return self._global_selection_state

    def get_session_model(self, session_id: str) -> tuple[str, str]:
        """Get the current model selection for a session"""
        session = self.get_session(session_id)
        if session:
            return (session.selected_provider, session.selected_model)
        return self.get_default_model()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} deleted")
            return True
        return False

    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        if len(self._sessions) < self.MAX_SESSIONS // 2:
            return  # Only cleanup when approaching limit

        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.is_expired(self.SESSION_TTL_MINUTES)
        ]

        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics"""
        active_sessions = sum(
            1 for s in self._sessions.values() if not s.is_expired(self.SESSION_TTL_MINUTES)
        )

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "max_sessions": self.MAX_SESSIONS,
            "session_ttl_minutes": self.SESSION_TTL_MINUTES,
        }


# Global singleton instance
session_service = SessionService()


def get_session_service() -> SessionService:
    """Get the session service singleton"""
    return session_service
