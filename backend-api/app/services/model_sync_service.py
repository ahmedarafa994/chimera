"""
Model synchronization service with Redis caching and routing.
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis.asyncio as redis

from app.core.config import settings
from app.core.database import db_manager
from app.core.unified_errors import ChimeraError
from app.domain.model_sync import (
    AvailableModelsResponse,
    ModelInfo,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from app.models.model_sync import (
    ModelAvailabilityCache,
    cleanup_expired_sessions,
    get_user_session,
    log_model_change_error,
    update_user_session_model,
)

logger = logging.getLogger(__name__)

# Redis client for caching
redis_client: redis.Redis | None = None

# Rate limiting: 10 model changes per minute per user
MODEL_CHANGE_RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60  # seconds

# Cache TTL settings
MODELS_CACHE_TTL = 300  # 5 minutes
MODEL_AVAILABILITY_TTL = 60  # 1 minute


def datetime_serializer(obj: Any) -> str:
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def get_redis_client() -> redis.Redis:
    """Get or create Redis client."""
    global redis_client
    if redis_client is None:
        try:
            redis_url = (
                settings.REDIS_URL if hasattr(settings, "REDIS_URL") else "redis://localhost:6379/0"
            )
            redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            # Test connection
            await redis_client.ping()
            logger.info("Redis client initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory caching: {e}")
            redis_client = None
    return redis_client


class ModelSyncService:
    """Service for model synchronization with caching and routing."""

    def __init__(self):
        self.default_model = getattr(settings, "DEFAULT_MODEL_ID", "deepseek-chat")
        self._models_cache: list[ModelInfo] | None = None
        self._cache_timestamp: datetime | None = None

    async def initialize_models(self):
        """Initialize models from configuration."""
        try:
            redis_client = await get_redis_client()

            # Try to get from cache first
            if redis_client:
                cached_models = await redis_client.get("available_models")
                if cached_models:
                    models_data = json.loads(cached_models)
                    self._models_cache = [ModelInfo(**model) for model in models_data]
                    self._cache_timestamp = datetime.utcnow()
                    logger.info(f"Loaded {len(self._models_cache)} models from Redis cache")
                    return

            # Load from configuration if not in cache
            models = []
            provider_models = settings.get_provider_models()

            for provider_id, model_list in provider_models.items():
                for model_name in model_list:
                    model_id = f"{provider_id}:{model_name}"

                    # Determine capabilities based on model name
                    capabilities = self._determine_model_capabilities(model_name)
                    max_tokens = self._determine_max_tokens(model_name)

                    model_info = ModelInfo(
                        id=model_id,
                        name=model_name,
                        description=f"{provider_id.title()} {model_name}",
                        capabilities=capabilities,
                        max_tokens=max_tokens,
                        # All config models are active by default
                        is_active=True,
                        provider=provider_id,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    models.append(model_info)

            self._models_cache = models
            self._cache_timestamp = datetime.utcnow()

            # Cache in Redis with proper datetime serialization
            if redis_client:
                models_json = json.dumps(
                    [model.model_dump(mode="json") for model in models], default=datetime_serializer
                )
                await redis_client.setex("available_models", MODELS_CACHE_TTL, models_json)

            logger.info(f"Initialized {len(models)} models from configuration")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise ChimeraError(status_code=500, message=f"Model initialization failed: {e!s}")

    def _determine_model_capabilities(self, model_name: str) -> list[str]:
        """Determine model capabilities based on name."""
        capabilities = ["text_generation"]
        lower_name = model_name.lower()

        if "image" in lower_name:
            capabilities.append("image_generation")
        if "computer-use" in lower_name:
            capabilities.append("computer_use")
        if "thinking" in lower_name or "o1" in lower_name or "o3" in lower_name:
            capabilities.append("reasoning")
        if "flash" in lower_name or "haiku" in lower_name or "mini" in lower_name:
            capabilities.append("fast_inference")
        if "pro" in lower_name or "opus" in lower_name:
            capabilities.append("high_quality")

        return capabilities

    def _determine_max_tokens(self, model_name: str) -> int:
        """Determine max tokens based on model name."""
        lower_name = model_name.lower()
        if "pro" in lower_name or "opus" in lower_name:
            return 8192
        elif "flash" in lower_name or "haiku" in lower_name or "mini" in lower_name:
            return 4096
        elif "thinking" in lower_name or "o1" in lower_name or "o3" in lower_name:
            return 32768
        else:
            return 2048  # Default

    async def get_available_models(self) -> AvailableModelsResponse:
        """Get all available models with caching."""
        try:
            if not self._models_cache or not self._cache_timestamp:
                await self.initialize_models()

            # Check if cache is still valid
            cache_age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
            if cache_age > MODELS_CACHE_TTL:
                await self.initialize_models()  # Refresh cache

            return AvailableModelsResponse(
                models=self._models_cache or [],
                count=len(self._models_cache or []),
                timestamp=datetime.utcnow().isoformat(),
                cache_ttl=MODELS_CACHE_TTL,
            )

        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise ChimeraError(status_code=500, message=f"Failed to retrieve models: {e!s}")

    async def validate_model_id(self, model_id: str) -> bool:
        """Validate if model ID exists in available models."""
        if not self._models_cache:
            await self.initialize_models()

        return any(model.id == model_id for model in self._models_cache)

    async def select_model(
        self,
        request: ModelSelectionRequest,
        user_id: str,
        session_id: str,
        ip_address: str,
        user_agent: str | None = None,
    ) -> ModelSelectionResponse:
        """Select a model for the user session."""
        try:
            # Rate limiting check
            if not await self._check_rate_limit(user_id, session_id):
                return ModelSelectionResponse(
                    success=False,
                    model_id=request.model_id,
                    message=("Rate limit exceeded. Maximum 10 model changes per minute."),
                    timestamp=datetime.utcnow().isoformat(),
                )

            # Validate model ID
            if not await self.validate_model_id(request.model_id):
                # Return fallback model
                fallback_model = self.default_model
                if self._models_cache:
                    fallback_model = (
                        self._models_cache[0].id if self._models_cache else self.default_model
                    )

                return ModelSelectionResponse(
                    success=False,
                    model_id=request.model_id,
                    message="Invalid model ID",
                    timestamp=datetime.utcnow().isoformat(),
                    fallback_model=fallback_model,
                )

            # Update user session
            async with db_manager.session() as db_session:
                success = await update_user_session_model(db_session, session_id, request.model_id)

                if success:
                    # Update rate limit counter
                    await self._update_rate_limit(user_id, session_id)

                    return ModelSelectionResponse(
                        success=True,
                        model_id=request.model_id,
                        message="Model selected successfully",
                        timestamp=datetime.utcnow().isoformat(),
                    )
                else:
                    # Log error
                    await log_model_change_error(
                        db_session,
                        user_id,
                        session_id,
                        request.model_id,
                        "Failed to update user session",
                        ip_address,
                        user_agent,
                    )

                    return ModelSelectionResponse(
                        success=False,
                        model_id=request.model_id,
                        message="Failed to update session",
                        timestamp=datetime.utcnow().isoformat(),
                    )

        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return ModelSelectionResponse(
                success=False,
                model_id=request.model_id,
                message=f"Selection failed: {e!s}",
                timestamp=datetime.utcnow().isoformat(),
            )

    async def get_user_model(self, session_id: str) -> ModelInfo | None:
        """Get the currently selected model for a user session."""
        try:
            async with db_manager.session() as db_session:
                session = await get_user_session(db_session, session_id)
                if session and session.selected_model_id:
                    # Find model info
                    if self._models_cache:
                        for model in self._models_cache:
                            if model.id == session.selected_model_id:
                                return model
                    return None
                return None

        except Exception as e:
            logger.error(f"Failed to get user model: {e}")
            return None

    async def _check_rate_limit(self, user_id: str, session_id: str) -> bool:
        """Check if user has exceeded rate limit for model changes."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return True  # No rate limiting if Redis unavailable

            rate_key = f"model_change_rate:{user_id}:{session_id}"
            current_count = await redis_client.get(rate_key)

            return not (current_count and int(current_count) >= MODEL_CHANGE_RATE_LIMIT)

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True  # Allow on error

    async def _update_rate_limit(self, user_id: str, session_id: str):
        """Update rate limit counter for user."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return

            rate_key = f"model_change_rate:{user_id}:{session_id}"
            current_count = await redis_client.get(rate_key)

            new_count = int(current_count) + 1 if current_count else 1

            await redis_client.setex(rate_key, RATE_LIMIT_WINDOW, str(new_count))

        except Exception as e:
            logger.warning(f"Failed to update rate limit: {e}")

    async def update_model_availability(
        self, model_id: str, is_available: bool, error_message: str | None = None
    ):
        """Update model availability status."""
        try:
            redis_client = await get_redis_client()

            # Update in Redis cache
            if redis_client:
                availability_key = f"model_availability:{model_id}"
                availability_data = {
                    "is_available": is_available,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_message": error_message,
                }
                await redis_client.setex(
                    availability_key,
                    MODEL_AVAILABILITY_TTL,
                    json.dumps(availability_data),
                )

            # Update in database
            async with db_manager.session() as db_session:
                cache_entry = await db_session.get(ModelAvailabilityCache, model_id)
                if cache_entry:
                    cache_entry.is_available = is_available
                    cache_entry.last_checked = datetime.utcnow()
                    cache_entry.error_message = error_message
                else:
                    cache_entry = ModelAvailabilityCache(
                        model_id=model_id,
                        is_available=is_available,
                        last_checked=datetime.utcnow(),
                        error_message=error_message,
                    )
                    db_session.add(cache_entry)

                await db_session.commit()

            logger.info(f"Updated model availability: {model_id} -> {is_available}")

        except Exception as e:
            logger.error(f"Failed to update model availability: {e}")

    async def cleanup_expired_data(self):
        """Clean up expired sessions and cache data."""
        try:
            async with db_manager.session() as db_session:
                await cleanup_expired_sessions(db_session)

            logger.info("Cleaned up expired model sync data")

        except Exception as e:
            logger.error(f"Failed to cleanup expired data: {e}")


# Global service instance
model_sync_service = ModelSyncService()
