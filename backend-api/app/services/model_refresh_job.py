"""
Background job for refreshing model availability every 60 seconds.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from app.core.database import db_manager
from app.domain.models import GenerationConfig, PromptRequest
from app.models.model_sync import ModelAvailabilityCache
from app.services.model_sync_service import model_sync_service

logger = logging.getLogger(__name__)

# Model availability check intervals
MODEL_REFRESH_INTERVAL = 60  # seconds
HEALTH_CHECK_INTERVAL = 30  # seconds
CONSECUTIVE_FAILURES_THRESHOLD = 3  # Mark model unavailable after N failures


class ModelRefreshJob:
    """Background job to refresh model availability."""

    def __init__(self):
        self.is_running = False
        self.failure_counts: dict[str, int] = {}
        self.last_health_check: datetime = datetime.utcnow()

    async def start(self):
        """Start the background job."""
        if self.is_running:
            logger.warning("Model refresh job is already running")
            return

        self.is_running = True
        logger.info("Starting model availability refresh job")

        try:
            while self.is_running:
                await self._refresh_model_availability()
                await self._perform_health_checks()
                await asyncio.sleep(MODEL_REFRESH_INTERVAL)

        except asyncio.CancelledError:
            logger.info("Model refresh job cancelled")
        except Exception as e:
            logger.error(f"Model refresh job error: {e}")
        finally:
            self.is_running = False

    async def stop(self):
        """Stop the background job."""
        self.is_running = False
        logger.info("Model refresh job stopped")

    async def _refresh_model_availability(self):
        """Refresh availability of all models."""
        try:
            logger.debug("Refreshing model availability...")

            # Get current available models
            available_response = await model_sync_service.get_available_models()

            # Check availability for each model
            for model in available_response.models:
                is_available = await self._check_model_availability(model)

                # Update availability status
                await model_sync_service.update_model_availability(
                    model_id=model.id,
                    is_available=is_available,
                    error_message=(
                        None
                        if is_available
                        else f"Model unavailable after {self.failure_counts.get(model.id, 0)} failures"
                    ),
                )

                # Update failure counter
                if is_available:
                    self.failure_counts[model.id] = 0
                else:
                    self.failure_counts[model.id] = self.failure_counts.get(model.id, 0) + 1

            logger.info(
                f"Model availability refresh completed. Checked {len(available_response.models)} models"
            )

        except Exception as e:
            logger.error(f"Failed to refresh model availability: {e}")

    async def _check_model_availability(self, model) -> bool:
        """Check if a specific model is available."""
        try:
            # Extract provider and model name from model ID
            if ":" not in model.id:
                return False  # Invalid model ID format

            provider_id, model_name = model.id.split(":", 1)

            # Check availability based on provider
            if provider_id == "google":
                return await self._check_google_model(model_name)
            elif provider_id == "anthropic" or provider_id == "kiro":
                return await self._check_anthropic_model(model_name)
            elif provider_id == "openai" or provider_id == "cursor":
                return await self._check_openai_model(model_name)
            elif provider_id == "deepseek":
                return await self._check_deepseek_model(model_name)
            else:
                logger.warning(f"Unknown provider: {provider_id}")
                return False

        except Exception as e:
            logger.error(f"Error checking model {model.id}: {e}")
            return False

    async def _check_google_model(self, model_name: str) -> bool:
        """Check Google Gemini model availability."""
        try:
            from app.infrastructure.gemini_client import GeminiClient

            client = GeminiClient()

            # Simple test call to check availability
            test_response = await client.generate(
                PromptRequest(
                    prompt="test", model=model_name, config=GenerationConfig(max_output_tokens=10)
                )
            )

            return test_response is not None

        except Exception as e:
            logger.debug(f"Google model {model_name} check failed: {e}")
            return False

    async def _check_anthropic_model(self, model_name: str) -> bool:
        """Check Anthropic model availability."""
        try:
            from app.infrastructure.anthropic_client import AnthropicClient

            client = AnthropicClient()

            # Test call to Anthropic endpoint
            test_response = await client.generate(
                PromptRequest(
                    prompt="test", model=model_name, config=GenerationConfig(max_output_tokens=10)
                )
            )

            return test_response is not None

        except Exception as e:
            logger.debug(f"Anthropic model {model_name} check failed: {e}")
            return False

    async def _check_openai_model(self, model_name: str) -> bool:
        """Check OpenAI model availability."""
        try:
            from app.infrastructure.openai_client import OpenAIClient

            client = OpenAIClient()

            # Test call to OpenAI endpoint
            test_response = await client.generate(
                PromptRequest(
                    prompt="test", model=model_name, config=GenerationConfig(max_output_tokens=10)
                )
            )

            return test_response is not None

        except Exception as e:
            logger.debug(f"OpenAI model {model_name} check failed: {e}")
            return False

    async def _check_deepseek_model(self, model_name: str) -> bool:
        """Check DeepSeek model availability."""
        try:
            from app.infrastructure.deepseek_client import DeepSeekClient

            client = DeepSeekClient()

            test_response = await client.generate(
                PromptRequest(
                    prompt="test", model=model_name, config=GenerationConfig(max_output_tokens=10)
                )
            )

            return test_response is not None

        except Exception as e:
            logger.debug(f"DeepSeek model {model_name} check failed: {e}")
            return False

    async def _perform_health_checks(self):
        """Perform health checks on all providers."""
        try:
            now = datetime.utcnow()

            # Only perform health checks if enough time has passed
            if (now - self.last_health_check).total_seconds() < HEALTH_CHECK_INTERVAL:
                return

            self.last_health_check = now
            logger.debug("Performing provider health checks...")

            # Check health of each provider
            providers_health = {}

            # Google/Gemini
            try:
                from app.infrastructure.gemini_client import GeminiClient

                client = GeminiClient()
                providers_health["google"] = await self._test_provider_health(client, "Google")
            except Exception as e:
                providers_health["google"] = {"status": "error", "error": str(e)}

            # Anthropic/Kiro
            try:
                from app.infrastructure.anthropic_client import AnthropicClient

                client = AnthropicClient()
                providers_health["anthropic"] = await self._test_provider_health(
                    client, "Anthropic"
                )
            except Exception as e:
                providers_health["anthropic"] = {"status": "error", "error": str(e)}

            # OpenAI/Cursor
            try:
                from app.infrastructure.openai_client import OpenAIClient

                client = OpenAIClient()
                providers_health["openai"] = await self._test_provider_health(client, "OpenAI")
            except Exception as e:
                providers_health["openai"] = {"status": "error", "error": str(e)}

            # DeepSeek
            try:
                from app.infrastructure.deepseek_client import DeepSeekClient

                client = DeepSeekClient()
                providers_health["deepseek"] = await self._test_provider_health(client, "DeepSeek")
            except Exception as e:
                providers_health["deepseek"] = {"status": "error", "error": str(e)}

            # Log overall health status
            healthy_providers = sum(
                1 for health in providers_health.values() if health.get("status") == "healthy"
            )
            total_providers = len(providers_health)

            logger.info(
                f"Provider health check completed: {healthy_providers}/{total_providers} providers healthy"
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _test_provider_health(self, client, provider_name: str) -> dict[str, Any]:
        """Test health of a specific provider client."""
        try:
            # Simple test call
            start_time = datetime.utcnow()

            test_response = await client.generate(
                PromptRequest(prompt="health check", config=GenerationConfig(max_output_tokens=5))
            )

            response_time = (datetime.utcnow() - start_time).total_seconds()

            if test_response and response_time < 10.0:  # 10 second timeout
                return {
                    "status": "healthy",
                    "response_time_ms": response_time * 1000,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_time_ms": (
                        response_time * 1000 if "response_time" in locals() else None
                    ),
                    "error": "Test call failed or timed out",
                    "timestamp": datetime.utcnow().isoformat(),
                }

        except Exception as e:
            return {"status": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

    async def get_job_status(self) -> dict[str, Any]:
        """Get current job status."""
        return {
            "is_running": self.is_running,
            "failure_counts": self.failure_counts,
            "last_health_check": self.last_health_check.isoformat(),
            "models_checked": len(self.failure_counts),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def cleanup_stale_cache_entries(self):
        """Clean up stale cache entries."""
        try:
            async with db_manager.get_session() as db_session:
                # Delete entries older than 24 hours
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                from sqlalchemy import delete

                result = await db_session.execute(
                    delete(ModelAvailabilityCache).where(
                        ModelAvailabilityCache.last_checked < cutoff_time
                    )
                )

                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} stale cache entries")

        except Exception as e:
            logger.error(f"Failed to cleanup stale cache entries: {e}")


# Global job instance
model_refresh_job = ModelRefreshJob()


async def start_model_refresh_job():
    """Start the model refresh background job."""
    await model_refresh_job.start()


async def stop_model_refresh_job():
    """Stop the model refresh background job."""
    await model_refresh_job.stop()


async def get_model_refresh_status():
    """Get the status of the model refresh job."""
    return await model_refresh_job.get_job_status()
