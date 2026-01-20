"""Comprehensive tests for model synchronization system.
Covers API endpoints, WebSocket functionality, and service layer.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.domain.model_sync import ModelSelectionRequest
from app.main import app
from app.services.model_sync_service import ModelSyncService


class TestModelSyncAPI:
    """Test model synchronization API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch(
            "app.services.model_sync_service.get_redis_client",
            new_callable=AsyncMock,
        ) as mock_get_redis:
            mock_redis_instance = AsyncMock()
            mock_get_redis.return_value = mock_redis_instance
            mock_redis_instance.get.return_value = None
            mock_redis_instance.setex.return_value = True
            mock_redis_instance.delete.return_value = True
            mock_redis_instance.exists.return_value = False
            yield mock_redis_instance

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch("app.core.database.db_manager.session") as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def sample_models(self):
        """Sample model data for testing."""
        return [
            {
                "id": "google:gemini-1.5-pro",
                "name": "gemini-1.5-pro",
                "description": "Google gemini-1.5-pro",
                "provider": "google",
                "capabilities": ["text_generation", "fast_inference", "high_quality"],
                "max_tokens": 8192,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
            {
                "id": "openai:gpt-4-turbo",
                "name": "gpt-4-turbo",
                "description": "OpenAI gpt-4-turbo",
                "provider": "openai",
                "capabilities": ["text_generation", "reasoning"],
                "max_tokens": 128000,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
        ]

    def test_get_available_models_success(
        self, client, mock_redis, mock_db_session, sample_models
    ) -> None:
        """Test successful retrieval of available models."""
        # Mock settings to return sample models
        # Correctly patch the method on the Settings CLASS
        with patch("app.core.config.Settings.get_provider_models") as mock_get_models:
            mock_get_models.return_value = {"google": ["gemini-1.5-pro"], "openai": ["gpt-4-turbo"]}

            # Mock Redis cache miss
            mock_redis.get.return_value = None

            response = client.get("/api/v1/models/available")

            # Since the implementation relies on settings and not DB for initial load:
            # It should succeed if settings are mocked correctly.
            # However, if it fails due to DB errors inside initialize_models (unlikely as it uses settings),
            # we should check the response.

            # Note: The implementation uses settings.get_provider_models()

            assert response.status_code == 200
            data = response.json()

            assert "models" in data
            assert "count" in data
            assert "timestamp" in data
            assert "cache_ttl" in data
            # Based on mocked settings, count should be 2
            assert data["count"] == 2
            assert len(data["models"]) == 2

    def test_select_model_success(self, client, mock_redis, mock_db_session) -> None:
        """Test successful model selection."""
        # Mock validation to succeed
        with patch(
            "app.services.model_sync_service.model_sync_service.validate_model_id",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = True

            # Mock update_user_session_model
            with patch(
                "app.services.model_sync_service.update_user_session_model",
                new_callable=AsyncMock,
            ) as mock_update:
                mock_update.return_value = True

                request_data = {
                    "model_id": "google:gemini-1.5-pro",
                    "timestamp": datetime.utcnow().isoformat(),
                }

                response = client.post(
                    "/api/v1/models/session/model",
                    json=request_data,
                    headers={"X-Session-ID": "test_session"},
                )

                assert response.status_code == 200
                data = response.json()

                assert data["success"] is True
                assert data["model_id"] == "google:gemini-1.5-pro"

    def test_select_model_invalid_model(self, client, mock_redis, mock_db_session) -> None:
        """Test model selection with invalid model ID."""
        with patch(
            "app.services.model_sync_service.model_sync_service.validate_model_id",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = False

            request_data = {
                "model_id": "invalid-model-id",
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = client.post(
                "/api/v1/models/session/model",
                json=request_data,
                headers={"X-Session-ID": "test_session"},
            )

            # The API returns 200 OK with success=False for logic failures handled in the service
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is False
            assert "invalid" in data["message"].lower()

    def test_validate_model_success(self, client, mock_db_session) -> None:
        """Test successful model validation."""
        with patch(
            "app.services.model_sync_service.model_sync_service.validate_model_id",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = True

            response = client.post("/api/v1/models/validate/google:gemini-1.5-pro")

            assert response.status_code == 200
            data = response.json()

            assert data["is_valid"] is True
            assert data["model_id"] == "google:gemini-1.5-pro"

    def test_validate_model_not_found(self, client, mock_db_session) -> None:
        """Test model validation with non-existent model."""
        with patch(
            "app.services.model_sync_service.model_sync_service.validate_model_id",
            new_callable=AsyncMock,
        ) as mock_validate:
            mock_validate.return_value = False

            response = client.post("/api/v1/models/validate/nonexistent-model")

            assert response.status_code == 200
            data = response.json()

            assert data["is_valid"] is False
            assert data["model_id"] == "nonexistent-model"

    def test_health_check(self, client) -> None:
        """Test model sync service health check."""
        response = client.get("/api/v1/models/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestModelSyncService:
    """Test model synchronization service layer."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        return mock_redis

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ModelSyncService()

    @pytest.mark.asyncio
    async def test_get_available_models_cache_miss(self, service, mock_redis) -> None:
        """Test getting available models with cache miss."""
        # Correctly patch the method on the Settings CLASS
        with (
            patch("app.services.model_sync_service.get_redis_client", return_value=mock_redis),
            patch("app.core.config.Settings.get_provider_models") as mock_get_models,
        ):
            mock_get_models.return_value = {"google": ["gemini-1.5-pro"], "openai": ["gpt-4-turbo"]}
            mock_redis.get.return_value = None

            result = await service.get_available_models()

            assert result.count == 2
            assert len(result.models) == 2

            # Verify cache was set
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_model_success(self, service, mock_redis, mock_db_session) -> None:
        """Test successful model selection."""
        session_id = "test_session"
        user_id = "test_user"
        model_id = "google:gemini-1.5-pro"
        ip_address = "127.0.0.1"

        with (
            patch("app.services.model_sync_service.get_redis_client", return_value=mock_redis),
            patch("app.core.database.db_manager.session") as mock_get_session,
            patch("app.services.model_sync_service.update_user_session_model", return_value=True),
            patch(
                "app.services.model_sync_service.ModelSyncService.validate_model_id",
                return_value=True,
            ),
        ):
            mock_get_session.return_value.__aenter__.return_value = mock_db_session

            request = ModelSelectionRequest(
                model_id=model_id,
                timestamp=datetime.utcnow().isoformat(),
            )
            result = await service.select_model(request, user_id, session_id, ip_address)

            assert result.success is True
            assert result.model_id == model_id

    @pytest.mark.asyncio
    async def test_select_model_rate_limited(self, service, mock_redis) -> None:
        """Test model selection with rate limiting."""
        session_id = "test_session"
        user_id = "test_user"
        model_id = "google:gemini-1.5-pro"
        ip_address = "127.0.0.1"

        with (
            patch("app.services.model_sync_service.get_redis_client", return_value=mock_redis),
            # Mock rate limit check to return False (limit exceeded)
            patch.object(service, "_check_rate_limit", return_value=False),
        ):
            request = ModelSelectionRequest(
                model_id=model_id,
                timestamp=datetime.utcnow().isoformat(),
            )
            result = await service.select_model(request, user_id, session_id, ip_address)

            assert result.success is False
            assert "rate limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_validate_model_success(self, service) -> None:
        """Test successful model validation."""
        model_id = "google:gemini-1.5-pro"

        # Mock initialize_models to populate cache
        with patch.object(service, "initialize_models"):
            service._models_cache = [MagicMock(id=model_id)]

            result = await service.validate_model_id(model_id)
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_model_not_found(self, service) -> None:
        """Test model validation with non-existent model."""
        model_id = "nonexistent-model"

        with patch.object(service, "initialize_models"):
            service._models_cache = [MagicMock(id="other-model")]

            result = await service.validate_model_id(model_id)
            assert result is False

    @pytest.mark.asyncio
    async def test_get_user_model_with_session(self, service, mock_db_session) -> None:
        """Test getting current model with valid session."""
        session_id = "test_session"
        model_id = "google:gemini-1.5-pro"

        with (
            patch("app.core.database.db_manager.session") as mock_get_session,
            patch("app.services.model_sync_service.get_user_session") as mock_get_user_session,
        ):
            mock_get_session.return_value.__aenter__.return_value = mock_db_session

            mock_session_obj = MagicMock()
            mock_session_obj.selected_model_id = model_id
            mock_get_user_session.return_value = mock_session_obj

            service._models_cache = [MagicMock(id=model_id)]

            result = await service.get_user_model(session_id)

            assert result is not None
            assert result.id == model_id

    @pytest.mark.asyncio
    async def test_get_user_model_no_session(self, service, mock_db_session) -> None:
        """Test getting current model with no session."""
        session_id = "nonexistent_session"

        with (
            patch("app.core.database.db_manager.session") as mock_get_session,
            patch("app.services.model_sync_service.get_user_session") as mock_get_user_session,
        ):
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_get_user_session.return_value = None

            result = await service.get_user_model(session_id)

            assert result is None


class TestWebSocketIntegration:
    """Test WebSocket functionality for real-time updates."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self) -> None:
        """Test WebSocket connection and message handling."""
        with (
            patch(
                "app.services.model_sync_service.get_redis_client",
                new_callable=AsyncMock,
            ),
            patch(
                "app.services.model_sync_service.model_sync_service.get_available_models",
            ) as mock_get_models,
        ):
            mock_get_models.return_value = MagicMock(count=5)

            client = TestClient(app)

            # The websocket endpoint is /api/v1/models/updates (as seen in model_sync.py)
            with client.websocket_connect("/api/v1/models/updates") as websocket:
                # Receive initial update
                response = websocket.receive_json()
                assert response["type"] == "model_availability"
                assert "Available models" in response["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
