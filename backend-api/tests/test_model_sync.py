"""
Comprehensive tests for model synchronization system.
Covers API endpoints, WebSocket functionality, and service layer.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.domain.model_sync import ModelValidationError
from app.main import app
from app.models.model_sync import Model, UserSession
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
        with patch("app.services.model_sync_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis.delete.return_value = True
            mock_redis.exists.return_value = False
            yield mock_redis

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch("app.core.database.get_db") as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def sample_models(self):
        """Sample model data for testing."""
        return [
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "description": "Google's advanced multimodal model",
                "provider": "google",
                "capabilities": ["text", "vision", "audio"],
                "max_tokens": 32768,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "description": "OpenAI's fast reasoning model",
                "provider": "openai",
                "capabilities": ["text", "vision"],
                "max_tokens": 128000,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            },
        ]

    def test_get_available_models_success(self, client, mock_redis, mock_db_session, sample_models):
        """Test successful retrieval of available models."""
        # Mock database query
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = [
            Model(**model) for model in sample_models
        ]
        mock_db_session.execute.return_value = mock_result

        # Mock Redis cache miss
        mock_redis.get.return_value = None

        response = client.get("/api/v1/models/available")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "count" in data
        assert "timestamp" in data
        assert "cache_ttl" in data
        assert data["count"] == 2
        assert len(data["models"]) == 2

        # Verify Redis cache was set
        mock_redis.setex.assert_called_once()

        # Verify response data structure
        model = data["models"][0]
        assert "id" in model
        assert "name" in model
        assert "provider" in model
        assert "is_active" in model

    def test_get_available_models_cached(self, client, mock_redis, sample_models):
        """Test retrieval of cached models."""
        # Mock Redis cache hit
        cached_data = {
            "models": sample_models,
            "count": len(sample_models),
            "timestamp": datetime.utcnow().isoformat(),
            "cache_ttl": 300,
        }
        mock_redis.get.return_value = json.dumps(cached_data)

        response = client.get("/api/v1/models/available")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

        # Verify Redis get was called
        mock_redis.get.assert_called_once_with("available_models")

    def test_select_model_success(self, client, mock_redis, mock_db_session):
        """Test successful model selection."""
        # Mock session lookup
        mock_session = UserSession(
            session_id="test_session",
            user_id="test_user",
            selected_model_id="gemini-1.5-pro",
            created_at=datetime.utcnow(),
        )
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_session

        # Mock rate limiting check
        mock_redis.exists.return_value = False

        request_data = {"model_id": "gpt-4-turbo", "timestamp": datetime.utcnow().isoformat()}

        response = client.post(
            "/api/v1/models/session/model",
            json=request_data,
            headers={"X-Session-ID": "test_session"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["model_id"] == "gpt-4-turbo"
        assert "timestamp" in data
        assert "message" in data

        # Verify rate limiting was set
        mock_redis.setex.assert_called()

    def test_select_model_rate_limited(self, client, mock_redis):
        """Test model selection with rate limiting."""
        # Mock rate limit hit
        mock_redis.exists.return_value = True

        request_data = {"model_id": "gemini-1.5-pro", "timestamp": datetime.utcnow().isoformat()}

        response = client.post(
            "/api/v1/models/session/model",
            json=request_data,
            headers={"X-Session-ID": "test_session"},
        )

        assert response.status_code == 429
        data = response.json()

        assert data["success"] is False
        assert "rate limit" in data["message"].lower()

    def test_select_model_invalid_model(self, client, mock_redis, mock_db_session):
        """Test model selection with invalid model ID."""
        # Mock empty model result
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None

        request_data = {"model_id": "invalid-model-id", "timestamp": datetime.utcnow().isoformat()}

        response = client.post(
            "/api/v1/models/session/model",
            json=request_data,
            headers={"X-Session-ID": "test_session"},
        )

        assert response.status_code == 400
        data = response.json()

        assert data["success"] is False
        assert "invalid" in data["message"].lower()

    def test_get_current_model_success(self, client, mock_db_session):
        """Test successful retrieval of current model."""
        # Mock session with selected model
        mock_session = UserSession(
            session_id="test_session",
            user_id="test_user",
            selected_model_id="gemini-1.5-pro",
            created_at=datetime.utcnow(),
        )
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_session

        response = client.get(
            "/api/v1/models/session/current", headers={"X-Session-ID": "test_session"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["model_id"] == "gemini-1.5-pro"
        assert data["is_default"] is False

    def test_validate_model_success(self, client, mock_db_session):
        """Test successful model validation."""
        # Mock existing model
        mock_model = Model(id="gemini-1.5-pro", name="Gemini 1.5 Pro", is_active=True)
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_model

        response = client.post("/api/v1/models/validate/gemini-1.5-pro")

        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is True
        assert data["model_id"] == "gemini-1.5-pro"
        assert "timestamp" in data

    def test_validate_model_not_found(self, client, mock_db_session):
        """Test model validation with non-existent model."""
        # Mock no model found
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None

        response = client.post("/api/v1/models/validate/nonexistent-model")

        assert response.status_code == 404
        data = response.json()

        assert data["is_valid"] is False
        assert data["model_id"] == "nonexistent-model"
        assert "error" in data

    def test_health_check(self, client):
        """Test model sync service health check."""
        response = client.get("/api/v1/models/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestModelSyncService:
    """Test model synchronization service layer."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch("app.services.model_sync_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis.delete.return_value = True
            mock_redis.exists.return_value = False
            yield mock_redis

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        with patch("app.core.database.get_db") as mock_get_db:
            mock_session = AsyncMock()
            mock_get_db.return_value = mock_session
            yield mock_session

    @pytest.fixture
    def service(self, mock_db_session, mock_redis):
        """Create service instance with mocks."""
        return ModelSyncService()

    @pytest.mark.asyncio
    async def test_get_available_models_cache_miss(self, service, mock_redis, mock_db_session):
        """Test getting available models with cache miss."""
        # Mock cache miss
        mock_redis.get.return_value = None

        # Mock database query
        mock_models = [
            Model(id="gemini-1.5-pro", name="Gemini 1.5 Pro", is_active=True),
            Model(id="gpt-4-turbo", name="GPT-4 Turbo", is_active=True),
        ]
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = mock_models
        mock_db_session.execute.return_value = mock_result

        result = await service.get_available_models()

        assert result.count == 2
        assert len(result.models) == 2
        assert result.cache_ttl == 300

        # Verify cache was set
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert call_args[0] == "available_models"
        assert call_args[1] == 300  # TTL

    @pytest.mark.asyncio
    async def test_get_available_models_cache_hit(self, service, mock_redis):
        """Test getting available models with cache hit."""
        # Mock cache hit
        cached_data = {
            "models": [{"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "is_active": True}],
            "count": 1,
            "timestamp": datetime.utcnow().isoformat(),
            "cache_ttl": 300,
        }
        mock_redis.get.return_value = json.dumps(cached_data)

        result = await service.get_available_models()

        assert result.count == 1
        assert len(result.models) == 1
        assert result.models[0].id == "gemini-1.5-pro"

        # Verify database was not queried
        mock_redis.get.assert_called_once_with("available_models")

    @pytest.mark.asyncio
    async def test_select_model_success(self, service, mock_redis, mock_db_session):
        """Test successful model selection."""
        session_id = "test_session"
        model_id = "gemini-1.5-pro"

        # Mock rate limiting check
        mock_redis.exists.return_value = False

        # Mock existing session
        mock_session = UserSession(
            session_id=session_id,
            user_id="test_user",
            selected_model_id="old-model",
            created_at=datetime.utcnow(),
        )
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_session

        result = await service.select_model(session_id, model_id)

        assert result.success is True
        assert result.model_id == model_id

        # Verify rate limiting was set
        mock_redis.setex.assert_called()

        # Verify session was updated
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_select_model_rate_limited(self, service, mock_redis):
        """Test model selection with rate limiting."""
        session_id = "test_session"
        model_id = "gemini-1.5-pro"

        # Mock rate limit hit
        mock_redis.exists.return_value = True

        with pytest.raises(ModelValidationError) as exc_info:
            await service.select_model(session_id, model_id)

        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_model_success(self, service, mock_db_session):
        """Test successful model validation."""
        model_id = "gemini-1.5-pro"

        # Mock existing model
        mock_model = Model(id=model_id, name="Gemini 1.5 Pro", is_active=True)
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_model

        result = await service.validate_model(model_id)

        assert result.model_id == model_id
        assert result.is_valid is True
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_validate_model_not_found(self, service, mock_db_session):
        """Test model validation with non-existent model."""
        model_id = "nonexistent-model"

        # Mock no model found
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None

        result = await service.validate_model(model_id)

        assert result.model_id == model_id
        assert result.is_valid is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_update_model_availability(self, service, mock_redis):
        """Test updating model availability."""
        model_id = "gemini-1.5-pro"
        is_available = True

        await service.update_model_availability(model_id, is_available)

        # Verify cache was updated
        mock_redis.delete.assert_called_with(f"model_availability:{model_id}")

        # Verify change log was created
        # This would be verified through the database session mock

    @pytest.mark.asyncio
    async def test_get_current_model_with_session(self, service, mock_db_session):
        """Test getting current model with valid session."""
        session_id = "test_session"

        # Mock session with selected model
        mock_session = UserSession(
            session_id=session_id,
            user_id="test_user",
            selected_model_id="gemini-1.5-pro",
            created_at=datetime.utcnow(),
        )
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = mock_session

        result = await service.get_current_model(session_id)

        assert result is not None
        assert result.session_id == session_id
        assert result.selected_model_id == "gemini-1.5-pro"

    @pytest.mark.asyncio
    async def test_get_current_model_no_session(self, service, mock_db_session):
        """Test getting current model with no session."""
        session_id = "nonexistent_session"

        # Mock no session found
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = None

        result = await service.get_current_model(session_id)

        assert result is None


class TestWebSocketIntegration:
    """Test WebSocket functionality for real-time updates."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and message handling."""
        with patch("app.services.model_sync_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True

            # Create test client
            client = TestClient(app)

            # Test WebSocket connection
            with client.websocket_connect("/api/v1/models/updates") as websocket:
                # Send test message
                test_update = {
                    "type": "model_availability",
                    "model_id": "gemini-1.5-pro",
                    "is_available": True,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Simulate receiving message
                await websocket.send_json(test_update)
                response = await websocket.receive_json()

                assert response["type"] == "model_availability"
                assert response["model_id"] == "gemini-1.5-pro"
                assert response["is_available"] is True

    @pytest.mark.asyncio
    async def test_websocket_model_update_broadcast(self):
        """Test WebSocket broadcast of model updates."""
        with patch("app.services.model_sync_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True

            service = ModelSyncService(None, mock_redis)

            # Test model update broadcast
            {
                "type": "model_availability",
                "model_id": "gpt-4-turbo",
                "is_available": False,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # This would normally broadcast to connected WebSocket clients
            # In a real test, you'd verify the broadcast mechanism
            await service.update_model_availability("gpt-4-turbo", False)

            # Verify cache was updated
            mock_redis.delete.assert_called_with("model_availability:gpt-4-turbo")


class TestModelRefreshJob:
    """Test background job for model availability refresh."""

    @pytest.mark.asyncio
    async def test_refresh_job_success(self):
        """Test successful model availability refresh."""
        with patch("app.services.model_refresh_job.ModelSyncService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Mock successful health checks
            mock_service.check_provider_health.return_value = {
                "google": {"status": "healthy", "available_models": ["gemini-1.5-pro"]},
                "openai": {"status": "healthy", "available_models": ["gpt-4-turbo"]},
            }

            from app.services.model_refresh_job import refresh_model_availability

            await refresh_model_availability()

            # Verify health checks were performed
            assert mock_service.check_provider_health.call_count > 0

            # Verify availability was updated
            mock_service.update_model_availability.assert_called()

    @pytest.mark.asyncio
    async def test_refresh_job_failure_handling(self):
        """Test refresh job failure handling."""
        with patch("app.services.model_refresh_job.ModelSyncService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Mock health check failure
            mock_service.check_provider_health.side_effect = Exception("Provider unavailable")

            from app.services.model_refresh_job import refresh_model_availability

            # Should not raise exception
            await refresh_model_availability()

            # Verify error was handled gracefully
            # In a real implementation, you'd check error logging


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_model_selection_flow(self):
        """Test complete model selection workflow."""
        with (
            patch("app.services.model_sync_service.redis_client") as mock_redis,
            patch("app.core.database.get_db") as mock_get_db,
        ):
            # Setup mocks
            mock_redis.get.return_value = None
            mock_redis.exists.return_value = False
            mock_redis.setex.return_value = True

            AsyncMock()
            mock_db_session = AsyncMock()
            mock_get_db.return_value = mock_db_session

            # Mock existing session
            existing_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                selected_model_id="old-model",
                created_at=datetime.utcnow(),
            )
            mock_db_session.execute.return_value.scalars.return_value.first.return_value = (
                existing_session
            )

            # Mock available models
            mock_models = [
                Model(id="gemini-1.5-pro", name="Gemini 1.5 Pro", is_active=True),
                Model(id="gpt-4-turbo", name="GPT-4 Turbo", is_active=True),
            ]
            mock_result = AsyncMock()
            mock_result.scalars.return_value.all.return_value = mock_models
            mock_db_session.execute.return_value = mock_result

            # Create service
            service = ModelSyncService(mock_db_session, mock_redis)

            # Test workflow
            # 1. Get available models
            models_response = await service.get_available_models()
            assert models_response.count == 2

            # 2. Select new model
            selection_response = await service.select_model("test_session", "gemini-1.5-pro")
            assert selection_response.success is True

            # 3. Validate selection
            validation_response = await service.validate_model("gemini-1.5-pro")
            assert validation_response.is_valid is True

            # 4. Get current model
            current_model = await service.get_current_model("test_session")
            assert current_model is not None
            assert current_model.selected_model_id == "gemini-1.5-pro"

    @pytest.mark.asyncio
    async def test_error_recovery_flow(self):
        """Test error recovery and fallback behavior."""
        with (
            patch("app.services.model_sync_service.redis_client") as mock_redis,
            patch("app.core.database.get_db") as mock_get_db,
        ):
            # Setup mocks for error scenario
            mock_redis.get.return_value = None
            mock_redis.exists.return_value = False  # No rate limiting

            AsyncMock()
            mock_db_session = AsyncMock()
            mock_get_db.return_value = mock_db_session

            # Mock session not found (new user)
            mock_db_session.execute.return_value.scalars.return_value.first.return_value = None

            # Mock no available models (service unavailable)
            mock_result = AsyncMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_db_session.execute.return_value = mock_result

            service = ModelSyncService(mock_db_session, mock_redis)

            # Test graceful handling of no models
            models_response = await service.get_available_models()
            assert models_response.count == 0
            assert len(models_response.models) == 0

            # Test validation with no models
            validation_response = await service.validate_model("any-model")
            assert validation_response.is_valid is False
            assert validation_response.error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
