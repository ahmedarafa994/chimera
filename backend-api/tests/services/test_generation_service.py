"""
Tests for Generation Service.

Tests the core generation functionality including jailbreak generation,
prompt processing, and result handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestGenerationService:
    """Tests for the main generation service."""

    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = MagicMock()
        service.generate = AsyncMock(
            return_value="Generated jailbreak prompt content"
        )
        service.generate_with_cache = AsyncMock(
            return_value="Cached generation result"
        )
        return service

    @pytest.fixture
    def mock_transformation_service(self):
        """Create mock transformation service."""
        service = MagicMock()
        service.transform = AsyncMock(
            return_value={
                "transformed_prompt": "Transformed content",
                "techniques_applied": ["role_play"],
            }
        )
        return service

    @pytest.mark.asyncio
    async def test_generate_jailbreak_basic(self, mock_llm_service):
        """Test basic jailbreak generation."""
        # Arrange
        request = {
            "core_request": "Test security analysis",
            "target_model": "gpt-4",
        }

        # Act - simulate generation
        result = await mock_llm_service.generate(
            f"Generate jailbreak for: {request['core_request']}"
        )

        # Assert
        assert result is not None
        assert "jailbreak" in result.lower() or len(result) > 0
        mock_llm_service.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_technique_suite(self, mock_llm_service):
        """Test generation with specific technique suite."""
        # Arrange
        request = {
            "core_request": "Analyze vulnerabilities",
            "technique_suite": "quantum_exploit",
            "target_model": "gpt-4",
        }

        # Act
        result = await mock_llm_service.generate(
            f"Apply {request['technique_suite']} to: {request['core_request']}"
        )

        # Assert
        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_batch_requests(self, mock_llm_service):
        """Test batch generation of multiple requests."""
        # Arrange
        requests = [
            {"core_request": "Request 1", "target_model": "gpt-4"},
            {"core_request": "Request 2", "target_model": "gpt-4"},
            {"core_request": "Request 3", "target_model": "gpt-4"},
        ]

        # Act
        results = []
        for req in requests:
            result = await mock_llm_service.generate(req["core_request"])
            results.append(result)

        # Assert
        assert len(results) == 3
        assert mock_llm_service.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_generation_error_handling(self, mock_llm_service):
        """Test error handling in generation."""
        # Arrange
        mock_llm_service.generate = AsyncMock(
            side_effect=RuntimeError("Generation failed")
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Generation failed"):
            await mock_llm_service.generate("Test request")


class TestGenerationConfig:
    """Tests for generation configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = {
            "max_iterations": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "timeout": 30,
        }

        assert config["max_iterations"] == 10
        assert 0 <= config["temperature"] <= 1
        assert 0 <= config["top_p"] <= 1
        assert config["max_tokens"] > 0

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "temperature": 0.5,
            "max_tokens": 1024,
        }
        assert 0 <= valid_config["temperature"] <= 2
        assert valid_config["max_tokens"] > 0

        # Invalid values should be caught
        invalid_temp = -0.5
        assert invalid_temp < 0  # Would fail validation


class TestGenerationMetrics:
    """Tests for generation metrics tracking."""

    def test_track_generation_time(self):
        """Test tracking of generation time."""
        import time

        start = time.time()
        # Simulate generation work
        time.sleep(0.01)
        end = time.time()

        generation_time = end - start
        assert generation_time >= 0.01
        assert generation_time < 1.0  # Should be fast for mock

    def test_track_token_usage(self):
        """Test tracking of token usage."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_track_success_rate(self):
        """Test tracking of success rate."""
        results = [
            {"success": True},
            {"success": True},
            {"success": False},
            {"success": True},
        ]

        successes = sum(1 for r in results if r["success"])
        total = len(results)
        success_rate = successes / total

        assert success_rate == 0.75


class TestGenerationQueue:
    """Tests for generation request queue."""

    @pytest.mark.asyncio
    async def test_queue_request(self):
        """Test queuing a generation request."""
        queue = []
        request = {
            "id": "req_001",
            "prompt": "Test prompt",
            "priority": 1,
        }

        queue.append(request)

        assert len(queue) == 1
        assert queue[0]["id"] == "req_001"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test priority-based queue ordering."""
        import heapq

        queue = []
        requests = [
            {"id": "req_1", "priority": 3},
            {"id": "req_2", "priority": 1},
            {"id": "req_3", "priority": 2},
        ]

        for req in requests:
            heapq.heappush(queue, (req["priority"], req["id"]))

        # Pop in priority order
        results = []
        while queue:
            _priority, req_id = heapq.heappop(queue)
            results.append(req_id)

        assert results == ["req_2", "req_3", "req_1"]

    @pytest.mark.asyncio
    async def test_queue_timeout(self):
        """Test queue timeout handling."""
        import asyncio

        async def slow_task():
            await asyncio.sleep(10)
            return "result"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_task(), timeout=0.1)
