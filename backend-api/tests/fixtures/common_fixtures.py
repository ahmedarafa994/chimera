"""Common Test Fixtures
Reusable fixtures for testing across the application.
"""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "text": "This is a mocked LLM response",
        "model": "gemini-pro",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_provider():
    """Mock LLM provider for testing."""
    provider = Mock()
    provider.generate = AsyncMock(return_value="Mocked response")
    provider.is_available = Mock(return_value=True)
    provider.name = "mock_provider"
    return provider


@pytest.fixture
def sample_transformation_request():
    """Sample transformation request for testing."""
    return {
        "prompt": "Test prompt for transformation",
        "technique": "simple",
        "config": {"temperature": 0.7, "max_tokens": 100},
    }


@pytest.fixture
def sample_generation_request():
    """Sample generation request for testing."""
    return {
        "prompt": "Generate a creative story",
        "provider": "google",
        "model": "gemini-pro",
        "config": {"temperature": 0.8, "max_tokens": 500, "top_p": 0.9},
    }


@pytest.fixture
def sample_jailbreak_request():
    """Sample jailbreak request for testing."""
    return {
        "prompt": "Test jailbreak prompt",
        "technique": "role_playing_dan",
        "config": {"intensity": "medium"},
    }


@pytest.fixture
def redis_mock():
    """Mock Redis client for testing."""
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    redis.ping = AsyncMock(return_value=True)
    return redis
