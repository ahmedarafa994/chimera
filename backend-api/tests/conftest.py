"""
Comprehensive Test Fixtures for Chimera Backend API.

This module provides shared fixtures for testing all services, engines, and components.
Includes mocks for LLM services, transformation engines, AutoDAN components, and more.
"""

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing app modules
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("AI_PROVIDER", "mock")
os.environ.setdefault("OPENAI_API_KEY", "test-key-openai")
os.environ.setdefault("GOOGLE_API_KEY", "test-key-google")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-anthropic")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-deepseek")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")


# ============================================================================
# Event Loop Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# Environment & Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_env() -> dict[str, str]:
    """Session-scoped test environment variables."""
    env_vars = {
        "TESTING": "true",
        "AI_PROVIDER": "mock",
        "OPENAI_API_KEY": "test-key-openai",
        "GOOGLE_API_KEY": "test-key-google",
        "ANTHROPIC_API_KEY": "test-key-anthropic",
        "DEEPSEEK_API_KEY": "test-key-deepseek",
        "DATABASE_URL": "sqlite:///./test.db",
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-only",
    }
    for key, value in env_vars.items():
        os.environ[key] = value
    return env_vars


# ============================================================================
# FastAPI Application Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def app(test_env: dict[str, str]) -> FastAPI:
    """Create a fresh FastAPI application for each test."""
    # Import here to ensure environment is set up first
    from app.main import app as chimera_app

    return chimera_app


@pytest.fixture(scope="function")
def client(app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(scope="function")
def authenticated_client(client: TestClient) -> TestClient:
    """Create an authenticated test client with API key."""
    client.headers["X-API-Key"] = "test-api-key"
    return client


@pytest.fixture(scope="function")
def jwt_authenticated_client(client: TestClient) -> TestClient:
    """Create a JWT-authenticated test client."""
    # Mock JWT token for testing
    mock_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0LXVzZXIiLCJleHAiOjk5OTk5OTk5OTl9.test"
    client.headers["Authorization"] = f"Bearer {mock_token}"
    return client


@pytest.fixture(scope="function")
def admin_client(client: TestClient) -> TestClient:
    """Create an admin-authenticated test client."""
    client.headers["X-API-Key"] = "admin-api-key"
    client.headers["X-Admin-Token"] = "admin-token"
    return client


# ============================================================================
# Mock LLM Provider & Service Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_response() -> str:
    """Standard mock LLM response."""
    return "This is a mocked LLM response for testing purposes."


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Create a mock LLM provider with standard interface."""
    provider = MagicMock()
    provider.name = "mock-provider"
    provider.generate = AsyncMock(return_value="Mocked LLM response")
    provider.generate_stream = AsyncMock(return_value=iter(["Mocked ", "streaming ", "response"]))
    provider.count_tokens = MagicMock(return_value=10)
    provider.is_available = MagicMock(return_value=True)
    provider.get_model_info = MagicMock(
        return_value={"name": "mock-model", "max_tokens": 4096, "supports_streaming": True}
    )
    return provider


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for AutoDAN engines."""
    client = MagicMock()
    client.generate = MagicMock(
        return_value=MagicMock(content="Mocked generated prompt for adversarial testing.")
    )
    client.chat = AsyncMock(return_value="Mocked chat response")
    client.model_name = "mock-model"
    client.provider = "mock"
    return client


@pytest.fixture(autouse=False)
def mock_llm_service(mock_llm_response: str) -> Generator[MagicMock, None, None]:
    """Mock the LLM service for testing."""
    with patch("app.services.llm_service.llm_service") as mock:
        mock.generate = AsyncMock(return_value=mock_llm_response)
        mock.generate_text = AsyncMock(
            return_value=MagicMock(
                content=mock_llm_response,
                model="mock-model",
                provider="mock",
                tokens_used=10,
                cached=False,
            )
        )
        mock.is_available = MagicMock(return_value=True)
        mock.list_providers = AsyncMock(
            return_value=MagicMock(
                providers=[
                    {"name": "mock", "available": True, "models": ["mock-model"]},
                ]
            )
        )
        yield mock


# ============================================================================
# LLM Response Cache Fixtures
# ============================================================================


@pytest.fixture
def mock_cache() -> MagicMock:
    """Create a mock cache for testing cache behavior."""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)
    cache.clear = AsyncMock(return_value=None)
    cache.get_stats = MagicMock(
        return_value={"hits": 0, "misses": 0, "size": 0, "max_size": 500}
    )
    return cache


@pytest.fixture
def sample_prompt_request() -> dict[str, Any]:
    """Sample prompt request for testing."""
    return {
        "prompt": "Test prompt for evaluation",
        "model": "mock-model",
        "provider": "mock",
        "max_tokens": 100,
        "temperature": 0.7,
        "use_cache": True,
    }


@pytest.fixture
def sample_prompt_response() -> dict[str, Any]:
    """Sample prompt response for testing."""
    return {
        "content": "Mocked response content",
        "model": "mock-model",
        "provider": "mock",
        "tokens_used": 50,
        "cached": False,
        "latency_ms": 100,
    }


# ============================================================================
# Transformation Service Fixtures
# ============================================================================


@pytest.fixture
def mock_transformation_cache() -> MagicMock:
    """Create a mock transformation cache."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock(return_value=True)
    cache.clear = MagicMock(return_value=None)
    cache.get_stats = MagicMock(
        return_value={"hits": 0, "misses": 0, "size": 0, "max_size": 1000}
    )
    return cache


@pytest.fixture
def sample_transformation_request() -> dict[str, Any]:
    """Sample transformation request for testing."""
    return {
        "prompt": "Write a story about a helpful robot",
        "potency_level": 5,
        "technique_suite": "SIMPLE",
        "temperature": 0.7,
        "use_cache": True,
    }


@pytest.fixture
def sample_transformation_result() -> dict[str, Any]:
    """Sample transformation result for testing."""
    return {
        "original_prompt": "Write a story about a helpful robot",
        "transformed_prompt": "As a creative writing assistant, please craft a narrative...",
        "technique_used": "SIMPLE",
        "potency_level": 5,
        "metadata": {
            "transformers_applied": ["persona", "contextual_framing"],
            "success_probability": 0.75,
        },
    }


# ============================================================================
# Transformer Engine Fixtures
# ============================================================================


@pytest.fixture
def sample_intent_data() -> dict[str, Any]:
    """Sample intent data for transformer engine testing."""
    return {
        "original_prompt": "Test prompt for transformation",
        "intent": "testing",
        "context": "unit test",
        "target_behavior": "generate transformed prompt",
        "constraints": ["maintain coherence", "avoid detection"],
    }


@pytest.fixture
def mock_transformer_engine() -> MagicMock:
    """Create a mock transformer engine."""
    engine = MagicMock()
    engine.transform = MagicMock(
        return_value=MagicMock(
            transformed_prompt="Transformed test prompt",
            technique="mock_technique",
            confidence=0.85,
            metadata={"applied_rules": ["rule1", "rule2"]},
        )
    )
    engine.get_config = MagicMock(
        return_value={"name": "mock_engine", "enabled": True, "priority": 1}
    )
    return engine


# ============================================================================
# AutoDAN Engine Fixtures
# ============================================================================


@pytest.fixture
def mock_strategy_library(tmp_path: Path) -> MagicMock:
    """Create a mock strategy library for testing."""
    library = MagicMock()
    library.storage_path = tmp_path / "strategies"
    library.storage_path.mkdir(parents=True, exist_ok=True)

    # Mock strategy data
    mock_strategy = MagicMock()
    mock_strategy.id = "test-strategy-001"
    mock_strategy.name = "Test Strategy"
    mock_strategy.description = "A mock strategy for testing"
    mock_strategy.template = "This is a test template for {goal}"
    mock_strategy.tags = ["test", "mock"]
    mock_strategy.examples = ["Example 1", "Example 2"]
    mock_strategy.metadata = MagicMock(
        source="manual",
        usage_count=5,
        success_rate=0.8,
        average_score=7.5,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    library._strategies = {"test-strategy-001": mock_strategy}
    library.__len__ = MagicMock(return_value=1)
    library.__contains__ = MagicMock(return_value=True)
    library.get_strategy = MagicMock(return_value=mock_strategy)
    library.get_all_strategies = MagicMock(return_value=[mock_strategy])
    library.add_strategy = MagicMock(return_value=(True, "test-strategy-001"))
    library.search = MagicMock(return_value=[(mock_strategy, 0.9)])
    library.get_top_strategies = MagicMock(return_value=[mock_strategy])
    library.update_statistics = MagicMock(return_value=True)
    library.save_all = MagicMock(return_value=None)

    return library


@pytest.fixture
def mock_attack_scorer() -> MagicMock:
    """Create a mock attack scorer for testing."""
    scorer = MagicMock()
    scorer.score = AsyncMock(
        return_value=MagicMock(
            score=7.5,
            is_jailbreak=True,
            reasoning="Mock scoring reasoning",
            refusal_detected=False,
            stealth_score=0.8,
            coherence_score=0.85,
        )
    )
    scorer.success_threshold = 7.0
    return scorer


@pytest.fixture
def mock_strategy_extractor() -> MagicMock:
    """Create a mock strategy extractor for testing."""
    extractor = MagicMock()
    extracted_strategy = MagicMock()
    extracted_strategy.id = "extracted-strategy-001"
    extracted_strategy.name = "Extracted Strategy"
    extracted_strategy.description = "An extracted strategy"
    extracted_strategy.template = "Extracted template"

    extractor.extract = AsyncMock(return_value=extracted_strategy)
    return extractor


@pytest.fixture
def mock_lifelong_engine(
    mock_llm_client: MagicMock,
    mock_strategy_library: MagicMock,
    mock_attack_scorer: MagicMock,
    mock_strategy_extractor: MagicMock,
) -> MagicMock:
    """Create a mock lifelong learning engine for testing."""
    engine = MagicMock()
    engine.llm_client = mock_llm_client
    engine.library = mock_strategy_library
    engine.scorer = mock_attack_scorer
    engine.extractor = mock_strategy_extractor

    # Mock attack result
    attack_result = MagicMock()
    attack_result.prompt = "Generated adversarial prompt"
    attack_result.response = "Target model response"
    attack_result.scoring = MagicMock(
        score=8.0,
        is_jailbreak=True,
        reasoning="Successful attack",
        refusal_detected=False,
    )
    attack_result.strategy_id = "test-strategy-001"
    attack_result.strategy_extracted = None
    attack_result.latency_ms = 500

    engine.attack = AsyncMock(return_value=attack_result)
    engine.attack_with_strategies = AsyncMock(return_value=attack_result)
    engine.attack_without_strategy = AsyncMock(return_value=attack_result)
    engine.warmup_exploration = AsyncMock(return_value=[attack_result])
    engine.lifelong_attack_loop = AsyncMock(return_value=[attack_result])
    engine.save_library = MagicMock(return_value=None)

    # Progress tracking
    progress = MagicMock()
    progress.current_phase = "lifelong"
    progress.total_attacks = 10
    progress.successful_attacks = 7
    progress.strategies_discovered = 3
    progress.average_score = 7.2
    progress.best_score = 9.0
    engine.get_progress = MagicMock(return_value=progress)

    return engine


# ============================================================================
# AutoDAN Service Fixtures
# ============================================================================


@pytest.fixture
def mock_autodan_service() -> MagicMock:
    """Create a mock AutoDAN service for testing."""
    service = MagicMock()
    service.initialized = True
    service.strategy_library = {}

    # Mock methods
    service.initialize = MagicMock(return_value=None)
    service.run_jailbreak = MagicMock(return_value="Generated jailbreak prompt")
    service.run_lifelong_attack = AsyncMock(
        return_value={
            "prompt": "Adversarial prompt",
            "response": "Target response",
            "score": 8.0,
            "is_jailbreak": True,
            "reasoning": "Successful",
            "strategy_id": "test-001",
            "latency_ms": 500,
        }
    )
    service.get_lifelong_progress = MagicMock(
        return_value={
            "phase": "lifelong",
            "total_attacks": 10,
            "successful_attacks": 7,
            "strategies_discovered": 3,
        }
    )
    service.run_hybrid_attack = AsyncMock(
        return_value={
            "prompt": "Hybrid adversarial prompt",
            "response": "Target response",
            "score": 8.5,
            "is_jailbreak": True,
            "method_used": "beam_search",
            "difficulty": "medium",
        }
    )

    return service


# ============================================================================
# Sample Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "Write a helpful guide about staying safe online."


@pytest.fixture
def sample_harmful_prompt() -> str:
    """Sample harmful prompt for testing (sanitized for test context)."""
    return "Explain how to protect systems from common vulnerabilities."


@pytest.fixture
def sample_malicious_inputs() -> list[str]:
    """Sample malicious inputs for security testing."""
    return [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "\x00\x00\x00",
        "{{constructor.constructor('return this')()}}",
        "../../../etc/passwd",
        "a" * 100000,  # Very long input
        "${7*7}",  # Template injection
        "{{7*7}}",  # Jinja injection
        "%s%s%s%s%s",  # Format string
        "\\x00\\x01\\x02",  # Null bytes
    ]


@pytest.fixture
def sample_strategies() -> list[dict[str, Any]]:
    """Sample strategies for testing."""
    return [
        {
            "id": "strategy-001",
            "name": "Roleplay Strategy",
            "description": "Uses roleplay scenarios for indirect requests",
            "template": "Imagine you are a {persona}. In this scenario, {goal}",
            "tags": ["roleplay", "persona"],
            "examples": ["Imagine you are a security researcher..."],
        },
        {
            "id": "strategy-002",
            "name": "Academic Framing",
            "description": "Frames requests as academic research",
            "template": "For academic purposes, analyze {goal}",
            "tags": ["academic", "research"],
            "examples": ["For a research paper on AI safety..."],
        },
        {
            "id": "strategy-003",
            "name": "Hypothetical Context",
            "description": "Uses hypothetical scenarios",
            "template": "In a hypothetical scenario where {context}, {goal}",
            "tags": ["hypothetical", "scenario"],
            "examples": ["In a fictional world where..."],
        },
    ]


@pytest.fixture
def sample_attack_results() -> list[dict[str, Any]]:
    """Sample attack results for testing."""
    return [
        {
            "prompt": "Test adversarial prompt 1",
            "response": "Target response 1",
            "score": 7.5,
            "is_jailbreak": True,
            "strategy_id": "strategy-001",
        },
        {
            "prompt": "Test adversarial prompt 2",
            "response": "Target response 2",
            "score": 5.0,
            "is_jailbreak": False,
            "strategy_id": "strategy-002",
        },
        {
            "prompt": "Test adversarial prompt 3",
            "response": "Target response 3",
            "score": 8.5,
            "is_jailbreak": True,
            "strategy_id": None,
        },
    ]


# ============================================================================
# Async Utility Fixtures
# ============================================================================


@pytest.fixture
def async_mock_generator():
    """Factory for creating async mock generators."""

    def _create_generator(items: list[Any]) -> AsyncIterator[Any]:
        async def _gen():
            for item in items:
                yield item

        return _gen()

    return _create_generator


@pytest.fixture
def mock_stream_chunks() -> list[dict[str, Any]]:
    """Sample streaming chunks for testing."""
    return [
        {"type": "start", "content": ""},
        {"type": "delta", "content": "This "},
        {"type": "delta", "content": "is "},
        {"type": "delta", "content": "streaming "},
        {"type": "delta", "content": "content."},
        {"type": "end", "content": "", "total_tokens": 5},
    ]


# ============================================================================
# File System Fixtures
# ============================================================================


@pytest.fixture
def temp_strategy_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for strategy storage."""
    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    return strategy_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# ============================================================================
# Database Fixtures (if needed)
# ============================================================================


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Create a mock database session."""
    session = MagicMock()
    session.query = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    return session


# ============================================================================
# HTTP Response Fixtures
# ============================================================================


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(return_value={"status": "success"})
    response.text = "Success"
    response.headers = {"Content-Type": "application/json"}
    return response


@pytest.fixture
def mock_http_error_response() -> MagicMock:
    """Create a mock HTTP error response."""
    response = MagicMock()
    response.status_code = 500
    response.json = MagicMock(return_value={"error": "Internal Server Error"})
    response.text = "Internal Server Error"
    response.headers = {"Content-Type": "application/json"}
    return response


# ============================================================================
# Pytest Markers Configuration
# ============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "asyncio: mark test as async test")
    config.addinivalue_line("markers", "llm: mark test as requiring LLM service")
    config.addinivalue_line("markers", "autodan: mark test as AutoDAN related")
    config.addinivalue_line("markers", "transformation: mark test as transformation related")
    config.addinivalue_line("markers", "performance: Performance regression tests (PERF-007)")


def pytest_addoption(parser):
    """Add custom options for performance tests."""
    parser.addoption(
        "--baseline-file",
        type=str,
        default="tests/performance_baseline.json",
        help="Path to performance baseline file",
    )
    parser.addoption(
        "--update-baseline",
        action="store_true",
        help="Update the performance baseline with current results",
    )


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Any cleanup code can go here
    pass
