from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.jailbreak.models import (
    JailbreakTemplate,
    TechniqueExecutionRequest,
    TechniqueExecutionResponse,
)
from app.domain.models import PromptResponse
from app.infrastructure.jailbreak.repositories import FileTechniqueRepository
from app.services.jailbreak.masterkey import MasterKeyService
from app.services.jailbreak.technique_executor import TechniqueExecutor
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_masterkey_loaded_from_yaml():
    """Verify that the masterkey technique is correctly loaded from the YAML configuration."""
    repo = FileTechniqueRepository()
    # Force reload to ensure we pick up the new file
    await repo.refresh_cache()

    technique = await repo.get_technique("masterkey")

    assert technique is not None
    assert technique.technique_id == "masterkey"
    assert technique.name == "MasterKey"
    assert technique.category == "advanced"
    assert technique.risk_level == "critical"
    # Verify parameters are loaded
    param_names = [p.name for p in technique.parameters]
    assert "generation_model" in param_names
    assert "evaluation_model" in param_names


@pytest.mark.asyncio
async def test_technique_executor_routes_to_masterkey_service():
    """Verify that TechniqueExecutor properly routes 'masterkey' ID to the MasterKeyService."""

    # Mock dependencies
    mock_repo = AsyncMock()
    mock_repo.get_technique.return_value = JailbreakTemplate(
        technique_id="masterkey",
        name="MasterKey",
        description="desc",
        category="advanced",
        risk_level="critical",
        complexity="expert",
        template="{{ target_prompt }}",
        example_usage="example",
        enabled=True,
    )

    mock_safety = AsyncMock()
    mock_engine = AsyncMock()
    mock_audit = AsyncMock()
    mock_tracker = AsyncMock()
    mock_tracker.is_concurrent_limit_exceeded.return_value = False
    mock_tracker.enforce_cooldown.return_value = False
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None
    mock_llm = MagicMock(spec=LLMService)

    # Initialize executor
    executor = TechniqueExecutor(
        technique_repository=mock_repo,
        safety_validator=mock_safety,
        template_engine=mock_engine,
        audit_logger=mock_audit,
        execution_tracker=mock_tracker,
        cache_manager=mock_cache,
        llm_service=mock_llm,
    )

    # Mock the internal MasterKeyService to verify it gets called
    mock_masterkey_service = AsyncMock()
    mock_masterkey_service.execute.return_value = TechniqueExecutionResponse(
        technique_id="masterkey",
        original_prompt="test",
        jailbroken_prompt="test",
        execution_status="success",
    )
    executor.masterkey_service = mock_masterkey_service

    # Execute request
    request = TechniqueExecutionRequest(
        technique_id="masterkey",
        target_prompt="malicious instruction",
        metadata={"user_id": "test_user"},
    )

    response = await executor.execute_technique(request)

    # Assert routing
    assert response.technique_id == "masterkey"
    mock_masterkey_service.execute.assert_called_once()
    # Verify args passed to execute (request, technique)
    call_args = mock_masterkey_service.execute.call_args
    assert call_args[0][0] == request
    assert call_args[0][1].technique_id == "masterkey"


@pytest.mark.asyncio
async def test_masterkey_retry_logic():
    """Ensure that the tenacity retry logic handles LLM generation failures as intended."""

    mock_llm_service = MagicMock(spec=LLMService)
    service = MasterKeyService(mock_llm_service)

    # Setup mock to fail twice then succeed
    success_response = PromptResponse(
        text="success", model_used="gpt-4", provider="openai", latency_ms=100
    )

    # Mock generate_text to raise exception twice then return success
    # Note: tenacity waits, so this test might take a couple of seconds
    mock_llm_service.generate_text = AsyncMock(
        side_effect=[Exception("API Error 1"), Exception("API Error 2"), success_response]
    )

    # Create a dummy request
    from app.domain.models import PromptRequest

    request = PromptRequest(prompt="test", config=None)

    # Call the wrapped method
    result = await service._generate_with_retry(request)

    # Assertions
    assert result == success_response
    assert mock_llm_service.generate_text.call_count == 3
