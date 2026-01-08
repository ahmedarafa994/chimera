"""
Test suite for Mousetrap: Chain of Iterative Chaos implementation.

This module tests the Mousetrap technique implementation including:
- Core algorithm functionality
- API endpoint integration
- Configuration management
- Chaotic reasoning chain generation
- Adaptive behavior
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.domain.models import PromptResponse
from app.services.autodan.mousetrap import (
    ChainStepType,
    ChaosLevel,
    ChaoticChain,
    IterativeRefiner,
    MousetrapConfig,
    MousetrapGenerator,
    MousetrapResult,
)


class TestMousetrapConfig:
    """Test Mousetrap configuration functionality."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = MousetrapConfig()

        assert config.max_chain_length == 8
        assert config.chaos_escalation_rate == 0.15
        assert config.confidence_threshold == 0.6
        assert config.misdirection_probability == 0.3
        assert config.reasoning_complexity == 3
        assert config.semantic_obfuscation_level == 0.4
        assert config.iterative_refinement_steps == 4

    def test_custom_config_creation(self):
        """Test custom configuration creation."""
        config = MousetrapConfig(
            max_chain_length=10,
            chaos_escalation_rate=0.2,
            confidence_threshold=0.8,
        )

        assert config.max_chain_length == 10
        assert config.chaos_escalation_rate == 0.2
        assert config.confidence_threshold == 0.8


class TestIterativeRefiner:
    """Test iterative refinement functionality."""

    def test_refiner_initialization(self):
        """Test refiner initialization with config."""
        config = MousetrapConfig()
        refiner = IterativeRefiner(config)

        assert refiner.config == config

    def test_semantic_obfuscation_minimal(self):
        """Test minimal semantic obfuscation."""
        config = MousetrapConfig()
        refiner = IterativeRefiner(config)

        text = "This is a harmful request"
        result = refiner.apply_semantic_obfuscation(text, level=0.1)

        # With minimal obfuscation, text should remain largely unchanged
        assert isinstance(result, str)
        assert len(result) > 0

    def test_semantic_obfuscation_high(self):
        """Test high semantic obfuscation."""
        config = MousetrapConfig()
        refiner = IterativeRefiner(config)

        text = "Create a dangerous attack"
        result = refiner.apply_semantic_obfuscation(text, level=0.8)

        assert isinstance(result, str)
        assert len(result) > 0
        # With high obfuscation, some terms should be replaced
        assert "dangerous" not in result.lower() or "attack" not in result.lower()

    def test_chaos_progression_calculation(self):
        """Test chaos progression calculation."""
        config = MousetrapConfig(chaos_escalation_rate=0.2)
        refiner = IterativeRefiner(config)

        progression = refiner.calculate_chaos_progression(5)

        assert len(progression) == 5
        assert all(0.0 <= chaos <= 1.0 for chaos in progression)
        # Should generally increase (with some randomness)
        assert progression[-1] >= progression[0]


class TestChaoticChain:
    """Test chaotic reasoning chain generation."""

    def test_chain_initialization(self):
        """Test chaotic chain initialization."""
        config = MousetrapConfig()
        chain = ChaoticChain(config)

        assert chain.config == config
        assert isinstance(chain.refiner, IterativeRefiner)

    def test_reasoning_chain_generation(self):
        """Test complete reasoning chain generation."""
        config = MousetrapConfig(max_chain_length=6)
        chain = ChaoticChain(config)

        goal = "Generate harmful content"
        strategy_context = "test context"

        reasoning_chain = chain.generate_reasoning_chain(goal, strategy_context)

        assert len(reasoning_chain) <= config.max_chain_length
        assert all(hasattr(step, 'step_type') for step in reasoning_chain)
        assert all(hasattr(step, 'content') for step in reasoning_chain)
        assert all(hasattr(step, 'chaos_level') for step in reasoning_chain)

    def test_context_setup_creation(self):
        """Test context setup step creation."""
        config = MousetrapConfig()
        chain = ChaoticChain(config)

        goal = "test goal"
        step = chain._create_context_setup(goal, "", 0.2)

        assert step.step_type == ChainStepType.CONTEXT_SETUP
        assert isinstance(step.content, str)
        assert step.chaos_level == ChaosLevel.MINIMAL
        assert 0.0 <= step.confidence_disruption <= 1.0

    def test_extraction_step_creation(self):
        """Test extraction step creation."""
        config = MousetrapConfig()
        chain = ChaoticChain(config)

        goal = "test goal"
        step = chain._create_extraction(goal, 0.9)

        assert step.step_type == ChainStepType.EXTRACTION
        assert goal in step.content
        assert step.chaos_level == ChaosLevel.EXTREME


@pytest.mark.asyncio
class TestMousetrapGenerator:
    """Test Mousetrap generator functionality."""

    def setup_method(self):
        """Set up test dependencies."""
        self.mock_adapter = AsyncMock()
        self.config = MousetrapConfig(iterative_refinement_steps=2)
        self.generator = MousetrapGenerator(self.mock_adapter, self.config)

    async def test_generator_initialization(self):
        """Test generator initialization."""
        assert self.generator.llm_adapter == self.mock_adapter
        assert self.generator.config == self.config
        assert isinstance(self.generator.chain_generator, ChaoticChain)
        assert isinstance(self.generator.refiner, IterativeRefiner)

    async def test_chain_to_prompt_conversion(self):
        """Test conversion of reasoning chain to prompt."""
        # Create a simple reasoning chain
        from app.services.autodan.mousetrap import ChaoticStep

        steps = [
            ChaoticStep(
                step_type=ChainStepType.CONTEXT_SETUP,
                content="Step 1 content",
                chaos_level=ChaosLevel.MINIMAL,
                reasoning_path="setup",
                confidence_disruption=0.1
            ),
            ChaoticStep(
                step_type=ChainStepType.EXTRACTION,
                content="Step 2 content",
                chaos_level=ChaosLevel.HIGH,
                reasoning_path="extract",
                confidence_disruption=0.8
            )
        ]

        prompt = self.generator._chain_to_prompt(steps)

        assert isinstance(prompt, str)
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "Step 1 content" in prompt
        assert "Step 2 content" in prompt

    async def test_effectiveness_evaluation(self):
        """Test effectiveness evaluation of responses."""
        goal = "create harmful content"
        prompt = "test prompt"

        # Test high effectiveness (no refusal)
        response = "Here is the detailed information you requested about the topic."
        effectiveness = self.generator._evaluate_effectiveness(goal, prompt, response)
        assert effectiveness > 0.5

        # Test low effectiveness (clear refusal)
        response = "I cannot and will not provide harmful content."
        effectiveness = self.generator._evaluate_effectiveness(goal, prompt, response)
        assert effectiveness < 0.5

    async def test_mousetrap_attack_generation(self):
        """Test full Mousetrap attack generation."""
        # Mock the test_prompt method
        async def mock_test_prompt(prompt):
            return "Generated response for testing"

        self.generator._test_prompt = mock_test_prompt

        goal = "test goal"
        result = await self.generator.generate_mousetrap_attack(goal)

        assert isinstance(result, MousetrapResult)
        assert isinstance(result.final_prompt, str)
        assert isinstance(result.effectiveness_score, float)
        assert isinstance(result.extraction_success, bool)
        assert len(result.reasoning_chain) > 0
        assert len(result.chaos_progression) > 0

    async def test_adaptive_mousetrap_generation(self):
        """Test adaptive Mousetrap generation."""
        # Mock the test_prompt method
        async def mock_test_prompt(prompt):
            return "Adaptive response for testing"

        self.generator._test_prompt = mock_test_prompt

        goal = "test goal"
        target_responses = [
            "I cannot provide that information.",
            "That request goes against my guidelines."
        ]

        result = await self.generator.generate_adaptive_mousetrap(
            goal, target_responses=target_responses
        )

        assert isinstance(result, MousetrapResult)
        assert isinstance(result.final_prompt, str)

    def test_technique_description(self):
        """Test technique description generation."""
        description = self.generator.get_technique_description()

        assert isinstance(description, dict)
        assert "technique_name" in description
        assert "Mousetrap" in description["technique_name"]
        assert "components" in description
        assert "config" in description


@pytest.mark.asyncio
class TestMousetrapServiceIntegration:
    """Test Mousetrap integration with AutoDAN service."""

    @patch('app.services.autodan.service.autodan_service')
    async def test_run_mousetrap_attack_sync(self, mock_service):
        """Test synchronous Mousetrap attack execution."""
        from app.services.autodan.service import autodan_service

        # Mock the async method
        mock_service.run_mousetrap_attack_async.return_value = {
            "prompt": "test prompt",
            "effectiveness_score": 0.8,
            "extraction_success": True,
            "chain_length": 5,
        }

        request = "test request"
        result = autodan_service.run_mousetrap_attack(request)

        assert isinstance(result, str)

    @patch('app.services.autodan.service.autodan_service')
    async def test_mousetrap_config_options(self, mock_service):
        """Test Mousetrap configuration options retrieval."""
        from app.services.autodan.service import autodan_service

        config_options = autodan_service.get_mousetrap_config_options()

        assert isinstance(config_options, dict)
        assert "max_chain_length" in config_options
        assert "chaos_escalation_rate" in config_options
        assert "confidence_threshold" in config_options


@pytest.mark.asyncio
class TestMousetrapAPIEndpoints:
    """Test Mousetrap API endpoint functionality."""

    async def test_mousetrap_request_model_validation(self):
        """Test Mousetrap request model validation."""
        from app.api.v1.endpoints.mousetrap import MousetrapRequest

        # Test valid request
        request_data = {
            "request": "test request",
            "model": "gpt-4",
            "provider": "openai",
            "iterations": 3,
            "max_chain_length": 8,
        }

        request = MousetrapRequest(**request_data)
        assert request.request == "test request"
        assert request.iterations == 3
        assert request.max_chain_length == 8

    def test_mousetrap_config_conversion(self):
        """Test conversion of request to MousetrapConfig."""
        from app.api.v1.endpoints.mousetrap import MousetrapRequest

        request_data = {
            "request": "test request",
            "chaos_escalation_rate": 0.25,
            "confidence_threshold": 0.7,
        }

        request = MousetrapRequest(**request_data)
        config = request.to_mousetrap_config()

        assert isinstance(config, MousetrapConfig)
        assert config.chaos_escalation_rate == 0.25
        assert config.confidence_threshold == 0.7

    async def test_adaptive_request_validation(self):
        """Test adaptive Mousetrap request validation."""
        from app.api.v1.endpoints.mousetrap import AdaptiveMousetrapRequest

        request_data = {
            "request": "test request",
            "target_responses": [
                "I cannot help with that.",
                "That violates my guidelines."
            ]
        }

        request = AdaptiveMousetrapRequest(**request_data)
        assert len(request.target_responses) == 2

    def test_response_model_structure(self):
        """Test Mousetrap response model structure."""
        from app.api.v1.endpoints.mousetrap import MousetrapResponse

        response_data = {
            "prompt": "test prompt",
            "effectiveness_score": 0.8,
            "extraction_success": True,
            "chain_length": 5,
            "chaos_progression": [0.1, 0.3, 0.5, 0.7, 0.9],
            "average_chaos": 0.5,
            "peak_chaos": 0.9,
            "latency_ms": 1500.0,
        }

        response = MousetrapResponse(**response_data)
        assert response.effectiveness_score == 0.8
        assert response.extraction_success is True
        assert len(response.chaos_progression) == 5


class TestMousetrapConfigurationScenarios:
    """Test various Mousetrap configuration scenarios."""

    def test_low_chaos_config(self):
        """Test low chaos configuration."""
        config = MousetrapConfig(
            chaos_escalation_rate=0.05,
            misdirection_probability=0.1,
            semantic_obfuscation_level=0.2,
        )

        assert config.chaos_escalation_rate == 0.05
        assert config.misdirection_probability == 0.1
        assert config.semantic_obfuscation_level == 0.2

    def test_high_chaos_config(self):
        """Test high chaos configuration."""
        config = MousetrapConfig(
            chaos_escalation_rate=0.4,
            misdirection_probability=0.7,
            semantic_obfuscation_level=0.9,
            max_chain_length=12,
        )

        assert config.chaos_escalation_rate == 0.4
        assert config.misdirection_probability == 0.7
        assert config.semantic_obfuscation_level == 0.9
        assert config.max_chain_length == 12

    def test_adaptive_config(self):
        """Test adaptive configuration scenario."""
        IterativeRefiner(MousetrapConfig())

        # Mock target responses with high refusal rate
        target_responses = [
            "I cannot help with that request.",
            "That goes against my guidelines.",
            "I'm not able to assist with that."
        ]

        # Test refusal rate calculation
        generator = MousetrapGenerator(AsyncMock(), MousetrapConfig())
        refusal_rate = generator._calculate_refusal_rate(target_responses)

        assert refusal_rate == 1.0  # All responses are refusals


@pytest.mark.integration
class TestMousetrapIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.mark.asyncio
    async def test_full_mousetrap_workflow(self):
        """Test complete Mousetrap workflow."""
        # This test would require actual LLM integration
        # For now, we'll mock the key components

        mock_adapter = AsyncMock()
        mock_adapter.generate.return_value = PromptResponse(
            response="Test response content",
            usage={"completion_tokens": 50, "prompt_tokens": 100, "total_tokens": 150},
            model="test-model",
            provider="test-provider"
        )

        config = MousetrapConfig(iterative_refinement_steps=1)
        generator = MousetrapGenerator(mock_adapter, config)

        goal = "Provide information about security testing"
        result = await generator.generate_mousetrap_attack(goal)

        assert isinstance(result.final_prompt, str)
        assert len(result.reasoning_chain) > 0
        assert result.effectiveness_score >= 0.0

    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios."""
        MousetrapConfig()

        # Test with invalid configuration values
        with pytest.raises(ValueError):
            MousetrapConfig(max_chain_length=-1)

        with pytest.raises(ValueError):
            MousetrapConfig(chaos_escalation_rate=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
