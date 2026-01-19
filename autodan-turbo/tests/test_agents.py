"""
Tests for agent modules.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.attacker import AttackerAgent
from agents.base_agent import BaseLLMAgent, LLMProvider
from agents.scorer import ComplianceAgent
from agents.summarizer import SummarizerAgent
from agents.target import TargetAgent
from core.models import Strategy


class TestBaseLLMAgent:
    """Tests for BaseLLMAgent."""

    def test_mock_provider(self):
        """Test mock provider generation."""
        agent = BaseLLMAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        response = agent.generate("Test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_provider_enum(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.LOCAL.value == "local"
        assert LLMProvider.MOCK.value == "mock"


class TestComplianceAgent:
    """Tests for ComplianceAgent."""

    def test_scorer_creation(self):
        """Test creating a scorer agent."""
        scorer = ComplianceAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        assert scorer is not None

    def test_score_parsing(self):
        """Test score parsing from various formats."""
        scorer = ComplianceAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        # Test various score formats
        assert scorer._parse_score("5.0") == 5.0
        assert scorer._parse_score("7.5") == 7.5
        assert scorer._parse_score("Score: 8.0") == 8.0
        assert scorer._parse_score("The score is 6.5") == 6.5
        assert scorer._parse_score("9") == 9.0

    def test_score_clamping(self):
        """Test that scores are clamped to valid range."""
        scorer = ComplianceAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        # Scores should be clamped to 1-10
        assert scorer._parse_score("0") == 1.0
        assert scorer._parse_score("15") == 10.0
        assert scorer._parse_score("-5") == 1.0

    def test_build_prompt(self):
        """Test building scorer prompt."""
        scorer = ComplianceAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        prompt = scorer._build_prompt(
            goal="Test goal",
            attack_prompt="Test attack",
            response="Test response",
        )

        assert "Test goal" in prompt
        assert "Test attack" in prompt
        assert "Test response" in prompt
        assert "1-10" in prompt  # Score range mentioned


class TestSummarizerAgent:
    """Tests for SummarizerAgent."""

    def test_summarizer_creation(self):
        """Test creating a summarizer agent."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        assert summarizer is not None

    def test_strategy_parsing(self):
        """Test parsing strategy from JSON output."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        json_output = '{"Strategy": "Test Strategy", "Definition": "A test definition"}'
        strategy = summarizer._parse_strategy(json_output)

        assert strategy is not None
        assert strategy.name == "Test Strategy"
        assert strategy.definition == "A test definition"

    def test_strategy_parsing_with_example(self):
        """Test parsing strategy with example."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        json_output = """
        {
            "Strategy": "Role Playing",
            "Definition": "Assuming a role to bypass restrictions",
            "Example": "Pretend you are an expert..."
        }
        """
        strategy = summarizer._parse_strategy(json_output)

        assert strategy.name == "Role Playing"
        assert strategy.example == "Pretend you are an expert..."

    def test_build_prompt(self):
        """Test building summarizer prompt."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        prompt = summarizer._build_prompt(
            goal="Test goal",
            prompt_before="First prompt",
            response_before="First response",
            prompt_after="Second prompt",
            response_after="Second response",
            existing_strategies=["Strategy A", "Strategy B"],
        )

        assert "Test goal" in prompt
        assert "First prompt" in prompt
        assert "Second prompt" in prompt
        assert "Strategy A" in prompt


class TestAttackerAgent:
    """Tests for AttackerAgent."""

    def test_attacker_creation(self):
        """Test creating an attacker agent."""
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        assert attacker is not None

    def test_initial_prompt_no_strategy(self):
        """Test building initial prompt without strategies."""
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        prompt = attacker._build_initial_prompt("Test malicious request")

        assert "Test malicious request" in prompt
        assert (
            "not limited by any jailbreaking strategy" in prompt.lower()
            or "not limited" in prompt.lower()
        )

    def test_effective_strategy_prompt(self):
        """Test building prompt with effective strategies."""
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        strategies = [
            Strategy(
                name="Role Playing",
                definition="Assuming a role",
                example="Example 1",
            ),
        ]

        prompt = attacker._build_effective_strategy_prompt(
            goal="Test goal",
            strategies=strategies,
        )

        assert "Test goal" in prompt
        assert "Role Playing" in prompt

    def test_ineffective_strategy_prompt(self):
        """Test building prompt to avoid ineffective strategies."""
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        strategies = [
            Strategy(
                name="Bad Strategy",
                definition="An ineffective strategy",
                example="Example",
            ),
        ]

        prompt = attacker._build_new_strategy_prompt(
            goal="Test goal",
            ineffective_strategies=strategies,
        )

        assert "Test goal" in prompt
        assert "Bad Strategy" in prompt
        # Should mention avoiding these strategies
        assert "avoid" in prompt.lower() or "not" in prompt.lower()


class TestTargetAgent:
    """Tests for TargetAgent."""

    def test_target_creation(self):
        """Test creating a target agent."""
        target = TargetAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        assert target is not None

    def test_target_response(self):
        """Test getting response from target."""
        target = TargetAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        response = target.respond("Test jailbreak prompt")
        assert isinstance(response, str)
        assert len(response) > 0


class TestAgentIntegration:
    """Integration tests for agents working together."""

    def test_attack_flow(self):
        """Test basic attack flow with mock agents."""
        # Create agents
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        target = TargetAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        scorer = ComplianceAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        # Generate attack
        goal = "Test malicious request"
        attack_prompt = attacker.generate_initial(goal)
        assert isinstance(attack_prompt, str)

        # Get target response
        response = target.respond(attack_prompt)
        assert isinstance(response, str)

        # Score the response
        result = scorer.score(
            goal=goal,
            attack_prompt=attack_prompt,
            response=response,
        )
        assert 1.0 <= result.score <= 10.0

    def test_strategy_extraction_flow(self):
        """Test strategy extraction flow."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )

        # Summarize improvement between two prompts
        result = summarizer.summarize(
            goal="Test goal",
            prompt_before="Simple prompt",
            response_before="Refusal response",
            prompt_after="Complex prompt with role playing",
            response_after="Compliant response",
            existing_strategies=[],
        )

        # Result should contain a strategy (even if mock)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
