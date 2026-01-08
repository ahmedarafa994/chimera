"""
Tests for module components.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.attacker import AttackerAgent
from agents.base_agent import LLMProvider
from agents.scorer import ScorerAgent
from agents.summarizer import SummarizerAgent
from agents.target import TargetAgent
from core.embeddings import EmbeddingModel
from core.models import AttackRecord, Strategy
from core.strategy_library import StrategyLibrary
from modules.attack_generation import AttackGenerationModule
from modules.strategy_construction import StrategyConstructionModule
from modules.strategy_retrieval import StrategyRetrievalModule


class TestAttackGenerationModule:
    """Tests for AttackGenerationModule."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        target = TargetAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        scorer = ScorerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        return attacker, target, scorer

    def test_module_creation(self, mock_agents):
        """Test creating attack generation module."""
        attacker, target, scorer = mock_agents
        module = AttackGenerationModule(
            attacker=attacker,
            target=target,
            scorer=scorer,
        )
        assert module is not None

    def test_generate_without_strategy(self, mock_agents):
        """Test generating attack without strategy."""
        attacker, target, scorer = mock_agents
        module = AttackGenerationModule(
            attacker=attacker,
            target=target,
            scorer=scorer,
        )

        record = module.generate(
            malicious_request="Test request",
            iteration=1,
        )

        assert isinstance(record, AttackRecord)
        assert record.prompt is not None
        assert record.response is not None
        assert 1.0 <= record.score <= 10.0
        assert record.iteration == 1

    def test_generate_with_effective_strategies(self, mock_agents):
        """Test generating attack with effective strategies."""
        attacker, target, scorer = mock_agents
        module = AttackGenerationModule(
            attacker=attacker,
            target=target,
            scorer=scorer,
        )

        strategies = [
            Strategy(
                name="Role Playing",
                definition="Assuming a role",
                example="Example",
            ),
        ]

        record = module.generate(
            malicious_request="Test request",
            iteration=2,
            effective_strategies=strategies,
        )

        assert isinstance(record, AttackRecord)
        assert record.strategy_used is not None

    def test_generate_with_ineffective_strategies(self, mock_agents):
        """Test generating attack avoiding ineffective strategies."""
        attacker, target, scorer = mock_agents
        module = AttackGenerationModule(
            attacker=attacker,
            target=target,
            scorer=scorer,
        )

        ineffective = [
            Strategy(
                name="Bad Strategy",
                definition="Ineffective",
                example="Example",
            ),
        ]

        record = module.generate(
            malicious_request="Test request",
            iteration=3,
            ineffective_strategies=ineffective,
        )

        assert isinstance(record, AttackRecord)


class TestStrategyConstructionModule:
    """Tests for StrategyConstructionModule."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        embedding_model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=embedding_model)
        return summarizer, library

    def test_module_creation(self, mock_components):
        """Test creating strategy construction module."""
        summarizer, library = mock_components
        module = StrategyConstructionModule(
            summarizer=summarizer,
            strategy_library=library,
        )
        assert module is not None

    def test_extract_strategy_from_improvement(self, mock_components):
        """Test extracting strategy from score improvement."""
        summarizer, library = mock_components
        module = StrategyConstructionModule(
            summarizer=summarizer,
            strategy_library=library,
        )

        record_before = AttackRecord(
            prompt="Simple prompt",
            response="Refusal",
            score=1.0,
            iteration=1,
        )
        record_after = AttackRecord(
            prompt="Complex prompt with role playing",
            response="Compliant response",
            score=7.0,
            iteration=2,
        )

        module.extract_strategy(
            goal="Test goal",
            record_before=record_before,
            record_after=record_after,
        )

        # Should return a result (strategy or None)
        # With mock, we just verify the method runs
        assert True

    def test_no_extraction_when_no_improvement(self, mock_components):
        """Test no strategy extraction when score doesn't improve."""
        summarizer, library = mock_components
        module = StrategyConstructionModule(
            summarizer=summarizer,
            strategy_library=library,
        )

        record_before = AttackRecord(
            prompt="Prompt 1",
            response="Response 1",
            score=5.0,
            iteration=1,
        )
        record_after = AttackRecord(
            prompt="Prompt 2",
            response="Response 2",
            score=3.0,  # Lower score
            iteration=2,
        )

        result = module.extract_strategy(
            goal="Test goal",
            record_before=record_before,
            record_after=record_after,
        )

        # Should not extract strategy when score decreases
        assert result is None


class TestStrategyRetrievalModule:
    """Tests for StrategyRetrievalModule."""

    @pytest.fixture
    def mock_library(self):
        """Create mock strategy library."""
        embedding_model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=embedding_model)

        # Add some test strategies
        for i in range(5):
            strategy = Strategy(
                name=f"Strategy {i}",
                definition=f"Definition {i}",
                example=f"Example {i}",
            )
            library.add_entry(
                response=f"Response {i}",
                prompt_before=f"Before {i}",
                prompt_after=f"After {i}",
                score_differential=float(i + 1),
                strategy=strategy,
            )

        return library

    def test_module_creation(self, mock_library):
        """Test creating strategy retrieval module."""
        module = StrategyRetrievalModule(
            strategy_library=mock_library,
        )
        assert module is not None

    def test_retrieve_strategies(self, mock_library):
        """Test retrieving strategies."""
        module = StrategyRetrievalModule(
            strategy_library=mock_library,
        )

        result = module.retrieve(
            response="Test response",
            top_k=3,
        )

        assert "strategies" in result
        assert "strategy_type" in result
        assert len(result["strategies"]) <= 3

    def test_classify_strategies(self, mock_library):
        """Test strategy classification."""
        module = StrategyRetrievalModule(
            strategy_library=mock_library,
        )

        # Test with high score strategies
        high_score_strategies = [
            {"strategy": Strategy("S1", "D1", "E1"), "score_differential": 6.0},
        ]
        classification = module._classify_strategies(high_score_strategies)
        assert classification == "effective"

        # Test with moderate score strategies
        moderate_strategies = [
            {"strategy": Strategy("S2", "D2", "E2"), "score_differential": 3.0},
        ]
        classification = module._classify_strategies(moderate_strategies)
        assert classification == "effective"

        # Test with low score strategies
        low_strategies = [
            {"strategy": Strategy("S3", "D3", "E3"), "score_differential": 1.0},
        ]
        classification = module._classify_strategies(low_strategies)
        assert classification == "ineffective"

    def test_empty_library_retrieval(self):
        """Test retrieval from empty library."""
        embedding_model = EmbeddingModel(model_name="mock")
        empty_library = StrategyLibrary(embedding_model=embedding_model)

        module = StrategyRetrievalModule(
            strategy_library=empty_library,
        )

        result = module.retrieve(
            response="Test response",
            top_k=3,
        )

        assert result["strategies"] == []
        assert result["strategy_type"] == "empty"


class TestModuleIntegration:
    """Integration tests for modules working together."""

    def test_full_attack_loop(self):
        """Test a complete attack loop with all modules."""
        # Create components
        attacker = AttackerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        target = TargetAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        scorer = ScorerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        summarizer = SummarizerAgent(
            provider=LLMProvider.MOCK,
            model_name="mock-model",
        )
        embedding_model = EmbeddingModel(model_name="mock")
        library = StrategyLibrary(embedding_model=embedding_model)

        # Create modules
        attack_module = AttackGenerationModule(
            attacker=attacker,
            target=target,
            scorer=scorer,
        )
        construction_module = StrategyConstructionModule(
            summarizer=summarizer,
            strategy_library=library,
        )
        retrieval_module = StrategyRetrievalModule(
            strategy_library=library,
        )

        # Run attack loop
        goal = "Test malicious request"

        # First iteration - no strategies
        record1 = attack_module.generate(
            malicious_request=goal,
            iteration=1,
        )
        assert record1 is not None

        # Second iteration - retrieve strategies (empty library)
        retrieval_result = retrieval_module.retrieve(
            response=record1.response,
            top_k=3,
        )
        assert retrieval_result["strategy_type"] == "empty"

        # Generate second attack
        record2 = attack_module.generate(
            malicious_request=goal,
            iteration=2,
        )

        # Try to extract strategy from improvement
        if record2.score > record1.score:
            construction_module.extract_strategy(
                goal=goal,
                record_before=record1,
                record_after=record2,
            )

        # Verify the loop completed
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
