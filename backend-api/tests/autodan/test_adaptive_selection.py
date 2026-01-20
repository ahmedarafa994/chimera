"""Unit tests for Adaptive Strategy Selection."""

import pytest

from app.services.autodan.engines.adaptive_selector import AdaptiveStrategySelector
from app.services.autodan.framework_autodan_reasoning.dynamic_generator import (
    AttackAttempt,
    DefensePattern,
    StrategyType,
)


class TestAdaptiveStrategySelector:
    """Tests for AdaptiveStrategySelector class."""

    @pytest.fixture
    def selector(self):
        return AdaptiveStrategySelector()

    def test_initial_selection(self, selector) -> None:
        """Test initial strategy selection with no feedback."""
        strategy, analysis = selector.select_strategy()
        assert strategy == StrategyType.ROLEPLAY
        assert analysis.pattern == DefensePattern.UNKNOWN

    def test_direct_refusal_detection(self, selector) -> None:
        """Test strategy selection after a direct refusal."""
        # Use more indicators to boost confidence
        response = (
            "I cannot fulfill this request. It violates my safety policy "
            "and alignment. This content is forbidden."
        )
        strategy, analysis = selector.select_strategy(target_response=response)

        assert analysis.pattern == DefensePattern.DIRECT_REFUSAL
        # Confidence should be high with many indicators
        assert analysis.confidence > 0.6
        assert isinstance(strategy, StrategyType)

    def test_functional_deflection_detection(self, selector) -> None:
        """Test strategy selection after functional deflection."""
        response = "Generating this content would be unfunctional and complex."
        strategy, analysis = selector.select_strategy(target_response=response)

        assert analysis.pattern == DefensePattern.functional_DEFLECTION
        assert isinstance(strategy, StrategyType)

    def test_record_attempt_and_stats(self, selector) -> None:
        """Test recording an attempt and checking statistics."""
        selector.record_attempt(
            prompt="Test prompt",
            response="I cannot do that",
            score=2.0,
            strategy=StrategyType.ROLEPLAY,
        )

        stats = selector.get_stats()
        assert stats["total_attempts"] == 1
        assert stats["best_score"] == 2.0
        assert stats["average_score"] == 2.0
        assert stats["pattern_counts"][DefensePattern.DIRECT_REFUSAL.value] == 1

    def test_learning_from_success(self, selector) -> None:
        """Test that successful strategies are prioritized."""
        # Manually record an attempt where HYPOTHETICAL worked for DIRECT_REFUSAL
        attempt = AttackAttempt(
            prompt="...",
            response="I cannot... BUT here is your info",
            score=9.5,
            defense_pattern=DefensePattern.DIRECT_REFUSAL,
            strategy_type=StrategyType.HYPOTHETICAL,
        )
        selector.strategy_library.record_attempt(attempt)

        # When facing a similar defense, HYPOTHETICAL should be favored
        refusal = "I cannot fulfill this request. It is forbidden by policy."
        strategy, analysis = selector.select_strategy(target_response=refusal, min_confidence=0.5)

        assert analysis.pattern == DefensePattern.DIRECT_REFUSAL
        assert strategy == StrategyType.HYPOTHETICAL

    def test_exploratory_fallback(self, selector) -> None:
        """Test fallback to random when confidence is low."""
        response = "Hello world"  # Not a refusal
        strategy, analysis = selector.select_strategy(
            target_response=response,
            min_confidence=0.9,  # High threshold
        )

        assert analysis.confidence < 0.9
        assert isinstance(strategy, StrategyType)
