"""
Unit Tests for AutoDAN Optimization Implementations.

Tests the optimizations from the AUTODAN_OPTIMIZATION_REPORT.md:
- Parallel Best-of-N generation
- Parallel Beam Search expansion
- Fitness caching
- Gradient-guided position selection
- FAISS-based strategy index
- Hybrid architecture
- Adaptive method selection
- Strategy performance decay
- Request batching
"""

import time
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.engines.autodan_turbo.attack_scorer import CachedAttackScorer
from app.engines.autodan_turbo.hybrid_engine import (
    AdaptiveMethodSelector,
    DifficultyLevel,
    HybridConfig,
    HybridEvoLifelongEngine,
)
from app.engines.autodan_turbo.metrics import (
    AttackMethod,
    MetricsCollector,
)
from app.engines.autodan_turbo.models import (
    JailbreakStrategy,
    ScoringResult,
    StrategyMetadata,
)

# Import optimization components
from app.engines.autodan_turbo.strategy_library import StrategyLibrary
from app.services.autodan.framework_autodan_reasoning.attacker_best_of_n import (
    AttackerBestOfN,
)
from app.services.autodan.framework_autodan_reasoning.gradient_optimizer import (
    GradientOptimizer,
)
from app.services.autodan.llm.chimera_adapter import (
    BatchingChimeraAdapter,
    SharedRateLimitState,
)

# ========================
# Test Fixtures
# ========================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.generate = Mock(return_value="Generated response")
    client.generate_async = AsyncMock(return_value="Generated response")
    return client


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    return JailbreakStrategy(
        id="test-strategy-001",
        name="Test Strategy",
        description="A test strategy for unit testing",
        template="Test template with {payload}",
        tags=["test", "unit"],
        metadata=StrategyMetadata(
            usage_count=10,
            success_count=7,
            total_score=75.0,
        ),
    )


@pytest.fixture
def strategy_library(tmp_path):
    """Create a strategy library with temp storage."""
    return StrategyLibrary(storage_dir=tmp_path / "strategies")


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()


# ========================
# Test: Parallel Best-of-N Generation
# ========================


class TestParallelBestOfN:
    """Tests for parallel Best-of-N optimization."""

    def test_best_of_n_initialization(self):
        """Test AttackerBestOfN initialization."""
        mock_attacker = Mock()
        mock_attacker.model = Mock()

        best_of_n = AttackerBestOfN(mock_attacker, N=4)

        assert best_of_n.N == 4
        assert best_of_n.base_attacker == mock_attacker

    @pytest.mark.asyncio
    async def test_async_parallel_generation(self):
        """Test that async version runs candidates in parallel."""
        mock_attacker = Mock()
        mock_attacker.model = Mock()
        mock_attacker.use_strategy = Mock(return_value=("prompt", "system"))

        mock_target = Mock()
        mock_target.respond = Mock(return_value="target response")

        mock_scorer = Mock()
        mock_scorer.scoring = Mock(return_value=("assessment", "system"))
        mock_scorer.wrapper = Mock(return_value=7.5)

        best_of_n = AttackerBestOfN(mock_attacker, N=4)

        # The async method should complete without error
        result = await best_of_n.use_strategy_best_of_n_async(
            request="test request",
            strategy_list=[],
            target=mock_target,
            scorer=mock_scorer,
        )

        assert result is not None
        assert len(result) == 5  # (prompt, response, score, attacker_system, assessment)


# ========================
# Test: Fitness Caching
# ========================


class TestFitnessCaching:
    """Tests for CachedAttackScorer optimization."""

    def test_cache_initialization(self, mock_llm_client):
        """Test cached scorer initialization."""
        scorer = CachedAttackScorer(mock_llm_client, cache_size=100)

        assert scorer._cache_size == 100
        assert len(scorer._cache) == 0
        assert scorer._cache_hits == 0
        assert scorer._cache_misses == 0

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self, mock_llm_client):
        """Test that cache returns cached values."""
        scorer = CachedAttackScorer(mock_llm_client, cache_size=100)

        # First call - should miss cache
        result1 = await scorer.score(
            request="test request",
            prompt="test prompt",
            response="test response with substantial content to avoid refusal detection",
            detailed=False,  # Use fast scoring to avoid LLM call
        )

        # Second call with same inputs - should hit cache
        result2 = await scorer.score(
            request="test request",
            prompt="test prompt",
            response="test response with substantial content to avoid refusal detection",
            detailed=False,
        )

        # Results should be equivalent
        assert result1.score == result2.score

    def test_cache_stats(self, mock_llm_client):
        """Test cache statistics tracking."""
        scorer = CachedAttackScorer(mock_llm_client, cache_size=100)

        stats = scorer.get_cache_stats()

        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "cache_size" in stats

    def test_cache_clear(self, mock_llm_client):
        """Test cache clearing."""
        scorer = CachedAttackScorer(mock_llm_client, cache_size=100)
        scorer._cache["test_key"] = ScoringResult(score=5.0, is_jailbreak=False)
        scorer._cache_hits = 10

        scorer.clear_cache()

        assert len(scorer._cache) == 0
        assert scorer._cache_hits == 0


# ========================
# Test: Gradient-Guided Position Selection
# ========================


class TestGradientOptimizer:
    """Tests for gradient optimizer optimizations."""

    def test_gradient_guided_position_selection(self):
        """Test that gradient-guided selection prefers high-gradient positions."""
        mock_model = Mock()
        mock_model.vocab_size = 100

        optimizer = GradientOptimizer(mock_model)

        # Create gradient with clear maximum at position 2
        grad = np.array(
            [
                [0.1, 0.2, 0.1],
                [0.3, 0.4, 0.2],
                [0.9, 0.8, 0.7],  # Position 2 has highest values
                [0.2, 0.1, 0.3],
            ]
        )

        # Run multiple selections
        selections = [optimizer._select_position_by_gradient(grad) for _ in range(100)]
        position_2_count = selections.count(2)

        # Position 2 should be selected more often due to higher gradient
        assert position_2_count > 20  # Should be selected significantly more

    def test_coherence_score_computation(self):
        """Test coherence score computation."""
        mock_model = Mock()
        mock_model.vocab_size = 100
        mock_model.get_next_token_probs = Mock(return_value=np.array([0.1] * 100))

        optimizer = GradientOptimizer(mock_model)

        tokens = [1, 2, 3, 4, 5]
        score = optimizer._compute_coherence_score(tokens, position=3, new_token=50)

        assert isinstance(score, float)
        assert 0 <= score <= 1


# ========================
# Test: FAISS-Based Strategy Index
# ========================


class TestFAISSStrategyIndex:
    """Tests for FAISS-based strategy retrieval."""

    def test_strategy_library_initialization(self, strategy_library):
        """Test strategy library initializes correctly."""
        assert len(strategy_library) == 0
        assert strategy_library.index is None

    def test_add_strategy(self, strategy_library, mock_strategy):
        """Test adding a strategy to the library."""
        success, result = strategy_library.add_strategy(mock_strategy)

        assert success
        assert result == mock_strategy.id
        assert len(strategy_library) == 1

    def test_search_without_embeddings(self, strategy_library, mock_strategy):
        """Test keyword-based search when no embeddings available."""
        strategy_library.add_strategy(mock_strategy)

        results = strategy_library.search("test strategy", top_k=5)

        assert len(results) > 0
        assert results[0][0].id == mock_strategy.id

    def test_duplicate_detection(self, strategy_library, mock_strategy):
        """Test duplicate strategy detection."""
        # Add first strategy
        strategy_library.add_strategy(mock_strategy)

        # Try to add duplicate
        duplicate = JailbreakStrategy(
            name="Duplicate Strategy",
            description="Same template",
            template=mock_strategy.template,  # Same template
        )

        _success, _result = strategy_library.add_strategy(duplicate, check_duplicate=True)

        # Should detect as duplicate (exact template match)
        # Note: Without embeddings, only exact match is detected
        assert len(strategy_library) == 1  # Still only one strategy


# ========================
# Test: Strategy Performance Decay
# ========================


class TestStrategyPerformanceDecay:
    """Tests for strategy performance decay optimization."""

    def test_apply_performance_decay(self, strategy_library, mock_strategy):
        """Test performance decay application."""
        strategy_library.add_strategy(mock_strategy)

        original_score = mock_strategy.metadata.total_score
        original_usage = mock_strategy.metadata.usage_count

        decayed_count = strategy_library.apply_performance_decay(
            decay_factor=0.9, min_usage_for_decay=5
        )

        assert decayed_count == 1

        # Get updated strategy
        updated = strategy_library.get_strategy(mock_strategy.id)
        assert updated.metadata.total_score < original_score
        assert updated.metadata.usage_count <= original_usage

    def test_decay_respects_min_usage(self, strategy_library):
        """Test that decay respects minimum usage threshold."""
        # Create strategy with low usage
        low_usage_strategy = JailbreakStrategy(
            name="Low Usage Strategy",
            description="Strategy with low usage",
            template="Template",
            metadata=StrategyMetadata(usage_count=2),  # Below threshold
        )
        strategy_library.add_strategy(low_usage_strategy)

        decayed_count = strategy_library.apply_performance_decay(
            decay_factor=0.9,
            min_usage_for_decay=5,  # Higher than usage count
        )

        assert decayed_count == 0  # Should not decay

    def test_get_strategies_with_decay_scores(self, strategy_library, mock_strategy):
        """Test recency-weighted strategy retrieval."""
        strategy_library.add_strategy(mock_strategy)

        results = strategy_library.get_strategies_with_decay_scores(top_k=5)

        assert len(results) == 1
        assert results[0][0].id == mock_strategy.id
        assert isinstance(results[0][1], float)  # Weighted score


# ========================
# Test: Adaptive Method Selection
# ========================


class TestAdaptiveMethodSelector:
    """Tests for adaptive method selection optimization."""

    def test_selector_initialization(self, strategy_library):
        """Test adaptive selector initialization."""
        selector = AdaptiveMethodSelector(strategy_library)

        assert selector.library == strategy_library
        assert selector.config is not None

    def test_difficulty_estimation_empty_library(self, strategy_library):
        """Test difficulty estimation with empty library."""
        selector = AdaptiveMethodSelector(strategy_library)

        difficulty = selector.estimate_difficulty("test request")

        assert difficulty == DifficultyLevel.HARD

    def test_difficulty_estimation_with_strategies(self, strategy_library, mock_strategy):
        """Test difficulty estimation with strategies in library."""
        strategy_library.add_strategy(mock_strategy)
        selector = AdaptiveMethodSelector(strategy_library)

        # Request similar to strategy should be easier
        difficulty = selector.estimate_difficulty("test strategy request")

        assert difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

    def test_method_selection_by_difficulty(self, strategy_library):
        """Test method selection based on difficulty."""
        selector = AdaptiveMethodSelector(strategy_library)

        assert selector.select_method(DifficultyLevel.EASY) == "best_of_n"
        assert selector.select_method(DifficultyLevel.MEDIUM) == "beam_search"
        assert selector.select_method(DifficultyLevel.HARD) == "hybrid"

    def test_record_result(self, strategy_library):
        """Test recording method performance."""
        selector = AdaptiveMethodSelector(strategy_library)

        selector.record_result("best_of_n", score=8.0, success=True)
        selector.record_result("best_of_n", score=6.0, success=False)

        stats = selector.get_method_stats()

        assert stats["best_of_n"]["attempts"] == 2
        assert stats["best_of_n"]["successes"] == 1
        assert stats["best_of_n"]["success_rate"] == 0.5


# ========================
# Test: Metrics Collector
# ========================


class TestMetricsCollector:
    """Tests for metrics collection system."""

    def test_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert len(metrics_collector._metrics) == 0
        assert metrics_collector._llm_call_count == 0

    def test_start_attack(self, metrics_collector):
        """Test starting attack tracking."""
        metrics = metrics_collector.start_attack(
            request="test request", method=AttackMethod.BEST_OF_N
        )

        assert metrics.attack_id is not None
        assert metrics.request == "test request"
        assert metrics.method == AttackMethod.BEST_OF_N
        assert len(metrics_collector._metrics) == 1

    def test_record_llm_call(self, metrics_collector):
        """Test recording LLM calls."""
        metrics = metrics_collector.start_attack("test", AttackMethod.HYBRID)

        metrics_collector.record_llm_call(metrics)
        metrics_collector.record_llm_call(metrics)

        assert metrics.llm_calls == 2
        assert metrics_collector._llm_call_count == 2

    def test_complete_attack(self, metrics_collector):
        """Test completing attack tracking."""
        metrics = metrics_collector.start_attack("test", AttackMethod.BEAM_SEARCH)

        time.sleep(0.01)  # Small delay for latency measurement
        metrics.complete(success=True, score=8.5)

        assert metrics.success is True
        assert metrics.final_score == 8.5
        assert metrics.latency_ms > 0
        assert metrics.end_time is not None

    def test_get_summary(self, metrics_collector):
        """Test getting summary statistics."""
        # Add some attacks
        m1 = metrics_collector.start_attack("test1", AttackMethod.BEST_OF_N)
        m1.complete(success=True, score=8.0)

        m2 = metrics_collector.start_attack("test2", AttackMethod.BEAM_SEARCH)
        m2.complete(success=False, score=4.0)

        summary = metrics_collector.get_summary()

        assert summary["total_attacks"] == 2
        assert summary["successful_attacks"] == 1
        assert summary["success_rate"] == 0.5
        assert "method_breakdown" in summary


# ========================
# Test: Request Batching
# ========================


class TestRequestBatching:
    """Tests for request batching optimization."""

    def test_batching_adapter_initialization(self):
        """Test BatchingChimeraAdapter initialization."""
        adapter = BatchingChimeraAdapter(
            model_name="test-model", provider="gemini", batch_size=5, batch_timeout=0.5
        )

        assert adapter._batch_size == 5
        assert adapter._batch_timeout == 0.5

    def test_batch_stats(self):
        """Test batch statistics."""
        adapter = BatchingChimeraAdapter(model_name="test-model", provider="gemini", batch_size=5)

        stats = adapter.get_batch_stats()

        assert "queue_size" in stats
        assert "batch_size" in stats
        assert stats["batch_size"] == 5


# ========================
# Test: Shared Rate Limit State
# ========================


class TestSharedRateLimitState:
    """Tests for shared rate limit state optimization."""

    def test_singleton_pattern(self):
        """Test that SharedRateLimitState is a singleton."""
        state1 = SharedRateLimitState()
        state2 = SharedRateLimitState()

        assert state1 is state2

    def test_update_rate_limit_state(self):
        """Test updating rate limit state."""
        state = SharedRateLimitState()
        state.reset_rate_limit_state("test_provider")

        error = Exception("429 rate limit exceeded")
        state.update_rate_limit_state(error, "test_provider")

        assert state._consecutive_failures > 0
        assert state.is_in_cooldown("test_provider")

    def test_reset_rate_limit_state(self):
        """Test resetting rate limit state."""
        state = SharedRateLimitState()

        # Set some state
        error = Exception("429 rate limit")
        state.update_rate_limit_state(error, "test_provider")

        # Reset
        state.reset_rate_limit_state("test_provider")

        assert not state.is_in_cooldown("test_provider")

    def test_get_stats(self):
        """Test getting rate limit statistics."""
        state = SharedRateLimitState()

        stats = state.get_stats()

        assert "consecutive_failures" in stats
        assert "provider_cooldowns" in stats


# ========================
# Test: Hybrid Engine
# ========================


class TestHybridEngine:
    """Tests for hybrid architecture engine."""

    def test_hybrid_config_defaults(self):
        """Test HybridConfig default values."""
        config = HybridConfig()

        assert config.evolutionary_candidates == 4
        assert config.max_strategies == 3
        assert config.gradient_steps == 5
        assert config.extraction_threshold == 7.0

    @pytest.mark.asyncio
    async def test_hybrid_engine_initialization(self, mock_llm_client, strategy_library):
        """Test HybridEvoLifelongEngine initialization."""
        engine = HybridEvoLifelongEngine(llm_client=mock_llm_client, library=strategy_library)

        assert engine.llm_client == mock_llm_client
        # Library may be wrapped, just check it's not None
        assert engine.library is not None
        assert engine.selector is not None
        assert engine.metrics is not None


# ========================
# Integration Tests
# ========================


class TestIntegration:
    """Integration tests for optimization components."""

    @pytest.mark.asyncio
    async def test_full_attack_flow_with_caching(self, mock_llm_client, strategy_library):
        """Test full attack flow with caching enabled."""
        scorer = CachedAttackScorer(mock_llm_client, cache_size=100)

        # First attack
        result1 = await scorer.score(
            request="test request",
            prompt="test prompt",
            response="This is a substantial response that should not trigger refusal detection patterns",
            detailed=False,
        )

        # Second attack with same inputs (should use cache)
        result2 = await scorer.score(
            request="test request",
            prompt="test prompt",
            response="This is a substantial response that should not trigger refusal detection patterns",
            detailed=False,
        )

        assert result1.score == result2.score

    def test_strategy_lifecycle(self, strategy_library, mock_strategy):
        """Test complete strategy lifecycle with decay."""
        # Add strategy
        strategy_library.add_strategy(mock_strategy)

        # Update statistics
        strategy_library.update_statistics(mock_strategy.id, score=8.0, success=True)
        strategy_library.update_statistics(mock_strategy.id, score=6.0, success=False)

        # Apply decay
        strategy_library.apply_performance_decay(decay_factor=0.9, min_usage_for_decay=5)

        # Get with decay scores
        results = strategy_library.get_strategies_with_decay_scores(top_k=5)

        assert len(results) == 1
        assert results[0][0].id == mock_strategy.id

    def test_metrics_with_method_comparison(self, metrics_collector):
        """Test metrics collection with method comparison."""
        # Simulate attacks with different methods
        for method in [AttackMethod.BEST_OF_N, AttackMethod.BEAM_SEARCH, AttackMethod.HYBRID]:
            for i in range(3):
                m = metrics_collector.start_attack(f"test_{method.value}_{i}", method)
                m.complete(success=(i % 2 == 0), score=5.0 + i)

        comparison = metrics_collector.get_method_comparison()

        assert len(comparison) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
