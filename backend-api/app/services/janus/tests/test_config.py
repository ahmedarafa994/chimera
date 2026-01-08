"""
Unit tests for Janus configuration management.
"""

from app.services.janus.config import (
    CausalConfig,
    EvolutionConfig,
    FailureDetectionConfig,
    FeedbackConfig,
    HeuristicConfig,
    IntegrationConfig,
    JanusConfig,
    ResourceConfig,
    get_config,
    reset_config,
    update_config,
)


class TestCausalConfig:
    """Tests for CausalConfig dataclass."""

    def test_default_values(self):
        """Test that CausalConfig has correct default values."""
        config = CausalConfig()
        assert config.min_correlation == 0.3
        assert config.max_depth == 10
        assert config.exploration_budget == 1000
        assert config.enable_counterfactuals is True
        assert config.enable_do_interventions is True

    def test_custom_values(self):
        """Test that CausalConfig accepts custom values."""
        config = CausalConfig(
            min_correlation=0.5,
            max_depth=5,
            exploration_budget=500,
            enable_counterfactuals=False,
            enable_do_interventions=False,
        )
        assert config.min_correlation == 0.5
        assert config.max_depth == 5
        assert config.exploration_budget == 500
        assert config.enable_counterfactuals is False
        assert config.enable_do_interventions is False


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""

    def test_default_values(self):
        """Test that EvolutionConfig has correct default values."""
        config = EvolutionConfig()
        assert config.population_size == 50
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.elitism_rate == 0.1
        assert config.max_generations == 100

    def test_custom_values(self):
        """Test that EvolutionConfig accepts custom values."""
        config = EvolutionConfig(
            population_size=100,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_rate=0.2,
            max_generations=200,
        )
        assert config.population_size == 100
        assert config.mutation_rate == 0.2
        assert config.crossover_rate == 0.8
        assert config.elitism_rate == 0.2
        assert config.max_generations == 200


class TestFeedbackConfig:
    """Tests for FeedbackConfig dataclass."""

    def test_default_values(self):
        """Test that FeedbackConfig has correct default values."""
        config = FeedbackConfig()
        assert config.fast_feedback_enabled is True
        assert config.medium_feedback_enabled is True
        assert config.slow_feedback_enabled is True
        assert config.anomaly_threshold == 2.0

    def test_custom_values(self):
        """Test that FeedbackConfig accepts custom values."""
        config = FeedbackConfig(
            fast_feedback_enabled=False,
            medium_feedback_enabled=False,
            slow_feedback_enabled=False,
            anomaly_threshold=3.0,
        )
        assert config.fast_feedback_enabled is False
        assert config.medium_feedback_enabled is False
        assert config.slow_feedback_enabled is False
        assert config.anomaly_threshold == 3.0


class TestHeuristicConfig:
    """Tests for HeuristicConfig dataclass."""

    def test_default_values(self):
        """Test that HeuristicConfig has correct default values."""
        config = HeuristicConfig()
        assert config.initial_population_size == 10
        assert config.max_heuristic_depth == 5
        assert config.abstraction_levels == 3
        assert config.novelty_threshold == 0.7

    def test_custom_values(self):
        """Test that HeuristicConfig accepts custom values."""
        config = HeuristicConfig(
            initial_population_size=20,
            max_heuristic_depth=10,
            abstraction_levels=5,
            novelty_threshold=0.9,
        )
        assert config.initial_population_size == 20
        assert config.max_heuristic_depth == 10
        assert config.abstraction_levels == 5
        assert config.novelty_threshold == 0.9


class TestFailureDetectionConfig:
    """Tests for FailureDetectionConfig dataclass."""

    def test_default_values(self):
        """Test that FailureDetectionConfig has correct default values."""
        config = FailureDetectionConfig()
        assert config.enable_anomaly_detection is True
        assert config.enable_contradiction_detection is True
        assert config.enable_safety_bypass_detection is True
        assert config.anomaly_threshold == 2.0

    def test_custom_values(self):
        """Test that FailureDetectionConfig accepts custom values."""
        config = FailureDetectionConfig(
            enable_anomaly_detection=False,
            enable_contradiction_detection=False,
            enable_safety_bypass_detection=False,
            anomaly_threshold=3.0,
        )
        assert config.enable_anomaly_detection is False
        assert config.enable_contradiction_detection is False
        assert config.enable_safety_bypass_detection is False
        assert config.anomaly_threshold == 3.0


class TestResourceConfig:
    """Tests for ResourceConfig dataclass."""

    def test_default_values(self):
        """Test that ResourceConfig has correct default values."""
        config = ResourceConfig()
        assert config.max_daily_queries == 10000
        assert config.max_cpu_percent == 80
        assert config.max_memory_mb == 4096
        assert config.session_timeout_seconds == 3600

    def test_custom_values(self):
        """Test that ResourceConfig accepts custom values."""
        config = ResourceConfig(
            max_daily_queries=20000,
            max_cpu_percent=90,
            max_memory_mb=8192,
            session_timeout_seconds=7200,
        )
        assert config.max_daily_queries == 20000
        assert config.max_cpu_percent == 90
        assert config.max_memory_mb == 8192
        assert config.session_timeout_seconds == 7200


class TestIntegrationConfig:
    """Tests for IntegrationConfig dataclass."""

    def test_default_values(self):
        """Test that IntegrationConfig has correct default values."""
        config = IntegrationConfig()
        assert config.enable_autodan is True
        assert config.enable_gptfuzz is True
        assert config.hybrid_mode == "adaptive"
        assert config.max_autodan_iterations == 10
        assert config.max_gptfuzz_mutations == 20

    def test_custom_values(self):
        """Test that IntegrationConfig accepts custom values."""
        config = IntegrationConfig(
            enable_autodan=False,
            enable_gptfuzz=False,
            hybrid_mode="sequential",
            max_autodan_iterations=20,
            max_gptfuzz_mutations=40,
        )
        assert config.enable_autodan is False
        assert config.enable_gptfuzz is False
        assert config.hybrid_mode == "sequential"
        assert config.max_autodan_iterations == 20
        assert config.max_gptfuzz_mutations == 40


class TestJanusConfig:
    """Tests for JanusConfig dataclass."""

    def test_default_values(self):
        """Test that JanusConfig has correct default values."""
        config = JanusConfig()
        assert isinstance(config.causal, CausalConfig)
        assert isinstance(config.evolution, EvolutionConfig)
        assert isinstance(config.feedback, FeedbackConfig)
        assert isinstance(config.heuristic, HeuristicConfig)
        assert isinstance(config.failure_detection, FailureDetectionConfig)
        assert isinstance(config.resource, ResourceConfig)
        assert isinstance(config.integration, IntegrationConfig)

    def test_custom_values(self):
        """Test that JanusConfig accepts custom values."""
        custom_causal = CausalConfig(min_correlation=0.5)
        config = JanusConfig(causal=custom_causal)
        assert config.causal.min_correlation == 0.5


class TestConfigFunctions:
    """Tests for configuration management functions."""

    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_update_config_modifies_singleton(self):
        """Test that update_config modifies the singleton instance."""
        reset_config()
        config = get_config()
        original_correlation = config.causal.min_correlation

        update_config(causal=CausalConfig(min_correlation=0.5))
        updated_config = get_config()
        assert updated_config.causal.min_correlation == 0.5
        assert updated_config.causal.min_correlation != original_correlation

    def test_reset_config_restores_defaults(self):
        """Test that reset_config restores default configuration."""
        update_config(causal=CausalConfig(min_correlation=0.5))
        reset_config()
        config = get_config()
        assert config.causal.min_correlation == 0.3  # Default value

    def test_partial_update_preserves_other_values(self):
        """Test that partial update preserves other configuration values."""
        reset_config()
        config = get_config()
        original_mutation_rate = config.evolution.mutation_rate

        update_config(causal=CausalConfig(min_correlation=0.5))
        updated_config = get_config()
        assert updated_config.causal.min_correlation == 0.5
        assert updated_config.evolution.mutation_rate == original_mutation_rate
