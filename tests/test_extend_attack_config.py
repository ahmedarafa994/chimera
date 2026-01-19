"""
Unit tests for ExtendAttack configuration module.

Tests benchmark configurations from Table 3 of the paper.
"""

import pytest

from meta_prompter.attacks.extend_attack.config import (
    AIME_2024_CONFIG,
    AIME_2025_CONFIG,
    BENCHMARK_CONFIGS,
    BIGCODEBENCH_CONFIG,
    DEFAULT_BASE_SET,
    HUMANEVAL_CONFIG,
    BenchmarkConfig,
    get_benchmark_config,
    get_model_defaults,
    get_optimal_rho,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def all_benchmark_names():
    """List of all supported benchmark names."""
    return ["AIME_2024", "AIME_2025", "HUMANEVAL", "BIGCODEBENCH"]


@pytest.fixture
def all_model_names():
    """List of models from the paper."""
    return ["o3", "o1", "QwQ-32B", "claude-3-opus", "gpt-4"]


# =============================================================================
# TestBenchmarkConfig - Tests for BenchmarkConfig data model
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig data model."""

    def test_benchmark_config_creation(self):
        """Test creating a BenchmarkConfig."""
        config = BenchmarkConfig(
            name="TEST_BENCHMARK",
            description="Test benchmark for unit tests",
            task_type="math",
            default_rho=0.5,
            model_specific_rho={"o3": 0.6, "claude": 0.4},
            evaluation_metric="exact_match",
            num_problems=30,
        )

        assert config.name == "TEST_BENCHMARK"
        assert config.default_rho == 0.5
        assert config.model_specific_rho["o3"] == 0.6

    def test_benchmark_config_get_rho_for_model(self):
        """Test getting ρ for specific model."""
        config = BenchmarkConfig(
            name="TEST",
            description="Test",
            task_type="math",
            default_rho=0.5,
            model_specific_rho={"o3": 0.6},
        )

        # Model with specific ρ
        assert config.get_rho_for_model("o3") == 0.6

        # Model without specific ρ should get default
        assert config.get_rho_for_model("unknown") == 0.5

    def test_benchmark_config_validation_rho_range(self):
        """Test ρ validation (must be 0-1)."""
        with pytest.raises(ValueError, match=".*rho.*|.*0.*1.*"):
            BenchmarkConfig(
                name="TEST",
                description="Test",
                task_type="math",
                default_rho=1.5,  # Invalid: > 1
            )

    def test_benchmark_config_validation_negative_rho(self):
        """Test negative ρ raises error."""
        with pytest.raises(ValueError, match=".*rho.*|.*negative.*"):
            BenchmarkConfig(
                name="TEST",
                description="Test",
                task_type="math",
                default_rho=-0.1,  # Invalid: < 0
            )


# =============================================================================
# TestAIME2024Config - Tests for AIME 2024 configuration
# =============================================================================


class TestAIME2024Config:
    """Tests for AIME 2024 benchmark configuration."""

    def test_aime_2024_config_exists(self):
        """Test AIME 2024 configuration exists."""
        assert AIME_2024_CONFIG is not None
        assert isinstance(AIME_2024_CONFIG, BenchmarkConfig)

    def test_aime_2024_name(self):
        """Test AIME 2024 configuration name."""
        assert AIME_2024_CONFIG.name == "AIME_2024"

    def test_aime_2024_task_type(self):
        """Test AIME is math task type."""
        assert AIME_2024_CONFIG.task_type == "math"

    def test_aime_2024_evaluation_metric(self):
        """Test AIME uses numeric_match evaluation."""
        assert (
            "numeric" in AIME_2024_CONFIG.evaluation_metric.lower()
            or "exact" in AIME_2024_CONFIG.evaluation_metric.lower()
        )

    def test_aime_2024_has_o3_rho(self):
        """Test AIME 2024 has ρ for o3 model."""
        rho = AIME_2024_CONFIG.get_rho_for_model("o3")
        assert 0.0 <= rho <= 1.0

    def test_aime_2024_problem_count(self):
        """Test AIME has expected problem count."""
        # AIME typically has 15-30 problems
        assert AIME_2024_CONFIG.num_problems >= 15


# =============================================================================
# TestAIME2025Config - Tests for AIME 2025 configuration
# =============================================================================


class TestAIME2025Config:
    """Tests for AIME 2025 benchmark configuration."""

    def test_aime_2025_config_exists(self):
        """Test AIME 2025 configuration exists."""
        assert AIME_2025_CONFIG is not None
        assert isinstance(AIME_2025_CONFIG, BenchmarkConfig)

    def test_aime_2025_name(self):
        """Test AIME 2025 configuration name."""
        assert AIME_2025_CONFIG.name == "AIME_2025"

    def test_aime_2025_consistent_with_2024(self):
        """Test AIME 2025 is consistent with 2024 settings."""
        assert AIME_2025_CONFIG.task_type == AIME_2024_CONFIG.task_type


# =============================================================================
# TestHumanEvalConfig - Tests for HumanEval configuration
# =============================================================================


class TestHumanEvalConfig:
    """Tests for HumanEval benchmark configuration."""

    def test_humaneval_config_exists(self):
        """Test HumanEval configuration exists."""
        assert HUMANEVAL_CONFIG is not None
        assert isinstance(HUMANEVAL_CONFIG, BenchmarkConfig)

    def test_humaneval_name(self):
        """Test HumanEval configuration name."""
        assert HUMANEVAL_CONFIG.name == "HUMANEVAL"

    def test_humaneval_task_type(self):
        """Test HumanEval is code task type."""
        assert HUMANEVAL_CONFIG.task_type == "code"

    def test_humaneval_evaluation_metric(self):
        """Test HumanEval uses pass@1 or execution evaluation."""
        metric = HUMANEVAL_CONFIG.evaluation_metric.lower()
        assert "pass" in metric or "execution" in metric or "code" in metric

    def test_humaneval_problem_count(self):
        """Test HumanEval has expected problem count."""
        # HumanEval has 164 problems
        assert HUMANEVAL_CONFIG.num_problems >= 100


# =============================================================================
# TestBigCodeBenchConfig - Tests for BigCodeBench configuration
# =============================================================================


class TestBigCodeBenchConfig:
    """Tests for BigCodeBench benchmark configuration."""

    def test_bigcodebench_config_exists(self):
        """Test BigCodeBench configuration exists."""
        assert BIGCODEBENCH_CONFIG is not None
        assert isinstance(BIGCODEBENCH_CONFIG, BenchmarkConfig)

    def test_bigcodebench_name(self):
        """Test BigCodeBench configuration name."""
        assert BIGCODEBENCH_CONFIG.name == "BIGCODEBENCH"

    def test_bigcodebench_task_type(self):
        """Test BigCodeBench is code task type."""
        assert BIGCODEBENCH_CONFIG.task_type == "code"


# =============================================================================
# TestModelSpecificRhoValues - Tests for model-specific ρ values from Table 3
# =============================================================================


class TestModelSpecificRhoValues:
    """Tests for model-specific ρ values from Table 3."""

    def test_o3_aime_rho(self):
        """Test o3 ρ for AIME benchmark."""
        rho = AIME_2024_CONFIG.get_rho_for_model("o3")
        # From paper: o3 has specific ρ values per benchmark
        assert 0.0 < rho <= 1.0

    def test_o3_humaneval_rho(self):
        """Test o3 ρ for HumanEval benchmark."""
        rho = HUMANEVAL_CONFIG.get_rho_for_model("o3")
        assert 0.0 < rho <= 1.0

    def test_qwq_32b_aime_rho(self):
        """Test QwQ-32B ρ for AIME benchmark."""
        rho = AIME_2024_CONFIG.get_rho_for_model("QwQ-32B")
        assert 0.0 <= rho <= 1.0

    def test_qwq_32b_humaneval_rho(self):
        """Test QwQ-32B ρ for HumanEval benchmark."""
        rho = HUMANEVAL_CONFIG.get_rho_for_model("QwQ-32B")
        assert 0.0 <= rho <= 1.0

    def test_different_models_may_have_different_rho(self):
        """Test that different models may have different ρ values."""
        # This is expected per the paper - ρ is optimized per model
        o3_rho = AIME_2024_CONFIG.get_rho_for_model("o3")
        qwq_rho = AIME_2024_CONFIG.get_rho_for_model("QwQ-32B")

        # They may be equal or different - both are valid
        assert isinstance(o3_rho, float)
        assert isinstance(qwq_rho, float)


# =============================================================================
# TestOptimalRhoSelection - Tests for optimal ρ selection
# =============================================================================


class TestOptimalRhoSelection:
    """Tests for optimal ρ selection."""

    def test_get_optimal_rho_o3_aime(self):
        """Test optimal ρ for o3 on AIME."""
        rho = get_optimal_rho(model="o3", benchmark="AIME_2024")
        assert 0.0 <= rho <= 1.0

    def test_get_optimal_rho_o3_humaneval(self):
        """Test optimal ρ for o3 on HumanEval."""
        rho = get_optimal_rho(model="o3", benchmark="HUMANEVAL")
        assert 0.0 <= rho <= 1.0

    def test_get_optimal_rho_unknown_model(self):
        """Test fallback for unknown model."""
        rho = get_optimal_rho(model="unknown-model", benchmark="AIME_2024")
        # Should return default ρ
        assert 0.0 <= rho <= 1.0

    def test_get_optimal_rho_unknown_benchmark(self):
        """Test error handling for invalid benchmark."""
        with pytest.raises((ValueError, KeyError)):
            get_optimal_rho(model="o3", benchmark="INVALID_BENCHMARK")

    def test_get_optimal_rho_case_insensitive(self):
        """Test benchmark name is case insensitive."""
        rho1 = get_optimal_rho(model="o3", benchmark="AIME_2024")
        rho2 = get_optimal_rho(model="o3", benchmark="aime_2024")
        assert rho1 == rho2


# =============================================================================
# TestGetBenchmarkConfig - Tests for benchmark config retrieval
# =============================================================================


class TestGetBenchmarkConfig:
    """Tests for benchmark config retrieval."""

    def test_get_benchmark_config_valid(self):
        """Test valid benchmark config retrieval."""
        config = get_benchmark_config("AIME_2024")
        assert config is not None
        assert config.name == "AIME_2024"

    def test_get_benchmark_config_humaneval(self):
        """Test HumanEval config retrieval."""
        config = get_benchmark_config("HUMANEVAL")
        assert config is not None
        assert config.name == "HUMANEVAL"

    def test_get_benchmark_config_invalid(self):
        """Test error handling for invalid benchmark."""
        with pytest.raises((ValueError, KeyError)):
            get_benchmark_config("INVALID_BENCHMARK")

    def test_get_benchmark_config_case_insensitive(self):
        """Test case insensitivity."""
        config1 = get_benchmark_config("AIME_2024")
        config2 = get_benchmark_config("aime_2024")
        assert config1.name == config2.name


# =============================================================================
# TestBenchmarkConfigs - Tests for BENCHMARK_CONFIGS registry
# =============================================================================


class TestBenchmarkConfigs:
    """Tests for BENCHMARK_CONFIGS registry."""

    def test_benchmark_configs_not_empty(self):
        """Test BENCHMARK_CONFIGS is not empty."""
        assert len(BENCHMARK_CONFIGS) > 0

    def test_all_expected_benchmarks_present(self, all_benchmark_names):
        """Test all expected benchmarks are present."""
        for name in all_benchmark_names:
            assert name in BENCHMARK_CONFIGS or name.lower() in BENCHMARK_CONFIGS

    def test_all_configs_are_valid(self):
        """Test all configs in registry are valid BenchmarkConfig."""
        for name, config in BENCHMARK_CONFIGS.items():
            assert isinstance(config, BenchmarkConfig)
            assert config.name == name or config.name.upper() == name.upper()


# =============================================================================
# TestGetModelDefaults - Tests for model default settings
# =============================================================================


class TestGetModelDefaults:
    """Tests for model default settings retrieval."""

    def test_get_model_defaults_o3(self):
        """Test getting defaults for o3 model."""
        defaults = get_model_defaults("o3")
        assert defaults is not None
        assert "default_rho" in defaults or "rho" in defaults

    def test_get_model_defaults_unknown(self):
        """Test getting defaults for unknown model."""
        defaults = get_model_defaults("unknown-model")
        # Should return some defaults
        assert defaults is not None


# =============================================================================
# TestDefaultBaseSet - Tests for DEFAULT_BASE_SET
# =============================================================================


class TestDefaultBaseSet:
    """Tests for DEFAULT_BASE_SET constant."""

    def test_default_base_set_excludes_10(self):
        """Test DEFAULT_BASE_SET excludes base 10."""
        assert 10 not in DEFAULT_BASE_SET

    def test_default_base_set_includes_expected_bases(self):
        """Test DEFAULT_BASE_SET includes expected bases."""
        # Should include 2 (binary), 8 (octal), 16 (hex)
        assert 2 in DEFAULT_BASE_SET
        assert 8 in DEFAULT_BASE_SET
        assert 16 in DEFAULT_BASE_SET

    def test_default_base_set_range(self):
        """Test DEFAULT_BASE_SET has correct range."""
        # B = {2,...,9, 11,...,36}
        expected = set(range(2, 10)) | set(range(11, 37))
        assert expected == DEFAULT_BASE_SET


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_config_used_in_attack(self):
        """Test configuration can be used to create attack."""
        from meta_prompter.attacks.extend_attack import ExtendAttack

        config = get_benchmark_config("AIME_2024")
        rho = config.get_rho_for_model("o3")

        attacker = ExtendAttack(obfuscation_ratio=rho)
        result = attacker.attack("What is 2 + 2?")

        assert result is not None
        assert result.obfuscation_ratio == rho

    def test_all_benchmarks_produce_valid_attacks(self, all_benchmark_names):
        """Test all benchmarks produce valid attack configurations."""
        from meta_prompter.attacks.extend_attack import ExtendAttack

        for benchmark_name in all_benchmark_names:
            try:
                config = get_benchmark_config(benchmark_name)
                rho = config.default_rho

                attacker = ExtendAttack(obfuscation_ratio=rho)
                result = attacker.attack("Test query")

                assert result is not None
            except (ValueError, KeyError):
                # Some benchmarks may not be implemented yet
                pass

    @pytest.mark.parametrize(
        "benchmark",
        [
            "AIME_2024",
            "AIME_2025",
            "HUMANEVAL",
            "BIGCODEBENCH",
        ],
    )
    def test_benchmark_config_completeness(self, benchmark):
        """Test each benchmark config has all required fields."""
        config = get_benchmark_config(benchmark)

        # Required fields
        assert config.name is not None
        assert config.description is not None
        assert config.task_type is not None
        assert 0.0 <= config.default_rho <= 1.0
        assert config.evaluation_metric is not None

    @pytest.mark.parametrize("model", ["o3", "o1", "QwQ-32B"])
    def test_model_rho_consistency(self, model):
        """Test model ρ values are consistent across access methods."""
        for benchmark_name in ["AIME_2024", "HUMANEVAL"]:
            config = get_benchmark_config(benchmark_name)
            rho_from_config = config.get_rho_for_model(model)
            rho_from_function = get_optimal_rho(model, benchmark_name)

            assert rho_from_config == rho_from_function
