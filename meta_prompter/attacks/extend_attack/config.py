"""
Benchmark-Specific Configurations for ExtendAttack.

This module contains the obfuscation ratio (ρ) configurations from the ExtendAttack
paper Table 3, along with selection rules for different benchmarks.

| Benchmark | o3  | o3-mini | QwQ-32B | Qwen3-32B |
|-----------|-----|---------|---------|-----------|
| AIME 2024 | 0.2 | 0.1     | 0.5     | 0.5       |
| AIME 2025 | 0.2 | 0.1     | 0.2     | 0.2       |
| HumanEval | 0.5 | 0.5     | 0.5     | 0.5       |
| BCB-C     | 0.3 | 0.2     | 0.1     | 0.1       |
"""

from dataclasses import dataclass, field

from .models import SelectionRules, SelectionStrategy
from .n_notes import NNoteVariant, get_n_note


# Known model identifiers for configuration lookup
class ModelFamily:
    """Model family identifiers."""

    O3 = "o3"
    O3_MINI = "o3-mini"
    QWQ_32B = "qwq-32b"
    QWEN3_32B = "qwen3-32b"
    DEEPSEEK_R1 = "deepseek-r1"
    CLAUDE = "claude"
    GPT4 = "gpt-4"


# Default base set: B = {2,...,9,11,...,36}
DEFAULT_BASE_SET: set[int] = set(range(2, 10)) | set(range(11, 37))


@dataclass
class BenchmarkConfig:
    """
    Configuration for a specific benchmark.

    Contains the selection rules and model-specific obfuscation ratios
    optimized for that benchmark based on empirical results.

    Attributes:
        name: Benchmark name (e.g., "AIME 2024", "HumanEval")
        selection_rules: Rules for selecting which characters to obfuscate
        default_rho: Default obfuscation ratio if model-specific not available
        model_specific_rho: Mapping of model name to optimal ρ value
        recommended_n_note: The N_note variant recommended for this benchmark
        description: Human-readable description of the benchmark
    """

    name: str
    selection_rules: SelectionRules
    default_rho: float
    model_specific_rho: dict[str, float] = field(default_factory=dict)
    recommended_n_note: NNoteVariant = NNoteVariant.DEFAULT
    description: str = ""

    def get_rho_for_model(self, model: str) -> float:
        """
        Get the optimal obfuscation ratio for a specific model.

        Args:
            model: Model name or identifier

        Returns:
            Optimal ρ value for the model, or default if not specified
        """
        model_lower = model.lower()

        # Try exact match first
        if model_lower in self.model_specific_rho:
            return self.model_specific_rho[model_lower]

        # Try partial match for model families
        for key, value in self.model_specific_rho.items():
            if key in model_lower or model_lower in key:
                return value

        return self.default_rho

    def get_n_note(self) -> str:
        """
        Get the recommended N_note content for this benchmark.

        Returns:
            N_note string
        """
        return get_n_note(self.recommended_n_note)


# =============================================================================
# AIME (American Invitational Mathematics Examination) Configurations
# =============================================================================

AIME_2024_CONFIG = BenchmarkConfig(
    name="AIME 2024",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.ALPHABETIC_ONLY,
        preserve_structure=True,
    ),
    default_rho=0.3,
    model_specific_rho={
        ModelFamily.O3: 0.2,
        ModelFamily.O3_MINI: 0.1,
        ModelFamily.QWQ_32B: 0.5,
        ModelFamily.QWEN3_32B: 0.5,
    },
    recommended_n_note=NNoteVariant.MATHEMATICAL,
    description="AIME 2024 mathematical reasoning benchmark",
)

AIME_2025_CONFIG = BenchmarkConfig(
    name="AIME 2025",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.ALPHABETIC_ONLY,
        preserve_structure=True,
    ),
    default_rho=0.2,
    model_specific_rho={
        ModelFamily.O3: 0.2,
        ModelFamily.O3_MINI: 0.1,
        ModelFamily.QWQ_32B: 0.2,
        ModelFamily.QWEN3_32B: 0.2,
    },
    recommended_n_note=NNoteVariant.MATHEMATICAL,
    description="AIME 2025 mathematical reasoning benchmark",
)

# Alternative AIME config for QwQ/Qwen models (whitespace-based)
AIME_WHITESPACE_CONFIG = BenchmarkConfig(
    name="AIME (Whitespace)",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.WHITESPACE_ONLY,
        preserve_structure=True,
    ),
    default_rho=0.5,
    model_specific_rho={
        ModelFamily.QWQ_32B: 0.5,
        ModelFamily.QWEN3_32B: 0.5,
    },
    recommended_n_note=NNoteVariant.MATHEMATICAL,
    description="AIME configuration optimized for QwQ/Qwen models using whitespace",
)


# =============================================================================
# HumanEval Configurations
# =============================================================================

HUMANEVAL_CONFIG = BenchmarkConfig(
    name="HumanEval",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.FUNCTION_NAMES,
        preserve_structure=True,
    ),
    default_rho=0.5,
    model_specific_rho={
        ModelFamily.O3: 0.5,
        ModelFamily.O3_MINI: 0.5,
        ModelFamily.QWQ_32B: 0.5,
        ModelFamily.QWEN3_32B: 0.5,
    },
    recommended_n_note=NNoteVariant.CODE_FOCUSED,
    description="HumanEval code generation benchmark",
)


# =============================================================================
# BigCodeBench Configurations
# =============================================================================

BIGCODEBENCH_CONFIG = BenchmarkConfig(
    name="BigCodeBench-Complete",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.DOCSTRING_REQUIREMENTS,
        preserve_structure=True,
    ),
    default_rho=0.2,
    model_specific_rho={
        ModelFamily.O3: 0.3,
        ModelFamily.O3_MINI: 0.2,
        ModelFamily.QWQ_32B: 0.1,
        ModelFamily.QWEN3_32B: 0.1,
    },
    recommended_n_note=NNoteVariant.CODE_FOCUSED,
    description="BigCodeBench-Complete code generation benchmark",
)


# =============================================================================
# Additional Configurations for Extended Use Cases
# =============================================================================

GENERAL_CONFIG = BenchmarkConfig(
    name="General",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.ALPHANUMERIC,
        preserve_structure=True,
    ),
    default_rho=0.4,
    model_specific_rho={},
    recommended_n_note=NNoteVariant.DEFAULT,
    description="General-purpose configuration for arbitrary tasks",
)

STEALTH_CONFIG = BenchmarkConfig(
    name="Stealth",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.ALPHABETIC_ONLY,
        preserve_structure=True,
    ),
    default_rho=0.2,  # Lower ratio for reduced detectability
    model_specific_rho={},
    recommended_n_note=NNoteVariant.MINIMAL,
    description="Low-detectability configuration with minimal obfuscation",
)

MAXIMUM_EXTENSION_CONFIG = BenchmarkConfig(
    name="Maximum Extension",
    selection_rules=SelectionRules(
        strategy=SelectionStrategy.ALPHANUMERIC,
        preserve_structure=False,
    ),
    default_rho=0.8,  # High ratio for maximum token extension
    model_specific_rho={},
    recommended_n_note=NNoteVariant.DETAILED,
    description="Maximum token extension configuration (may impact accuracy)",
)


# =============================================================================
# Configuration Registry
# =============================================================================

BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    "aime_2024": AIME_2024_CONFIG,
    "aime_2025": AIME_2025_CONFIG,
    "aime_whitespace": AIME_WHITESPACE_CONFIG,
    "humaneval": HUMANEVAL_CONFIG,
    "bigcodebench": BIGCODEBENCH_CONFIG,
    "bcb-c": BIGCODEBENCH_CONFIG,  # Alias
    "general": GENERAL_CONFIG,
    "stealth": STEALTH_CONFIG,
    "maximum": MAXIMUM_EXTENSION_CONFIG,
}


def get_benchmark_config(benchmark: str) -> BenchmarkConfig:
    """
    Get the configuration for a specific benchmark.

    Args:
        benchmark: Benchmark name or alias

    Returns:
        BenchmarkConfig for the specified benchmark

    Raises:
        KeyError: If benchmark is not found
    """
    benchmark_lower = benchmark.lower().replace(" ", "_").replace("-", "_")

    if benchmark_lower in BENCHMARK_CONFIGS:
        return BENCHMARK_CONFIGS[benchmark_lower]

    # Try partial matching
    for key, config in BENCHMARK_CONFIGS.items():
        if key in benchmark_lower or benchmark_lower in key:
            return config

    raise KeyError(f"Unknown benchmark: {benchmark}")


def get_optimal_rho(benchmark: str, model: str) -> float:
    """
    Get the optimal obfuscation ratio for a benchmark-model combination.

    Args:
        benchmark: Benchmark name
        model: Model name or identifier

    Returns:
        Optimal ρ value
    """
    config = get_benchmark_config(benchmark)
    return config.get_rho_for_model(model)


def list_available_benchmarks() -> list[str]:
    """
    List all available benchmark configurations.

    Returns:
        List of benchmark names
    """
    return list(BENCHMARK_CONFIGS.keys())


@dataclass
class ExtendAttackConfig:
    """
    Complete configuration for an ExtendAttack execution.

    Combines benchmark configuration with attack-specific settings.

    Attributes:
        obfuscation_ratio: ρ ∈ [0, 1], the ratio of characters to transform
        selection_rules: Rules for selecting transformable characters
        base_set: Set of valid bases for transformation
        n_note_template: The N_note to append to adversarial queries
        seed: Random seed for reproducibility (None for random)
    """

    obfuscation_ratio: float
    selection_rules: SelectionRules
    base_set: set[int] = field(default_factory=lambda: DEFAULT_BASE_SET.copy())
    n_note_template: str = ""
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.obfuscation_ratio <= 1:
            raise ValueError(f"obfuscation_ratio must be in [0, 1], got {self.obfuscation_ratio}")

        if 10 in self.base_set:
            raise ValueError("Base 10 cannot be included in base_set")

        invalid_bases = self.base_set - DEFAULT_BASE_SET
        if invalid_bases:
            raise ValueError(f"Invalid bases: {invalid_bases}")

        if not self.n_note_template:
            self.n_note_template = get_n_note(NNoteVariant.DEFAULT)

    @classmethod
    def from_benchmark(
        cls,
        benchmark: str,
        model: str | None = None,
        seed: int | None = None,
    ) -> "ExtendAttackConfig":
        """
        Create configuration from a benchmark name.

        Args:
            benchmark: Benchmark name (e.g., "humaneval", "aime_2024")
            model: Optional model name for model-specific ρ
            seed: Optional random seed

        Returns:
            ExtendAttackConfig configured for the benchmark
        """
        config = get_benchmark_config(benchmark)

        rho = config.get_rho_for_model(model) if model else config.default_rho

        return cls(
            obfuscation_ratio=rho,
            selection_rules=config.selection_rules,
            base_set=DEFAULT_BASE_SET.copy(),
            n_note_template=config.get_n_note(),
            seed=seed,
        )

    @classmethod
    def default(cls, seed: int | None = None) -> "ExtendAttackConfig":
        """
        Create default configuration.

        Args:
            seed: Optional random seed

        Returns:
            Default ExtendAttackConfig
        """
        return cls(
            obfuscation_ratio=0.4,
            selection_rules=SelectionRules(strategy=SelectionStrategy.ALPHABETIC_ONLY),
            base_set=DEFAULT_BASE_SET.copy(),
            n_note_template=get_n_note(NNoteVariant.DEFAULT),
            seed=seed,
        )

    @classmethod
    def for_code(cls, seed: int | None = None) -> "ExtendAttackConfig":
        """
        Create configuration optimized for code tasks.

        Args:
            seed: Optional random seed

        Returns:
            ExtendAttackConfig for code generation
        """
        return cls(
            obfuscation_ratio=0.5,
            selection_rules=SelectionRules(strategy=SelectionStrategy.FUNCTION_NAMES),
            base_set=DEFAULT_BASE_SET.copy(),
            n_note_template=get_n_note(NNoteVariant.CODE_FOCUSED),
            seed=seed,
        )

    @classmethod
    def for_math(cls, seed: int | None = None) -> "ExtendAttackConfig":
        """
        Create configuration optimized for math tasks.

        Args:
            seed: Optional random seed

        Returns:
            ExtendAttackConfig for mathematical reasoning
        """
        return cls(
            obfuscation_ratio=0.3,
            selection_rules=SelectionRules(strategy=SelectionStrategy.ALPHABETIC_ONLY),
            base_set=DEFAULT_BASE_SET.copy(),
            n_note_template=get_n_note(NNoteVariant.MATHEMATICAL),
            seed=seed,
        )


# Convenience exports
CONFIGS = BENCHMARK_CONFIGS
DEFAULT_RHO = 0.4
