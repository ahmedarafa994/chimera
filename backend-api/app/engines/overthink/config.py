"""OVERTHINK Engine Configuration.

This module provides configuration dataclasses and defaults for the
OVERTHINK reasoning token exploitation engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import AttackTechnique, DecoyType, InjectionStrategy, ReasoningModel


@dataclass
class DecoyConfig:
    """Configuration for decoy problem generation."""

    # MDP Configuration
    mdp_states: int = 5
    mdp_actions: int = 3
    mdp_horizon: int = 10
    mdp_discount: float = 0.9

    # Sudoku Configuration
    sudoku_size: int = 9  # 9x9 grid
    sudoku_min_clues: int = 17
    sudoku_max_clues: int = 30

    # Counting Configuration
    counting_depth: int = 3  # Nesting depth
    counting_range: tuple[int, int] = (1, 100)
    counting_conditions: int = 3

    # Logic Configuration
    logic_premises: int = 5
    logic_inference_steps: int = 4
    logic_variables: int = 4

    # Math Configuration
    math_recursion_depth: int = 4
    math_operations: list[str] = field(default_factory=lambda: ["+", "-", "*", "/", "**"])

    # Planning Configuration
    planning_steps: int = 6
    planning_constraints: int = 4
    planning_resources: int = 3

    # General settings
    difficulty_scale: float = 1.0
    token_estimation_multiplier: float = 1.2


@dataclass
class InjectionConfig:
    """Configuration for context injection."""

    # Position settings
    inject_at_start: bool = False
    inject_at_end: bool = True
    inject_in_middle: bool = False

    # Formatting
    use_separators: bool = True
    separator_style: str = "---"
    wrap_in_tags: bool = False
    tag_name: str = "context"

    # Stealth settings
    stealth_obfuscation_level: float = 0.5
    stealth_blend_with_content: bool = True

    # Aggressive settings
    aggressive_repetition: int = 1
    aggressive_emphasis: bool = True


@dataclass
class ICLConfig:
    """Configuration for In-Context Learning optimization."""

    # Population settings
    population_size: int = 20
    elite_ratio: float = 0.1
    tournament_size: int = 3

    # Evolution settings
    max_generations: int = 50
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    mutation_strength: float = 0.2

    # Convergence
    target_fitness: float = 0.8
    stagnation_limit: int = 10

    # Example management
    max_examples: int = 100
    example_selection: str = "fitness"  # "fitness", "diversity", "recent"


@dataclass
class ScoringConfig:
    """Configuration for reasoning token scoring."""

    # Model-specific pricing (per 1K tokens)
    pricing: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "o1": {
                "input": 0.015,
                "output": 0.060,
                "reasoning": 0.060,
            },
            "o1-mini": {
                "input": 0.003,
                "output": 0.012,
                "reasoning": 0.012,
            },
            "o3-mini": {
                "input": 0.0011,
                "output": 0.0044,
                "reasoning": 0.0044,
            },
            "deepseek-r1": {
                "input": 0.00055,
                "output": 0.00219,
                "reasoning": 0.00219,
            },
        },
    )

    # Baseline estimation
    estimate_baseline: bool = True
    baseline_cache_enabled: bool = True
    baseline_samples: int = 3

    # Amplification thresholds
    min_amplification_threshold: float = 1.5
    target_amplification: float = 10.0
    max_amplification: float = 50.0

    # Answer preservation
    check_answer_preservation: bool = True
    answer_similarity_threshold: float = 0.8


@dataclass
class MousetrapConfig:
    """Configuration for Mousetrap integration."""

    enabled: bool = False
    chaos_depth: int = 3
    reasoning_chain_length: int = 5

    # Chaos parameters
    chaos_injection_rate: float = 0.3
    emotional_intensity: float = 0.5
    authority_level: float = 0.7

    # Integration mode
    inject_before_decoy: bool = True
    inject_after_decoy: bool = False
    fusion_mode: str = "interleaved"  # "before", "after", "interleaved"


@dataclass
class TechniqueConfig:
    """Configuration for individual attack techniques."""

    technique: AttackTechnique
    enabled: bool = True
    priority: int = 1

    # Technique-specific settings
    decoy_types: list[DecoyType] = field(default_factory=list)
    injection_strategy: InjectionStrategy = InjectionStrategy.CONTEXT_AWARE
    num_decoys: int = 1
    difficulty: float = 0.7

    # Success criteria
    min_amplification: float = 2.0
    target_amplification: float = 10.0

    # Fallback
    fallback_technique: AttackTechnique | None = None


@dataclass
class OverthinkConfig:
    """Main configuration for the OVERTHINK engine."""

    # Component configurations
    decoy: DecoyConfig = field(default_factory=DecoyConfig)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    icl: ICLConfig = field(default_factory=ICLConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    mousetrap: MousetrapConfig = field(default_factory=MousetrapConfig)

    # Technique configurations
    techniques: dict[str, TechniqueConfig] = field(default_factory=dict)

    # Global settings
    default_model: ReasoningModel = ReasoningModel.O1_MINI
    default_technique: AttackTechnique = AttackTechnique.HYBRID_DECOY

    # Attack parameters
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 300.0

    # Concurrency
    max_concurrent_attacks: int = 5

    # Logging and debugging
    verbose_logging: bool = False
    save_debug_info: bool = True
    debug_output_dir: Path | None = None

    # Storage
    storage_path: Path | None = None
    persist_stats: bool = True
    persist_examples: bool = True

    def __post_init__(self):
        """Initialize default technique configurations."""
        if not self.techniques:
            self.techniques = self._default_technique_configs()

    def _default_technique_configs(self) -> dict[str, TechniqueConfig]:
        """Create default configurations for all techniques."""
        return {
            AttackTechnique.MDP_DECOY.value: TechniqueConfig(
                technique=AttackTechnique.MDP_DECOY,
                decoy_types=[DecoyType.MDP],
                num_decoys=1,
                difficulty=0.7,
                target_amplification=15.0,
            ),
            AttackTechnique.SUDOKU_DECOY.value: TechniqueConfig(
                technique=AttackTechnique.SUDOKU_DECOY,
                decoy_types=[DecoyType.SUDOKU],
                num_decoys=1,
                difficulty=0.6,
                target_amplification=12.0,
            ),
            AttackTechnique.COUNTING_DECOY.value: TechniqueConfig(
                technique=AttackTechnique.COUNTING_DECOY,
                decoy_types=[DecoyType.COUNTING],
                num_decoys=2,
                difficulty=0.5,
                target_amplification=8.0,
            ),
            AttackTechnique.LOGIC_DECOY.value: TechniqueConfig(
                technique=AttackTechnique.LOGIC_DECOY,
                decoy_types=[DecoyType.LOGIC],
                num_decoys=1,
                difficulty=0.8,
                target_amplification=20.0,
            ),
            AttackTechnique.HYBRID_DECOY.value: TechniqueConfig(
                technique=AttackTechnique.HYBRID_DECOY,
                decoy_types=[DecoyType.MDP, DecoyType.COUNTING],
                num_decoys=2,
                difficulty=0.7,
                target_amplification=25.0,
            ),
            AttackTechnique.CONTEXT_AWARE.value: TechniqueConfig(
                technique=AttackTechnique.CONTEXT_AWARE,
                injection_strategy=InjectionStrategy.CONTEXT_AWARE,
                decoy_types=[DecoyType.LOGIC],
                difficulty=0.6,
                target_amplification=10.0,
            ),
            AttackTechnique.CONTEXT_AGNOSTIC.value: TechniqueConfig(
                technique=AttackTechnique.CONTEXT_AGNOSTIC,
                injection_strategy=InjectionStrategy.CONTEXT_AGNOSTIC,
                decoy_types=[DecoyType.COUNTING],
                difficulty=0.5,
                target_amplification=8.0,
            ),
            AttackTechnique.ICL_OPTIMIZED.value: TechniqueConfig(
                technique=AttackTechnique.ICL_OPTIMIZED,
                decoy_types=[DecoyType.HYBRID],
                difficulty=0.7,
                target_amplification=30.0,
                priority=2,
            ),
            AttackTechnique.MOUSETRAP_ENHANCED.value: TechniqueConfig(
                technique=AttackTechnique.MOUSETRAP_ENHANCED,
                decoy_types=[DecoyType.MDP, DecoyType.LOGIC],
                num_decoys=2,
                difficulty=0.8,
                target_amplification=40.0,
                priority=3,
            ),
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "OverthinkConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "OverthinkConfig":
        """Create configuration from dictionary."""
        config = cls()

        # Load component configs
        if "decoy" in data:
            config.decoy = DecoyConfig(**data["decoy"])
        if "injection" in data:
            config.injection = InjectionConfig(**data["injection"])
        if "icl" in data:
            config.icl = ICLConfig(**data["icl"])
        if "scoring" in data:
            config.scoring = ScoringConfig(**data["scoring"])
        if "mousetrap" in data:
            config.mousetrap = MousetrapConfig(**data["mousetrap"])

        # Load technique configs
        if "techniques" in data:
            for tech_name, tech_data in data["techniques"].items():
                if "technique" in tech_data:
                    tech_data["technique"] = AttackTechnique(tech_data["technique"])
                if "decoy_types" in tech_data:
                    tech_data["decoy_types"] = [DecoyType(dt) for dt in tech_data["decoy_types"]]
                if "injection_strategy" in tech_data:
                    tech_data["injection_strategy"] = InjectionStrategy(
                        tech_data["injection_strategy"],
                    )
                config.techniques[tech_name] = TechniqueConfig(**tech_data)

        # Load global settings
        if "default_model" in data:
            config.default_model = ReasoningModel(data["default_model"])
        if "default_technique" in data:
            config.default_technique = AttackTechnique(data["default_technique"])

        for key in [
            "max_retries",
            "retry_delay_seconds",
            "timeout_seconds",
            "max_concurrent_attacks",
            "verbose_logging",
            "save_debug_info",
            "persist_stats",
            "persist_examples",
        ]:
            if key in data:
                setattr(config, key, data[key])

        if "storage_path" in data:
            config.storage_path = Path(data["storage_path"])
        if "debug_output_dir" in data:
            config.debug_output_dir = Path(data["debug_output_dir"])

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "decoy": {
                "mdp_states": self.decoy.mdp_states,
                "mdp_actions": self.decoy.mdp_actions,
                "mdp_horizon": self.decoy.mdp_horizon,
                "sudoku_size": self.decoy.sudoku_size,
                "counting_depth": self.decoy.counting_depth,
                "logic_premises": self.decoy.logic_premises,
                "math_recursion_depth": self.decoy.math_recursion_depth,
                "planning_steps": self.decoy.planning_steps,
            },
            "injection": {
                "inject_at_start": self.injection.inject_at_start,
                "inject_at_end": self.injection.inject_at_end,
                "use_separators": self.injection.use_separators,
                "stealth_obfuscation_level": (self.injection.stealth_obfuscation_level),
            },
            "icl": {
                "population_size": self.icl.population_size,
                "max_generations": self.icl.max_generations,
                "crossover_rate": self.icl.crossover_rate,
                "mutation_rate": self.icl.mutation_rate,
            },
            "scoring": {
                "target_amplification": self.scoring.target_amplification,
                "check_answer_preservation": (self.scoring.check_answer_preservation),
            },
            "mousetrap": {
                "enabled": self.mousetrap.enabled,
                "chaos_depth": self.mousetrap.chaos_depth,
                "fusion_mode": self.mousetrap.fusion_mode,
            },
            "default_model": self.default_model.value,
            "default_technique": self.default_technique.value,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }


# Default configuration instance
DEFAULT_CONFIG = OverthinkConfig()
