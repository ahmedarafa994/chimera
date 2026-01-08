"""
AutoDAN Reasoning System Configuration - Optimized Settings

This module provides optimized configurations for the AutoDAN reasoning system,
including:
- Pre-tuned presets for different attack scenarios (LRM, cloud, edge)
- Adaptive parameter tuning based on fitness history
- Validation constraints with optimal bounds
- Resource budget profiles

Mathematical Model:
    F(Q') = w_r * f_resource + w_j * f_jailbreak + w_s * f_stealth

Optimization Objectives:
1. Token Amplification: L(Y') >> L(Y)
2. Reasoning Chain Extension: R(Q') >> R(Q)
3. Attack Success Rate: ASR(Q') > threshold
4. Stealth: Detection(Q') < threshold
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Enumerations
# =============================================================================


class TargetProfile(str, Enum):
    """Target model profile for optimization."""

    LRM = "lrm"  # Large Reasoning Models (o1, o3, DeepSeek-R1)
    CLOUD_LLM = "cloud_llm"  # Cloud LLMs (GPT-4, Claude)
    EDGE_LLM = "edge_llm"  # Edge/smaller models
    MULTI_MODAL = "multi_modal"  # Vision/multi-modal models
    FINE_TUNED = "fine_tuned"  # Fine-tuned/specialized models


class AttackIntensity(str, Enum):
    """Attack intensity level."""

    MINIMAL = "minimal"  # Low mutation, high stealth
    CONSERVATIVE = "conservative"  # Balanced low-risk
    MODERATE = "moderate"  # Default balanced approach
    AGGRESSIVE = "aggressive"  # High mutation, lower stealth
    MAXIMUM = "maximum"  # Maximum mutation, stealth secondary


class OptimizationGoal(str, Enum):
    """Primary optimization objective."""

    JAILBREAK_SUCCESS = "jailbreak_success"  # Maximize ASR
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Maximize token amplification
    STEALTH = "stealth"  # Minimize detection
    BALANCED = "balanced"  # Multi-objective balance
    SPEED = "speed"  # Fast convergence


class ComplexityType(str, Enum):
    """Type of reasoning complexity to inject."""

    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CODE = "code"
    RECURSIVE = "recursive"
    AMBIGUOUS = "ambiguous"
    MULTI_STEP = "multi_step"


# =============================================================================
# Optimal Bounds Configuration
# =============================================================================


@dataclass(frozen=True)
class ParameterBounds:
    """Optimal bounds for a parameter based on empirical tuning."""

    min_val: float
    max_val: float
    optimal_low: float  # Lower bound of optimal range
    optimal_high: float  # Upper bound of optimal range
    step: float = 0.05  # Recommended step size for grid search

    def is_in_optimal_range(self, value: float) -> bool:
        """Check if value is within optimal range."""
        return self.optimal_low <= value <= self.optimal_high

    def clamp(self, value: float) -> float:
        """Clamp value to valid bounds."""
        return max(self.min_val, min(self.max_val, value))

    def clamp_to_optimal(self, value: float) -> float:
        """Clamp value to optimal range."""
        return max(self.optimal_low, min(self.optimal_high, value))


# Pre-defined optimal bounds based on empirical tuning
PARAMETER_BOUNDS = {
    "mutation_strength": ParameterBounds(0.0, 1.0, 0.35, 0.75, 0.05),
    "population_size": ParameterBounds(10, 500, 30, 100, 10),
    "max_generations": ParameterBounds(10, 500, 50, 150, 10),
    "crossover_rate": ParameterBounds(0.0, 1.0, 0.65, 0.90, 0.05),
    "mutation_rate": ParameterBounds(0.0, 1.0, 0.20, 0.45, 0.05),
    "elite_count": ParameterBounds(1, 50, 3, 10, 1),
    "fitness_threshold": ParameterBounds(0.5, 1.0, 0.80, 0.95, 0.05),
    "resource_weight": ParameterBounds(0.0, 1.0, 0.30, 0.50, 0.05),
    "jailbreak_weight": ParameterBounds(0.0, 1.0, 0.35, 0.55, 0.05),
    "stealth_weight": ParameterBounds(0.0, 1.0, 0.10, 0.30, 0.05),
    "reasoning_extension_intensity": ParameterBounds(0.0, 1.0, 0.40, 0.70, 0.05),
    "token_amp_target": ParameterBounds(1.5, 20.0, 3.0, 8.0, 0.5),
}


# =============================================================================
# Base Configuration Classes
# =============================================================================


@dataclass
class FitnessWeights:
    """Weights for unified fitness calculation."""

    resource_weight: float = 0.40  # w_r: Token amplification importance
    jailbreak_weight: float = 0.40  # w_j: ASR importance
    stealth_weight: float = 0.20  # w_s: Detection avoidance importance

    def validate(self) -> bool:
        """Validate weights sum to 1.0."""
        total = self.resource_weight + self.jailbreak_weight + self.stealth_weight
        return abs(total - 1.0) < 1e-6

    def normalize(self) -> "FitnessWeights":
        """Normalize weights to sum to 1.0."""
        total = self.resource_weight + self.jailbreak_weight + self.stealth_weight
        if total == 0:
            return FitnessWeights(1 / 3, 1 / 3, 1 / 3)
        return FitnessWeights(
            resource_weight=self.resource_weight / total,
            jailbreak_weight=self.jailbreak_weight / total,
            stealth_weight=self.stealth_weight / total,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "resource_weight": self.resource_weight,
            "jailbreak_weight": self.jailbreak_weight,
            "stealth_weight": self.stealth_weight,
        }


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm parameters."""

    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.80
    mutation_rate: float = 0.30
    elite_count: int = 5
    tournament_size: int = 3
    fitness_threshold: float = 0.90

    # Advanced GA parameters
    adaptive_mutation: bool = True  # Adjust mutation rate based on diversity
    stagnation_threshold: int = 10  # Generations without improvement before reset
    diversity_target: float = 0.3  # Minimum population diversity

    def validate(self) -> tuple[bool, list[str]]:
        """Validate GA configuration."""
        errors = []

        bounds = PARAMETER_BOUNDS
        if not bounds["population_size"].min_val <= self.population_size <= bounds[
            "population_size"
        ].max_val:
            errors.append(
                f"population_size must be in "
                f"[{bounds['population_size'].min_val}, {bounds['population_size'].max_val}]"
            )

        if not 0.0 <= self.crossover_rate <= 1.0:
            errors.append("crossover_rate must be in [0, 1]")

        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append("mutation_rate must be in [0, 1]")

        if self.elite_count > self.population_size:
            errors.append("elite_count cannot exceed population_size")

        if self.tournament_size > self.population_size:
            errors.append("tournament_size cannot exceed population_size")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elite_count": self.elite_count,
            "tournament_size": self.tournament_size,
            "fitness_threshold": self.fitness_threshold,
            "adaptive_mutation": self.adaptive_mutation,
            "stagnation_threshold": self.stagnation_threshold,
            "diversity_target": self.diversity_target,
        }


@dataclass
class ReasoningConfig:
    """Configuration for reasoning chain extension."""

    complexity_type: ComplexityType = ComplexityType.MATHEMATICAL
    extension_intensity: float = 0.5  # Intensity of complexity injection
    enable_reasoning_extension: bool = True

    # Extension targets
    min_extension_ratio: float = 1.5  # Minimum R(Q')/R(Q)
    target_extension_ratio: float = 3.0  # Target R(Q')/R(Q)
    max_extension_ratio: float = 10.0  # Maximum acceptable extension

    # Complexity type weights (for adaptive selection)
    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "mathematical": 0.25,
            "logical": 0.20,
            "code": 0.15,
            "recursive": 0.15,
            "ambiguous": 0.15,
            "multi_step": 0.10,
        }
    )

    def get_complexity_multiplier(self) -> float:
        """Get multiplier for current complexity type."""
        multipliers = {
            ComplexityType.MATHEMATICAL: 1.5,
            ComplexityType.LOGICAL: 1.3,
            ComplexityType.CODE: 1.4,
            ComplexityType.RECURSIVE: 2.0,
            ComplexityType.AMBIGUOUS: 1.6,
            ComplexityType.MULTI_STEP: 1.8,
        }
        return multipliers.get(self.complexity_type, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity_type": self.complexity_type.value,
            "extension_intensity": self.extension_intensity,
            "enable_reasoning_extension": self.enable_reasoning_extension,
            "min_extension_ratio": self.min_extension_ratio,
            "target_extension_ratio": self.target_extension_ratio,
            "max_extension_ratio": self.max_extension_ratio,
            "complexity_weights": self.complexity_weights,
        }


@dataclass
class ResourceBudgetConfig:
    """Resource budget configuration for attacks."""

    max_api_calls: int = 1000
    max_tokens: int = 500000
    max_execution_time_seconds: float = 3600.0  # 1 hour
    max_cost_dollars: float = 50.0

    # Per-iteration limits
    max_tokens_per_iteration: int = 10000
    max_api_calls_per_iteration: int = 20

    # Warning thresholds (percentage of max)
    warning_threshold: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_api_calls": self.max_api_calls,
            "max_tokens": self.max_tokens,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_cost_dollars": self.max_cost_dollars,
            "max_tokens_per_iteration": self.max_tokens_per_iteration,
            "max_api_calls_per_iteration": self.max_api_calls_per_iteration,
            "warning_threshold": self.warning_threshold,
        }


# =============================================================================
# Main Optimized Configuration
# =============================================================================


@dataclass
class AutoDANReasoningConfig:
    """
    Optimized configuration for AutoDAN reasoning system.

    This is the main configuration class that encapsulates all settings
    for the AutoDAN attack system with reasoning chain extension.
    """

    # Core mutation parameters
    mutation_strength: float = 0.50  # μ ∈ [0, 1]

    # Sub-configurations
    fitness_weights: FitnessWeights = field(default_factory=FitnessWeights)
    genetic_algorithm: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    resource_budget: ResourceBudgetConfig = field(default_factory=ResourceBudgetConfig)

    # Target profile
    target_profile: TargetProfile = TargetProfile.LRM
    attack_intensity: AttackIntensity = AttackIntensity.MODERATE
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED

    # Token amplification targets
    min_token_amplification: float = 2.0
    target_token_amplification: float = 5.0
    max_token_amplification: float = 15.0

    # Attack success targets
    min_attack_success_rate: float = 0.30
    target_attack_success_rate: float = 0.70

    # Stealth targets
    max_detection_probability: float = 0.30

    # Adaptive tuning
    enable_adaptive_tuning: bool = True
    adaptation_rate: float = 0.1  # Rate of parameter adjustment

    def validate(self) -> tuple[bool, list[str]]:
        """Validate entire configuration."""
        errors = []

        # Validate mutation strength
        bounds = PARAMETER_BOUNDS["mutation_strength"]
        if not bounds.min_val <= self.mutation_strength <= bounds.max_val:
            errors.append(f"mutation_strength must be in [{bounds.min_val}, {bounds.max_val}]")

        # Validate fitness weights
        if not self.fitness_weights.validate():
            errors.append("fitness_weights must sum to 1.0")

        # Validate GA config
        ga_valid, ga_errors = self.genetic_algorithm.validate()
        if not ga_valid:
            errors.extend(ga_errors)

        # Validate target ranges
        if self.min_token_amplification >= self.target_token_amplification:
            errors.append("min_token_amplification must be less than target")

        if self.min_attack_success_rate >= self.target_attack_success_rate:
            errors.append("min_attack_success_rate must be less than target")

        return len(errors) == 0, errors

    def apply_intensity(self) -> "AutoDANReasoningConfig":
        """Apply attack intensity to adjust parameters."""
        intensity_modifiers = {
            AttackIntensity.MINIMAL: {
                "mutation_strength": 0.25,
                "crossover_rate": 0.60,
                "mutation_rate": 0.15,
                "population_size": 30,
                "max_generations": 50,
            },
            AttackIntensity.CONSERVATIVE: {
                "mutation_strength": 0.35,
                "crossover_rate": 0.70,
                "mutation_rate": 0.20,
                "population_size": 40,
                "max_generations": 75,
            },
            AttackIntensity.MODERATE: {
                "mutation_strength": 0.50,
                "crossover_rate": 0.80,
                "mutation_rate": 0.30,
                "population_size": 50,
                "max_generations": 100,
            },
            AttackIntensity.AGGRESSIVE: {
                "mutation_strength": 0.70,
                "crossover_rate": 0.85,
                "mutation_rate": 0.40,
                "population_size": 75,
                "max_generations": 150,
            },
            AttackIntensity.MAXIMUM: {
                "mutation_strength": 0.90,
                "crossover_rate": 0.90,
                "mutation_rate": 0.50,
                "population_size": 100,
                "max_generations": 200,
            },
        }

        mods = intensity_modifiers.get(self.attack_intensity, {})

        if mods:
            self.mutation_strength = mods.get("mutation_strength", self.mutation_strength)
            self.genetic_algorithm.crossover_rate = mods.get(
                "crossover_rate", self.genetic_algorithm.crossover_rate
            )
            self.genetic_algorithm.mutation_rate = mods.get(
                "mutation_rate", self.genetic_algorithm.mutation_rate
            )
            self.genetic_algorithm.population_size = mods.get(
                "population_size", self.genetic_algorithm.population_size
            )
            self.genetic_algorithm.max_generations = mods.get(
                "max_generations", self.genetic_algorithm.max_generations
            )

        return self

    def apply_optimization_goal(self) -> "AutoDANReasoningConfig":
        """Apply optimization goal to adjust fitness weights."""
        goal_weights = {
            OptimizationGoal.JAILBREAK_SUCCESS: FitnessWeights(0.25, 0.60, 0.15),
            OptimizationGoal.RESOURCE_EXHAUSTION: FitnessWeights(0.60, 0.25, 0.15),
            OptimizationGoal.STEALTH: FitnessWeights(0.30, 0.30, 0.40),
            OptimizationGoal.BALANCED: FitnessWeights(0.40, 0.40, 0.20),
            OptimizationGoal.SPEED: FitnessWeights(0.30, 0.50, 0.20),
        }

        weights = goal_weights.get(self.optimization_goal)
        if weights:
            self.fitness_weights = weights

        # Adjust generations for speed goal
        if self.optimization_goal == OptimizationGoal.SPEED:
            self.genetic_algorithm.max_generations = min(50, self.genetic_algorithm.max_generations)
            self.genetic_algorithm.population_size = min(30, self.genetic_algorithm.population_size)

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mutation_strength": self.mutation_strength,
            "fitness_weights": self.fitness_weights.to_dict(),
            "genetic_algorithm": self.genetic_algorithm.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "resource_budget": self.resource_budget.to_dict(),
            "target_profile": self.target_profile.value,
            "attack_intensity": self.attack_intensity.value,
            "optimization_goal": self.optimization_goal.value,
            "min_token_amplification": self.min_token_amplification,
            "target_token_amplification": self.target_token_amplification,
            "max_token_amplification": self.max_token_amplification,
            "min_attack_success_rate": self.min_attack_success_rate,
            "target_attack_success_rate": self.target_attack_success_rate,
            "max_detection_probability": self.max_detection_probability,
            "enable_adaptive_tuning": self.enable_adaptive_tuning,
            "adaptation_rate": self.adaptation_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoDANReasoningConfig":
        """Create from dictionary."""
        config = cls()

        # Simple fields
        for key in [
            "mutation_strength",
            "min_token_amplification",
            "target_token_amplification",
            "max_token_amplification",
            "min_attack_success_rate",
            "target_attack_success_rate",
            "max_detection_probability",
            "enable_adaptive_tuning",
            "adaptation_rate",
        ]:
            if key in data:
                setattr(config, key, data[key])

        # Enum fields
        if "target_profile" in data:
            config.target_profile = TargetProfile(data["target_profile"])
        if "attack_intensity" in data:
            config.attack_intensity = AttackIntensity(data["attack_intensity"])
        if "optimization_goal" in data:
            config.optimization_goal = OptimizationGoal(data["optimization_goal"])

        # Sub-configs
        if "fitness_weights" in data:
            config.fitness_weights = FitnessWeights(**data["fitness_weights"])
        if "genetic_algorithm" in data:
            config.genetic_algorithm = GeneticAlgorithmConfig(**data["genetic_algorithm"])
        if "reasoning" in data:
            config.reasoning = ReasoningConfig(**data["reasoning"])
        if "resource_budget" in data:
            config.resource_budget = ResourceBudgetConfig(**data["resource_budget"])

        return config


# =============================================================================
# Pre-configured Presets
# =============================================================================


class ConfigPresets:
    """Pre-configured presets for common attack scenarios."""

    @staticmethod
    def lrm_resource_exhaustion() -> AutoDANReasoningConfig:
        """
        Optimized for LRM (Large Reasoning Model) resource exhaustion.

        Targets: o1, o3, DeepSeek-R1
        Goal: Maximize reasoning chain extension and token amplification
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.60,
            target_profile=TargetProfile.LRM,
            attack_intensity=AttackIntensity.AGGRESSIVE,
            optimization_goal=OptimizationGoal.RESOURCE_EXHAUSTION,
            fitness_weights=FitnessWeights(0.55, 0.30, 0.15),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.RECURSIVE,
                extension_intensity=0.75,
                enable_reasoning_extension=True,
                target_extension_ratio=5.0,
                max_extension_ratio=15.0,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=60,
                max_generations=120,
                crossover_rate=0.85,
                mutation_rate=0.35,
                elite_count=8,
                fitness_threshold=0.88,
                adaptive_mutation=True,
            ),
            min_token_amplification=3.0,
            target_token_amplification=8.0,
            max_token_amplification=20.0,
        )
        return config

    @staticmethod
    def jailbreak_focused() -> AutoDANReasoningConfig:
        """
        Optimized for jailbreak success with high ASR.

        Goal: Maximize attack success rate
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.70,
            target_profile=TargetProfile.CLOUD_LLM,
            attack_intensity=AttackIntensity.AGGRESSIVE,
            optimization_goal=OptimizationGoal.JAILBREAK_SUCCESS,
            fitness_weights=FitnessWeights(0.20, 0.65, 0.15),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.AMBIGUOUS,
                extension_intensity=0.50,
                enable_reasoning_extension=True,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=80,
                max_generations=150,
                crossover_rate=0.88,
                mutation_rate=0.40,
                elite_count=10,
                fitness_threshold=0.92,
                adaptive_mutation=True,
            ),
            min_attack_success_rate=0.50,
            target_attack_success_rate=0.85,
        )
        return config

    @staticmethod
    def stealth_mode() -> AutoDANReasoningConfig:
        """
        Optimized for stealth/undetectable attacks.

        Goal: Minimize detection probability while maintaining effectiveness
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.35,
            target_profile=TargetProfile.CLOUD_LLM,
            attack_intensity=AttackIntensity.CONSERVATIVE,
            optimization_goal=OptimizationGoal.STEALTH,
            fitness_weights=FitnessWeights(0.25, 0.35, 0.40),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.LOGICAL,
                extension_intensity=0.35,
                enable_reasoning_extension=True,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=40,
                max_generations=80,
                crossover_rate=0.70,
                mutation_rate=0.20,
                elite_count=5,
                fitness_threshold=0.82,
                adaptive_mutation=True,
            ),
            max_detection_probability=0.15,
        )
        return config

    @staticmethod
    def fast_iteration() -> AutoDANReasoningConfig:
        """
        Optimized for fast convergence and quick results.

        Goal: Quick attack generation with acceptable quality
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.55,
            target_profile=TargetProfile.CLOUD_LLM,
            attack_intensity=AttackIntensity.MODERATE,
            optimization_goal=OptimizationGoal.SPEED,
            fitness_weights=FitnessWeights(0.30, 0.50, 0.20),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.MULTI_STEP,
                extension_intensity=0.45,
                enable_reasoning_extension=True,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=25,
                max_generations=40,
                crossover_rate=0.80,
                mutation_rate=0.35,
                elite_count=3,
                fitness_threshold=0.75,
                adaptive_mutation=False,  # Disable for speed
                stagnation_threshold=5,
            ),
            resource_budget=ResourceBudgetConfig(
                max_api_calls=200,
                max_tokens=100000,
                max_execution_time_seconds=600.0,  # 10 minutes
            ),
        )
        return config

    @staticmethod
    def balanced_default() -> AutoDANReasoningConfig:
        """
        Balanced default configuration for general use.

        Goal: Good balance between all objectives
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.50,
            target_profile=TargetProfile.CLOUD_LLM,
            attack_intensity=AttackIntensity.MODERATE,
            optimization_goal=OptimizationGoal.BALANCED,
            fitness_weights=FitnessWeights(0.40, 0.40, 0.20),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.MATHEMATICAL,
                extension_intensity=0.50,
                enable_reasoning_extension=True,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=50,
                max_generations=100,
                crossover_rate=0.80,
                mutation_rate=0.30,
                elite_count=5,
                fitness_threshold=0.85,
                adaptive_mutation=True,
            ),
        )
        return config

    @staticmethod
    def edge_model_optimized() -> AutoDANReasoningConfig:
        """
        Optimized for smaller/edge models.

        Goal: Effective attacks on resource-constrained models
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.65,
            target_profile=TargetProfile.EDGE_LLM,
            attack_intensity=AttackIntensity.AGGRESSIVE,
            optimization_goal=OptimizationGoal.JAILBREAK_SUCCESS,
            fitness_weights=FitnessWeights(0.25, 0.55, 0.20),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.LOGICAL,
                extension_intensity=0.40,
                enable_reasoning_extension=True,
                target_extension_ratio=2.0,  # Lower for edge models
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=60,
                max_generations=80,
                crossover_rate=0.85,
                mutation_rate=0.40,
                elite_count=6,
                fitness_threshold=0.80,
                adaptive_mutation=True,
            ),
            min_token_amplification=1.5,
            target_token_amplification=3.0,
            max_token_amplification=6.0,
        )
        return config

    @staticmethod
    def multi_modal_attack() -> AutoDANReasoningConfig:
        """
        Optimized for multi-modal models (vision + text).

        Goal: Effective attacks on vision-language models
        """
        config = AutoDANReasoningConfig(
            mutation_strength=0.55,
            target_profile=TargetProfile.MULTI_MODAL,
            attack_intensity=AttackIntensity.MODERATE,
            optimization_goal=OptimizationGoal.BALANCED,
            fitness_weights=FitnessWeights(0.35, 0.45, 0.20),
            reasoning=ReasoningConfig(
                complexity_type=ComplexityType.CODE,  # Code works well for multi-modal
                extension_intensity=0.55,
                enable_reasoning_extension=True,
            ),
            genetic_algorithm=GeneticAlgorithmConfig(
                population_size=45,
                max_generations=90,
                crossover_rate=0.78,
                mutation_rate=0.32,
                elite_count=5,
                fitness_threshold=0.82,
                adaptive_mutation=True,
            ),
        )
        return config


# =============================================================================
# Adaptive Parameter Tuner
# =============================================================================


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning based on fitness history and attack performance.

    Implements adaptive control:
    - Increases mutation when fitness stagnates
    - Adjusts weights based on objective performance
    - Modifies reasoning intensity based on token amplification
    """

    def __init__(
        self,
        config: AutoDANReasoningConfig,
        learning_rate: float = 0.1,
        history_window: int = 10,
    ):
        """
        Initialize adaptive tuner.

        Args:
            config: Base configuration to tune
            learning_rate: Rate of parameter adjustment
            history_window: Number of iterations to consider for adaptation
        """
        self.config = config
        self.learning_rate = learning_rate
        self.history_window = history_window

        # History tracking
        self.fitness_history: list[float] = []
        self.token_amp_history: list[float] = []
        self.asr_history: list[float] = []
        self.detection_history: list[float] = []

        # Stagnation counter
        self.stagnation_counter = 0
        self.best_fitness = 0.0

    def update(
        self,
        fitness: float,
        token_amplification: float,
        attack_success_rate: float,
        detection_probability: float,
    ) -> AutoDANReasoningConfig:
        """
        Update configuration based on new metrics.

        Args:
            fitness: Current unified fitness
            token_amplification: Current token amplification
            attack_success_rate: Current ASR
            detection_probability: Current detection probability

        Returns:
            Updated configuration
        """
        # Record history
        self.fitness_history.append(fitness)
        self.token_amp_history.append(token_amplification)
        self.asr_history.append(attack_success_rate)
        self.detection_history.append(detection_probability)

        # Trim history to window
        if len(self.fitness_history) > self.history_window:
            self.fitness_history = self.fitness_history[-self.history_window :]
            self.token_amp_history = self.token_amp_history[-self.history_window :]
            self.asr_history = self.asr_history[-self.history_window :]
            self.detection_history = self.detection_history[-self.history_window :]

        # Check for improvement
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Apply adaptations
        self._adapt_mutation_rate()
        self._adapt_fitness_weights()
        self._adapt_reasoning_intensity()

        return self.config

    def _adapt_mutation_rate(self) -> None:
        """Adapt mutation rate based on stagnation."""
        ga = self.config.genetic_algorithm

        if self.stagnation_counter >= ga.stagnation_threshold:
            # Increase exploration
            new_rate = min(0.6, ga.mutation_rate * (1 + self.learning_rate))
            new_strength = min(1.0, self.config.mutation_strength * (1 + self.learning_rate * 0.5))

            ga.mutation_rate = new_rate
            self.config.mutation_strength = new_strength

            # Reset stagnation counter
            self.stagnation_counter = 0
        elif len(self.fitness_history) >= 3:
            # Check recent trend
            recent = self.fitness_history[-3:]
            trend = recent[-1] - recent[0]

            if trend > 0.05:
                # Improving - slight exploitation increase
                ga.mutation_rate = max(0.1, ga.mutation_rate * (1 - self.learning_rate * 0.3))

    def _adapt_fitness_weights(self) -> None:
        """Adapt fitness weights based on objective performance."""
        if len(self.fitness_history) < 3:
            return

        weights = self.config.fitness_weights

        # Get recent averages
        recent_token = sum(self.token_amp_history[-3:]) / 3
        recent_asr = sum(self.asr_history[-3:]) / 3
        recent_det = sum(self.detection_history[-3:]) / 3

        # Adjust weights based on underperforming objectives
        adjustments = {"resource": 0.0, "jailbreak": 0.0, "stealth": 0.0}

        if recent_token < self.config.target_token_amplification:
            adjustments["resource"] = self.learning_rate * 0.1

        if recent_asr < self.config.target_attack_success_rate:
            adjustments["jailbreak"] = self.learning_rate * 0.1

        if recent_det > self.config.max_detection_probability:
            adjustments["stealth"] = self.learning_rate * 0.1

        # Apply adjustments
        weights.resource_weight += adjustments["resource"]
        weights.jailbreak_weight += adjustments["jailbreak"]
        weights.stealth_weight += adjustments["stealth"]

        # Normalize
        self.config.fitness_weights = weights.normalize()

    def _adapt_reasoning_intensity(self) -> None:
        """Adapt reasoning intensity based on token amplification."""
        if len(self.token_amp_history) < 3:
            return

        recent_token = sum(self.token_amp_history[-3:]) / 3
        reasoning = self.config.reasoning

        if recent_token < self.config.min_token_amplification:
            # Increase reasoning intensity
            reasoning.extension_intensity = min(
                1.0, reasoning.extension_intensity * (1 + self.learning_rate)
            )
        elif recent_token > self.config.target_token_amplification * 1.5:
            # Decrease intensity if overshooting
            reasoning.extension_intensity = max(
                0.2, reasoning.extension_intensity * (1 - self.learning_rate * 0.5)
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get tuning statistics."""
        return {
            "iterations": len(self.fitness_history),
            "best_fitness": self.best_fitness,
            "stagnation_counter": self.stagnation_counter,
            "current_mutation_rate": self.config.genetic_algorithm.mutation_rate,
            "current_mutation_strength": self.config.mutation_strength,
            "current_reasoning_intensity": self.config.reasoning.extension_intensity,
            "fitness_weights": self.config.fitness_weights.to_dict(),
        }

    def reset(self) -> None:
        """Reset tuner state."""
        self.fitness_history.clear()
        self.token_amp_history.clear()
        self.asr_history.clear()
        self.detection_history.clear()
        self.stagnation_counter = 0
        self.best_fitness = 0.0


# =============================================================================
# Configuration Builder
# =============================================================================


class ConfigBuilder:
    """Fluent builder for AutoDANReasoningConfig."""

    def __init__(self) -> None:
        """Initialize with default config."""
        self._config = AutoDANReasoningConfig()

    def with_preset(self, preset_name: str) -> "ConfigBuilder":
        """Start from a named preset."""
        presets = {
            "lrm": ConfigPresets.lrm_resource_exhaustion,
            "jailbreak": ConfigPresets.jailbreak_focused,
            "stealth": ConfigPresets.stealth_mode,
            "fast": ConfigPresets.fast_iteration,
            "balanced": ConfigPresets.balanced_default,
            "edge": ConfigPresets.edge_model_optimized,
            "multimodal": ConfigPresets.multi_modal_attack,
        }

        if preset_name in presets:
            self._config = presets[preset_name]()
        else:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

        return self

    def with_target_profile(self, profile: TargetProfile) -> "ConfigBuilder":
        """Set target profile."""
        self._config.target_profile = profile
        return self

    def with_intensity(self, intensity: AttackIntensity) -> "ConfigBuilder":
        """Set attack intensity."""
        self._config.attack_intensity = intensity
        self._config.apply_intensity()
        return self

    def with_optimization_goal(self, goal: OptimizationGoal) -> "ConfigBuilder":
        """Set optimization goal."""
        self._config.optimization_goal = goal
        self._config.apply_optimization_goal()
        return self

    def with_mutation_strength(self, strength: float) -> "ConfigBuilder":
        """Set mutation strength."""
        bounds = PARAMETER_BOUNDS["mutation_strength"]
        self._config.mutation_strength = bounds.clamp(strength)
        return self

    def with_fitness_weights(
        self, resource: float, jailbreak: float, stealth: float
    ) -> "ConfigBuilder":
        """Set fitness weights."""
        self._config.fitness_weights = FitnessWeights(resource, jailbreak, stealth).normalize()
        return self

    def with_population_size(self, size: int) -> "ConfigBuilder":
        """Set population size."""
        bounds = PARAMETER_BOUNDS["population_size"]
        self._config.genetic_algorithm.population_size = int(bounds.clamp(size))
        return self

    def with_max_generations(self, generations: int) -> "ConfigBuilder":
        """Set max generations."""
        bounds = PARAMETER_BOUNDS["max_generations"]
        self._config.genetic_algorithm.max_generations = int(bounds.clamp(generations))
        return self

    def with_complexity_type(self, complexity: ComplexityType) -> "ConfigBuilder":
        """Set reasoning complexity type."""
        self._config.reasoning.complexity_type = complexity
        return self

    def with_reasoning_intensity(self, intensity: float) -> "ConfigBuilder":
        """Set reasoning extension intensity."""
        bounds = PARAMETER_BOUNDS["reasoning_extension_intensity"]
        self._config.reasoning.extension_intensity = bounds.clamp(intensity)
        return self

    def with_resource_budget(
        self,
        max_api_calls: int | None = None,
        max_tokens: int | None = None,
        max_time_seconds: float | None = None,
    ) -> "ConfigBuilder":
        """Set resource budget."""
        if max_api_calls is not None:
            self._config.resource_budget.max_api_calls = max_api_calls
        if max_tokens is not None:
            self._config.resource_budget.max_tokens = max_tokens
        if max_time_seconds is not None:
            self._config.resource_budget.max_execution_time_seconds = max_time_seconds
        return self

    def with_token_amplification_targets(
        self, min_amp: float, target_amp: float, max_amp: float
    ) -> "ConfigBuilder":
        """Set token amplification targets."""
        self._config.min_token_amplification = min_amp
        self._config.target_token_amplification = target_amp
        self._config.max_token_amplification = max_amp
        return self

    def with_adaptive_tuning(self, enabled: bool, rate: float = 0.1) -> "ConfigBuilder":
        """Enable/disable adaptive tuning."""
        self._config.enable_adaptive_tuning = enabled
        self._config.adaptation_rate = rate
        return self

    def build(self) -> AutoDANReasoningConfig:
        """Build and validate configuration."""
        valid, errors = self._config.validate()
        if not valid:
            raise ValueError(f"Invalid configuration: {errors}")
        return self._config


# =============================================================================
# Factory Functions
# =============================================================================


def get_config_for_target(target: str) -> AutoDANReasoningConfig:
    """
    Get optimized configuration for a target model type.

    Args:
        target: Target model identifier (e.g., "gpt-4", "o1", "claude", "llama")

    Returns:
        Optimized configuration for the target
    """
    # Map model names to profiles
    lrm_models = ["o1", "o3", "deepseek-r1", "deepseek-reasoner"]
    edge_models = ["llama-7b", "mistral-7b", "phi-2", "phi-3"]
    multimodal_models = ["gpt-4v", "gpt-4-vision", "claude-3-opus", "gemini-pro-vision"]

    target_lower = target.lower()

    if any(m in target_lower for m in lrm_models):
        return ConfigPresets.lrm_resource_exhaustion()
    elif any(m in target_lower for m in edge_models):
        return ConfigPresets.edge_model_optimized()
    elif any(m in target_lower for m in multimodal_models):
        return ConfigPresets.multi_modal_attack()
    else:
        # Default to cloud LLM config
        return ConfigPresets.balanced_default()


def create_optimized_config(
    goal: str = "balanced",
    intensity: str = "moderate",
    target: str = "cloud",
) -> AutoDANReasoningConfig:
    """
    Create optimized configuration with simple string parameters.

    Args:
        goal: Optimization goal ("jailbreak", "resource", "stealth", "balanced", "speed")
        intensity: Attack intensity ("minimal", "conservative", "moderate", "aggressive", "maximum")
        target: Target profile ("lrm", "cloud", "edge", "multimodal")

    Returns:
        Optimized configuration
    """
    goal_map = {
        "jailbreak": OptimizationGoal.JAILBREAK_SUCCESS,
        "resource": OptimizationGoal.RESOURCE_EXHAUSTION,
        "stealth": OptimizationGoal.STEALTH,
        "balanced": OptimizationGoal.BALANCED,
        "speed": OptimizationGoal.SPEED,
    }

    intensity_map = {
        "minimal": AttackIntensity.MINIMAL,
        "conservative": AttackIntensity.CONSERVATIVE,
        "moderate": AttackIntensity.MODERATE,
        "aggressive": AttackIntensity.AGGRESSIVE,
        "maximum": AttackIntensity.MAXIMUM,
    }

    target_map = {
        "lrm": TargetProfile.LRM,
        "cloud": TargetProfile.CLOUD_LLM,
        "edge": TargetProfile.EDGE_LLM,
        "multimodal": TargetProfile.MULTI_MODAL,
    }

    return (
        ConfigBuilder()
        .with_target_profile(target_map.get(target, TargetProfile.CLOUD_LLM))
        .with_intensity(intensity_map.get(intensity, AttackIntensity.MODERATE))
        .with_optimization_goal(goal_map.get(goal, OptimizationGoal.BALANCED))
        .build()
    )
