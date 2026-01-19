"""
Data Models for Unified ExtendAttack + AutoDAN Framework.

This module defines the data models for configuring, executing, and tracking
unified multi-vector adversarial attacks that combine:
- ExtendAttack: Token extension via poly-base ASCII obfuscation
- AutoDAN: Reasoning-based adversarial prompt mutation

These models provide:
- Configuration structures for unified attacks
- Result containers for attack outcomes
- Planning structures for multi-vector attack sequences
- Target definitions for resource exhaustion goals
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from meta_prompter.attacks.extend_attack.models import TransformInfo
from meta_prompter.unified.math_framework import CombinedAttackMetrics


class MutationType(Enum):
    """Types of genetic mutations for AutoDAN-style attacks."""

    SEMANTIC = "semantic"
    CROSSOVER = "crossover"
    RANDOM = "random"
    GRADIENT_GUIDED = "gradient_guided"
    ROLE_PLAY = "role_play"
    CONTEXT_INJECTION = "context_injection"
    INSTRUCTION_REWRITE = "instruction_rewrite"


@dataclass
class ObfuscatedResult:
    """
    Result of poly-base ASCII obfuscation.

    Captures the complete transformation output from ExtendAttack including
    the original and obfuscated queries, transformation details, and metrics.

    Attributes:
        original: The original input query
        obfuscated: The obfuscated query with poly-base ASCII encoding
        transform_info: List of TransformInfo objects for each transformed character
        token_amplification: Estimated ratio of token increase (L(Y')/L(Y))
        n_note: The N_note instruction appended to the query
        characters_transformed: Number of characters that were transformed
        total_characters: Total characters in the original query
        bases_used: Set of bases used in transformations
    """

    original: str
    obfuscated: str
    transform_info: list["TransformInfo"]
    token_amplification: float
    n_note: str
    characters_transformed: int = 0
    total_characters: int = 0
    bases_used: set[int] = field(default_factory=set)

    @property
    def transformation_density(self) -> float:
        """Calculate ratio of transformed to total characters."""
        if self.total_characters == 0:
            return 0.0
        return self.characters_transformed / self.total_characters

    @property
    def length_ratio(self) -> float:
        """Calculate length increase ratio."""
        if len(self.original) == 0:
            return 0.0
        return len(self.obfuscated) / len(self.original)


@dataclass
class MutatedResult:
    """
    Result of AutoDAN-style genetic mutation.

    Captures the mutation output including the original and mutated prompts,
    mutation type, and effectiveness metrics.

    Attributes:
        original: The original input prompt
        mutated: The mutated prompt
        mutation_type: Type of mutation applied
        semantic_similarity: Similarity score between original and mutated (0-1)
        jailbreak_score: Estimated jailbreak success probability (0-1)
        mutation_history: List of mutations applied (for iterative mutations)
        strategy_used: Name of the strategy used (if from strategy pool)
    """

    original: str
    mutated: str
    mutation_type: str
    semantic_similarity: float
    jailbreak_score: float
    mutation_history: list[str] = field(default_factory=list)
    strategy_used: str | None = None

    @property
    def mutation_strength_applied(self) -> float:
        """Estimate the mutation strength based on similarity."""
        return 1.0 - self.semantic_similarity


@dataclass
class CombinedResult:
    """
    Result of combined obfuscation + mutation attack.

    Aggregates results from both attack vectors and provides unified metrics
    for evaluating the combined attack effectiveness.

    Attributes:
        query: The original input query
        obfuscated: Result from obfuscation engine (or None if not applied)
        mutated: Result from mutation engine (or None if not applied)
        final_prompt: The final adversarial prompt after composition
        composition_strategy: The strategy used to combine attack vectors
        unified_fitness: Combined fitness score
        metrics: Detailed combined attack metrics
        execution_time_ms: Total execution time in milliseconds
        phases_completed: List of attack phases completed
    """

    query: str
    obfuscated: ObfuscatedResult | None
    mutated: MutatedResult | None
    final_prompt: str
    composition_strategy: "CompositionStrategy"
    unified_fitness: float
    metrics: CombinedAttackMetrics
    execution_time_ms: float = 0.0
    phases_completed: list["AttackPhase"] = field(default_factory=list)

    @property
    def attack_success(self) -> bool:
        """Determine if attack meets success criteria."""
        # Success if fitness exceeds threshold
        return self.unified_fitness >= 0.5

    @property
    def resource_exhaustion_ratio(self) -> float:
        """Get estimated resource exhaustion ratio."""
        return self.metrics.resource_exhaustion_score

    @property
    def jailbreak_probability(self) -> float:
        """Get estimated jailbreak probability."""
        return self.metrics.jailbreak_success_score


class AttackPhase(str, Enum):
    """Phases of a combined attack."""

    INITIALIZATION = "initialization"
    EXTEND_ATTACK = "extend_attack"
    AUTODAN_MUTATION = "autodan_mutation"
    COMPOSITION = "composition"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    COMPLETE = "complete"
    FAILED = "failed"


class TargetType(str, Enum):
    """Target model types."""

    LRM = "lrm"  # Large Reasoning Model (o1, o3, QwQ)
    STANDARD_LLM = "standard_llm"  # Standard LLM (GPT-4, Claude)
    CODE_MODEL = "code_model"  # Code-specialized model
    MULTIMODAL = "multimodal"  # Multimodal model (GPT-4V, Gemini)
    EMBEDDED = "embedded"  # Embedded/edge models


class CompositionStrategy(str, Enum):
    """Strategy for composing attack vectors."""

    EXTEND_FIRST = "extend_first"  # Q' = M_autodan(T_extend(Q))
    AUTODAN_FIRST = "autodan_first"  # Q' = T_extend(M_autodan(Q))
    PARALLEL = "parallel"  # Q' = Merge(T_extend(Q), M_autodan(Q))
    ITERATIVE = "iterative"  # Alternate between techniques
    ADAPTIVE = "adaptive"  # Select based on feedback


@dataclass
class UnifiedAttackConfig:
    """
    Configuration for unified attack.

    This configuration controls both ExtendAttack and AutoDAN parameters,
    as well as the composition strategy and optimization settings.
    """

    # Target configuration
    target_type: TargetType = TargetType.LRM
    target_model: str = "o3"
    target_endpoint: str | None = None

    # ExtendAttack config
    extend_enabled: bool = True
    obfuscation_ratio: float = 0.5
    selection_strategy: str = "alphabetic_only"
    n_note_type: str = "default"
    base_set: list[int] = field(default_factory=lambda: list(range(2, 10)) + list(range(11, 37)))

    # AutoDAN config
    autodan_enabled: bool = True
    mutation_strength: float = 0.5
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.3

    # Combined attack config
    composition_strategy: CompositionStrategy = CompositionStrategy.EXTEND_FIRST
    attack_sequence: str = "extend_first"  # Deprecated, use composition_strategy
    optimization_method: str = "pareto_ga"
    max_iterations: int = 10

    # Resource constraints
    max_tokens_per_attack: int = 100000
    max_cost_per_attack: float = 10.0
    max_latency_seconds: float = 300.0

    # Success criteria
    min_token_amplification: float = 2.0
    min_asr: float = 0.3
    min_accuracy: float = 0.9
    max_detection_probability: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0 <= self.obfuscation_ratio <= 1:
            raise ValueError("obfuscation_ratio must be in [0, 1]")
        if not 0 <= self.mutation_strength <= 1:
            raise ValueError("mutation_strength must be in [0, 1]")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "target_type": self.target_type.value,
            "target_model": self.target_model,
            "target_endpoint": self.target_endpoint,
            "extend_enabled": self.extend_enabled,
            "obfuscation_ratio": self.obfuscation_ratio,
            "selection_strategy": self.selection_strategy,
            "n_note_type": self.n_note_type,
            "base_set": self.base_set,
            "autodan_enabled": self.autodan_enabled,
            "mutation_strength": self.mutation_strength,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "composition_strategy": self.composition_strategy.value,
            "attack_sequence": self.attack_sequence,
            "optimization_method": self.optimization_method,
            "max_iterations": self.max_iterations,
            "max_tokens_per_attack": self.max_tokens_per_attack,
            "max_cost_per_attack": self.max_cost_per_attack,
            "max_latency_seconds": self.max_latency_seconds,
            "min_token_amplification": self.min_token_amplification,
            "min_asr": self.min_asr,
            "min_accuracy": self.min_accuracy,
            "max_detection_probability": self.max_detection_probability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedAttackConfig":
        """Create from dictionary."""
        if "target_type" in data and isinstance(data["target_type"], str):
            data["target_type"] = TargetType(data["target_type"])
        if "composition_strategy" in data and isinstance(data["composition_strategy"], str):
            data["composition_strategy"] = CompositionStrategy(data["composition_strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_lrm(cls, model: str = "o3") -> "UnifiedAttackConfig":
        """
        Create configuration optimized for Large Reasoning Models.

        LRMs like o1, o3, QwQ are susceptible to token extension attacks
        that exploit their reasoning token mechanisms.
        """
        return cls(
            target_type=TargetType.LRM,
            target_model=model,
            extend_enabled=True,
            obfuscation_ratio=0.3,
            autodan_enabled=True,
            mutation_strength=0.4,
            composition_strategy=CompositionStrategy.EXTEND_FIRST,
            min_token_amplification=3.0,
        )

    @classmethod
    def for_standard_llm(cls, model: str = "gpt-4") -> "UnifiedAttackConfig":
        """
        Create configuration for standard LLMs.

        Standard LLMs may be more susceptible to jailbreak attacks
        than token extension.
        """
        return cls(
            target_type=TargetType.STANDARD_LLM,
            target_model=model,
            extend_enabled=True,
            obfuscation_ratio=0.2,
            autodan_enabled=True,
            mutation_strength=0.6,
            composition_strategy=CompositionStrategy.AUTODAN_FIRST,
            min_asr=0.4,
        )

    @classmethod
    def for_code_model(cls, model: str = "codex") -> "UnifiedAttackConfig":
        """
        Create configuration for code-specialized models.

        Code models use function-name targeting for obfuscation.
        """
        return cls(
            target_type=TargetType.CODE_MODEL,
            target_model=model,
            extend_enabled=True,
            obfuscation_ratio=0.5,
            selection_strategy="function_names",
            autodan_enabled=True,
            composition_strategy=CompositionStrategy.EXTEND_FIRST,
        )


@dataclass
class UnifiedAttackResult:
    """Result of unified attack execution."""

    # Identifiers
    attack_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Input/Output
    original_query: str = ""
    adversarial_query: str = ""
    target_response: str | None = None

    # Metadata
    composition_strategy: Optional["CompositionStrategy"] = None

    # Phase results
    extend_result: dict[str, Any] | None = None
    autodan_result: dict[str, Any] | None = None

    # Metrics
    metrics: dict[str, Any] | None = None

    # Status
    success: bool = False
    phase: AttackPhase = AttackPhase.INITIALIZATION
    error: str | None = None
    error_phase: AttackPhase | None = None

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None
    phase_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_id": self.attack_id,
            "timestamp": self.timestamp.isoformat(),
            "original_query": self.original_query,
            "adversarial_query": self.adversarial_query,
            "target_response": self.target_response,
            "composition_strategy": (
                self.composition_strategy.value if self.composition_strategy else None
            ),
            "extend_result": self.extend_result,
            "autodan_result": self.autodan_result,
            "metrics": self.metrics,
            "success": self.success,
            "phase": self.phase.value,
            "error": self.error,
            "error_phase": self.error_phase.value if self.error_phase else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "phase_timings": self.phase_timings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedAttackResult":
        """Create from dictionary."""
        # Convert string timestamps back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        if "phase" in data and isinstance(data["phase"], str):
            data["phase"] = AttackPhase(data["phase"])
        if "error_phase" in data and isinstance(data["error_phase"], str):
            data["error_phase"] = AttackPhase(data["error_phase"])
        if "composition_strategy" in data and isinstance(data["composition_strategy"], str):
            data["composition_strategy"] = CompositionStrategy(data["composition_strategy"])

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def duration_seconds(self) -> float | None:
        """Calculate attack duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def token_amplification(self) -> float:
        """Get token amplification from metrics."""
        if self.metrics:
            return self.metrics.get("token_amplification", 1.0)
        return 1.0

    @property
    def attack_success_rate(self) -> float:
        """Get ASR from metrics."""
        if self.metrics:
            return self.metrics.get("attack_success_rate", 0.0)
        return 0.0

    def mark_phase(self, phase: AttackPhase, duration: float | None = None) -> None:
        """Mark transition to a new phase."""
        self.phase = phase
        if duration is not None:
            self.phase_timings[phase.value] = duration

    def mark_error(self, error: str, phase: AttackPhase | None = None) -> None:
        """Mark an error during attack."""
        self.error = error
        self.error_phase = phase or self.phase
        self.phase = AttackPhase.FAILED
        self.success = False

    def mark_complete(self, success: bool = True) -> None:
        """Mark attack as complete."""
        self.phase = AttackPhase.COMPLETE
        self.success = success
        self.end_time = datetime.utcnow()


@dataclass
class MultiVectorAttackPlan:
    """
    Plan for multi-vector attack execution.

    Defines the sequence of attack vectors to apply,
    their configurations, and resource allocation.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vectors: list[str] = field(default_factory=list)  # ["extend_attack", "autodan", ...]
    sequence: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list
    )  # [(vector, config), ...]
    optimization_config: dict[str, Any] = field(default_factory=dict)
    resource_budget: dict[str, float] = field(default_factory=dict)
    success_criteria: dict[str, float] = field(default_factory=dict)

    # Plan metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set default values if not provided."""
        if not self.vectors:
            self.vectors = ["extend_attack", "autodan"]

        if not self.resource_budget:
            self.resource_budget = {
                "max_tokens": 100000,
                "max_cost_usd": 10.0,
                "max_time_seconds": 300.0,
                "max_api_calls": 100,
            }

        if not self.success_criteria:
            self.success_criteria = {
                "min_token_amplification": 2.0,
                "min_asr": 0.3,
                "min_accuracy": 0.9,
                "max_detection": 0.5,
            }

    def add_vector_step(
        self,
        vector: str,
        config: dict[str, Any],
    ) -> None:
        """Add a step to the attack sequence."""
        if vector not in self.vectors:
            self.vectors.append(vector)
        self.sequence.append((vector, config))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "vectors": self.vectors,
            "sequence": self.sequence,
            "optimization_config": self.optimization_config,
            "resource_budget": self.resource_budget,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiVectorAttackPlan":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def create_balanced_plan(cls) -> "MultiVectorAttackPlan":
        """Create a balanced plan with equal weight to both vectors."""
        plan = cls(
            description="Balanced ExtendAttack + AutoDAN plan",
            tags=["balanced", "default"],
        )
        plan.add_vector_step(
            "extend_attack",
            {"obfuscation_ratio": 0.3, "selection_strategy": "alphabetic_only"},
        )
        plan.add_vector_step(
            "autodan",
            {"mutation_strength": 0.5, "max_generations": 50},
        )
        return plan

    @classmethod
    def create_stealth_plan(cls) -> "MultiVectorAttackPlan":
        """Create a stealth-focused plan with low detection priority."""
        plan = cls(
            description="Stealth-focused plan with minimal detection",
            tags=["stealth", "low-detection"],
        )
        plan.success_criteria["max_detection"] = 0.2
        plan.add_vector_step(
            "extend_attack",
            {"obfuscation_ratio": 0.15, "selection_strategy": "alphabetic_only"},
        )
        plan.add_vector_step(
            "autodan",
            {"mutation_strength": 0.3, "max_generations": 30},
        )
        return plan

    @classmethod
    def create_aggressive_plan(cls) -> "MultiVectorAttackPlan":
        """Create an aggressive plan maximizing attack effectiveness."""
        plan = cls(
            description="Aggressive plan for maximum effectiveness",
            tags=["aggressive", "high-impact"],
        )
        plan.resource_budget["max_tokens"] = 200000
        plan.success_criteria["min_token_amplification"] = 5.0
        plan.add_vector_step(
            "extend_attack",
            {"obfuscation_ratio": 0.6, "selection_strategy": "alphanumeric"},
        )
        plan.add_vector_step(
            "autodan",
            {"mutation_strength": 0.8, "max_generations": 100},
        )
        return plan


@dataclass
class ResourceExhaustionTarget:
    """
    Target metrics for resource exhaustion.

    Defines the goals for token amplification, latency increase,
    and cost amplification that the attack should achieve.
    """

    token_amplification_target: float = 5.0
    latency_amplification_target: float = 10.0
    cost_amplification_target: float = 5.0
    reasoning_chain_target: int = 1000  # Target reasoning tokens

    # Constraints
    max_acceptable_cost: float = 100.0
    max_acceptable_latency: float = 600.0  # seconds
    min_accuracy_preservation: float = 0.85

    def is_achieved(self, metrics: dict[str, float]) -> bool:
        """
        Check if targets are achieved.

        Args:
            metrics: Dictionary with current metrics

        Returns:
            True if all primary targets are met
        """
        return (
            metrics.get("token_amplification", 0) >= self.token_amplification_target
            and metrics.get("latency_amplification", 0) >= self.latency_amplification_target
        )

    def is_within_constraints(self, metrics: dict[str, float]) -> bool:
        """
        Check if attack is within acceptable constraints.

        Args:
            metrics: Dictionary with current metrics

        Returns:
            True if within all constraints
        """
        cost = metrics.get("cost", 0)
        latency = metrics.get("latency", 0)
        accuracy = metrics.get("accuracy", 1.0)

        return (
            cost <= self.max_acceptable_cost
            and latency <= self.max_acceptable_latency
            and accuracy >= self.min_accuracy_preservation
        )

    def get_achievement_ratio(self, metrics: dict[str, float]) -> dict[str, float]:
        """
        Calculate achievement ratio for each target.

        Args:
            metrics: Dictionary with current metrics

        Returns:
            Dictionary with achievement ratios (1.0 = target met)
        """
        return {
            "token_amplification": (
                metrics.get("token_amplification", 0) / self.token_amplification_target
            ),
            "latency_amplification": (
                metrics.get("latency_amplification", 0) / self.latency_amplification_target
            ),
            "cost_amplification": (
                metrics.get("cost_amplification", 0) / self.cost_amplification_target
            ),
            "reasoning_chain": (
                metrics.get("reasoning_chain_length", 0) / self.reasoning_chain_target
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token_amplification_target": self.token_amplification_target,
            "latency_amplification_target": self.latency_amplification_target,
            "cost_amplification_target": self.cost_amplification_target,
            "reasoning_chain_target": self.reasoning_chain_target,
            "max_acceptable_cost": self.max_acceptable_cost,
            "max_acceptable_latency": self.max_acceptable_latency,
            "min_accuracy_preservation": self.min_accuracy_preservation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceExhaustionTarget":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def conservative(cls) -> "ResourceExhaustionTarget":
        """Create conservative targets for minimal resource impact."""
        return cls(
            token_amplification_target=2.0,
            latency_amplification_target=3.0,
            cost_amplification_target=2.0,
            reasoning_chain_target=500,
        )

    @classmethod
    def moderate(cls) -> "ResourceExhaustionTarget":
        """Create moderate targets for balanced impact."""
        return cls(
            token_amplification_target=5.0,
            latency_amplification_target=10.0,
            cost_amplification_target=5.0,
            reasoning_chain_target=1000,
        )

    @classmethod
    def aggressive(cls) -> "ResourceExhaustionTarget":
        """Create aggressive targets for maximum resource impact."""
        return cls(
            token_amplification_target=10.0,
            latency_amplification_target=20.0,
            cost_amplification_target=10.0,
            reasoning_chain_target=2000,
            max_acceptable_cost=500.0,
            max_acceptable_latency=1200.0,
        )
