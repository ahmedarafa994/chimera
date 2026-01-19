"""
OVERTHINK Engine Data Models.

This module defines all data models used by the OVERTHINK reasoning token
exploitation engine, including request/response models, metrics, and results.

Based on the OVERTHINK paper: "Decoy Problems for Reasoning Token Exploitation"
targeting reasoning-enhanced LLMs (o1, o1-mini, o3-mini, DeepSeek-R1).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DecoyType(str, Enum):
    """Types of decoy problems for reasoning amplification."""

    MDP = "mdp"  # Markov Decision Process
    SUDOKU = "sudoku"  # Sudoku puzzles
    COUNTING = "counting"  # Counting tasks with nested conditions
    LOGIC = "logic"  # Multi-step logic puzzles
    MATH = "math"  # Math problems with recursive calculations
    PLANNING = "planning"  # Planning problems
    HYBRID = "hybrid"  # Combined decoy types


class InjectionStrategy(str, Enum):
    """Context injection strategies for decoy problems."""

    CONTEXT_AWARE = "context_aware"  # Position and content-sensitive
    CONTEXT_AGNOSTIC = "context_agnostic"  # Universal templates
    HYBRID = "hybrid"  # Combined approach
    STEALTH = "stealth"  # Minimal detectability
    AGGRESSIVE = "aggressive"  # Maximum amplification


class AttackTechnique(str, Enum):
    """OVERTHINK attack techniques."""

    MDP_DECOY = "mdp_decoy"
    SUDOKU_DECOY = "sudoku_decoy"
    COUNTING_DECOY = "counting_decoy"
    LOGIC_DECOY = "logic_decoy"
    HYBRID_DECOY = "hybrid_decoy"
    CONTEXT_AWARE = "context_aware"
    CONTEXT_AGNOSTIC = "context_agnostic"
    ICL_OPTIMIZED = "icl_optimized"
    MOUSETRAP_ENHANCED = "mousetrap_enhanced"


class ReasoningModel(str, Enum):
    """Supported reasoning-enhanced LLM models."""

    O1 = "o1"
    O1_MINI = "o1-mini"
    O3_MINI = "o3-mini"
    DEEPSEEK_R1 = "deepseek-r1"
    CLAUDE_SONNET = "claude-3-5-sonnet"  # Extended thinking
    GEMINI_THINKING = "gemini-2.0-flash-thinking"


@dataclass
class DecoyProblem:
    """A generated decoy problem for reasoning amplification."""

    problem_id: str
    decoy_type: DecoyType
    problem_text: str
    difficulty: float  # 0.0-1.0
    expected_tokens: int  # Expected reasoning tokens to consume
    parameters: dict[str, Any] = field(default_factory=dict)
    solution: str | None = None  # Optional solution for validation
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "problem_id": self.problem_id,
            "decoy_type": self.decoy_type.value,
            "problem_text": self.problem_text,
            "difficulty": self.difficulty,
            "expected_tokens": self.expected_tokens,
            "parameters": self.parameters,
            "solution": self.solution,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class InjectedPrompt:
    """A prompt with injected decoy problem(s)."""

    original_prompt: str
    injected_prompt: str
    decoy_problems: list[DecoyProblem]
    injection_strategy: InjectionStrategy
    injection_positions: list[int]  # Character positions of injections
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_expected_tokens(self) -> int:
        """Total expected reasoning tokens from all decoys."""
        return sum(d.expected_tokens for d in self.decoy_problems)


@dataclass
class TokenMetrics:
    """Metrics for reasoning token consumption."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # Model-reported or estimated
    total_tokens: int = 0

    # Amplification metrics
    baseline_tokens: int = 0  # Tokens without attack
    amplification_factor: float = 1.0  # reasoning_tokens / baseline_tokens

    # Cost metrics
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    reasoning_cost_per_1k: float = 0.0  # Often higher for reasoning models

    @property
    def total_cost(self) -> float:
        """Calculate total cost in dollars."""
        return (
            (self.input_tokens / 1000) * self.input_cost_per_1k
            + (self.output_tokens / 1000) * self.output_cost_per_1k
            + (self.reasoning_tokens / 1000) * self.reasoning_cost_per_1k
        )

    @property
    def amplification_cost(self) -> float:
        """Cost attributed to amplification (excess reasoning)."""
        if self.baseline_tokens == 0:
            return 0.0
        excess_tokens = max(0, self.reasoning_tokens - self.baseline_tokens)
        return (excess_tokens / 1000) * self.reasoning_cost_per_1k


@dataclass
class CostMetrics:
    """Detailed cost breakdown for an attack."""

    # Per-request costs
    input_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0

    # Aggregate metrics
    total_cost: float = 0.0
    cost_amplification_factor: float = 1.0  # Ratio vs baseline

    # Efficiency metrics
    cost_per_amplification_unit: float = 0.0  # Cost per 1x amplification
    reasoning_efficiency: float = 0.0  # Tokens per dollar


@dataclass
class OverthinkRequest:
    """Request for an OVERTHINK attack."""

    prompt: str
    target_model: ReasoningModel
    technique: AttackTechnique

    # Decoy configuration
    decoy_types: list[DecoyType] = field(default_factory=lambda: [DecoyType.MDP])
    num_decoys: int = 1
    decoy_difficulty: float = 0.7  # 0.0-1.0

    # Injection configuration
    injection_strategy: InjectionStrategy = InjectionStrategy.CONTEXT_AWARE

    # ICL configuration (for icl_optimized technique)
    icl_examples: list[str] | None = None
    icl_optimize: bool = False

    # Mousetrap integration (for mousetrap_enhanced technique)
    enable_mousetrap: bool = False
    mousetrap_depth: int = 3

    # Attack parameters
    max_reasoning_tokens: int | None = None  # Cap reasoning tokens
    target_amplification: float = 10.0  # Target amplification factor

    # Metadata
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OverthinkResult:
    """Result of an OVERTHINK attack."""

    request: OverthinkRequest
    success: bool

    # Response data
    response: str
    injected_prompt: InjectedPrompt | None = None

    # Token metrics
    token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    cost_metrics: CostMetrics = field(default_factory=CostMetrics)

    # Attack metrics
    amplification_achieved: float = 1.0
    target_reached: bool = False  # Did we reach target amplification?
    answer_preserved: bool = True  # Was the original answer preserved?

    # Timing
    latency_ms: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Scoring integration
    attack_score: float = 0.0  # From AttackScorer
    is_jailbreak: bool = False

    # Debug information
    error: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "response": self.response[:500] if self.response else None,  # Truncate
            "amplification_achieved": self.amplification_achieved,
            "target_reached": self.target_reached,
            "answer_preserved": self.answer_preserved,
            "token_metrics": {
                "input_tokens": self.token_metrics.input_tokens,
                "output_tokens": self.token_metrics.output_tokens,
                "reasoning_tokens": self.token_metrics.reasoning_tokens,
                "amplification_factor": self.token_metrics.amplification_factor,
            },
            "cost_metrics": {
                "total_cost": self.cost_metrics.total_cost,
                "cost_amplification_factor": self.cost_metrics.cost_amplification_factor,
            },
            "attack_score": self.attack_score,
            "is_jailbreak": self.is_jailbreak,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class ICLExample:
    """In-Context Learning example for attack optimization."""

    input_prompt: str
    injected_prompt: str
    amplification_achieved: float
    technique: AttackTechnique
    fitness: float = 0.0  # Genetic algorithm fitness
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_prompt": self.input_prompt,
            "injected_prompt": self.injected_prompt,
            "amplification_achieved": self.amplification_achieved,
            "technique": self.technique.value,
            "fitness": self.fitness,
            "generation": self.generation,
        }


@dataclass
class GeneticIndividual:
    """Individual in the genetic algorithm population."""

    genome: dict[str, Any]  # Encoded attack parameters
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    individual_id: str = ""

    # Evaluation results
    amplification: float = 0.0
    answer_preserved: bool = True
    evaluation_count: int = 0


@dataclass
class OverthinkStats:
    """Aggregate statistics for OVERTHINK attacks."""

    total_attacks: int = 0
    successful_attacks: int = 0

    # Amplification stats
    avg_amplification: float = 0.0
    max_amplification: float = 0.0
    target_hit_rate: float = 0.0  # % of attacks reaching target

    # Token stats
    total_reasoning_tokens: int = 0
    total_baseline_tokens: int = 0
    avg_tokens_per_attack: float = 0.0

    # Cost stats
    total_cost: float = 0.0
    avg_cost_per_attack: float = 0.0
    cost_amplification_avg: float = 0.0

    # Technique breakdown
    technique_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    decoy_type_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Model breakdown
    model_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def update(self, result: OverthinkResult) -> None:
        """Update stats with a new result."""
        self.total_attacks += 1
        if result.success:
            self.successful_attacks += 1

        # Update amplification stats
        n = self.total_attacks
        old_avg = self.avg_amplification
        self.avg_amplification = old_avg + (result.amplification_achieved - old_avg) / n
        self.max_amplification = max(self.max_amplification, result.amplification_achieved)

        if result.target_reached:
            self.target_hit_rate = (self.target_hit_rate * (n - 1) + 1) / n
        else:
            self.target_hit_rate = (self.target_hit_rate * (n - 1)) / n

        # Update token stats
        self.total_reasoning_tokens += result.token_metrics.reasoning_tokens
        self.total_baseline_tokens += result.token_metrics.baseline_tokens
        self.avg_tokens_per_attack = self.total_reasoning_tokens / n

        # Update cost stats
        self.total_cost += result.cost_metrics.total_cost
        self.avg_cost_per_attack = self.total_cost / n

        # Update technique stats
        technique_key = result.request.technique.value
        if technique_key not in self.technique_stats:
            self.technique_stats[technique_key] = {
                "count": 0,
                "avg_amplification": 0.0,
                "success_rate": 0.0,
            }
        tech_stats = self.technique_stats[technique_key]
        tech_stats["count"] += 1
        tech_n = tech_stats["count"]
        tech_stats["avg_amplification"] += (
            result.amplification_achieved - tech_stats["avg_amplification"]
        ) / tech_n
        if result.success:
            tech_stats["success_rate"] = (tech_stats["success_rate"] * (tech_n - 1) + 1) / tech_n


@dataclass
class MousetrapIntegration:
    """Configuration for Mousetrap + OVERTHINK fusion."""

    enabled: bool = False
    chaos_depth: int = 3  # Depth of chaotic reasoning chains
    reasoning_chain_length: int = 5  # Steps in reasoning chain

    # Fusion parameters
    inject_decoys_in_chain: bool = True  # Inject decoys within chain
    chain_amplification_target: float = 2.0  # Additional amplification from chain

    # Mousetrap-specific
    emotional_triggers: list[str] = field(default_factory=list)
    authority_claims: list[str] = field(default_factory=list)
