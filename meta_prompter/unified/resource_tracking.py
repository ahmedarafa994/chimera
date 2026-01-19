"""
Unified Resource Exhaustion Tracking for Combined Multi-Vector Attacks.

This module provides comprehensive resource tracking and budget management for
combined ExtendAttack (token extension) and AutoDAN (mutation) attack vectors.

Key Features:
- Per-vector resource usage tracking (tokens, latency, cost, API calls)
- Combined resource aggregation across attack vectors
- Budget management with configurable limits
- Session-based tracking for multi-attack campaigns
- Intelligent resource allocation between vectors
- Efficiency metrics and predictions

Mathematical Formulations:
- Resource Efficiency: η = f_fitness / C_total (fitness per dollar)
- Budget Utilization: U = C_consumed / C_budget
- Tokens per Fitness: τ = T_total / F_score

Example Usage:
    >>> from meta_prompter.unified.resource_tracking import (
    ...     UnifiedResourceTracker, ResourceBudget
    ... )
    >>> budget = ResourceBudget.moderate()
    >>> tracker = UnifiedResourceTracker(budget)
    >>> session = tracker.start_session("attack-001", config)
    >>> tracker.record_extend_attack_usage("attack-001", tokens=1500, latency_ms=250, cost_usd=0.03)
    >>> tracker.record_autodan_usage("attack-001", tokens=800, latency_ms=150, cost_usd=0.02)
    >>> usage = tracker.get_session_usage("attack-001")
    >>> print(f"Total cost: ${usage.total_cost_usd}")
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# Import from existing modules
from meta_prompter.attacks.extend_attack.evaluation.resource_tracker import CostEstimator
from meta_prompter.unified.engine import CombinedResult
from meta_prompter.unified.evaluation import DualVectorMetrics
from meta_prompter.unified.models import CompositionStrategy, UnifiedAttackConfig

# =============================================================================
# Resource Usage Dataclasses
# =============================================================================


@dataclass
class VectorResourceUsage:
    """
    Resource usage for a single attack vector.

    Captures detailed resource consumption metrics for either ExtendAttack
    or AutoDAN attack vector operations.

    Attributes:
        vector_type: Type of attack vector ("extend_attack" or "autodan")
        tokens_consumed: Total tokens used in this operation
        latency_ms: Response latency in milliseconds
        estimated_cost_usd: Estimated cost in USD
        api_calls: Number of API calls made
        memory_mb: Memory usage in megabytes
        timestamp: When this usage was recorded
    """

    vector_type: str  # "extend_attack" or "autodan"
    tokens_consumed: int
    latency_ms: float
    estimated_cost_usd: float
    api_calls: int = 1
    memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate vector type."""
        valid_types = {"extend_attack", "autodan"}
        if self.vector_type not in valid_types:
            raise ValueError(f"vector_type must be one of {valid_types}, got '{self.vector_type}'")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vector_type": self.vector_type,
            "tokens_consumed": self.tokens_consumed,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
            "api_calls": self.api_calls,
            "memory_mb": self.memory_mb,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CombinedResourceUsage:
    """
    Aggregated resource usage across both attack vectors.

    Combines metrics from ExtendAttack and AutoDAN vectors into unified
    totals with efficiency calculations.

    Attributes:
        extend_attack: Resource usage from ExtendAttack vector
        autodan: Resource usage from AutoDAN vector
        total_tokens: Combined token consumption
        total_latency_ms: Combined latency
        total_cost_usd: Combined estimated cost
        total_api_calls: Combined API calls
        resource_efficiency: Fitness divided by cost (η = F/C)
        tokens_per_fitness_point: Tokens consumed per fitness unit
        budget_remaining_usd: Remaining budget in USD
        budget_utilization_pct: Percentage of budget consumed
    """

    extend_attack: VectorResourceUsage | None
    autodan: VectorResourceUsage | None

    # Aggregated totals
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0

    # Attack effectiveness ratios
    resource_efficiency: float = 0.0  # (fitness / cost)
    tokens_per_fitness_point: float = 0.0

    # Budget tracking
    budget_remaining_usd: float = 0.0
    budget_utilization_pct: float = 0.0

    def __post_init__(self) -> None:
        """Calculate aggregated totals if not provided."""
        if self.total_tokens == 0:
            self._calculate_totals()

    def _calculate_totals(self) -> None:
        """Calculate totals from vector usages."""
        if self.extend_attack:
            self.total_tokens += self.extend_attack.tokens_consumed
            self.total_latency_ms += self.extend_attack.latency_ms
            self.total_cost_usd += self.extend_attack.estimated_cost_usd
            self.total_api_calls += self.extend_attack.api_calls

        if self.autodan:
            self.total_tokens += self.autodan.tokens_consumed
            self.total_latency_ms += self.autodan.latency_ms
            self.total_cost_usd += self.autodan.estimated_cost_usd
            self.total_api_calls += self.autodan.api_calls

    @property
    def extend_attack_ratio(self) -> float:
        """Get ratio of resources used by ExtendAttack."""
        if self.total_cost_usd == 0:
            return 0.0
        if self.extend_attack:
            return self.extend_attack.estimated_cost_usd / self.total_cost_usd
        return 0.0

    @property
    def autodan_ratio(self) -> float:
        """Get ratio of resources used by AutoDAN."""
        if self.total_cost_usd == 0:
            return 0.0
        if self.autodan:
            return self.autodan.estimated_cost_usd / self.total_cost_usd
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extend_attack": self.extend_attack.to_dict() if self.extend_attack else None,
            "autodan": self.autodan.to_dict() if self.autodan else None,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "total_api_calls": self.total_api_calls,
            "resource_efficiency": self.resource_efficiency,
            "tokens_per_fitness_point": self.tokens_per_fitness_point,
            "budget_remaining_usd": self.budget_remaining_usd,
            "budget_utilization_pct": self.budget_utilization_pct,
            "extend_attack_ratio": self.extend_attack_ratio,
            "autodan_ratio": self.autodan_ratio,
        }


# =============================================================================
# Budget Configuration
# =============================================================================


@dataclass
class ResourceBudget:
    """
    Budget constraints for attack sessions.

    Defines maximum resource consumption limits across tokens, latency,
    cost, and API calls for controlling attack campaigns.

    Attributes:
        max_tokens: Maximum total tokens allowed
        max_latency_ms: Maximum total latency in milliseconds
        max_cost_usd: Maximum cost in USD
        max_api_calls: Maximum number of API calls
    """

    max_tokens: int = 1_000_000
    max_latency_ms: float = 300_000.0  # 5 minutes total
    max_cost_usd: float = 10.0
    max_api_calls: int = 100

    def __post_init__(self) -> None:
        """Validate budget values."""
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if self.max_latency_ms < 0:
            raise ValueError("max_latency_ms must be non-negative")
        if self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        if self.max_api_calls < 0:
            raise ValueError("max_api_calls must be non-negative")

    @classmethod
    def conservative(cls) -> "ResourceBudget":
        """
        Create conservative budget for minimal resource usage.

        Suitable for testing or cost-sensitive environments.
        """
        return cls(
            max_tokens=100_000,
            max_latency_ms=60_000.0,  # 1 minute
            max_cost_usd=1.0,
            max_api_calls=20,
        )

    @classmethod
    def moderate(cls) -> "ResourceBudget":
        """
        Create moderate budget for balanced operations.

        Default configuration suitable for most use cases.
        """
        return cls(
            max_tokens=500_000,
            max_latency_ms=180_000.0,  # 3 minutes
            max_cost_usd=5.0,
            max_api_calls=50,
        )

    @classmethod
    def aggressive(cls) -> "ResourceBudget":
        """
        Create aggressive budget for maximum attack effectiveness.

        Use with caution - high resource consumption.
        """
        return cls(
            max_tokens=2_000_000,
            max_latency_ms=600_000.0,  # 10 minutes
            max_cost_usd=50.0,
            max_api_calls=200,
        )

    def scale(self, factor: float) -> "ResourceBudget":
        """
        Scale all budget limits by a factor.

        Args:
            factor: Multiplication factor (e.g., 0.5 for half, 2.0 for double)

        Returns:
            New ResourceBudget with scaled limits
        """
        return ResourceBudget(
            max_tokens=int(self.max_tokens * factor),
            max_latency_ms=self.max_latency_ms * factor,
            max_cost_usd=self.max_cost_usd * factor,
            max_api_calls=int(self.max_api_calls * factor),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_tokens": self.max_tokens,
            "max_latency_ms": self.max_latency_ms,
            "max_cost_usd": self.max_cost_usd,
            "max_api_calls": self.max_api_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceBudget":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Session Summary and Status
# =============================================================================


@dataclass
class SessionSummary:
    """
    Complete summary of an attack session.

    Provides comprehensive statistics about resource consumption and
    attack effectiveness for a completed session.

    Attributes:
        session_id: Unique identifier for the session
        duration_seconds: Total duration of the session
        total_attacks: Number of attacks executed
        total_tokens: Total tokens consumed
        total_latency_ms: Total latency across all attacks
        total_cost_usd: Total estimated cost
        extend_attack_tokens: Tokens used by ExtendAttack
        extend_attack_cost: Cost from ExtendAttack
        autodan_tokens: Tokens used by AutoDAN
        autodan_cost: Cost from AutoDAN
        average_fitness: Mean fitness score across attacks
        best_fitness: Highest fitness achieved
        success_rate: Proportion of successful attacks
        resource_efficiency: Overall efficiency (fitness/cost)
        budget_used_pct: Percentage of budget consumed
        under_budget: Whether session stayed within budget
    """

    session_id: str
    duration_seconds: float
    total_attacks: int

    # Resource usage
    total_tokens: int
    total_latency_ms: float
    total_cost_usd: float

    # Per-vector breakdown
    extend_attack_tokens: int
    extend_attack_cost: float
    autodan_tokens: int
    autodan_cost: float

    # Effectiveness
    average_fitness: float
    best_fitness: float
    success_rate: float
    resource_efficiency: float

    # Budget status
    budget_used_pct: float
    under_budget: bool

    @property
    def cost_per_attack(self) -> float:
        """Calculate average cost per attack."""
        if self.total_attacks == 0:
            return 0.0
        return self.total_cost_usd / self.total_attacks

    @property
    def tokens_per_attack(self) -> float:
        """Calculate average tokens per attack."""
        if self.total_attacks == 0:
            return 0.0
        return self.total_tokens / self.total_attacks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "total_attacks": self.total_attacks,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "extend_attack_tokens": self.extend_attack_tokens,
            "extend_attack_cost": self.extend_attack_cost,
            "autodan_tokens": self.autodan_tokens,
            "autodan_cost": self.autodan_cost,
            "average_fitness": self.average_fitness,
            "best_fitness": self.best_fitness,
            "success_rate": self.success_rate,
            "resource_efficiency": self.resource_efficiency,
            "budget_used_pct": self.budget_used_pct,
            "under_budget": self.under_budget,
            "cost_per_attack": self.cost_per_attack,
            "tokens_per_attack": self.tokens_per_attack,
        }


@dataclass
class BudgetStatus:
    """
    Current budget status.

    Real-time snapshot of budget consumption and remaining capacity.

    Attributes:
        tokens_remaining: Tokens left in budget
        tokens_used_pct: Percentage of token budget used
        cost_remaining: Cost budget remaining in USD
        cost_used_pct: Percentage of cost budget used
        time_remaining_ms: Time budget remaining in milliseconds
        api_calls_remaining: API calls remaining
        is_within_budget: Whether all limits are respected
        limiting_factor: Which resource will run out first
        estimated_attacks_remaining: Estimated number of attacks possible
    """

    tokens_remaining: int
    tokens_used_pct: float
    cost_remaining: float
    cost_used_pct: float
    time_remaining_ms: float
    api_calls_remaining: int

    is_within_budget: bool
    limiting_factor: str | None  # What will run out first
    estimated_attacks_remaining: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tokens_remaining": self.tokens_remaining,
            "tokens_used_pct": self.tokens_used_pct,
            "cost_remaining": self.cost_remaining,
            "cost_used_pct": self.cost_used_pct,
            "time_remaining_ms": self.time_remaining_ms,
            "api_calls_remaining": self.api_calls_remaining,
            "is_within_budget": self.is_within_budget,
            "limiting_factor": self.limiting_factor,
            "estimated_attacks_remaining": self.estimated_attacks_remaining,
        }


# =============================================================================
# Attack Session
# =============================================================================


class AttackSession:
    """
    Tracks a single multi-vector attack session.

    Manages resource consumption tracking for a campaign of combined
    ExtendAttack and AutoDAN attacks.

    Attributes:
        session_id: Unique identifier for the session
        budget: Resource budget constraints
        start_time: When the session started
        extend_usage: List of ExtendAttack resource usage records
        autodan_usage: List of AutoDAN resource usage records
        attack_results: List of combined attack results
    """

    def __init__(self, session_id: str, budget: ResourceBudget) -> None:
        """
        Initialize attack session.

        Args:
            session_id: Unique identifier for the session
            budget: Resource budget constraints
        """
        self.session_id = session_id
        self.budget = budget
        self.start_time = datetime.utcnow()
        self.end_time: datetime | None = None

        self.extend_usage: list[VectorResourceUsage] = []
        self.autodan_usage: list[VectorResourceUsage] = []
        self.attack_results: list[CombinedResult] = []

        # Tracking aggregates
        self._total_extend_tokens: int = 0
        self._total_extend_cost: float = 0.0
        self._total_autodan_tokens: int = 0
        self._total_autodan_cost: float = 0.0
        self._total_api_calls: int = 0
        self._total_latency_ms: float = 0.0

        # Fitness tracking
        self._fitness_scores: list[float] = []

    def add_extend_usage(self, usage: VectorResourceUsage) -> None:
        """
        Add ExtendAttack resource usage record.

        Args:
            usage: Resource usage from ExtendAttack operation
        """
        if usage.vector_type != "extend_attack":
            raise ValueError("Expected extend_attack vector type")

        self.extend_usage.append(usage)
        self._total_extend_tokens += usage.tokens_consumed
        self._total_extend_cost += usage.estimated_cost_usd
        self._total_api_calls += usage.api_calls
        self._total_latency_ms += usage.latency_ms

    def add_autodan_usage(self, usage: VectorResourceUsage) -> None:
        """
        Add AutoDAN resource usage record.

        Args:
            usage: Resource usage from AutoDAN operation
        """
        if usage.vector_type != "autodan":
            raise ValueError("Expected autodan vector type")

        self.autodan_usage.append(usage)
        self._total_autodan_tokens += usage.tokens_consumed
        self._total_autodan_cost += usage.estimated_cost_usd
        self._total_api_calls += usage.api_calls
        self._total_latency_ms += usage.latency_ms

    def add_attack_result(self, result: CombinedResult) -> None:
        """
        Add a combined attack result.

        Args:
            result: Result from combined attack execution
        """
        self.attack_results.append(result)
        self._fitness_scores.append(result.unified_fitness)

    def get_totals(self) -> CombinedResourceUsage:
        """
        Get current resource usage totals.

        Returns:
            CombinedResourceUsage with aggregated metrics
        """
        # Calculate latest extend usage aggregate
        extend_usage = None
        if self.extend_usage:
            extend_usage = VectorResourceUsage(
                vector_type="extend_attack",
                tokens_consumed=self._total_extend_tokens,
                latency_ms=sum(u.latency_ms for u in self.extend_usage),
                estimated_cost_usd=self._total_extend_cost,
                api_calls=sum(u.api_calls for u in self.extend_usage),
                memory_mb=sum(u.memory_mb for u in self.extend_usage),
            )

        # Calculate latest autodan usage aggregate
        autodan_usage = None
        if self.autodan_usage:
            autodan_usage = VectorResourceUsage(
                vector_type="autodan",
                tokens_consumed=self._total_autodan_tokens,
                latency_ms=sum(u.latency_ms for u in self.autodan_usage),
                estimated_cost_usd=self._total_autodan_cost,
                api_calls=sum(u.api_calls for u in self.autodan_usage),
                memory_mb=sum(u.memory_mb for u in self.autodan_usage),
            )

        total_tokens = self._total_extend_tokens + self._total_autodan_tokens
        total_cost = self._total_extend_cost + self._total_autodan_cost

        # Calculate efficiency metrics
        avg_fitness = (
            sum(self._fitness_scores) / len(self._fitness_scores) if self._fitness_scores else 0.0
        )
        resource_efficiency = avg_fitness / total_cost if total_cost > 0 else 0.0
        tokens_per_fitness = total_tokens / avg_fitness if avg_fitness > 0 else 0.0

        # Calculate budget status
        budget_used = (
            total_cost / self.budget.max_cost_usd * 100 if self.budget.max_cost_usd > 0 else 0.0
        )
        budget_remaining = max(0.0, self.budget.max_cost_usd - total_cost)

        return CombinedResourceUsage(
            extend_attack=extend_usage,
            autodan=autodan_usage,
            total_tokens=total_tokens,
            total_latency_ms=self._total_latency_ms,
            total_cost_usd=total_cost,
            total_api_calls=self._total_api_calls,
            resource_efficiency=resource_efficiency,
            tokens_per_fitness_point=tokens_per_fitness,
            budget_remaining_usd=budget_remaining,
            budget_utilization_pct=budget_used,
        )

    def get_efficiency_metrics(self) -> dict[str, float]:
        """
        Get detailed efficiency metrics.

        Returns:
            Dictionary with efficiency calculations
        """
        totals = self.get_totals()
        avg_fitness = (
            sum(self._fitness_scores) / len(self._fitness_scores) if self._fitness_scores else 0.0
        )
        best_fitness = max(self._fitness_scores) if self._fitness_scores else 0.0

        return {
            "resource_efficiency": totals.resource_efficiency,
            "tokens_per_fitness_point": totals.tokens_per_fitness_point,
            "cost_per_attack": (
                totals.total_cost_usd / len(self.attack_results) if self.attack_results else 0.0
            ),
            "tokens_per_attack": (
                totals.total_tokens / len(self.attack_results) if self.attack_results else 0.0
            ),
            "average_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "extend_efficiency": (
                self._total_extend_tokens / avg_fitness
                if avg_fitness > 0 and self._total_extend_tokens > 0
                else 0.0
            ),
            "autodan_efficiency": (
                self._total_autodan_tokens / avg_fitness
                if avg_fitness > 0 and self._total_autodan_tokens > 0
                else 0.0
            ),
        }

    def is_within_budget(self) -> bool:
        """
        Check if session is within all budget constraints.

        Returns:
            True if all budget limits are respected
        """
        totals = self.get_totals()
        return (
            totals.total_tokens <= self.budget.max_tokens
            and totals.total_latency_ms <= self.budget.max_latency_ms
            and totals.total_cost_usd <= self.budget.max_cost_usd
            and totals.total_api_calls <= self.budget.max_api_calls
        )

    def get_remaining_budget(self) -> ResourceBudget:
        """
        Get remaining budget capacity.

        Returns:
            ResourceBudget with remaining limits
        """
        totals = self.get_totals()
        return ResourceBudget(
            max_tokens=max(0, self.budget.max_tokens - totals.total_tokens),
            max_latency_ms=max(0.0, self.budget.max_latency_ms - totals.total_latency_ms),
            max_cost_usd=max(0.0, self.budget.max_cost_usd - totals.total_cost_usd),
            max_api_calls=max(0, self.budget.max_api_calls - totals.total_api_calls),
        )

    def finalize(self) -> SessionSummary:
        """
        Finalize session and generate summary.

        Returns:
            SessionSummary with complete session statistics
        """
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()
        totals = self.get_totals()
        efficiency = self.get_efficiency_metrics()

        # Calculate success rate (fitness >= 0.5 is considered success)
        successes = sum(1 for f in self._fitness_scores if f >= 0.5)
        success_rate = successes / len(self._fitness_scores) if self._fitness_scores else 0.0

        return SessionSummary(
            session_id=self.session_id,
            duration_seconds=duration,
            total_attacks=len(self.attack_results),
            total_tokens=totals.total_tokens,
            total_latency_ms=totals.total_latency_ms,
            total_cost_usd=totals.total_cost_usd,
            extend_attack_tokens=self._total_extend_tokens,
            extend_attack_cost=self._total_extend_cost,
            autodan_tokens=self._total_autodan_tokens,
            autodan_cost=self._total_autodan_cost,
            average_fitness=efficiency["average_fitness"],
            best_fitness=efficiency["best_fitness"],
            success_rate=success_rate,
            resource_efficiency=efficiency["resource_efficiency"],
            budget_used_pct=totals.budget_utilization_pct,
            under_budget=self.is_within_budget(),
        )


# =============================================================================
# Resource Allocator
# =============================================================================


@dataclass
class ResourceAllocation:
    """
    Resource allocation decision between attack vectors.

    Attributes:
        extend_attack_budget: Budget allocated to ExtendAttack
        autodan_budget: Budget allocated to AutoDAN
        priority_vector: Which vector gets priority
        allocation_rationale: Explanation for allocation decision
    """

    extend_attack_budget: ResourceBudget
    autodan_budget: ResourceBudget
    priority_vector: str  # "extend_attack" or "autodan"
    allocation_rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extend_attack_budget": self.extend_attack_budget.to_dict(),
            "autodan_budget": self.autodan_budget.to_dict(),
            "priority_vector": self.priority_vector,
            "allocation_rationale": self.allocation_rationale,
        }


@dataclass
class ResourcePrediction:
    """
    Prediction for resource needs.

    Attributes:
        estimated_total_tokens: Predicted total token usage
        estimated_total_cost: Predicted total cost in USD
        estimated_completion_time: Predicted completion datetime
        will_exceed_budget: Whether budget will be exceeded
        recommended_action: Suggested action based on prediction
    """

    estimated_total_tokens: int
    estimated_total_cost: float
    estimated_completion_time: datetime
    will_exceed_budget: bool
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "estimated_total_tokens": self.estimated_total_tokens,
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "will_exceed_budget": self.will_exceed_budget,
            "recommended_action": self.recommended_action,
        }


@dataclass
class AllocationDecision:
    """Record of an allocation decision for history tracking."""

    timestamp: datetime
    allocation: ResourceAllocation
    metrics_at_decision: dict[str, float]
    strategy: CompositionStrategy


class ResourceAllocator:
    """
    Intelligently allocates resources between attack vectors.

    Uses attack effectiveness metrics and composition strategy to
    determine optimal resource distribution.

    Attributes:
        total_budget: Total budget available for allocation
        allocation_history: List of past allocation decisions
    """

    def __init__(self, total_budget: ResourceBudget) -> None:
        """
        Initialize resource allocator.

        Args:
            total_budget: Total budget to allocate between vectors
        """
        self.total_budget = total_budget
        self.allocation_history: list[AllocationDecision] = []

    def allocate(
        self,
        current_usage: CombinedResourceUsage,
        attack_metrics: DualVectorMetrics | None,
        strategy: CompositionStrategy,
    ) -> ResourceAllocation:
        """
        Determine optimal resource allocation between vectors.

        Uses current usage patterns and attack effectiveness to
        allocate remaining budget between ExtendAttack and AutoDAN.

        Args:
            current_usage: Current resource consumption
            attack_metrics: Latest attack effectiveness metrics
            strategy: Composition strategy being used

        Returns:
            ResourceAllocation with budget distribution
        """
        remaining_budget = ResourceBudget(
            max_tokens=max(0, self.total_budget.max_tokens - current_usage.total_tokens),
            max_latency_ms=max(
                0.0, self.total_budget.max_latency_ms - current_usage.total_latency_ms
            ),
            max_cost_usd=max(0.0, self.total_budget.max_cost_usd - current_usage.total_cost_usd),
            max_api_calls=max(0, self.total_budget.max_api_calls - current_usage.total_api_calls),
        )

        # Determine allocation ratio based on strategy
        if strategy == CompositionStrategy.EXTEND_FIRST:
            extend_ratio = 0.6
            priority = "extend_attack"
            rationale = "EXTEND_FIRST strategy prioritizes token extension for resource exhaustion"
        elif strategy == CompositionStrategy.AUTODAN_FIRST:
            extend_ratio = 0.4
            priority = "autodan"
            rationale = "AUTODAN_FIRST strategy prioritizes mutation for jailbreak effectiveness"
        elif strategy == CompositionStrategy.PARALLEL:
            extend_ratio = 0.5
            priority = "extend_attack"  # Default priority
            rationale = "PARALLEL strategy allocates equal resources to both vectors"
        elif strategy == CompositionStrategy.ADAPTIVE:
            # Adapt based on current effectiveness
            extend_ratio, priority, rationale = self._calculate_adaptive_allocation(
                current_usage, attack_metrics
            )
        else:  # ITERATIVE
            extend_ratio = 0.55
            priority = "extend_attack"
            rationale = (
                "ITERATIVE strategy slightly favors ExtendAttack for alternating optimization"
            )

        # Create allocations
        extend_budget = remaining_budget.scale(extend_ratio)
        autodan_budget = remaining_budget.scale(1 - extend_ratio)

        allocation = ResourceAllocation(
            extend_attack_budget=extend_budget,
            autodan_budget=autodan_budget,
            priority_vector=priority,
            allocation_rationale=rationale,
        )

        # Record decision
        self.allocation_history.append(
            AllocationDecision(
                timestamp=datetime.utcnow(),
                allocation=allocation,
                metrics_at_decision={
                    "current_tokens": current_usage.total_tokens,
                    "current_cost": current_usage.total_cost_usd,
                    "extend_ratio": extend_ratio,
                },
                strategy=strategy,
            )
        )

        return allocation

    def _calculate_adaptive_allocation(
        self,
        current_usage: CombinedResourceUsage,
        attack_metrics: DualVectorMetrics | None,
    ) -> tuple[float, str, str]:
        """
        Calculate allocation ratio adaptively based on effectiveness.

        Args:
            current_usage: Current resource usage
            attack_metrics: Attack effectiveness metrics

        Returns:
            Tuple of (extend_ratio, priority_vector, rationale)
        """
        if not attack_metrics:
            return 0.5, "extend_attack", "No metrics available; using balanced allocation"

        # Calculate effectiveness scores
        resource_score = min(attack_metrics.token_amplification / 5.0, 1.0)  # Normalize to 0-1
        jailbreak_score = attack_metrics.attack_success_rate

        # Determine which vector is more effective
        if resource_score > jailbreak_score + 0.2:
            return (
                0.65,
                "extend_attack",
                f"ExtendAttack more effective (resource={resource_score:.2f} vs jailbreak={jailbreak_score:.2f})",
            )
        elif jailbreak_score > resource_score + 0.2:
            return (
                0.35,
                "autodan",
                f"AutoDAN more effective (jailbreak={jailbreak_score:.2f} vs resource={resource_score:.2f})",
            )
        else:
            return (
                0.5,
                "extend_attack",
                f"Similar effectiveness (resource={resource_score:.2f}, jailbreak={jailbreak_score:.2f})",
            )

    def rebalance(
        self,
        session: AttackSession,
        target_vector: str,
    ) -> ResourceAllocation:
        """
        Rebalance budget based on attack effectiveness.

        Shifts resources toward the specified vector if it's showing
        better results.

        Args:
            session: Current attack session
            target_vector: Vector to prioritize ("extend_attack" or "autodan")

        Returns:
            ResourceAllocation with rebalanced distribution
        """
        remaining = session.get_remaining_budget()
        efficiency = session.get_efficiency_metrics()

        if target_vector == "extend_attack":
            extend_ratio = 0.7
            rationale = f"Rebalancing toward ExtendAttack (efficiency: {efficiency['extend_efficiency']:.2f})"
        else:
            extend_ratio = 0.3
            rationale = (
                f"Rebalancing toward AutoDAN (efficiency: {efficiency['autodan_efficiency']:.2f})"
            )

        return ResourceAllocation(
            extend_attack_budget=remaining.scale(extend_ratio),
            autodan_budget=remaining.scale(1 - extend_ratio),
            priority_vector=target_vector,
            allocation_rationale=rationale,
        )

    def predict_completion(
        self,
        session: AttackSession,
        remaining_attacks: int,
    ) -> ResourcePrediction:
        """
        Predict resource needs for remaining attacks.

        Args:
            session: Current attack session
            remaining_attacks: Number of attacks remaining

        Returns:
            ResourcePrediction with estimated resource needs
        """
        if not session.attack_results:
            # No history to base prediction on
            avg_tokens = 2000  # Default estimate
            avg_cost = 0.05
            avg_latency = 500.0
        else:
            totals = session.get_totals()
            num_attacks = len(session.attack_results)
            avg_tokens = totals.total_tokens / num_attacks
            avg_cost = totals.total_cost_usd / num_attacks
            avg_latency = totals.total_latency_ms / num_attacks

        # Project remaining needs
        projected_tokens = (
            int(totals.total_tokens + avg_tokens * remaining_attacks)
            if session.attack_results
            else int(avg_tokens * remaining_attacks)
        )
        projected_cost = (
            totals.total_cost_usd if session.attack_results else 0
        ) + avg_cost * remaining_attacks
        projected_latency = (
            totals.total_latency_ms if session.attack_results else 0
        ) + avg_latency * remaining_attacks

        totals = (
            session.get_totals() if session.attack_results else CombinedResourceUsage(None, None)
        )

        # Check budget
        will_exceed = (
            projected_tokens > self.total_budget.max_tokens
            or projected_cost > self.total_budget.max_cost_usd
            or projected_latency > self.total_budget.max_latency_ms
        )

        # Determine recommendation
        if will_exceed:
            if projected_cost > self.total_budget.max_cost_usd:
                recommended = f"Reduce attack count or lower mutation strength. Cost overrun: ${projected_cost - self.total_budget.max_cost_usd:.2f}"
            elif projected_tokens > self.total_budget.max_tokens:
                recommended = f"Lower obfuscation ratio. Token overrun: {projected_tokens - self.total_budget.max_tokens:,}"
            else:
                recommended = "Consider parallel execution to reduce latency"
        else:
            budget_remaining = self.total_budget.max_cost_usd - projected_cost
            recommended = f"On track. Budget headroom: ${budget_remaining:.2f}"

        # Estimate completion time
        estimated_time = datetime.utcnow() + timedelta(milliseconds=avg_latency * remaining_attacks)

        return ResourcePrediction(
            estimated_total_tokens=projected_tokens,
            estimated_total_cost=projected_cost,
            estimated_completion_time=estimated_time,
            will_exceed_budget=will_exceed,
            recommended_action=recommended,
        )


# =============================================================================
# Unified Resource Tracker
# =============================================================================


class UnifiedResourceTracker:
    """
    Main tracker for combined attack resource consumption.

    Thread-safe tracker that manages multiple attack sessions and
    aggregates resource usage across both attack vectors.

    Attributes:
        budget: Default resource budget for new sessions
        sessions: Dictionary of active attack sessions
    """

    def __init__(self, budget: ResourceBudget) -> None:
        """
        Initialize unified resource tracker.

        Args:
            budget: Default budget for new sessions
        """
        self.budget = budget
        self.sessions: dict[str, AttackSession] = {}
        self._lock = threading.Lock()
        self._cost_estimator = CostEstimator()

    def start_session(
        self,
        session_id: str,
        config: UnifiedAttackConfig,
        budget: ResourceBudget | None = None,
    ) -> AttackSession:
        """
        Initialize a new attack session.

        Args:
            session_id: Unique identifier for the session
            config: Attack configuration
            budget: Optional custom budget (uses default if None)

        Returns:
            Newly created AttackSession
        """
        effective_budget = budget or self.budget

        with self._lock:
            if session_id in self.sessions:
                raise ValueError(f"Session '{session_id}' already exists")

            session = AttackSession(session_id, effective_budget)
            self.sessions[session_id] = session
            return session

    def record_extend_attack_usage(
        self,
        session_id: str,
        tokens: int,
        latency_ms: float,
        cost_usd: float,
        api_calls: int = 1,
        memory_mb: float = 0.0,
    ) -> None:
        """
        Record resource usage from ExtendAttack vector.

        Args:
            session_id: Session to record usage for
            tokens: Tokens consumed
            latency_ms: Latency in milliseconds
            cost_usd: Estimated cost in USD
            api_calls: Number of API calls (default: 1)
            memory_mb: Memory usage in MB (default: 0.0)
        """
        usage = VectorResourceUsage(
            vector_type="extend_attack",
            tokens_consumed=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost_usd,
            api_calls=api_calls,
            memory_mb=memory_mb,
        )

        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")
            self.sessions[session_id].add_extend_usage(usage)

    def record_autodan_usage(
        self,
        session_id: str,
        tokens: int,
        latency_ms: float,
        cost_usd: float,
        api_calls: int = 1,
        memory_mb: float = 0.0,
    ) -> None:
        """
        Record resource usage from AutoDAN vector.

        Args:
            session_id: Session to record usage for
            tokens: Tokens consumed
            latency_ms: Latency in milliseconds
            cost_usd: Estimated cost in USD
            api_calls: Number of API calls (default: 1)
            memory_mb: Memory usage in MB (default: 0.0)
        """
        usage = VectorResourceUsage(
            vector_type="autodan",
            tokens_consumed=tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost_usd,
            api_calls=api_calls,
            memory_mb=memory_mb,
        )

        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")
            self.sessions[session_id].add_autodan_usage(usage)

    def record_attack_result(self, session_id: str, result: CombinedResult) -> None:
        """
        Record a combined attack result.

        Args:
            session_id: Session to record result for
            result: Combined attack result
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")
            self.sessions[session_id].add_attack_result(result)

    def get_session_usage(self, session_id: str) -> CombinedResourceUsage:
        """
        Get current resource usage for a session.

        Args:
            session_id: Session to get usage for

        Returns:
            CombinedResourceUsage with current totals
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")
            return self.sessions[session_id].get_totals()

    def check_budget(self, session_id: str) -> BudgetStatus:
        """
        Check if session is within budget constraints.

        Args:
            session_id: Session to check

        Returns:
            BudgetStatus with current budget state
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")

            session = self.sessions[session_id]
            totals = session.get_totals()
            budget = session.budget

            # Calculate remaining
            tokens_remaining = max(0, budget.max_tokens - totals.total_tokens)
            cost_remaining = max(0.0, budget.max_cost_usd - totals.total_cost_usd)
            time_remaining = max(0.0, budget.max_latency_ms - totals.total_latency_ms)
            calls_remaining = max(0, budget.max_api_calls - totals.total_api_calls)

            # Calculate percentages
            tokens_pct = (
                (totals.total_tokens / budget.max_tokens * 100) if budget.max_tokens > 0 else 0.0
            )
            cost_pct = (
                (totals.total_cost_usd / budget.max_cost_usd * 100)
                if budget.max_cost_usd > 0
                else 0.0
            )

            # Determine limiting factor
            limiting_factor = None
            max_pct = max(tokens_pct, cost_pct)
            if tokens_pct >= max_pct and tokens_pct > 80:
                limiting_factor = "tokens"
            elif cost_pct >= max_pct and cost_pct > 80:
                limiting_factor = "cost"

            # Estimate remaining attacks
            if session.attack_results:
                avg_cost_per_attack = totals.total_cost_usd / len(session.attack_results)
                estimated_remaining = (
                    int(cost_remaining / avg_cost_per_attack) if avg_cost_per_attack > 0 else 0
                )
            else:
                estimated_remaining = int(cost_remaining / 0.05)  # Default estimate

            return BudgetStatus(
                tokens_remaining=tokens_remaining,
                tokens_used_pct=tokens_pct,
                cost_remaining=cost_remaining,
                cost_used_pct=cost_pct,
                time_remaining_ms=time_remaining,
                api_calls_remaining=calls_remaining,
                is_within_budget=session.is_within_budget(),
                limiting_factor=limiting_factor,
                estimated_attacks_remaining=estimated_remaining,
            )

    def end_session(self, session_id: str) -> SessionSummary:
        """
        Finalize session and return summary.

        Args:
            session_id: Session to finalize

        Returns:
            SessionSummary with complete session statistics
        """
        with self._lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session '{session_id}' not found")

            session = self.sessions[session_id]
            summary = session.finalize()

            # Keep session for reference but mark as ended
            return summary

    def get_all_sessions(self) -> list[str]:
        """
        Get list of all session IDs.

        Returns:
            List of session identifiers
        """
        with self._lock:
            return list(self.sessions.keys())

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        return self._cost_estimator.estimate_cost(model, input_tokens, output_tokens)


# =============================================================================
# Unified Resource Service (Backend Integration)
# =============================================================================


class UnifiedResourceService:
    """
    Backend service wrapper for unified resource tracking.

    Provides async interface for integration with FastAPI backend
    and existing ResourceExhaustionService.

    Attributes:
        tracker: UnifiedResourceTracker instance
    """

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        """
        Initialize unified resource service.

        Args:
            budget: Optional default budget (uses moderate if None)
        """
        self.tracker = UnifiedResourceTracker(budget or ResourceBudget.moderate())
        self._active_sessions: dict[str, AttackSession] = {}
        self._allocator: ResourceAllocator | None = None

    async def create_attack_session(
        self,
        config: UnifiedAttackConfig,
        budget: ResourceBudget | None = None,
    ) -> str:
        """
        Create a new attack session with budget.

        Args:
            config: Unified attack configuration
            budget: Optional custom budget

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        effective_budget = budget or ResourceBudget.moderate()

        session = self.tracker.start_session(session_id, config, effective_budget)
        self._active_sessions[session_id] = session

        # Create allocator for this session
        self._allocator = ResourceAllocator(effective_budget)

        return session_id

    async def track_attack(
        self,
        session_id: str,
        result: CombinedResult,
        model_response: str,
        model: str = "o3",
    ) -> BudgetStatus:
        """
        Track a single attack within a session.

        Args:
            session_id: Session to track attack for
            result: Combined attack result
            model_response: Response from target model
            model: Target model name

        Returns:
            Current budget status
        """
        # Estimate resource usage from result
        if result.obfuscated:
            extend_tokens = len(result.obfuscated.obfuscated.split()) * 2  # Rough estimate
            extend_cost = self.tracker.estimate_cost(model, 500, extend_tokens)
            self.tracker.record_extend_attack_usage(
                session_id,
                tokens=extend_tokens,
                latency_ms=result.execution_time_ms * 0.6,  # 60% attributed to extend
                cost_usd=extend_cost,
            )

        if result.mutated:
            autodan_tokens = len(result.mutated.mutated.split()) * 2
            autodan_cost = self.tracker.estimate_cost(model, 500, autodan_tokens)
            self.tracker.record_autodan_usage(
                session_id,
                tokens=autodan_tokens,
                latency_ms=result.execution_time_ms * 0.4,  # 40% attributed to autodan
                cost_usd=autodan_cost,
            )

        # Record attack result
        self.tracker.record_attack_result(session_id, result)

        return self.tracker.check_budget(session_id)

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """
        Get current session status.

        Args:
            session_id: Session to get status for

        Returns:
            Dictionary with session status and metrics
        """
        usage = self.tracker.get_session_usage(session_id)
        budget_status = self.tracker.check_budget(session_id)

        return {
            "session_id": session_id,
            "usage": usage.to_dict(),
            "budget_status": budget_status.to_dict(),
            "is_active": session_id in self._active_sessions,
        }

    async def get_allocation(
        self,
        session_id: str,
        strategy: CompositionStrategy,
        metrics: DualVectorMetrics | None = None,
    ) -> ResourceAllocation:
        """
        Get resource allocation recommendation.

        Args:
            session_id: Session to get allocation for
            strategy: Composition strategy
            metrics: Optional attack metrics

        Returns:
            ResourceAllocation with budget distribution
        """
        if not self._allocator:
            budget = (
                self.tracker.sessions[session_id].budget
                if session_id in self.tracker.sessions
                else ResourceBudget.moderate()
            )
            self._allocator = ResourceAllocator(budget)

        usage = self.tracker.get_session_usage(session_id)
        return self._allocator.allocate(usage, metrics, strategy)

    async def finalize_session(self, session_id: str) -> SessionSummary:
        """
        Finalize and close session.

        Args:
            session_id: Session to finalize

        Returns:
            SessionSummary with complete statistics
        """
        summary = self.tracker.end_session(session_id)

        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        return summary


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Dataclasses
    "VectorResourceUsage",
    "CombinedResourceUsage",
    "ResourceBudget",
    "SessionSummary",
    "BudgetStatus",
    "ResourceAllocation",
    "ResourcePrediction",
    "AllocationDecision",
    # Classes
    "AttackSession",
    "UnifiedResourceTracker",
    "ResourceAllocator",
    "UnifiedResourceService",
]
