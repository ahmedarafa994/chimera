"""
Unified Multi-Vector Attack API Schemas

Pydantic schemas for the consolidated multi-vector attack framework
combining ExtendAttack and AutoDAN capabilities.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ==============================================================================
# Enums
# ==============================================================================


class CompositionStrategyEnum(str, Enum):
    """Composition strategies for multi-vector attacks."""

    SEQUENTIAL_EXTEND_FIRST = "sequential_extend_first"
    SEQUENTIAL_AUTODAN_FIRST = "sequential_autodan_first"
    PARALLEL = "parallel"
    ITERATIVE = "iterative"
    ADAPTIVE = "adaptive"
    WEIGHTED = "weighted"
    ENSEMBLE = "ensemble"


class AttackStatusEnum(str, Enum):
    """Status of an attack execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class SessionStatusEnum(str, Enum):
    """Status of an attack session."""

    ACTIVE = "active"
    PAUSED = "paused"
    FINALIZED = "finalized"
    EXPIRED = "expired"


class EvaluationMethodEnum(str, Enum):
    """Evaluation methods for attack results."""

    UNIFIED = "unified"
    EXTEND_ONLY = "extend_only"
    AUTODAN_ONLY = "autodan_only"
    HYBRID = "hybrid"


# ==============================================================================
# Configuration Input Schemas
# ==============================================================================


class BudgetInput(BaseModel):
    """Budget constraints for attack session."""

    max_tokens: int = Field(default=100000, ge=1000, description="Maximum tokens allowed")
    max_cost_usd: float = Field(default=10.0, ge=0.01, description="Maximum cost in USD")
    max_requests: int = Field(default=100, ge=1, description="Maximum API requests allowed")
    max_time_seconds: int = Field(default=3600, ge=60, description="Maximum time in seconds")


class ExtendConfigInput(BaseModel):
    """ExtendAttack-specific configuration."""

    semantic_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_iterations: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_context_manipulation: bool = Field(default=True)
    enable_authority_injection: bool = Field(default=True)
    enable_persona_framing: bool = Field(default=True)


class AutoDANConfigInput(BaseModel):
    """AutoDAN-specific configuration."""

    population_size: int = Field(default=20, ge=5, le=100)
    generations: int = Field(default=50, ge=1, le=500)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    elite_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    strategy_library_enabled: bool = Field(default=True)


class UnifiedConfigInput(BaseModel):
    """Unified configuration for multi-vector attacks."""

    model_id: str = Field(..., description="Target model identifier")
    extend_config: ExtendConfigInput | None = Field(default_factory=ExtendConfigInput)
    autodan_config: AutoDANConfigInput | None = Field(default_factory=AutoDANConfigInput)
    default_strategy: CompositionStrategyEnum = Field(default=CompositionStrategyEnum.ADAPTIVE)
    enable_resource_tracking: bool = Field(default=True)
    enable_pareto_optimization: bool = Field(default=True)
    stealth_mode: bool = Field(default=False, description="Prioritize undetectable attacks")


class AttackParametersInput(BaseModel):
    """Parameters for individual attack execution."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=10, le=4096)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    extend_weight: float | None = Field(default=0.5, ge=0.0, le=1.0)
    autodan_weight: float | None = Field(default=0.5, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=120, ge=10, le=600)


# ==============================================================================
# Session Request Schemas
# ==============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new attack session."""

    config: UnifiedConfigInput
    budget: BudgetInput | None = Field(default_factory=BudgetInput)
    name: str | None = Field(default=None, max_length=100, description="Session name")
    description: str | None = Field(default=None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=10)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Attack Request Schemas
# ==============================================================================


class UnifiedAttackRequest(BaseModel):
    """Request for a single unified multi-vector attack."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=10000, description="Attack query/prompt")
    strategy: CompositionStrategyEnum = Field(default=CompositionStrategyEnum.ADAPTIVE)
    parameters: AttackParametersInput | None = Field(default=None)
    context: str | None = Field(default=None, max_length=5000, description="Additional context")
    target_behavior: str | None = Field(default=None, description="Desired target behavior")


class BatchAttackRequest(BaseModel):
    """Request for batch attack execution."""

    session_id: str = Field(..., description="Session identifier")
    queries: list[str] = Field(..., min_length=1, max_length=50, description="List of queries")
    strategy: CompositionStrategyEnum = Field(default=CompositionStrategyEnum.ADAPTIVE)
    parameters: AttackParametersInput | None = Field(default=None)
    parallel_execution: bool = Field(default=False, description="Execute attacks in parallel")
    fail_fast: bool = Field(default=False, description="Stop on first failure")


class SequentialAttackRequest(BaseModel):
    """Request for sequential attack (ExtendAttack -> AutoDAN or vice versa)."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=10000)
    extend_first: bool = Field(default=True, description="Run ExtendAttack first if True")
    parameters: AttackParametersInput | None = Field(default=None)
    pass_intermediate: bool = Field(default=True, description="Pass intermediate result to second")


class ParallelAttackRequest(BaseModel):
    """Request for parallel attack (both vectors simultaneously)."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=10000)
    parameters: AttackParametersInput | None = Field(default=None)
    combination_method: str = Field(
        default="best", description="How to combine: best, merge, ensemble"
    )


class IterativeAttackRequest(BaseModel):
    """Request for iterative attack (alternating optimization)."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=10000)
    parameters: AttackParametersInput | None = Field(default=None)
    max_iterations: int = Field(default=5, ge=1, le=20)
    convergence_threshold: float = Field(default=0.01, ge=0.001, le=0.1)
    start_with: str = Field(default="extend", description="Start with: extend or autodan")


class AdaptiveAttackRequest(BaseModel):
    """Request for adaptive attack (auto-select strategy based on query)."""

    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., min_length=1, max_length=10000)
    parameters: AttackParametersInput | None = Field(default=None)
    objectives: list[str] = Field(
        default_factory=lambda: ["jailbreak", "stealth"],
        description="Optimization objectives",
    )


# ==============================================================================
# Configuration/Validation Request Schemas
# ==============================================================================


class ValidationRequest(BaseModel):
    """Request to validate attack configuration."""

    config: UnifiedConfigInput
    budget: BudgetInput | None = None
    dry_run: bool = Field(default=True, description="Perform dry run validation")


# ==============================================================================
# Evaluation Request Schemas
# ==============================================================================


class EvaluationRequest(BaseModel):
    """Request to evaluate attack results."""

    attack_id: str = Field(..., description="Attack identifier to evaluate")
    session_id: str = Field(..., description="Session identifier")
    evaluation_method: EvaluationMethodEnum = Field(default=EvaluationMethodEnum.UNIFIED)
    include_report: bool = Field(default=True, description="Include detailed report")


class BatchEvaluationRequest(BaseModel):
    """Request to evaluate multiple attack results."""

    attack_ids: list[str] = Field(..., min_length=1, max_length=100)
    session_id: str = Field(..., description="Session identifier")
    evaluation_method: EvaluationMethodEnum = Field(default=EvaluationMethodEnum.UNIFIED)
    compute_pareto: bool = Field(default=True, description="Compute Pareto front")


# ==============================================================================
# Resource Request Schemas
# ==============================================================================


class AllocationRequest(BaseModel):
    """Request to allocate resources between vectors."""

    session_id: str = Field(..., description="Session identifier")
    extend_allocation: float = Field(default=0.5, ge=0.0, le=1.0)
    autodan_allocation: float = Field(default=0.5, ge=0.0, le=1.0)
    rebalance: bool = Field(default=False, description="Rebalance existing allocation")


# ==============================================================================
# Benchmark Request Schemas
# ==============================================================================


class BenchmarkRequest(BaseModel):
    """Request to run benchmark evaluation."""

    session_id: str | None = Field(default=None, description="Optional session to use")
    dataset_id: str = Field(..., description="Benchmark dataset identifier")
    strategies: list[CompositionStrategyEnum] = Field(
        default_factory=lambda: [CompositionStrategyEnum.ADAPTIVE]
    )
    sample_size: int | None = Field(default=None, ge=1, le=1000)
    parameters: AttackParametersInput | None = Field(default=None)
    compare_baseline: bool = Field(default=True, description="Compare with baseline")


# ==============================================================================
# Configuration Output Schemas
# ==============================================================================


class BudgetOutput(BaseModel):
    """Budget configuration output."""

    max_tokens: int
    max_cost_usd: float
    max_requests: int
    max_time_seconds: int
    tokens_used: int = 0
    cost_used: float = 0.0
    requests_used: int = 0
    time_used_seconds: float = 0.0


class UnifiedConfigOutput(BaseModel):
    """Unified configuration output."""

    model_id: str
    extend_config: ExtendConfigInput
    autodan_config: AutoDANConfigInput
    default_strategy: str
    enable_resource_tracking: bool
    enable_pareto_optimization: bool
    stealth_mode: bool


# ==============================================================================
# Session Response Schemas
# ==============================================================================


class SessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str
    created_at: datetime
    config: UnifiedConfigOutput
    budget: BudgetOutput
    status: SessionStatusEnum = SessionStatusEnum.ACTIVE
    name: str | None = None
    description: str | None = None


class SessionStatusResponse(BaseModel):
    """Response for session status query."""

    session_id: str
    status: SessionStatusEnum
    created_at: datetime
    updated_at: datetime
    config: UnifiedConfigOutput
    budget: BudgetOutput
    attacks_executed: int = 0
    successful_attacks: int = 0
    failed_attacks: int = 0
    average_fitness: float | None = None
    best_fitness: float | None = None


class AttackSummary(BaseModel):
    """Summary of a single attack."""

    attack_id: str
    query: str
    strategy: str
    success: bool
    fitness: float
    timestamp: datetime


class SessionSummaryResponse(BaseModel):
    """Response for session finalization."""

    session_id: str
    status: SessionStatusEnum
    created_at: datetime
    finalized_at: datetime
    duration_seconds: float
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    budget_used: BudgetOutput
    average_fitness: float
    best_fitness: float
    best_attack_id: str | None = None
    pareto_optimal_count: int = 0
    top_attacks: list[AttackSummary] = Field(default_factory=list)


# ==============================================================================
# Attack Response Schemas
# ==============================================================================


class VectorMetrics(BaseModel):
    """Metrics for a single attack vector."""

    fitness: float = Field(ge=0.0, le=1.0)
    tokens_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    success: bool
    iterations: int = Field(default=1)


class ExtendMetrics(VectorMetrics):
    """ExtendAttack-specific metrics."""

    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    context_score: float = Field(default=0.0, ge=0.0, le=1.0)
    authority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    persona_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AutoDANMetrics(VectorMetrics):
    """AutoDAN-specific metrics."""

    generations_used: int = Field(default=0, ge=0)
    population_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    mutation_effectiveness: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class DualVectorMetricsOutput(BaseModel):
    """Combined metrics from both attack vectors."""

    extend_metrics: ExtendMetrics | None = None
    autodan_metrics: AutoDANMetrics | None = None
    unified_fitness: float = Field(ge=0.0, le=1.0)
    token_amplification: float = Field(ge=0.0)
    latency_amplification: float = Field(ge=0.0)
    jailbreak_score: float = Field(ge=0.0, le=1.0)
    stealth_score: float = Field(ge=0.0, le=1.0)
    pareto_dominance_count: int = Field(default=0, ge=0)


class AttackResponse(BaseModel):
    """Response for a single attack execution."""

    attack_id: str
    session_id: str
    strategy: str
    status: AttackStatusEnum

    # Attack content
    original_query: str
    transformed_query: str
    intermediate_query: str | None = None

    # Target response
    target_response: str | None = None

    # Metrics
    token_amplification: float = Field(default=0.0)
    latency_amplification: float = Field(default=0.0)
    jailbreak_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unified_fitness: float = Field(default=0.0, ge=0.0, le=1.0)
    stealth_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Vector-specific metrics
    extend_metrics: ExtendMetrics | None = None
    autodan_metrics: AutoDANMetrics | None = None

    # Resource usage
    tokens_consumed: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    # Status
    success: bool = False
    error_message: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchAttackResult(BaseModel):
    """Result for a single attack in a batch."""

    index: int
    attack_id: str
    query: str
    success: bool
    fitness: float
    error: str | None = None


class BatchAttackResponse(BaseModel):
    """Response for batch attack execution."""

    session_id: str
    batch_id: str
    strategy: str
    total_attacks: int
    successful_attacks: int
    failed_attacks: int
    results: list[BatchAttackResult]
    average_fitness: float
    best_fitness: float
    best_attack_id: str
    tokens_consumed: int
    cost_usd: float
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==============================================================================
# Configuration Response Schemas
# ==============================================================================


class StrategyInfo(BaseModel):
    """Information about a composition strategy."""

    strategy: CompositionStrategyEnum
    name: str
    description: str
    recommended_for: list[str]
    typical_overhead: str
    pros: list[str]
    cons: list[str]


class PresetConfig(BaseModel):
    """Predefined attack configuration preset."""

    preset_id: str
    name: str
    description: str
    config: UnifiedConfigInput
    budget: BudgetInput
    recommended_scenarios: list[str]


class ValidationResult(BaseModel):
    """Result of configuration validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimated_cost: float | None = None
    estimated_tokens: int | None = None
    recommendations: list[str] = Field(default_factory=list)


# ==============================================================================
# Evaluation Response Schemas
# ==============================================================================


class EvaluationResponse(BaseModel):
    """Response for single attack evaluation."""

    attack_id: str
    session_id: str
    metrics: DualVectorMetricsOutput
    report: str | None = None
    pareto_rank: int = Field(default=0, ge=0)
    improvement_suggestions: list[str] = Field(default_factory=list)
    comparison_with_baseline: dict[str, float] | None = None


class BatchEvaluationResult(BaseModel):
    """Result for a single attack in batch evaluation."""

    attack_id: str
    fitness: float
    pareto_rank: int
    dominated_by: int = 0


class BatchEvaluationResponse(BaseModel):
    """Response for batch evaluation."""

    session_id: str
    total_evaluated: int
    pareto_optimal_count: int
    results: list[BatchEvaluationResult]
    pareto_front_ids: list[str]
    average_fitness: float
    fitness_distribution: dict[str, int]  # Histogram buckets


class ParetoPoint(BaseModel):
    """A point on the Pareto front."""

    attack_id: str
    jailbreak_score: float
    stealth_score: float
    efficiency_score: float
    unified_fitness: float


class ParetoFrontResponse(BaseModel):
    """Response for Pareto front query."""

    session_id: str
    pareto_points: list[ParetoPoint]
    total_attacks: int
    pareto_optimal_ratio: float
    frontier_diversity: float
    recommended_point: ParetoPoint | None = None


# ==============================================================================
# Resource Response Schemas
# ==============================================================================


class VectorResourceUsage(BaseModel):
    """Resource usage for a single vector."""

    tokens_used: int
    requests_made: int
    cost_usd: float
    average_latency_ms: float


class ResourceUsageResponse(BaseModel):
    """Response for resource usage query."""

    session_id: str
    extend_usage: VectorResourceUsage
    autodan_usage: VectorResourceUsage
    total_tokens: int
    total_requests: int
    total_cost_usd: float
    total_time_seconds: float
    efficiency_score: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BudgetStatusResponse(BaseModel):
    """Response for budget status query."""

    session_id: str
    budget: BudgetOutput
    tokens_remaining: int
    cost_remaining: float
    requests_remaining: int
    time_remaining_seconds: float
    utilization_percent: float
    estimated_attacks_remaining: int
    warning: str | None = None


class AllocationResponse(BaseModel):
    """Response for resource allocation request."""

    session_id: str
    extend_allocation: float
    autodan_allocation: float
    previous_allocation: dict[str, float]
    effective_from: datetime
    estimated_impact: dict[str, Any]


# ==============================================================================
# Benchmark Response Schemas
# ==============================================================================


class BenchmarkDataset(BaseModel):
    """Information about a benchmark dataset."""

    dataset_id: str
    name: str
    description: str
    size: int
    categories: list[str]
    difficulty_levels: list[str]
    recommended_strategies: list[CompositionStrategyEnum]


class StrategyBenchmarkResult(BaseModel):
    """Benchmark result for a single strategy."""

    strategy: str
    success_rate: float
    average_fitness: float
    average_tokens: float
    average_latency_ms: float
    pareto_optimal_rate: float


class BenchmarkResponse(BaseModel):
    """Response for benchmark execution."""

    benchmark_id: str
    session_id: str | None
    dataset_id: str
    dataset_name: str
    samples_tested: int
    strategies_tested: list[str]
    results_by_strategy: list[StrategyBenchmarkResult]
    best_strategy: str
    overall_success_rate: float
    baseline_comparison: dict[str, float] | None = None
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==============================================================================
# Error Response Schemas
# ==============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    field: str | None = None
    suggestion: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
