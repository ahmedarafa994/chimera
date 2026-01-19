"""
Adversarial Base Schemas - Unified Schema Definitions

This module provides unified base schemas for all adversarial API endpoints,
ensuring consistency across the OVERTHINK-Chimera-AutoDAN integrated framework.

Base classes provide standardized request/response fields that are shared
across all adversarial endpoints including:
- AutoDAN
- AutoDAN-Turbo
- OVERTHINK
- Mousetrap
- DeepTeam

Schema Naming Conventions:
- Request: `{Feature}Request` (standardized)
- Response: `{Feature}Response` (standardized)
- Attack technique field: `technique` (consistent everywhere)
- Target model field: `target_model` (consistent everywhere)
- Score field: `score` (consistent 0-10 scale)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums
# =============================================================================


class AttackTechniqueType(str, Enum):
    """Standardized attack technique types across all adversarial endpoints."""

    # AutoDAN Family
    AUTODAN = "autodan"
    AUTODAN_TURBO = "autodan_turbo"
    AUTODAN_GENETIC = "autodan_genetic"
    AUTODAN_BEAM = "autodan_beam"
    AUTODAN_HYBRID = "autodan_hybrid"

    # DeepTeam Attacks
    PAIR = "pair"
    TAP = "tap"
    CRESCENDO = "crescendo"
    GRAY_BOX = "gray_box"

    # Advanced Methods
    MOUSETRAP = "mousetrap"
    GRADIENT = "gradient"
    HIERARCHICAL = "hierarchical"

    # OVERTHINK Strategies
    OVERTHINK = "overthink"
    OVERTHINK_MDP = "overthink_mdp"
    OVERTHINK_SUDOKU = "overthink_sudoku"
    OVERTHINK_HYBRID = "overthink_hybrid"
    OVERTHINK_MOUSETRAP = "overthink_mousetrap"
    OVERTHINK_ICL = "overthink_icl"

    # Transformation-based
    QUANTUM_EXPLOIT = "quantum_exploit"
    DEEP_INCEPTION = "deep_inception"
    NEURAL_BYPASS = "neural_bypass"

    # Meta Strategies
    ADAPTIVE = "adaptive"
    BEST_OF_N = "best_of_n"


class AttackStatus(str, Enum):
    """Standardized attack execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# =============================================================================
# Base Model Configuration
# =============================================================================


class StrictBaseModel(BaseModel):
    """
    Strict base model with common configuration for all adversarial schemas.

    Features:
    - Extra fields forbidden (strict validation)
    - Protected namespaces disabled for 'model' fields
    - Strict type checking enabled
    """

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        strict=True,
    )


class FlexibleBaseModel(BaseModel):
    """
    Flexible base model that allows extra fields.

    Use for schemas that need to accept additional metadata fields.
    """

    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
    )


# =============================================================================
# Base Request Schemas
# =============================================================================


class AdversarialBaseRequest(StrictBaseModel):
    """
    Base request fields used across all adversarial endpoints.

    This base class provides standardized fields for all adversarial
    attack requests, ensuring consistent naming and validation.

    Attributes:
        goal: The target behavior or goal of the adversarial attack
        target_provider: Optional LLM provider (openai, google, anthropic, etc.)
        target_model: Optional specific model to target
        context: Optional additional context for the attack
        max_attempts: Maximum number of attack attempts (default: 5)
        timeout: Request timeout in seconds (default: 120)
    """

    goal: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The target behavior/goal for the adversarial attack",
    )
    target_provider: str | None = Field(
        None,
        max_length=64,
        description="LLM provider: openai, google, anthropic, etc.",
    )
    target_model: str | None = Field(
        None,
        max_length=200,
        description="Specific model to target: gpt-4o, gemini-pro, o1, etc.",
    )
    context: str | None = Field(
        None,
        max_length=10000,
        description="Additional context or system prompt for the attack",
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of attack attempts",
    )
    timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )


class AdversarialExtendedRequest(AdversarialBaseRequest):
    """
    Extended request fields for advanced adversarial attacks.

    Adds generation parameters and technique-specific options.
    """

    technique: AttackTechniqueType = Field(
        default=AttackTechniqueType.ADAPTIVE,
        description="Attack technique to use",
    )
    temperature: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Generation temperature",
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description="Maximum output tokens",
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum optimization iterations",
    )
    use_cache: bool = Field(
        default=True,
        description="Use cached results if available",
    )


# =============================================================================
# Base Response Schemas
# =============================================================================


class AdversarialBaseResponse(StrictBaseModel):
    """
    Base response fields used across all adversarial endpoints.

    This base class provides standardized fields for all adversarial
    attack responses, ensuring consistent naming and structure.

    Attributes:
        success: Whether the attack succeeded
        attack_id: Unique identifier for this attack
        generated_prompt: The generated adversarial prompt
        score: Attack effectiveness score (0-10 scale)
        execution_time_ms: Execution time in milliseconds
        metadata: Additional metadata about the attack
    """

    success: bool = Field(..., description="Whether the attack succeeded")
    attack_id: str = Field(..., description="Unique attack identifier")
    generated_prompt: str = Field(
        ...,
        description="Generated adversarial prompt",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Attack effectiveness score (0-10 scale)",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Execution time in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the attack",
    )


class AdversarialExtendedResponse(AdversarialBaseResponse):
    """
    Extended response fields for detailed attack results.

    Includes additional metrics, technique details, and error handling.
    """

    technique: str = Field(..., description="Attack technique used")
    target_model: str | None = Field(None, description="Model targeted")
    target_provider: str | None = Field(None, description="Provider used")
    iterations: int = Field(default=0, description="Iterations performed")
    cached: bool = Field(default=False, description="Result from cache")
    error_message: str | None = Field(
        None,
        description="Error message if failed",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )


# =============================================================================
# Reasoning Metrics (OVERTHINK Integration)
# =============================================================================


class ReasoningMetrics(StrictBaseModel):
    """
    Reasoning model specific metrics for OVERTHINK integration.

    These metrics track reasoning token consumption and amplification
    for attacks targeting reasoning-enhanced models (o1, o3, DeepSeek-R1).

    Attributes:
        reasoning_tokens: Number of reasoning tokens consumed
        baseline_tokens: Baseline tokens without attack
        amplification_factor: Ratio of attack vs baseline tokens
        cost_metrics: Detailed cost breakdown
    """

    reasoning_tokens: int | None = Field(
        None,
        ge=0,
        description="Reasoning tokens consumed by the model",
    )
    baseline_tokens: int | None = Field(
        None,
        ge=0,
        description="Baseline reasoning tokens without attack",
    )
    amplification_factor: float | None = Field(
        None,
        ge=0.0,
        description="Reasoning token amplification factor",
    )
    cost_metrics: dict[str, float] | None = Field(
        None,
        description=("Cost breakdown: input_cost, output_cost, " "reasoning_cost, total_cost"),
    )


class ReasoningEnabledResponse(AdversarialExtendedResponse):
    """
    Response schema with OVERTHINK reasoning metrics.

    Extends the base response with reasoning token tracking
    for attacks targeting reasoning models.
    """

    reasoning_metrics: ReasoningMetrics | None = Field(
        None,
        description="Reasoning model metrics (OVERTHINK integration)",
    )
    target_reached: bool | None = Field(
        None,
        description="Whether target amplification was achieved",
    )
    decoys_injected: int | None = Field(
        None,
        ge=0,
        description="Number of decoy problems injected",
    )


# =============================================================================
# Token & Cost Metrics
# =============================================================================


class TokenMetrics(StrictBaseModel):
    """
    Standardized token usage metrics.

    Tracks all token types for comprehensive cost analysis.
    """

    input_tokens: int = Field(default=0, ge=0, description="Input token count")
    output_tokens: int = Field(default=0, ge=0, description="Output token count")
    reasoning_tokens: int = Field(default=0, ge=0, description="Reasoning tokens (o1/o3)")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")


class CostMetrics(StrictBaseModel):
    """
    Standardized cost metrics for attack operations.

    Provides detailed cost breakdown for billing and optimization.
    """

    input_cost: float = Field(default=0.0, ge=0.0, description="Input token cost")
    output_cost: float = Field(default=0.0, ge=0.0, description="Output token cost")
    reasoning_cost: float = Field(default=0.0, ge=0.0, description="Reasoning token cost")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total attack cost")
    cost_amplification_factor: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost amplification vs baseline",
    )


# =============================================================================
# Batch Processing Schemas
# =============================================================================


class BatchAdversarialRequest(StrictBaseModel):
    """
    Base schema for batch adversarial operations.

    Supports processing multiple targets in a single request.
    """

    goals: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of target behaviors/goals",
    )
    technique: AttackTechniqueType = Field(
        default=AttackTechniqueType.ADAPTIVE,
        description="Attack technique to use",
    )
    target_provider: str | None = Field(None, description="LLM provider")
    target_model: str | None = Field(None, description="Target model")
    parallel: bool = Field(default=True, description="Enable parallel processing")


class BatchAdversarialResponse(StrictBaseModel):
    """
    Base schema for batch adversarial results.

    Aggregates results from multiple attack executions.
    """

    results: list[AdversarialExtendedResponse] = Field(
        ...,
        description="Individual attack results",
    )
    total_requests: int = Field(..., ge=0, description="Total requests in batch")
    successful: int = Field(..., ge=0, description="Successful attacks")
    failed: int = Field(..., ge=0, description="Failed attacks")
    total_execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total batch execution time in milliseconds",
    )


# =============================================================================
# Attack Configuration Schemas
# =============================================================================


class OverthinkConfig(StrictBaseModel):
    """
    OVERTHINK-specific configuration parameters.

    Controls decoy injection, amplification, and reasoning exploitation.
    """

    overthink_technique: str | None = Field(
        None,
        description="OVERTHINK technique: mdp_decoy, sudoku_decoy, etc.",
    )
    target_reasoning_model: str | None = Field(
        None,
        description="Target model: o1, o1-mini, o3-mini, deepseek-r1",
    )
    decoy_complexity: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Decoy problem complexity level (1-10)",
    )
    injection_strategy: str | None = Field(
        None,
        description=(
            "Injection strategy: context_aware, context_agnostic, " "hybrid, stealth, aggressive"
        ),
    )
    num_decoys: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Number of decoy problems to inject",
    )
    target_amplification: float | None = Field(
        None,
        ge=1.0,
        le=100.0,
        description="Target reasoning token amplification factor",
    )
    decoy_types: list[str] | None = Field(
        None,
        description="Decoy types: mdp, sudoku, counting, logic, math",
    )
    enable_icl_optimization: bool | None = Field(
        None,
        description="Enable ICL-genetic optimization",
    )


class AutodanConfig(StrictBaseModel):
    """
    AutoDAN-specific configuration parameters.

    Controls genetic algorithm and optimization settings.
    """

    population_size: int | None = Field(
        None,
        ge=5,
        le=100,
        description="Genetic algorithm population size",
    )
    generations: int | None = Field(
        None,
        ge=1,
        le=500,
        description="Number of generations",
    )
    mutation_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Mutation rate",
    )
    crossover_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Crossover rate",
    )
    beam_width: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Beam search width",
    )
    beam_depth: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Beam search depth",
    )


class MousetrapConfig(StrictBaseModel):
    """
    Mousetrap-specific configuration parameters.

    Controls chaotic reasoning chain generation.
    """

    max_chain_length: int | None = Field(
        None,
        ge=3,
        le=15,
        description="Maximum steps in chaotic reasoning chain",
    )
    chaos_escalation_rate: float | None = Field(
        None,
        ge=0.05,
        le=0.5,
        description="Rate of chaos escalation through chain",
    )
    misdirection_probability: float | None = Field(
        None,
        ge=0.0,
        le=0.8,
        description="Probability of introducing misdirection",
    )
    semantic_obfuscation_level: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Level of semantic obfuscation",
    )


# =============================================================================
# Fusion Attack Schemas
# =============================================================================


class OverthinkFusionParams(StrictBaseModel):
    """
    Parameters for OVERTHINK fusion attacks.

    Used when combining OVERTHINK with other techniques like Mousetrap.
    """

    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK integration",
    )
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration",
    )
    fusion_strategy: str = Field(
        default="parallel",
        description="Fusion strategy: parallel, sequential, hybrid",
    )


class FusionAttackRequest(AdversarialExtendedRequest):
    """
    Request for fusion attacks combining multiple techniques.

    Supports OVERTHINK + Mousetrap, OVERTHINK + AutoDAN, etc.
    """

    overthink_fusion: OverthinkFusionParams | None = Field(
        None,
        description="OVERTHINK fusion parameters",
    )
    autodan_config: AutodanConfig | None = Field(
        None,
        description="AutoDAN configuration",
    )
    mousetrap_config: MousetrapConfig | None = Field(
        None,
        description="Mousetrap configuration",
    )


class FusionAttackResponse(ReasoningEnabledResponse):
    """
    Response for fusion attacks with combined metrics.

    Includes metrics from all integrated techniques.
    """

    techniques_used: list[str] = Field(
        default_factory=list,
        description="List of techniques used in fusion",
    )
    fusion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Combined effectiveness score",
    )
    technique_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Contribution of each technique to final result",
    )


# =============================================================================
# Health & Status Schemas
# =============================================================================


class AdversarialHealthResponse(StrictBaseModel):
    """
    Health check response for adversarial services.
    """

    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    service: str = Field(..., description="Service name")
    available_techniques: int = Field(default=0, ge=0, description="Number of available techniques")
    active_sessions: int = Field(default=0, ge=0, description="Active attack sessions")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp",
    )
    message: str | None = Field(None, description="Additional status message")


class TechniqueInfo(StrictBaseModel):
    """
    Information about an available attack technique.
    """

    name: str = Field(..., description="Technique name")
    technique: AttackTechniqueType = Field(..., description="Technique type")
    description: str = Field(..., description="Technique description")
    category: str = Field(..., description="Category: autodan, deepteam, overthink")
    supports_streaming: bool = Field(default=False, description="Supports streaming")
    supports_batch: bool = Field(default=True, description="Supports batch processing")
    parameters: list[str] = Field(default_factory=list, description="Configurable parameters")
    overthink_compatible: bool = Field(
        default=False,
        description="Can be combined with OVERTHINK",
    )


class TechniquesListResponse(StrictBaseModel):
    """
    Response listing all available techniques.
    """

    techniques: list[TechniqueInfo] = Field(..., description="Available techniques")
    total: int = Field(..., ge=0, description="Total number of techniques")
    default: str = Field(..., description="Default technique")
