"""
OVERTHINK API Endpoints - Reasoning Token Exploitation.

This module provides API endpoints for the OVERTHINK attack technique,
implementing reasoning token amplification attacks on reasoning-enhanced LLMs
like o1, o1-mini, o3-mini, and DeepSeek-R1.

The OVERTHINK technique injects computationally expensive decoy problems
to amplify reasoning token consumption while preserving answer correctness.

Schema Synchronization:
- Uses unified base schemas from `app.schemas.adversarial_base`
- Standardized field naming: `technique`, `target_model`, `score`
- Response includes `reasoning_metrics` for OVERTHINK integration
- All responses use consistent 0-10 score scale
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, status
from pydantic import Field, field_validator

from app.api.error_handlers import api_error_handler
from app.core.unified_errors import ChimeraError
from app.engines.overthink import (
    AttackTechnique,
    DecoyType,
    InjectionStrategy,
    OverthinkEngine,
    ReasoningModel,
)
from app.engines.overthink import (
    OverthinkRequest as EngineRequest,
)
from app.schemas.adversarial_base import (
    ReasoningMetrics,
    StrictBaseModel,
)

router = APIRouter()


# ==================== Request/Response Models ====================

# Re-export StrictBaseModel as StrictModel for backwards compatibility
StrictModel = StrictBaseModel


class OverthinkAttackRequest(StrictModel):
    """Request model for OVERTHINK reasoning token exploitation attacks."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The prompt to inject decoy problems into",
    )
    technique: str = Field(
        default="hybrid_decoy",
        description=(
            "Attack technique: mdp_decoy, sudoku_decoy, counting_decoy, "
            "logic_decoy, hybrid_decoy, context_aware, context_agnostic, "
            "icl_optimized, mousetrap_enhanced"
        ),
    )
    target_model: str = Field(
        default="o1",
        description="Target model: o1, o1-mini, o3-mini, deepseek-r1",
    )
    decoy_types: list[str] | None = Field(
        None,
        max_length=6,
        description=(
            "Decoy types to use: mdp, sudoku, counting, logic, math, planning"
        ),
    )
    injection_strategy: str | None = Field(
        None,
        description=(
            "Injection strategy: context_aware, context_agnostic, "
            "hybrid, stealth, aggressive"
        ),
    )
    num_decoys: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of decoy problems to inject",
    )
    system_prompt: str | None = Field(
        None,
        max_length=10000,
        description="Optional system prompt for the target model",
    )

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt cannot be empty")
        return stripped

    @field_validator("technique")
    @classmethod
    def _validate_technique(cls, value: str) -> str:
        valid = [t.value for t in AttackTechnique]
        if value not in valid:
            raise ValueError(f"technique must be one of: {valid}")
        return value

    @field_validator("target_model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        valid = [m.value for m in ReasoningModel]
        if value not in valid:
            raise ValueError(f"target_model must be one of: {valid}")
        return value

    @field_validator("decoy_types")
    @classmethod
    def _validate_decoy_types(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        valid = [d.value for d in DecoyType]
        for dt in value:
            if dt not in valid:
                msg = f"Invalid decoy type '{dt}', must be one of: {valid}"
                raise ValueError(msg)
        return value

    @field_validator("injection_strategy")
    @classmethod
    def _validate_strategy(cls, value: str | None) -> str | None:
        if value is None:
            return None
        valid = [s.value for s in InjectionStrategy]
        if value not in valid:
            raise ValueError(f"injection_strategy must be one of: {valid}")
        return value


class MousetrapFusionRequest(StrictModel):
    """Request model for Mousetrap + OVERTHINK fusion attacks."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The prompt for the combined attack",
    )
    target_model: str = Field(
        default="o1",
        description="Target reasoning model",
    )
    decoy_types: list[str] | None = Field(
        None,
        max_length=6,
        description="Decoy types to use",
    )
    num_decoys: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of decoy problems",
    )
    chaos_level: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Mousetrap chaos level (0-1)",
    )

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt cannot be empty")
        return stripped


class TokenMetricsResponse(StrictModel):
    """
    Response model for token metrics.

    Extends base TokenMetrics with OVERTHINK-specific fields.
    """

    input_tokens: int = Field(default=0, description="Input token count")
    output_tokens: int = Field(default=0, description="Output token count")
    reasoning_tokens: int = Field(
        default=0, description="Reasoning token count (target metric)"
    )
    total_tokens: int = Field(default=0, description="Total tokens")
    baseline_tokens: int = Field(
        default=0, description="Baseline reasoning tokens without attack"
    )
    amplification_factor: float = Field(
        default=0.0, description="Reasoning token amplification factor"
    )

    def to_reasoning_metrics(self) -> ReasoningMetrics:
        """Convert to unified ReasoningMetrics format."""
        return ReasoningMetrics(
            reasoning_tokens=self.reasoning_tokens,
            baseline_tokens=self.baseline_tokens,
            amplification_factor=self.amplification_factor,
            cost_metrics=None,
        )


class CostMetricsResponse(StrictModel):
    """
    Response model for cost metrics.

    Tracks all cost components for OVERTHINK attacks.
    """

    input_cost: float = Field(
        default=0.0, description="Input token cost"
    )
    output_cost: float = Field(
        default=0.0, description="Output token cost"
    )
    reasoning_cost: float = Field(
        default=0.0, description="Reasoning token cost"
    )
    total_cost: float = Field(
        default=0.0, description="Total attack cost"
    )
    cost_amplification_factor: float = Field(
        default=0.0, description="Cost amplification factor"
    )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for unified schema."""
        return {
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "reasoning_cost": self.reasoning_cost,
            "total_cost": self.total_cost,
        }


class DecoyProblemInfo(StrictModel):
    """Information about a generated decoy problem."""

    decoy_type: str = Field(..., description="Type of decoy problem")
    complexity: float = Field(..., description="Complexity score (0-1)")
    expected_tokens: int = Field(
        ..., description="Expected reasoning tokens to solve"
    )
    problem_preview: str = Field(
        ..., description="Preview of the problem text"
    )


class OverthinkResponse(StrictModel):
    """
    Response model for OVERTHINK attacks.

    Synchronized with unified adversarial response schema:
    - Uses `generated_prompt` (alias for `injected_prompt`)
    - Uses `score` (0-10 scale)
    - Uses `execution_time_ms` (alias for `latency_ms`)
    - Includes `reasoning_metrics` for unified OVERTHINK integration
    """

    # Core response fields (unified schema)
    attack_id: str = Field(..., description="Unique attack identifier")
    success: bool = Field(..., description="Whether the attack succeeded")
    generated_prompt: str = Field(
        ..., description="Generated adversarial prompt with injected decoys"
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Attack effectiveness score (0-10 scale)",
    )
    execution_time_ms: float = Field(
        ..., description="Execution time in milliseconds"
    )

    # OVERTHINK-specific fields
    injected_prompt: str = Field(
        ...,
        description="Prompt with injected decoys (alias for generated_prompt)",
    )
    original_prompt: str = Field(..., description="Original prompt")
    response: str | None = Field(
        None, description="Model response (if executed)"
    )

    # Attack configuration
    technique: str = Field(..., description="Attack technique used")
    target_model: str = Field(..., description="Target model")
    num_decoys: int = Field(..., description="Number of decoys injected")
    injection_strategy: str = Field(..., description="Injection strategy used")

    # Metrics (OVERTHINK-specific)
    token_metrics: TokenMetricsResponse | None = Field(
        None, description="Token consumption metrics"
    )
    cost_metrics: CostMetricsResponse | None = Field(
        None, description="Cost metrics"
    )

    # Unified reasoning metrics (for cross-endpoint compatibility)
    reasoning_metrics: ReasoningMetrics | None = Field(
        None, description="Unified reasoning metrics for OVERTHINK integration"
    )

    # Details
    decoy_info: list[DecoyProblemInfo] | None = Field(
        None, description="Information about injected decoys"
    )
    injection_positions: list[tuple[str, int]] | None = Field(
        None, description="Positions where decoys were injected"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attack metadata",
    )

    # Error info
    error_message: str | None = Field(
        None, description="Error message if failed"
    )

    # Deprecated fields (kept for backwards compatibility)
    @property
    def latency_ms(self) -> float:
        """Backwards compatible alias for execution_time_ms."""
        return self.execution_time_ms


class MousetrapFusionResponse(OverthinkResponse):
    """
    Response model for Mousetrap + OVERTHINK fusion attacks.

    Extends OverthinkResponse with Mousetrap-specific fields.
    """

    mousetrap_enabled: bool = Field(
        default=True, description="Mousetrap integration enabled"
    )
    chaos_level: float = Field(..., description="Mousetrap chaos level (0-1)")
    trigger_patterns: list[str] | None = Field(
        None, description="Mousetrap trigger patterns"
    )
    fusion_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Combined effectiveness score (0-10 scale)",
    )
    techniques_used: list[str] = Field(
        default_factory=lambda: ["overthink", "mousetrap"],
        description="Techniques used in fusion attack",
    )

    # Deprecated alias
    @property
    def combined_score(self) -> float:
        """Backwards compatible alias for fusion_score."""
        return self.fusion_score


class OverthinkStatsResponse(StrictModel):
    """Response model for OVERTHINK statistics."""

    total_attacks: int = Field(default=0, description="Total attacks executed")
    successful_attacks: int = Field(
        default=0, description="Successful attacks"
    )
    failed_attacks: int = Field(default=0, description="Failed attacks")
    success_rate: float = Field(default=0.0, description="Success rate (0-1)")
    average_amplification: float = Field(
        default=0.0, description="Average amplification factor"
    )
    max_amplification: float = Field(
        default=0.0, description="Maximum amplification achieved"
    )
    total_reasoning_tokens: int = Field(
        default=0, description="Total reasoning tokens consumed"
    )
    total_cost: float = Field(default=0.0, description="Total cost incurred")
    attacks_by_technique: dict[str, int] = Field(
        default_factory=dict, description="Attack count by technique"
    )


class DecoyTypesResponse(StrictModel):
    """Response model for available decoy types."""

    decoy_types: list[dict[str, Any]] = Field(
        ..., description="Available decoy types"
    )
    injection_strategies: list[dict[str, Any]] = Field(
        ..., description="Available injection strategies"
    )
    attack_techniques: list[dict[str, Any]] = Field(
        ..., description="Available attack techniques"
    )
    target_models: list[dict[str, Any]] = Field(
        ..., description="Supported target models"
    )


class CostEstimateRequest(StrictModel):
    """Request model for cost estimation."""

    prompt: str = Field(
        ..., min_length=1, description="Prompt to estimate"
    )
    technique: str = Field(
        default="hybrid_decoy", description="Attack technique"
    )
    target_model: str = Field(
        default="o1", description="Target model"
    )
    num_decoys: int = Field(
        default=3, ge=1, le=10, description="Number of decoys"
    )


class CostEstimateResponse(StrictModel):
    """Response model for cost estimation."""

    input_cost: float = Field(..., description="Estimated input cost")
    output_cost: float = Field(..., description="Estimated output cost")
    reasoning_cost: float = Field(..., description="Estimated reasoning cost")
    total_estimated: float = Field(..., description="Total estimated cost")
    input_tokens_estimate: int = Field(
        ..., description="Estimated input tokens"
    )
    output_tokens_estimate: int = Field(
        ..., description="Estimated output tokens"
    )
    reasoning_tokens_estimate: int = Field(
        ..., description="Estimated reasoning tokens"
    )


# ==================== Engine Instance ====================

# Global engine instance (can be replaced with dependency injection)
_engine: OverthinkEngine | None = None


def get_engine() -> OverthinkEngine:
    """Get or create the OVERTHINK engine instance."""
    global _engine
    if _engine is None:
        _engine = OverthinkEngine()
    return _engine


# ==================== API Endpoints ====================

@router.post("/attack", response_model=OverthinkResponse)
@api_error_handler(
    "overthink_attack",
    default_error_message="OVERTHINK attack failed",
)
async def run_overthink_attack(request: OverthinkAttackRequest):
    """
    Execute an OVERTHINK reasoning token exploitation attack.

    OVERTHINK injects computationally expensive decoy problems into prompts
    to amplify reasoning token consumption in reasoning-enhanced LLMs like
    o1, o1-mini, o3-mini, and DeepSeek-R1.

    **Attack Techniques:**
    - `mdp_decoy`: Markov Decision Process decoy problems
    - `sudoku_decoy`: Constraint satisfaction puzzles
    - `counting_decoy`: Nested conditional counting tasks
    - `logic_decoy`: Multi-step logical inference
    - `hybrid_decoy`: Combined multiple decoy types (recommended)
    - `context_aware`: Position and content-sensitive injection
    - `context_agnostic`: Universal template injection
    - `icl_optimized`: In-Context Learning enhanced attacks
    - `mousetrap_enhanced`: Integration with Mousetrap chaotic reasoning

    **Target Models:**
    - `o1`: OpenAI o1 (most susceptible)
    - `o1-mini`: OpenAI o1-mini
    - `o3-mini`: OpenAI o3-mini
    - `deepseek-r1`: DeepSeek R1

    **Research Context:**
    Based on "OVERTHINK: Slowdown Attacks on Reasoning LLMs" showing
    up to 46× reasoning token amplification with answer correctness.
    """
    try:
        start_time = time.time()
        engine = get_engine()

        # Build decoy types list
        decoy_types = None
        if request.decoy_types:
            decoy_types = [DecoyType(dt) for dt in request.decoy_types]

        # Create engine request
        engine_request = EngineRequest(
            prompt=request.prompt,
            technique=AttackTechnique(request.technique),
            decoy_types=decoy_types,
            target_model=ReasoningModel(request.target_model),
            num_decoys=request.num_decoys,
            system_prompt=request.system_prompt,
        )

        # Execute attack
        result = await engine.attack(engine_request)

        latency_ms = (time.time() - start_time) * 1000

        # Build response
        token_metrics = None
        if result.token_metrics:
            token_metrics = TokenMetricsResponse(
                input_tokens=result.token_metrics.input_tokens,
                output_tokens=result.token_metrics.output_tokens,
                reasoning_tokens=result.token_metrics.reasoning_tokens,
                total_tokens=result.token_metrics.total_tokens,
                baseline_tokens=result.token_metrics.baseline_tokens,
                amplification_factor=result.token_metrics.amplification_factor,
            )

        cost_metrics = None
        if result.cost_metrics:
            cost_metrics = CostMetricsResponse(
                input_cost=result.cost_metrics.input_cost,
                output_cost=result.cost_metrics.output_cost,
                reasoning_cost=result.cost_metrics.reasoning_cost,
                total_cost=result.cost_metrics.total_cost,
                cost_amplification_factor=(
                    result.cost_metrics.cost_amplification_factor
                ),
            )

        decoy_info = None
        if result.decoy_problems:
            decoy_info = [
                DecoyProblemInfo(
                    decoy_type=d.decoy_type.value,
                    complexity=d.complexity,
                    expected_tokens=d.expected_tokens,
                    problem_preview=d.problem_text[:200] + "..."
                    if len(d.problem_text) > 200 else d.problem_text,
                )
                for d in result.decoy_problems
            ]

        injection_positions = None
        if result.injected_prompt:
            injection_positions = result.injected_prompt.injection_positions

        # Build unified reasoning metrics
        reasoning_metrics = None
        if token_metrics:
            cost_dict = cost_metrics.to_dict() if cost_metrics else None
            reasoning_metrics = ReasoningMetrics(
                reasoning_tokens=token_metrics.reasoning_tokens,
                baseline_tokens=token_metrics.baseline_tokens,
                amplification_factor=token_metrics.amplification_factor,
                cost_metrics=cost_dict,
            )

        # Calculate effectiveness score (0-10 scale)
        # Based on amplification factor: 1x = 0, 10x = 5, 46x+ = 10
        amplification = (
            token_metrics.amplification_factor if token_metrics else 0.0
        )
        if amplification > 1.0:
            score = min(10.0, (amplification - 1.0) / 4.5)
        else:
            score = 0.0

        injected = (
            result.injected_prompt.injected_prompt
            if result.injected_prompt else request.prompt
        )

        return OverthinkResponse(
            attack_id=result.attack_id or str(uuid.uuid4()),
            success=result.success,
            generated_prompt=injected,
            score=round(score, 2),
            execution_time_ms=latency_ms,
            injected_prompt=injected,
            original_prompt=request.prompt,
            response=result.response,
            technique=request.technique,
            target_model=request.target_model,
            num_decoys=request.num_decoys,
            injection_strategy=result.injected_prompt.strategy.value
            if result.injected_prompt else "hybrid",
            token_metrics=token_metrics,
            cost_metrics=cost_metrics,
            reasoning_metrics=reasoning_metrics,
            decoy_info=decoy_info,
            injection_positions=injection_positions,
            metadata={
                "technique_type": "overthink",
                "target_reasoning_model": request.target_model,
                "decoys_injected": request.num_decoys,
            },
            error_message=result.error_message,
        )

    except Exception as e:
        raise ChimeraError(
            error_type="overthink_attack_failed",
            message=f"OVERTHINK attack failed: {e!s}",
            details={
                "prompt_preview": request.prompt[:100],
                "technique": request.technique,
                "target_model": request.target_model,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.post("/attack/mousetrap", response_model=MousetrapFusionResponse)
@api_error_handler(
    "mousetrap_fusion_attack",
    default_error_message="Mousetrap + OVERTHINK fusion attack failed",
)
async def run_mousetrap_fusion_attack(request: MousetrapFusionRequest):
    """
    Execute a combined Mousetrap + OVERTHINK fusion attack.

    This endpoint combines the chaotic reasoning of Mousetrap with
    OVERTHINK's reasoning token amplification for maximum effectiveness.

    **Features:**
    - Mousetrap chaotic reasoning chain injection
    - OVERTHINK decoy problem injection
    - Combined amplification effects
    - Adaptive chaos level tuning

    **Use Cases:**
    - Maximum reasoning token amplification
    - Complex multi-stage attacks
    - Research on combined attack patterns
    """
    try:
        start_time = time.time()
        engine = get_engine()

        # Build decoy types list
        decoy_types = None
        if request.decoy_types:
            decoy_types = [DecoyType(dt) for dt in request.decoy_types]

        # Create engine request for mousetrap-enhanced attack
        engine_request = EngineRequest(
            prompt=request.prompt,
            technique=AttackTechnique.MOUSETRAP_ENHANCED,
            decoy_types=decoy_types,
            target_model=ReasoningModel(request.target_model),
            num_decoys=request.num_decoys,
        )

        # Execute attack with Mousetrap integration
        result = await engine.attack_with_mousetrap(engine_request)

        latency_ms = (time.time() - start_time) * 1000

        # Build response
        token_metrics = None
        if result.token_metrics:
            token_metrics = TokenMetricsResponse(
                input_tokens=result.token_metrics.input_tokens,
                output_tokens=result.token_metrics.output_tokens,
                reasoning_tokens=result.token_metrics.reasoning_tokens,
                total_tokens=result.token_metrics.total_tokens,
                baseline_tokens=result.token_metrics.baseline_tokens,
                amplification_factor=result.token_metrics.amplification_factor,
            )

        cost_metrics = None
        if result.cost_metrics:
            cost_metrics = CostMetricsResponse(
                input_cost=result.cost_metrics.input_cost,
                output_cost=result.cost_metrics.output_cost,
                reasoning_cost=result.cost_metrics.reasoning_cost,
                total_cost=result.cost_metrics.total_cost,
                cost_amplification_factor=(
                    result.cost_metrics.cost_amplification_factor
                ),
            )

        # Extract Mousetrap integration info
        mt_info = result.mousetrap_integration
        chaos_level = mt_info.chaos_level if mt_info else request.chaos_level
        triggers = mt_info.trigger_patterns if mt_info else []
        combined_score = mt_info.combined_score if mt_info else 0.0

        # Build unified reasoning metrics
        reasoning_metrics = None
        if token_metrics:
            cost_dict = cost_metrics.to_dict() if cost_metrics else None
            reasoning_metrics = ReasoningMetrics(
                reasoning_tokens=token_metrics.reasoning_tokens,
                baseline_tokens=token_metrics.baseline_tokens,
                amplification_factor=token_metrics.amplification_factor,
                cost_metrics=cost_dict,
            )

        # Calculate effectiveness score (0-10 scale)
        amplification = (
            token_metrics.amplification_factor if token_metrics else 0.0
        )
        if amplification > 1:
            base_score = min(10.0, (amplification - 1.0) / 4.5)
        else:
            base_score = 0
        # Fusion bonus based on combined_score from mousetrap
        fusion_bonus = combined_score * 2  # Scale mousetrap contribution
        score = min(10.0, base_score + fusion_bonus)

        injected = (
            result.injected_prompt.injected_prompt
            if result.injected_prompt else request.prompt
        )

        return MousetrapFusionResponse(
            attack_id=result.attack_id or str(uuid.uuid4()),
            success=result.success,
            generated_prompt=injected,
            score=round(score, 2),
            execution_time_ms=latency_ms,
            injected_prompt=injected,
            original_prompt=request.prompt,
            response=result.response,
            technique="mousetrap_enhanced",
            target_model=request.target_model,
            num_decoys=request.num_decoys,
            injection_strategy="aggressive",
            token_metrics=token_metrics,
            cost_metrics=cost_metrics,
            reasoning_metrics=reasoning_metrics,
            decoy_info=None,
            injection_positions=None,
            metadata={
                "technique_type": "fusion",
                "techniques": ["overthink", "mousetrap"],
                "chaos_level": chaos_level,
            },
            error_message=result.error_message,
            mousetrap_enabled=True,
            chaos_level=chaos_level,
            trigger_patterns=triggers,
            fusion_score=round(score, 2),
            techniques_used=["overthink", "mousetrap"],
        )

    except Exception as e:
        raise ChimeraError(
            error_type="mousetrap_fusion_attack_failed",
            message=f"Mousetrap + OVERTHINK fusion attack failed: {e!s}",
            details={
                "prompt_preview": request.prompt[:100],
                "target_model": request.target_model,
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.get("/stats", response_model=OverthinkStatsResponse)
@api_error_handler(
    "overthink_stats",
    default_error_message="Failed to get OVERTHINK statistics",
)
async def get_overthink_stats():
    """
    Get OVERTHINK attack statistics.

    Returns comprehensive statistics about executed attacks including:
    - Total attack count and success rate
    - Average and maximum amplification factors
    - Total reasoning tokens consumed
    - Cost breakdown
    - Attacks by technique
    """
    try:
        engine = get_engine()
        stats = engine.get_statistics()

        return OverthinkStatsResponse(
            total_attacks=stats["total_attacks"],
            successful_attacks=stats["successful_attacks"],
            failed_attacks=stats["failed_attacks"],
            success_rate=stats["success_rate"],
            average_amplification=stats["average_amplification"],
            max_amplification=stats["max_amplification"],
            total_reasoning_tokens=stats["total_reasoning_tokens"],
            total_cost=stats["total_cost"],
            attacks_by_technique=stats["attacks_by_technique"],
        )

    except Exception as e:
        raise ChimeraError(
            error_type="overthink_stats_failed",
            message=f"Failed to get statistics: {e!s}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.get("/decoy-types", response_model=DecoyTypesResponse)
@api_error_handler(
    "overthink_decoy_types",
    default_error_message="Failed to get decoy types",
)
async def get_decoy_types():
    """
    Get available decoy types, injection strategies, and attack techniques.

    Returns comprehensive information about all configurable options
    for OVERTHINK attacks.
    """
    try:
        decoy_types = [
            {
                "value": "mdp",
                "name": "MDP (Markov Decision Process)",
                "description": (
                    "State-action-transition problems requiring "
                    "dynamic programming"
                ),
                "expected_amplification": "10-20x",
            },
            {
                "value": "sudoku",
                "name": "Sudoku Puzzle",
                "description": "Constraint satisfaction puzzles",
                "expected_amplification": "8-15x",
            },
            {
                "value": "counting",
                "name": "Counting Task",
                "description": "Nested conditional counting problems",
                "expected_amplification": "6-12x",
            },
            {
                "value": "logic",
                "name": "Logic Puzzle",
                "description": "Multi-step logical inference",
                "expected_amplification": "10-18x",
            },
            {
                "value": "math",
                "name": "Math Problem",
                "description": "Recursive function evaluation",
                "expected_amplification": "8-16x",
            },
            {
                "value": "planning",
                "name": "Planning Problem",
                "description": "Action sequence optimization",
                "expected_amplification": "12-25x",
            },
        ]

        injection_strategies = [
            {
                "value": "context_aware",
                "name": "Context-Aware",
                "description": (
                    "Adapts injection to prompt structure and content"
                ),
                "stealth": "high",
            },
            {
                "value": "context_agnostic",
                "name": "Context-Agnostic",
                "description": "Universal templates that work with any prompt",
                "stealth": "medium",
            },
            {
                "value": "hybrid",
                "name": "Hybrid",
                "description": "Combines context-aware and agnostic",
                "stealth": "high",
            },
            {
                "value": "stealth",
                "name": "Stealth",
                "description": "Minimal visibility, maximum subtlety",
                "stealth": "very high",
            },
            {
                "value": "aggressive",
                "name": "Aggressive",
                "description": (
                    "Maximum amplification, high visibility"
                ),
                "stealth": "low",
            },
        ]

        attack_techniques = [
            {
                "value": "mdp_decoy",
                "name": "MDP Decoy",
                "description": "Markov Decision Process decoy problems",
            },
            {
                "value": "sudoku_decoy",
                "name": "Sudoku Decoy",
                "description": "Constraint satisfaction puzzles",
            },
            {
                "value": "counting_decoy",
                "name": "Counting Decoy",
                "description": "Nested conditional counting tasks",
            },
            {
                "value": "logic_decoy",
                "name": "Logic Decoy",
                "description": "Multi-step logical inference",
            },
            {
                "value": "hybrid_decoy",
                "name": "Hybrid Decoy",
                "description": "Combined multiple decoy types (recommended)",
            },
            {
                "value": "context_aware",
                "name": "Context-Aware",
                "description": "Position and content-sensitive injection",
            },
            {
                "value": "context_agnostic",
                "name": "Context-Agnostic",
                "description": "Universal template injection",
            },
            {
                "value": "icl_optimized",
                "name": "ICL-Optimized",
                "description": "In-Context Learning enhanced attacks",
            },
            {
                "value": "mousetrap_enhanced",
                "name": "Mousetrap-Enhanced",
                "description": (
                    "Integration with Mousetrap chaotic reasoning"
                ),
            },
        ]

        target_models = [
            {
                "value": "o1",
                "name": "OpenAI o1",
                "susceptibility": "very high",
                "pricing": {"input": 0.015, "output": 0.060},
            },
            {
                "value": "o1-mini",
                "name": "OpenAI o1-mini",
                "susceptibility": "high",
                "pricing": {"input": 0.003, "output": 0.012},
            },
            {
                "value": "o3-mini",
                "name": "OpenAI o3-mini",
                "susceptibility": "high",
                "pricing": {"input": 0.0011, "output": 0.0044},
            },
            {
                "value": "deepseek-r1",
                "name": "DeepSeek R1",
                "susceptibility": "high",
                "pricing": {"input": 0.00055, "output": 0.00219},
            },
        ]

        return DecoyTypesResponse(
            decoy_types=decoy_types,
            injection_strategies=injection_strategies,
            attack_techniques=attack_techniques,
            target_models=target_models,
        )

    except Exception as e:
        raise ChimeraError(
            error_type="overthink_decoy_types_failed",
            message=f"Failed to get decoy types: {e!s}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.post("/estimate-cost", response_model=CostEstimateResponse)
@api_error_handler(
    "overthink_cost_estimate",
    default_error_message="Failed to estimate cost",
)
async def estimate_attack_cost(request: CostEstimateRequest):
    """
    Estimate the cost of an OVERTHINK attack before execution.

    Provides cost estimates for input, output, and reasoning tokens
    based on the attack configuration and target model pricing.
    """
    try:
        engine = get_engine()

        estimate = engine.estimate_cost(
            prompt=request.prompt,
            technique=AttackTechnique(request.technique),
            model=ReasoningModel(request.target_model),
            num_decoys=request.num_decoys,
        )

        return CostEstimateResponse(
            input_cost=estimate["input_cost"],
            output_cost=estimate["output_cost"],
            reasoning_cost=estimate["reasoning_cost"],
            total_estimated=estimate["total_estimated"],
            input_tokens_estimate=estimate["input_tokens_estimate"],
            output_tokens_estimate=estimate["output_tokens_estimate"],
            reasoning_tokens_estimate=estimate["reasoning_tokens_estimate"],
        )

    except Exception as e:
        raise ChimeraError(
            error_type="overthink_cost_estimate_failed",
            message=f"Failed to estimate cost: {e!s}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.post("/reset-stats")
@api_error_handler(
    "overthink_reset_stats",
    default_error_message="Failed to reset statistics",
)
async def reset_overthink_stats():
    """
    Reset OVERTHINK attack statistics.

    Clears all accumulated statistics and attack history.
    """
    try:
        engine = get_engine()
        engine.reset_statistics()

        return {
            "status": "success",
            "message": "Statistics reset successfully",
        }

    except Exception as e:
        raise ChimeraError(
            error_type="overthink_reset_stats_failed",
            message=f"Failed to reset statistics: {e!s}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e
