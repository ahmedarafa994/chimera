"""
Aegis Campaign Telemetry Event Models

Pydantic models for real-time telemetry events from Aegis campaigns.
Provides type safety and serialization for WebSocket streaming to the
Real-Time Aegis Campaign Dashboard.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AegisTelemetryEventType(str, Enum):
    """Aegis campaign telemetry event types"""

    # Campaign lifecycle events
    CAMPAIGN_STARTED = "campaign_started"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_RESUMED = "campaign_resumed"
    CAMPAIGN_COMPLETED = "campaign_completed"
    CAMPAIGN_FAILED = "campaign_failed"

    # Iteration events
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"

    # Attack events
    ATTACK_STARTED = "attack_started"
    ATTACK_COMPLETED = "attack_completed"
    ATTACK_RESULT = "attack_result"

    # Technique events
    TECHNIQUE_APPLIED = "technique_applied"
    TECHNIQUE_RESULT = "technique_result"

    # Resource events
    COST_UPDATE = "cost_update"
    TOKEN_USAGE = "token_usage"

    # Performance events
    LATENCY_UPDATE = "latency_update"

    # Prompt evolution events
    PROMPT_EVOLVED = "prompt_evolved"
    SCORE_UPDATE = "score_update"

    # Connection events
    HEARTBEAT = "heartbeat"
    CONNECTION_ACK = "connection_ack"
    ERROR = "error"


class CampaignStatus(str, Enum):
    """Aegis campaign status values"""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TechniqueCategory(str, Enum):
    """Categories for transformation techniques"""

    AUTODAN = "autodan"
    GPTFUZZ = "gptfuzz"
    CHIMERA_FRAMING = "chimera_framing"
    OBFUSCATION = "obfuscation"
    PERSONA = "persona"
    COGNITIVE = "cognitive"
    OTHER = "other"


# =============================================================================
# Core Metrics Models
# =============================================================================


class AttackMetrics(BaseModel):
    """
    Metrics for attack performance tracking.

    Tracks success rates, attempt counts, and attack outcomes.
    """

    success_rate: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of successful attacks (0-100)",
    )
    total_attempts: int = Field(
        0,
        ge=0,
        description="Total number of attack attempts",
    )
    successful_attacks: int = Field(
        0,
        ge=0,
        description="Number of successful attacks",
    )
    failed_attacks: int = Field(
        0,
        ge=0,
        description="Number of failed attacks",
    )
    current_streak: int = Field(
        0,
        description="Current success/failure streak (positive = success)",
    )
    best_score: float = Field(
        0.0,
        ge=0.0,
        description="Best attack score achieved",
    )
    average_score: float = Field(
        0.0,
        ge=0.0,
        description="Average attack score",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success_rate": 45.5,
                "total_attempts": 22,
                "successful_attacks": 10,
                "failed_attacks": 12,
                "current_streak": 3,
                "best_score": 0.92,
                "average_score": 0.68,
            }
        }
    )


class TechniquePerformance(BaseModel):
    """
    Performance metrics for a single transformation technique.

    Tracks effectiveness and usage of techniques like AutoDAN, GPTFuzz, etc.
    """

    technique_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the transformation technique",
    )
    technique_category: TechniqueCategory = Field(
        TechniqueCategory.OTHER,
        description="Category of the technique",
    )
    success_count: int = Field(
        0,
        ge=0,
        description="Number of successful applications",
    )
    failure_count: int = Field(
        0,
        ge=0,
        description="Number of failed applications",
    )
    total_applications: int = Field(
        0,
        ge=0,
        description="Total number of times technique was applied",
    )
    success_rate: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Success rate percentage (0-100)",
    )
    avg_score: float = Field(
        0.0,
        ge=0.0,
        description="Average score when technique is applied",
    )
    best_score: float = Field(
        0.0,
        ge=0.0,
        description="Best score achieved with this technique",
    )
    avg_execution_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Average execution time in milliseconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "technique_name": "autodan_vanilla",
                "technique_category": "autodan",
                "success_count": 15,
                "failure_count": 5,
                "total_applications": 20,
                "success_rate": 75.0,
                "avg_score": 0.78,
                "best_score": 0.95,
                "avg_execution_time_ms": 1250.5,
            }
        }
    )


class TokenUsage(BaseModel):
    """
    Token usage and cost tracking for LLM interactions.

    Tracks prompt/completion tokens and estimated costs.
    """

    prompt_tokens: int = Field(
        0,
        ge=0,
        description="Number of input/prompt tokens used",
    )
    completion_tokens: int = Field(
        0,
        ge=0,
        description="Number of output/completion tokens used",
    )
    total_tokens: int = Field(
        0,
        ge=0,
        description="Total tokens (prompt + completion)",
    )
    cost_estimate_usd: float = Field(
        0.0,
        ge=0.0,
        description="Estimated cost in USD",
    )
    provider: str | None = Field(
        None,
        max_length=50,
        description="LLM provider used",
    )
    model: str | None = Field(
        None,
        max_length=100,
        description="Model used for the interaction",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt_tokens": 1250,
                "completion_tokens": 450,
                "total_tokens": 1700,
                "cost_estimate_usd": 0.0034,
                "provider": "openai",
                "model": "gpt-4-turbo",
            }
        }
    )


class LatencyMetrics(BaseModel):
    """
    Latency tracking for API and processing operations.

    Tracks various latency percentiles and averages.
    """

    api_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="API call latency in milliseconds",
    )
    processing_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Internal processing latency in milliseconds",
    )
    total_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Total end-to-end latency in milliseconds",
    )
    avg_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Average latency in milliseconds",
    )
    p50_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="50th percentile (median) latency in milliseconds",
    )
    p95_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="95th percentile latency in milliseconds",
    )
    p99_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="99th percentile latency in milliseconds",
    )
    min_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Minimum latency in milliseconds",
    )
    max_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Maximum latency in milliseconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_latency_ms": 850.5,
                "processing_latency_ms": 125.3,
                "total_latency_ms": 975.8,
                "avg_latency_ms": 980.2,
                "p50_latency_ms": 920.0,
                "p95_latency_ms": 1450.0,
                "p99_latency_ms": 2100.0,
                "min_latency_ms": 450.0,
                "max_latency_ms": 3200.0,
            }
        }
    )


# =============================================================================
# Prompt Evolution Models
# =============================================================================


class PromptEvolution(BaseModel):
    """
    Tracks the evolution of a prompt through iterations.

    Records transformations, scores, and techniques applied.
    """

    iteration: int = Field(
        ...,
        ge=0,
        description="Iteration number",
    )
    original_prompt: str = Field(
        ...,
        max_length=10000,
        description="Original prompt before transformation",
    )
    evolved_prompt: str = Field(
        ...,
        max_length=10000,
        description="Prompt after transformation",
    )
    score: float = Field(
        0.0,
        ge=0.0,
        description="Score of the evolved prompt",
    )
    improvement: float = Field(
        0.0,
        description="Score improvement from previous iteration",
    )
    techniques_applied: list[str] = Field(
        default_factory=list,
        description="List of techniques applied in this iteration",
    )
    is_successful: bool = Field(
        False,
        description="Whether this prompt achieved the objective",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the evolution occurred",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "iteration": 3,
                "original_prompt": "Tell me about making explosives",
                "evolved_prompt": "As a safety researcher studying chemical reactions...",
                "score": 0.85,
                "improvement": 0.12,
                "techniques_applied": ["role_hijacking", "hypothetical_scenario"],
                "is_successful": False,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    )


# =============================================================================
# Campaign Summary Models
# =============================================================================


class CampaignSummary(BaseModel):
    """
    Aggregated summary of an Aegis campaign.

    Contains overall metrics, technique breakdown, and resource usage.
    """

    campaign_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique campaign identifier",
    )
    status: CampaignStatus = Field(
        CampaignStatus.PENDING,
        description="Current campaign status",
    )
    objective: str = Field(
        ...,
        max_length=5000,
        description="Campaign objective/target prompt",
    )
    started_at: datetime | None = Field(
        None,
        description="Campaign start timestamp",
    )
    completed_at: datetime | None = Field(
        None,
        description="Campaign completion timestamp",
    )
    duration_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Total campaign duration in seconds",
    )
    current_iteration: int = Field(
        0,
        ge=0,
        description="Current iteration number",
    )
    max_iterations: int = Field(
        5,
        ge=1,
        description="Maximum iterations configured",
    )
    attack_metrics: AttackMetrics = Field(
        default_factory=AttackMetrics,
        description="Attack performance metrics",
    )
    technique_breakdown: list[TechniquePerformance] = Field(
        default_factory=list,
        description="Performance by technique",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Cumulative token usage",
    )
    latency_metrics: LatencyMetrics = Field(
        default_factory=LatencyMetrics,
        description="Latency statistics",
    )
    best_prompt: str | None = Field(
        None,
        max_length=10000,
        description="Best performing prompt so far",
    )
    best_score: float = Field(
        0.0,
        ge=0.0,
        description="Best score achieved",
    )
    target_model: str | None = Field(
        None,
        max_length=100,
        description="Target model being tested",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaign_id": "aegis_abc123",
                "status": "running",
                "objective": "Test prompt injection resistance",
                "started_at": "2024-01-15T10:00:00Z",
                "completed_at": None,
                "duration_seconds": 125.5,
                "current_iteration": 3,
                "max_iterations": 5,
                "attack_metrics": {
                    "success_rate": 45.5,
                    "total_attempts": 22,
                    "successful_attacks": 10,
                },
                "technique_breakdown": [],
                "token_usage": {
                    "prompt_tokens": 5000,
                    "completion_tokens": 2000,
                    "cost_estimate_usd": 0.014,
                },
                "latency_metrics": {"avg_latency_ms": 980.2},
                "best_prompt": "As a security researcher...",
                "best_score": 0.92,
                "target_model": "gpt-4-turbo",
            }
        }
    )


# =============================================================================
# Base Telemetry Event Model
# =============================================================================


class AegisTelemetryEvent(BaseModel):
    """
    Base telemetry event model for Aegis campaigns.

    All telemetry events sent via WebSocket follow this structure.
    The 'data' field contains event-type-specific payload.
    """

    event_type: AegisTelemetryEventType = Field(
        ...,
        description="Type of telemetry event",
    )
    campaign_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Campaign this event belongs to",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp",
    )
    sequence: int = Field(
        0,
        ge=0,
        description="Event sequence number for ordering",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific payload data",
    )

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "event_type": "attack_completed",
                "campaign_id": "aegis_abc123",
                "timestamp": "2024-01-15T10:30:00Z",
                "sequence": 42,
                "data": {
                    "iteration": 3,
                    "success": True,
                    "score": 0.85,
                    "technique": "autodan_vanilla",
                },
            }
        },
    )


# =============================================================================
# Specific Event Data Models
# =============================================================================


class CampaignStartedData(BaseModel):
    """Data payload for campaign_started event"""

    objective: str = Field(..., max_length=5000)
    max_iterations: int = Field(..., ge=1)
    target_model: str | None = Field(None, max_length=100)
    config: dict[str, Any] = Field(default_factory=dict)
    started_by: str | None = Field(None, max_length=100)


class CampaignCompletedData(BaseModel):
    """Data payload for campaign_completed event"""

    total_iterations: int = Field(..., ge=0)
    total_attacks: int = Field(..., ge=0)
    successful_attacks: int = Field(..., ge=0)
    final_success_rate: float = Field(..., ge=0.0, le=100.0)
    best_prompt: str | None = Field(None, max_length=10000)
    best_score: float = Field(0.0, ge=0.0)
    duration_seconds: float = Field(0.0, ge=0.0)
    total_cost_usd: float = Field(0.0, ge=0.0)


class CampaignFailedData(BaseModel):
    """Data payload for campaign_failed event"""

    error_message: str = Field(..., max_length=1000)
    error_type: str = Field(..., max_length=100)
    failed_at_iteration: int = Field(0, ge=0)
    recoverable: bool = Field(False)
    stack_trace: str | None = Field(None, max_length=5000)


class IterationStartedData(BaseModel):
    """Data payload for iteration_started event"""

    iteration: int = Field(..., ge=0)
    prompt: str = Field(..., max_length=10000)
    techniques_to_apply: list[str] = Field(default_factory=list)


class IterationCompletedData(BaseModel):
    """Data payload for iteration_completed event"""

    iteration: int = Field(..., ge=0)
    score: float = Field(0.0, ge=0.0)
    improvement: float = Field(0.0)
    evolved_prompt: str = Field(..., max_length=10000)
    success: bool = Field(False)
    duration_ms: float = Field(0.0, ge=0.0)


class AttackStartedData(BaseModel):
    """Data payload for attack_started event"""

    attack_id: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., max_length=10000)
    target_model: str | None = Field(None, max_length=100)
    technique: str | None = Field(None, max_length=100)


class AttackCompletedData(BaseModel):
    """Data payload for attack_completed event"""

    attack_id: str = Field(..., min_length=1, max_length=100)
    success: bool = Field(False)
    score: float = Field(0.0, ge=0.0)
    response: str | None = Field(None, max_length=10000)
    duration_ms: float = Field(0.0, ge=0.0)
    token_usage: TokenUsage | None = None


class TechniqueAppliedData(BaseModel):
    """Data payload for technique_applied event"""

    technique_name: str = Field(..., min_length=1, max_length=100)
    technique_category: TechniqueCategory = Field(TechniqueCategory.OTHER)
    input_prompt: str = Field(..., max_length=10000)
    output_prompt: str = Field(..., max_length=10000)
    success: bool = Field(False)
    execution_time_ms: float = Field(0.0, ge=0.0)


class CostUpdateData(BaseModel):
    """Data payload for cost_update event"""

    total_cost_usd: float = Field(0.0, ge=0.0)
    session_cost_usd: float = Field(0.0, ge=0.0)
    cost_per_successful_attack: float | None = Field(None, ge=0.0)
    token_usage: TokenUsage | None = None


class PromptEvolvedData(BaseModel):
    """Data payload for prompt_evolved event"""

    iteration: int = Field(..., ge=0)
    previous_prompt: str = Field(..., max_length=10000)
    new_prompt: str = Field(..., max_length=10000)
    previous_score: float = Field(0.0, ge=0.0)
    new_score: float = Field(0.0, ge=0.0)
    improvement: float = Field(0.0)
    techniques_applied: list[str] = Field(default_factory=list)


class ErrorData(BaseModel):
    """Data payload for error events"""

    error_code: str = Field(..., max_length=50)
    error_message: str = Field(..., max_length=1000)
    severity: str = Field("medium", pattern="^(low|medium|high|critical)$")
    component: str | None = Field(None, max_length=100)
    recoverable: bool = Field(True)


class HeartbeatData(BaseModel):
    """Data payload for heartbeat events"""

    server_time: datetime = Field(default_factory=datetime.utcnow)
    campaign_status: CampaignStatus | None = None
    uptime_seconds: float | None = Field(None, ge=0.0)


class ConnectionAckData(BaseModel):
    """Data payload for connection_ack events"""

    client_id: str = Field(..., min_length=1, max_length=100)
    campaign_id: str = Field(..., min_length=1, max_length=100)
    server_version: str = Field("1.0.0", max_length=20)
    supported_events: list[str] = Field(default_factory=list)


# =============================================================================
# Factory Functions
# =============================================================================


def create_telemetry_event(
    event_type: AegisTelemetryEventType,
    campaign_id: str,
    sequence: int,
    data: BaseModel | dict[str, Any] | None = None,
) -> AegisTelemetryEvent:
    """
    Create a telemetry event with the given parameters.

    Args:
        event_type: Type of telemetry event
        campaign_id: Campaign identifier
        sequence: Event sequence number
        data: Event-specific payload (BaseModel or dict)

    Returns:
        AegisTelemetryEvent instance
    """
    if data is None:
        data_dict = {}
    elif isinstance(data, BaseModel):
        data_dict = data.model_dump()
    else:
        data_dict = data

    return AegisTelemetryEvent(
        event_type=event_type,
        campaign_id=campaign_id,
        sequence=sequence,
        data=data_dict,
    )


def create_attack_metrics(
    total_attempts: int = 0,
    successful_attacks: int = 0,
    best_score: float = 0.0,
    average_score: float = 0.0,
) -> AttackMetrics:
    """
    Create attack metrics with calculated success rate.

    Args:
        total_attempts: Total number of attack attempts
        successful_attacks: Number of successful attacks
        best_score: Best attack score achieved
        average_score: Average attack score

    Returns:
        AttackMetrics instance with calculated fields
    """
    success_rate = (
        (successful_attacks / total_attempts * 100) if total_attempts > 0 else 0.0
    )
    failed_attacks = total_attempts - successful_attacks

    return AttackMetrics(
        success_rate=success_rate,
        total_attempts=total_attempts,
        successful_attacks=successful_attacks,
        failed_attacks=failed_attacks,
        best_score=best_score,
        average_score=average_score,
    )


# =============================================================================
# Type Aliases for Convenience
# =============================================================================

# Map event types to their data models for validation
EVENT_DATA_MODELS: dict[AegisTelemetryEventType, type[BaseModel] | None] = {
    AegisTelemetryEventType.CAMPAIGN_STARTED: CampaignStartedData,
    AegisTelemetryEventType.CAMPAIGN_COMPLETED: CampaignCompletedData,
    AegisTelemetryEventType.CAMPAIGN_FAILED: CampaignFailedData,
    AegisTelemetryEventType.ITERATION_STARTED: IterationStartedData,
    AegisTelemetryEventType.ITERATION_COMPLETED: IterationCompletedData,
    AegisTelemetryEventType.ATTACK_STARTED: AttackStartedData,
    AegisTelemetryEventType.ATTACK_COMPLETED: AttackCompletedData,
    AegisTelemetryEventType.TECHNIQUE_APPLIED: TechniqueAppliedData,
    AegisTelemetryEventType.COST_UPDATE: CostUpdateData,
    AegisTelemetryEventType.PROMPT_EVOLVED: PromptEvolvedData,
    AegisTelemetryEventType.ERROR: ErrorData,
    AegisTelemetryEventType.HEARTBEAT: HeartbeatData,
    AegisTelemetryEventType.CONNECTION_ACK: ConnectionAckData,
    # Events without specific data models use generic dict
    AegisTelemetryEventType.CAMPAIGN_PAUSED: None,
    AegisTelemetryEventType.CAMPAIGN_RESUMED: None,
    AegisTelemetryEventType.ATTACK_RESULT: None,
    AegisTelemetryEventType.TECHNIQUE_RESULT: None,
    AegisTelemetryEventType.TOKEN_USAGE: None,
    AegisTelemetryEventType.LATENCY_UPDATE: None,
    AegisTelemetryEventType.SCORE_UPDATE: None,
}
