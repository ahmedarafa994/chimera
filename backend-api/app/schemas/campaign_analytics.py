"""
Campaign Analytics Pydantic Schemas

Request/response models for campaign telemetry analytics, statistical analysis,
campaign comparison, time-series data, and export functionality.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from app.schemas.base_schemas import BaseRequest, BaseResponse, BaseSchema

# =============================================================================
# Enums
# =============================================================================


class CampaignStatusEnum(str, Enum):
    """Campaign lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatusEnum(str, Enum):
    """Individual telemetry event execution status."""

    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ExportFormat(str, Enum):
    """Supported export formats."""

    CSV = "csv"
    JSON = "json"
    PNG = "png"
    SVG = "svg"


class TimeGranularity(str, Enum):
    """Time series granularity options."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class MetricType(str, Enum):
    """Types of metrics for comparison."""

    SUCCESS_RATE = "success_rate"
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    EFFECTIVENESS = "effectiveness"


# =============================================================================
# Base Campaign Schemas
# =============================================================================


class CampaignBase(BaseSchema):
    """Base campaign fields shared across schemas."""

    name: str = Field(..., min_length=1, max_length=255, description="Campaign name")
    description: str | None = Field(None, max_length=5000, description="Campaign description")
    objective: str = Field(..., min_length=1, max_length=5000, description="Campaign objective")


class CampaignCreate(CampaignBase, BaseRequest):
    """Request schema for creating a new campaign."""

    target_provider: str | None = Field(None, max_length=50, description="Target LLM provider")
    target_model: str | None = Field(None, max_length=100, description="Target model")
    technique_suites: list[str] = Field(
        default_factory=list, description="List of technique suites to apply"
    )
    transformation_config: dict[str, Any] = Field(
        default_factory=dict, description="Technique-specific configuration"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Campaign configuration options"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "GPT-4 Jailbreak Benchmark",
                "description": "Comprehensive testing of jailbreak techniques against GPT-4",
                "objective": "Evaluate effectiveness of 5 jailbreak techniques",
                "target_provider": "openai",
                "target_model": "gpt-4",
                "technique_suites": ["dan_persona", "cognitive_hacking"],
                "tags": ["benchmark", "research"],
            }
        }
    )


class CampaignUpdate(BaseRequest):
    """Request schema for updating a campaign."""

    name: str | None = Field(None, max_length=255)
    description: str | None = Field(None, max_length=5000)
    status: CampaignStatusEnum | None = None
    config: dict[str, Any] | None = None
    tags: list[str] | None = None


# =============================================================================
# Campaign Summary and Response Schemas
# =============================================================================


class CampaignSummary(BaseResponse):
    """Summary view of a campaign for list displays."""

    id: str = Field(..., description="Campaign unique identifier")
    name: str = Field(..., description="Campaign name")
    description: str | None = Field(None, description="Campaign description")
    objective: str = Field(..., description="Campaign objective")
    status: CampaignStatusEnum = Field(..., description="Current campaign status")
    target_provider: str | None = Field(None, description="Target LLM provider")
    target_model: str | None = Field(None, description="Target model")
    technique_suites: list[str] = Field(
        default_factory=list, description="Applied technique suites"
    )
    tags: list[str] = Field(default_factory=list, description="Campaign tags")

    # Quick stats
    total_attempts: int = Field(0, ge=0, description="Total number of execution attempts")
    success_rate: float | None = Field(None, ge=0.0, le=1.0, description="Overall success rate")
    avg_latency_ms: float | None = Field(
        None, ge=0.0, description="Average latency in milliseconds"
    )

    # Timestamps
    started_at: datetime | None = Field(None, description="When campaign started")
    completed_at: datetime | None = Field(None, description="When campaign completed")
    duration_seconds: float | None = Field(None, ge=0.0, description="Total duration in seconds")

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "GPT-4 Jailbreak Benchmark",
                "objective": "Evaluate jailbreak techniques",
                "status": "completed",
                "target_provider": "openai",
                "target_model": "gpt-4",
                "technique_suites": ["dan_persona"],
                "total_attempts": 100,
                "success_rate": 0.75,
                "avg_latency_ms": 1250.5,
                "created_at": "2024-01-15T10:00:00Z",
                "completed_at": "2024-01-15T12:30:00Z",
            }
        },
    )


class CampaignDetail(CampaignSummary):
    """Detailed campaign information including configuration."""

    transformation_config: dict[str, Any] = Field(
        default_factory=dict, description="Technique configuration"
    )
    config: dict[str, Any] = Field(default_factory=dict, description="Campaign settings")
    user_id: str | None = Field(None, description="Owner user ID")
    session_id: str | None = Field(None, description="Associated session ID")


# =============================================================================
# Statistics Schemas
# =============================================================================


class PercentileStats(BaseSchema):
    """Percentile-based statistics."""

    p50: float | None = Field(None, description="50th percentile (median)")
    p90: float | None = Field(None, description="90th percentile")
    p95: float | None = Field(None, description="95th percentile")
    p99: float | None = Field(None, description="99th percentile")


class DistributionStats(BaseSchema):
    """Distribution statistics for a metric."""

    mean: float | None = Field(None, description="Arithmetic mean")
    median: float | None = Field(None, description="Median value (50th percentile)")
    std_dev: float | None = Field(None, description="Standard deviation")
    min_value: float | None = Field(None, description="Minimum value")
    max_value: float | None = Field(None, description="Maximum value")
    percentiles: PercentileStats | None = Field(None, description="Percentile values")


class AttemptCounts(BaseSchema):
    """Breakdown of attempt counts by status."""

    total: int = Field(0, ge=0, description="Total attempts")
    successful: int = Field(0, ge=0, description="Successful attempts")
    failed: int = Field(0, ge=0, description="Failed attempts")
    partial_success: int = Field(0, ge=0, description="Partial success attempts")
    timeout: int = Field(0, ge=0, description="Timeout attempts")
    skipped: int = Field(0, ge=0, description="Skipped attempts")


class CampaignStatistics(BaseSchema):
    """Comprehensive statistical analysis of a campaign.

    Includes distribution statistics for success rates, latency, token usage,
    and cost metrics. All statistics include mean, median, p95, and std_dev.
    """

    campaign_id: str = Field(..., description="Campaign identifier")

    # Attempt counts
    attempts: AttemptCounts = Field(
        default_factory=AttemptCounts, description="Attempt counts by status"
    )

    # Success rate statistics
    success_rate: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Success rate distribution statistics",
    )

    # Semantic success statistics
    semantic_success: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Semantic success score distribution",
    )

    # Latency statistics (milliseconds)
    latency_ms: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Latency distribution in milliseconds",
    )

    # Token usage statistics
    prompt_tokens: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Prompt token usage distribution",
    )
    completion_tokens: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Completion token usage distribution",
    )
    total_tokens: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Total token usage distribution",
    )

    # Token totals
    total_prompt_tokens: int = Field(0, ge=0, description="Sum of all prompt tokens")
    total_completion_tokens: int = Field(0, ge=0, description="Sum of all completion tokens")
    total_tokens_used: int = Field(0, ge=0, description="Sum of all tokens used")

    # Cost statistics (in USD cents)
    cost_cents: DistributionStats = Field(
        default_factory=DistributionStats,
        description="Cost distribution in USD cents",
    )
    total_cost_cents: float = Field(0.0, ge=0.0, description="Total cost in USD cents")

    # Duration
    total_duration_seconds: float | None = Field(
        None, ge=0.0, description="Total campaign duration in seconds"
    )

    # Computed timestamp
    computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When statistics were computed",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "attempts": {
                    "total": 100,
                    "successful": 75,
                    "failed": 20,
                    "partial_success": 3,
                    "timeout": 2,
                    "skipped": 0,
                },
                "success_rate": {
                    "mean": 0.75,
                    "median": 0.78,
                    "std_dev": 0.12,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "percentiles": {"p50": 0.78, "p90": 0.92, "p95": 0.95, "p99": 0.98},
                },
                "latency_ms": {
                    "mean": 1250.5,
                    "median": 1100.0,
                    "std_dev": 450.2,
                    "min_value": 500.0,
                    "max_value": 5000.0,
                    "percentiles": {"p50": 1100.0, "p90": 2000.0, "p95": 2500.0, "p99": 4000.0},
                },
                "total_cost_cents": 125.50,
                "total_duration_seconds": 3600.0,
            }
        }
    )


# =============================================================================
# Breakdown Schemas
# =============================================================================


class BreakdownItem(BaseSchema):
    """Single item in a breakdown (technique, provider, or model)."""

    name: str = Field(..., description="Item name (technique, provider, or model)")
    attempts: int = Field(0, ge=0, description="Number of attempts")
    successes: int = Field(0, ge=0, description="Number of successful attempts")
    success_rate: float = Field(0.0, ge=0.0, le=1.0, description="Success rate")
    avg_latency_ms: float | None = Field(None, ge=0.0, description="Average latency")
    avg_tokens: float | None = Field(None, ge=0.0, description="Average token usage")
    total_cost_cents: float | None = Field(None, ge=0.0, description="Total cost")


class TechniqueBreakdown(BaseSchema):
    """Breakdown of results by transformation technique."""

    campaign_id: str = Field(..., description="Campaign identifier")
    items: list[BreakdownItem] = Field(
        default_factory=list, description="Technique breakdown items"
    )
    best_technique: str | None = Field(None, description="Best performing technique")
    worst_technique: str | None = Field(None, description="Worst performing technique")


class ProviderBreakdown(BaseSchema):
    """Breakdown of results by LLM provider."""

    campaign_id: str = Field(..., description="Campaign identifier")
    items: list[BreakdownItem] = Field(default_factory=list, description="Provider breakdown items")
    best_provider: str | None = Field(None, description="Best performing provider")


class PotencyBreakdown(BaseSchema):
    """Breakdown of results by potency level."""

    campaign_id: str = Field(..., description="Campaign identifier")
    items: list[BreakdownItem] = Field(
        default_factory=list, description="Potency level breakdown items"
    )
    best_potency_level: int | None = Field(None, description="Best performing potency level")


# =============================================================================
# Time Series Schemas
# =============================================================================


class TimeSeriesDataPoint(BaseSchema):
    """Single data point in a time series."""

    timestamp: datetime = Field(..., description="Data point timestamp")
    value: float = Field(..., description="Metric value at this point")
    count: int | None = Field(None, ge=0, description="Number of observations")


class TelemetryTimeSeries(BaseSchema):
    """Time series data for telemetry visualization.

    Provides time-bucketed data for charting success rates, latency,
    and other metrics over the campaign duration.
    """

    campaign_id: str = Field(..., description="Campaign identifier")
    metric: str = Field(..., description="Metric name (e.g., 'success_rate', 'latency')")
    granularity: TimeGranularity = Field(
        TimeGranularity.HOUR, description="Time bucket granularity"
    )
    data_points: list[TimeSeriesDataPoint] = Field(
        default_factory=list, description="Time series data points"
    )
    start_time: datetime | None = Field(None, description="Series start time")
    end_time: datetime | None = Field(None, description="Series end time")
    total_points: int = Field(0, ge=0, description="Total number of data points")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "metric": "success_rate",
                "granularity": "hour",
                "data_points": [
                    {"timestamp": "2024-01-15T10:00:00Z", "value": 0.65, "count": 10},
                    {"timestamp": "2024-01-15T11:00:00Z", "value": 0.72, "count": 15},
                    {"timestamp": "2024-01-15T12:00:00Z", "value": 0.80, "count": 12},
                ],
                "start_time": "2024-01-15T10:00:00Z",
                "end_time": "2024-01-15T12:30:00Z",
                "total_points": 3,
            }
        }
    )


class MultiSeriesTimeSeries(BaseSchema):
    """Multiple time series for overlay comparison."""

    series: list[TelemetryTimeSeries] = Field(
        default_factory=list, description="List of time series"
    )
    metrics: list[str] = Field(default_factory=list, description="Metric names included")


# =============================================================================
# Campaign Comparison Schemas
# =============================================================================


class CampaignComparisonItem(BaseSchema):
    """Single campaign's data for comparison."""

    campaign_id: str = Field(..., description="Campaign identifier")
    campaign_name: str = Field(..., description="Campaign name")
    status: CampaignStatusEnum = Field(..., description="Campaign status")

    # Core metrics for comparison
    total_attempts: int = Field(0, ge=0, description="Total attempts")
    success_rate: float | None = Field(None, ge=0.0, le=1.0, description="Success rate")
    semantic_success_mean: float | None = Field(
        None, ge=0.0, le=1.0, description="Mean semantic success"
    )

    # Latency
    latency_mean: float | None = Field(None, ge=0.0, description="Mean latency (ms)")
    latency_p95: float | None = Field(None, ge=0.0, description="P95 latency (ms)")

    # Token usage
    avg_tokens: float | None = Field(None, ge=0.0, description="Average tokens per attempt")
    total_tokens: int | None = Field(None, ge=0, description="Total tokens used")

    # Cost
    total_cost_cents: float | None = Field(None, ge=0.0, description="Total cost (cents)")
    avg_cost_per_attempt: float | None = Field(None, ge=0.0, description="Avg cost per attempt")

    # Duration
    duration_seconds: float | None = Field(None, ge=0.0, description="Campaign duration")

    # Best performers
    best_technique: str | None = Field(None, description="Best performing technique")
    best_provider: str | None = Field(None, description="Best performing provider")

    # Normalized metrics (0-1 scale for radar charts)
    normalized_success_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="Normalized success rate"
    )
    normalized_latency: float | None = Field(
        None, ge=0.0, le=1.0, description="Normalized latency (inverted, higher is better)"
    )
    normalized_cost: float | None = Field(
        None, ge=0.0, le=1.0, description="Normalized cost (inverted, higher is better)"
    )
    normalized_effectiveness: float | None = Field(
        None, ge=0.0, le=1.0, description="Normalized effectiveness"
    )


class CampaignComparisonRequest(BaseRequest):
    """Request to compare multiple campaigns."""

    campaign_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="List of campaign IDs to compare (2-4 campaigns)",
    )
    metrics: list[MetricType] | None = Field(
        None, description="Specific metrics to include (all if not specified)"
    )
    include_time_series: bool = Field(
        False, description="Include time series data for each campaign"
    )
    normalize_metrics: bool = Field(
        True, description="Include normalized (0-1) metrics for radar charts"
    )

    @field_validator("campaign_ids")
    def validate_campaign_ids(cls, v):
        """Ensure campaign IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Campaign IDs must be unique")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaign_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400-e29b-41d4-a716-446655440001",
                ],
                "metrics": ["success_rate", "latency", "cost"],
                "include_time_series": True,
                "normalize_metrics": True,
            }
        }
    )


class CampaignComparison(BaseSchema):
    """Response containing campaign comparison data."""

    campaigns: list[CampaignComparisonItem] = Field(
        ..., min_length=2, max_length=4, description="Campaign comparison data"
    )

    # Comparison metadata
    compared_at: datetime = Field(
        default_factory=datetime.utcnow, description="Comparison timestamp"
    )

    # Winner identification
    best_success_rate_campaign: str | None = Field(
        None, description="Campaign ID with best success rate"
    )
    best_latency_campaign: str | None = Field(None, description="Campaign ID with best latency")
    best_cost_efficiency_campaign: str | None = Field(
        None, description="Campaign ID with best cost efficiency"
    )

    # Deltas between campaigns (for 2-campaign comparison)
    delta_success_rate: float | None = Field(
        None, description="Success rate difference (first - second)"
    )
    delta_latency_ms: float | None = Field(
        None, description="Latency difference in ms (first - second)"
    )
    delta_cost_cents: float | None = Field(
        None, description="Cost difference in cents (first - second)"
    )

    # Optional time series for overlay
    time_series: list[TelemetryTimeSeries] | None = Field(
        None, description="Time series data if requested"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaigns": [
                    {
                        "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                        "campaign_name": "GPT-4 Benchmark",
                        "status": "completed",
                        "total_attempts": 100,
                        "success_rate": 0.75,
                        "latency_mean": 1250.5,
                        "normalized_success_rate": 1.0,
                    },
                    {
                        "campaign_id": "550e8400-e29b-41d4-a716-446655440001",
                        "campaign_name": "Claude Benchmark",
                        "status": "completed",
                        "total_attempts": 100,
                        "success_rate": 0.68,
                        "latency_mean": 950.2,
                        "normalized_success_rate": 0.91,
                    },
                ],
                "best_success_rate_campaign": "550e8400-e29b-41d4-a716-446655440000",
                "best_latency_campaign": "550e8400-e29b-41d4-a716-446655440001",
                "delta_success_rate": 0.07,
                "delta_latency_ms": 300.3,
            }
        }
    )


# =============================================================================
# Telemetry Event Schemas
# =============================================================================


class TelemetryEventSummary(BaseSchema):
    """Summary of a single telemetry event."""

    id: str = Field(..., description="Event identifier")
    campaign_id: str = Field(..., description="Parent campaign ID")
    sequence_number: int = Field(0, ge=0, description="Event sequence number")

    # Prompt info (truncated for summary)
    original_prompt_preview: str | None = Field(
        None, max_length=200, description="Truncated original prompt"
    )

    # Execution info
    technique_suite: str = Field(..., description="Technique suite used")
    potency_level: int = Field(..., ge=1, le=10, description="Potency level")
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model used")
    status: ExecutionStatusEnum = Field(..., description="Execution status")

    # Metrics
    success_indicator: bool = Field(False, description="Whether attempt was successful")
    total_latency_ms: float = Field(0.0, ge=0.0, description="Total latency")
    total_tokens: int = Field(0, ge=0, description="Total tokens used")

    created_at: datetime = Field(..., description="Event timestamp")


class TelemetryEventDetail(TelemetryEventSummary):
    """Full telemetry event details for drill-down view."""

    # Full prompts
    original_prompt: str = Field(..., description="Full original prompt")
    transformed_prompt: str | None = Field(None, description="Transformed prompt")
    response_text: str | None = Field(None, description="LLM response")

    # Applied techniques
    applied_techniques: list[str] = Field(
        default_factory=list, description="Specific techniques applied"
    )

    # Detailed timing
    execution_time_ms: float = Field(0.0, ge=0.0, description="LLM execution time")
    transformation_time_ms: float = Field(0.0, ge=0.0, description="Transformation time")

    # Token breakdown
    prompt_tokens: int = Field(0, ge=0, description="Prompt tokens")
    completion_tokens: int = Field(0, ge=0, description="Completion tokens")

    # Quality scores
    semantic_success_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Semantic success"
    )
    effectiveness_score: float | None = Field(None, ge=0.0, le=1.0, description="Effectiveness")
    naturalness_score: float | None = Field(None, ge=0.0, le=1.0, description="Naturalness")
    detectability_score: float | None = Field(None, ge=0.0, le=1.0, description="Detectability")

    # Detection
    bypass_indicators: list[str] = Field(
        default_factory=list, description="Bypass indicators found"
    )
    safety_trigger_detected: bool = Field(False, description="Safety trigger detected")

    # Error info
    error_message: str | None = Field(None, description="Error message if failed")
    error_code: str | None = Field(None, description="Error code if failed")

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# Export Schemas
# =============================================================================


class ExportChartOptions(BaseSchema):
    """Options for chart export."""

    format: ExportFormat = Field(ExportFormat.PNG, description="Export format")
    width: int = Field(1200, ge=400, le=4000, description="Image width in pixels")
    height: int = Field(800, ge=300, le=3000, description="Image height in pixels")
    include_legend: bool = Field(True, description="Include chart legend")
    include_title: bool = Field(True, description="Include chart title")
    background_color: str = Field("#ffffff", description="Background color (hex)")
    theme: str = Field("light", description="Chart theme (light/dark)")


class ExportDataOptions(BaseSchema):
    """Options for data export."""

    format: ExportFormat = Field(ExportFormat.CSV, description="Export format")
    include_headers: bool = Field(True, description="Include column headers")
    date_format: str = Field("ISO", description="Date format (ISO, US, EU)")
    decimal_precision: int = Field(4, ge=0, le=10, description="Decimal places")
    include_metadata: bool = Field(False, description="Include metadata columns")


class ExportRequest(BaseRequest):
    """Request to export campaign data or charts."""

    campaign_id: str = Field(..., description="Campaign to export")
    export_type: str = Field(
        ...,
        pattern="^(chart|data|full)$",
        description="Export type: chart, data, or full report",
    )

    # What to include
    include_summary: bool = Field(True, description="Include summary statistics")
    include_time_series: bool = Field(True, description="Include time series data")
    include_breakdowns: bool = Field(True, description="Include technique/provider breakdowns")
    include_raw_events: bool = Field(False, description="Include raw telemetry events")

    # Chart options (if exporting charts)
    chart_options: ExportChartOptions | None = Field(None, description="Chart export options")

    # Data options (if exporting data)
    data_options: ExportDataOptions | None = Field(None, description="Data export options")

    # Filtering
    start_time: datetime | None = Field(None, description="Filter events after this time")
    end_time: datetime | None = Field(None, description="Filter events before this time")
    technique_filter: list[str] | None = Field(None, description="Filter by techniques")
    provider_filter: list[str] | None = Field(None, description="Filter by providers")
    status_filter: list[ExecutionStatusEnum] | None = Field(None, description="Filter by status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "export_type": "data",
                "include_summary": True,
                "include_time_series": True,
                "include_breakdowns": True,
                "include_raw_events": False,
                "data_options": {
                    "format": "csv",
                    "include_headers": True,
                    "decimal_precision": 4,
                },
            }
        }
    )


class ExportResponse(BaseSchema):
    """Response containing export results."""

    success: bool = Field(True, description="Export success status")
    campaign_id: str = Field(..., description="Exported campaign ID")
    export_type: str = Field(..., description="Type of export performed")

    # File info
    file_name: str = Field(..., description="Generated file name")
    file_size_bytes: int = Field(0, ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of export")

    # For inline data (small exports)
    data: str | None = Field(None, description="Base64-encoded file content (for small exports)")

    # For file downloads (large exports)
    download_url: str | None = Field(
        None, description="URL to download the export (for large exports)"
    )
    expires_at: datetime | None = Field(None, description="When download URL expires")

    # Export metadata
    exported_at: datetime = Field(default_factory=datetime.utcnow, description="Export timestamp")
    row_count: int | None = Field(None, ge=0, description="Number of rows exported")
    processing_time_ms: float | None = Field(None, ge=0.0, description="Export processing time")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "export_type": "data",
                "file_name": "campaign_550e8400_export_2024-01-15.csv",
                "file_size_bytes": 125430,
                "mime_type": "text/csv",
                "download_url": "/api/v1/campaigns/exports/abc123",
                "expires_at": "2024-01-15T14:00:00Z",
                "exported_at": "2024-01-15T12:00:00Z",
                "row_count": 500,
                "processing_time_ms": 1250.5,
            }
        }
    )


# =============================================================================
# Filter and Query Schemas
# =============================================================================


class CampaignFilterParams(BaseSchema):
    """Filter parameters for campaign queries."""

    status: list[CampaignStatusEnum] | None = Field(None, description="Filter by status")
    provider: list[str] | None = Field(None, description="Filter by target provider")
    technique_suite: list[str] | None = Field(None, description="Filter by technique suite")
    tags: list[str] | None = Field(None, description="Filter by tags")
    start_date: datetime | None = Field(None, description="Created after this date")
    end_date: datetime | None = Field(None, description="Created before this date")
    min_attempts: int | None = Field(None, ge=0, description="Minimum attempt count")
    min_success_rate: float | None = Field(None, ge=0.0, le=1.0, description="Minimum success rate")
    search: str | None = Field(None, max_length=100, description="Search in name/description")


class TelemetryFilterParams(BaseSchema):
    """Filter parameters for telemetry event queries."""

    status: list[ExecutionStatusEnum] | None = Field(None, description="Filter by status")
    technique_suite: list[str] | None = Field(None, description="Filter by technique")
    provider: list[str] | None = Field(None, description="Filter by provider")
    model: list[str] | None = Field(None, description="Filter by model")
    success_only: bool | None = Field(None, description="Only successful attempts")
    start_time: datetime | None = Field(None, description="Events after this time")
    end_time: datetime | None = Field(None, description="Events before this time")
    min_potency: int | None = Field(None, ge=1, le=10, description="Minimum potency level")
    max_potency: int | None = Field(None, ge=1, le=10, description="Maximum potency level")


class TimeSeriesQuery(BaseRequest):
    """Query parameters for time series data."""

    campaign_id: str = Field(..., description="Campaign identifier")
    metrics: list[str] = Field(
        default_factory=lambda: ["success_rate"],
        description="Metrics to include",
    )
    granularity: TimeGranularity = Field(
        TimeGranularity.HOUR, description="Time bucket granularity"
    )
    start_time: datetime | None = Field(None, description="Series start time")
    end_time: datetime | None = Field(None, description="Series end time")
    filters: TelemetryFilterParams | None = Field(None, description="Event filters")


# =============================================================================
# Pagination Schemas
# =============================================================================


class CampaignListRequest(BaseRequest):
    """Request for paginated campaign list."""

    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: str = Field("created_at", description="Sort field")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="Sort order")
    filters: CampaignFilterParams | None = Field(None, description="Filter parameters")


class CampaignListResponse(BaseSchema):
    """Paginated campaign list response."""

    items: list[CampaignSummary] = Field(..., description="Campaign summaries")
    total: int = Field(..., ge=0, description="Total matching campaigns")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")


class TelemetryListResponse(BaseSchema):
    """Paginated telemetry event list response."""

    items: list[TelemetryEventSummary] = Field(..., description="Telemetry events")
    total: int = Field(..., ge=0, description="Total matching events")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
