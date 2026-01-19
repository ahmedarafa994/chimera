"""
Schema Layer

This module provides unified schema definitions for API requests and responses.
"""

# Re-export domain models that endpoints expect from schemas
# Re-export request/response models from domain for backward compatibility
from app.domain.models import (
    GenerationConfig,
    PromptRequest,
    PromptResponse,
    TechniqueSuite,
    TransformationRequest,
    TransformationResponse,
)
from app.schemas.aegis_telemetry import (
    AegisTelemetryEvent,
    AegisTelemetryEventType,
    AttackMetrics,
    CampaignStatus,
    CampaignSummary,
    LatencyMetrics,
    PromptEvolution,
    TechniqueCategory,
    TechniquePerformance,
    TokenUsage,
    create_attack_metrics,
    create_telemetry_event,
)
from app.schemas.api_schemas import (
    EvasionAttemptResult,
    EvasionTaskConfig,
    EvasionTaskResult,
    EvasionTaskStatusEnum,
    EvasionTaskStatusResponse,
    HealthCheckResponse,
    LLMModel,
    LLMModelBase,
    LLMModelCreate,
    LLMProvider,
    MetamorphosisStrategyConfig,
    MetamorphosisStrategyInfo,
)
from app.schemas.base_schemas import (
    BaseRequest,
    BaseResponse,
    BaseSchema,
    ErrorResponse,
    SuccessResponse,
)
from app.schemas.campaign_analytics import (  # Statistics schemas; Breakdown schemas; Campaign schemas; Enums; Export schemas; Time series schemas; Telemetry schemas
    AttemptCounts,
    BreakdownItem,
    CampaignBase,
    CampaignComparison,
    CampaignComparisonItem,
    CampaignComparisonRequest,
    CampaignCreate,
    CampaignDetail,
    CampaignFilterParams,
    CampaignListRequest,
    CampaignListResponse,
    CampaignStatistics,
    CampaignStatusEnum,
    CampaignUpdate,
    DistributionStats,
    ExecutionStatusEnum,
    ExportChartOptions,
    ExportDataOptions,
    ExportFormat,
    ExportRequest,
    ExportResponse,
    MetricType,
    MultiSeriesTimeSeries,
    PercentileStats,
    PotencyBreakdown,
    ProviderBreakdown,
    TechniqueBreakdown,
    TelemetryEventDetail,
    TelemetryEventSummary,
    TelemetryFilterParams,
    TelemetryListResponse,
    TelemetryTimeSeries,
    TimeGranularity,
    TimeSeriesDataPoint,
    TimeSeriesQuery,
)
from app.schemas.prompt_library import (
    CreateTemplateRequest,
    CreateVersionRequest,
    RateTemplateRequest,
    RatingListResponse,
    RatingResponse,
    RatingStatisticsResponse,
    SaveFromCampaignRequest,
    SearchTemplatesRequest,
    TemplateDeleteResponse,
    TemplateListResponse,
    TemplateResponse,
    TemplateStatsResponse,
    TemplateVersionListResponse,
    TemplateVersionResponse,
    TopRatedTemplatesResponse,
    UpdateRatingRequest,
    UpdateTemplateRequest,
)

# Aliases for backward compatibility with endpoints expecting these names
ExecuteRequest = PromptRequest
ExecuteResponse = PromptResponse
TransformRequest = TransformationRequest
TransformResponse = TransformationResponse

# Placeholder classes for missing schemas
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class LLMResult(BaseModel):
    content: str
    model: str | None = None
    provider: str | None = None
    usage: dict[str, Any] | None = None


class TransformationDetail(BaseModel):
    technique: str
    description: str | None = None
    applied: bool = True


class TransformResultMetadata(BaseModel):
    technique_suite: str | None = None
    potency_level: int = 5
    potency: int | None = None
    timestamp: str | None = None
    strategy: str | None = None
    cached: bool | None = None
    layers_applied: list[str] = []
    execution_time_ms: float | None = None
    applied_techniques: list[str] = []
    techniques_applied: list[str] = []
    bypass_probability: float | None = None


class Provider(BaseModel):
    name: str
    status: str = "available"
    is_default: bool = False


class ProviderInfo(BaseModel):
    name: str
    status: str = "available"
    models: list[str] = []


class ProviderListResponse(BaseModel):
    providers: list[ProviderInfo] = []


class TechniqueInfo(BaseModel):
    name: str
    description: str | None = None
    category: str | None = None


class TechniqueListResponse(BaseModel):
    techniques: list[TechniqueInfo] = []


__all__ = [
    # Aegis telemetry schemas
    "AegisTelemetryEvent",
    "AegisTelemetryEventType",
    "AttackMetrics",
    # Campaign Analytics - Statistics schemas
    "AttemptCounts",
    # Base schemas
    "BaseRequest",
    "BaseResponse",
    "BaseSchema",
    # Campaign Analytics - Breakdown schemas
    "BreakdownItem",
    # Campaign Analytics - Campaign schemas
    "CampaignBase",
    "CampaignComparison",
    "CampaignComparisonItem",
    "CampaignComparisonRequest",
    "CampaignCreate",
    "CampaignDetail",
    "CampaignFilterParams",
    "CampaignListRequest",
    "CampaignListResponse",
    "CampaignStatistics",
    "CampaignStatus",
    # Campaign Analytics - Enums
    "CampaignStatusEnum",
    "CampaignSummary",
    "CampaignUpdate",
    # Prompt Library schemas
    "CreateTemplateRequest",
    "CreateVersionRequest",
    "DistributionStats",
    "ErrorResponse",
    # Evasion schemas
    "EvasionAttemptResult",
    "EvasionTaskConfig",
    "EvasionTaskResult",
    "EvasionTaskStatusEnum",
    "EvasionTaskStatusResponse",
    # Execution aliases
    "ExecuteRequest",
    "ExecuteResponse",
    "ExecutionStatusEnum",
    # Campaign Analytics - Export schemas
    "ExportChartOptions",
    "ExportDataOptions",
    "ExportFormat",
    "ExportRequest",
    "ExportResponse",
    # Health check
    "HealthCheckResponse",
    # LLM schemas
    "LLMModel",
    "LLMModelBase",
    "LLMModelCreate",
    "LLMProvider",
    "LLMResult",
    "LatencyMetrics",
    # Metamorphosis schemas
    "MetamorphosisStrategyConfig",
    "MetamorphosisStrategyInfo",
    "MetricType",
    # Campaign Analytics - Time series schemas
    "MultiSeriesTimeSeries",
    "PercentileStats",
    "PotencyBreakdown",
    "PromptEvolution",
    # Provider schemas
    "Provider",
    "ProviderBreakdown",
    "ProviderInfo",
    "ProviderListResponse",
    "RateTemplateRequest",
    "RatingListResponse",
    "RatingResponse",
    "RatingStatisticsResponse",
    "SaveFromCampaignRequest",
    "SearchTemplatesRequest",
    "SuccessResponse",
    "TechniqueBreakdown",
    "TechniqueCategory",
    # Technique schemas
    "TechniqueInfo",
    "TechniqueListResponse",
    "TechniquePerformance",
    "TechniqueSuite",
    # Campaign Analytics - Telemetry schemas
    "TelemetryEventDetail",
    "TelemetryEventSummary",
    "TelemetryFilterParams",
    "TelemetryListResponse",
    "TelemetryTimeSeries",
    "TemplateDeleteResponse",
    "TemplateListResponse",
    "TemplateResponse",
    "TemplateStatsResponse",
    "TemplateVersionListResponse",
    "TemplateVersionResponse",
    "TimeGranularity",
    "TimeSeriesDataPoint",
    "TimeSeriesQuery",
    "TokenUsage",
    "TopRatedTemplatesResponse",
    # Transform aliases
    "TransformRequest",
    "TransformResponse",
    "TransformResultMetadata",
    "TransformationDetail",
    "UpdateRatingRequest",
    "UpdateTemplateRequest",
    "create_attack_metrics",
    "create_telemetry_event",
]
