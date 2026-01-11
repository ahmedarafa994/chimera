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
from app.schemas.campaign_analytics import (
    # Enums
    CampaignStatusEnum,
    ExecutionStatusEnum,
    ExportFormat,
    MetricType,
    TimeGranularity,
    # Campaign schemas
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
    CampaignSummary,
    CampaignUpdate,
    # Breakdown schemas
    BreakdownItem,
    PotencyBreakdown,
    ProviderBreakdown,
    TechniqueBreakdown,
    # Statistics schemas
    AttemptCounts,
    DistributionStats,
    PercentileStats,
    # Time series schemas
    MultiSeriesTimeSeries,
    TelemetryTimeSeries,
    TimeSeriesDataPoint,
    TimeSeriesQuery,
    # Telemetry schemas
    TelemetryEventDetail,
    TelemetryEventSummary,
    TelemetryFilterParams,
    TelemetryListResponse,
    # Export schemas
    ExportChartOptions,
    ExportDataOptions,
    ExportRequest,
    ExportResponse,
)

# Aliases for backward compatibility with endpoints expecting these names
ExecuteRequest = PromptRequest
ExecuteResponse = PromptResponse
TransformRequest = PromptRequest
TransformResponse = PromptResponse

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
    techniques_applied: list[str] = []
    potency_level: int = 5


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
    # Base schemas
    "BaseRequest",
    "BaseResponse",
    "BaseSchema",
    "ErrorResponse",
    "SuccessResponse",
    # Evasion schemas
    "EvasionAttemptResult",
    "EvasionTaskConfig",
    "EvasionTaskResult",
    "EvasionTaskStatusEnum",
    "EvasionTaskStatusResponse",
    # Execution aliases
    "ExecuteRequest",
    "ExecuteResponse",
    # Health check
    "HealthCheckResponse",
    # LLM schemas
    "LLMModel",
    "LLMModelBase",
    "LLMModelCreate",
    "LLMProvider",
    "LLMResult",
    # Metamorphosis schemas
    "MetamorphosisStrategyConfig",
    "MetamorphosisStrategyInfo",
    # Provider schemas
    "Provider",
    "ProviderInfo",
    "ProviderListResponse",
    # Technique schemas
    "TechniqueInfo",
    "TechniqueListResponse",
    "TechniqueSuite",
    # Transform aliases
    "TransformRequest",
    "TransformResponse",
    "TransformResultMetadata",
    "TransformationDetail",
    # Campaign Analytics - Enums
    "CampaignStatusEnum",
    "ExecutionStatusEnum",
    "ExportFormat",
    "MetricType",
    "TimeGranularity",
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
    "CampaignSummary",
    "CampaignUpdate",
    # Campaign Analytics - Breakdown schemas
    "BreakdownItem",
    "PotencyBreakdown",
    "ProviderBreakdown",
    "TechniqueBreakdown",
    # Campaign Analytics - Statistics schemas
    "AttemptCounts",
    "DistributionStats",
    "PercentileStats",
    # Campaign Analytics - Time series schemas
    "MultiSeriesTimeSeries",
    "TelemetryTimeSeries",
    "TimeSeriesDataPoint",
    "TimeSeriesQuery",
    # Campaign Analytics - Telemetry schemas
    "TelemetryEventDetail",
    "TelemetryEventSummary",
    "TelemetryFilterParams",
    "TelemetryListResponse",
    # Campaign Analytics - Export schemas
    "ExportChartOptions",
    "ExportDataOptions",
    "ExportRequest",
    "ExportResponse",
]
