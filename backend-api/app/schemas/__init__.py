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
    create_telemetry_event,
    create_attack_metrics,
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
    # Aegis telemetry schemas
    "AegisTelemetryEvent",
    "AegisTelemetryEventType",
    "AttackMetrics",
    "CampaignStatus",
    "CampaignSummary",
    "LatencyMetrics",
    "PromptEvolution",
    "TechniqueCategory",
    "TechniquePerformance",
    "TokenUsage",
    "create_telemetry_event",
    "create_attack_metrics",
    # API schemas
    "EvasionAttemptResult",
    "EvasionTaskConfig",
    "EvasionTaskResult",
    "EvasionTaskStatusEnum",
    "EvasionTaskStatusResponse",
    "ExecuteRequest",
    "ExecuteResponse",
    "HealthCheckResponse",
    "LLMModel",
    "LLMModelBase",
    "LLMModelCreate",
    "LLMProvider",
    "LLMResult",
    "MetamorphosisStrategyConfig",
    "MetamorphosisStrategyInfo",
    "Provider",
    "ProviderInfo",
    "ProviderListResponse",
    "TechniqueInfo",
    "TechniqueListResponse",
    "TechniqueSuite",
    "TransformRequest",
    "TransformResponse",
    "TransformResultMetadata",
    "TransformationDetail",
]
