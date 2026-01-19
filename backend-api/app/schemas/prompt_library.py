from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.domain.prompt_library_models import (
    PromptTemplate,
    SharingLevel,
    TechniqueType,
    TemplateRating,
    TemplateStatus,
    TemplateVersion,
    VulnerabilityType,
)


class CreateTemplateRequest(BaseModel):
    title: str
    description: str
    prompt_text: str
    technique_types: list[TechniqueType] = []
    vulnerability_types: list[VulnerabilityType] = []
    sharing_level: SharingLevel = SharingLevel.PRIVATE
    target_models: list[str] = []
    tags: list[str] = []
    custom_data: dict[str, Any] = {}


class UpdateTemplateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    sharing_level: SharingLevel | None = None
    status: TemplateStatus | None = None
    technique_types: list[TechniqueType] | None = None
    vulnerability_types: list[VulnerabilityType] | None = None
    target_models: list[str] | None = None
    tags: list[str] | None = None
    custom_data: dict[str, Any] | None = None


class SearchTemplatesRequest(BaseModel):
    query: str | None = None
    technique_type: TechniqueType | None = None
    vulnerability_type: VulnerabilityType | None = None
    sharing_level: SharingLevel | None = None
    tags: list[str] | None = None
    owner_id: str | None = None
    min_rating: float | None = None
    limit: int = 20
    offset: int = 0


class RateTemplateRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    effectiveness_vote: bool
    comment: str | None = None


class UpdateRatingRequest(RateTemplateRequest):
    pass


class CreateVersionRequest(BaseModel):
    prompt_text: str
    description: str | None = None
    metadata_overrides: dict[str, Any] | None = None


class TemplateListItem(BaseModel):
    id: str
    title: str
    description: str
    technique_types: list[TechniqueType]
    vulnerability_types: list[VulnerabilityType]
    sharing_level: SharingLevel
    status: TemplateStatus
    avg_rating: float
    total_ratings: int
    effectiveness_score: float
    tags: list[str]
    created_at: datetime
    owner_id: str


class TemplateDetailResponse(PromptTemplate):
    avg_rating: float
    total_ratings: int
    effectiveness_score: float


class SearchTemplatesResponse(BaseModel):
    items: list[TemplateListItem]
    total: int
    limit: int
    offset: int


class SaveFromCampaignRequest(BaseModel):
    campaign_id: str
    attack_id: str | None = None
    title: str
    description: str
    sharing_level: SharingLevel = SharingLevel.PRIVATE


# Additional Response Models for Schema Index Compatibility
class TemplateResponse(BaseModel):
    success: bool
    template: PromptTemplate


class TemplateListResponse(BaseModel):
    items: list[TemplateListItem]
    total: int


class TemplateVersionResponse(BaseModel):
    version: TemplateVersion


class TemplateVersionListResponse(BaseModel):
    versions: list[TemplateVersion]


class RatingResponse(BaseModel):
    rating: TemplateRating


class RatingStatistics(BaseModel):
    avg_rating: float
    total_ratings: int
    effectiveness_score: float
    rating_distribution: dict[int, int]


class RatingListResponse(BaseModel):
    ratings: list[TemplateRating]
    statistics: RatingStatistics | None = None


class RatingStatisticsResponse(BaseModel):
    stats: RatingStatistics


class TopRatedTemplatesResponse(BaseModel):
    items: list[TemplateListItem]


class TemplateDeleteResponse(BaseModel):
    success: bool
    id: str


class TemplateStatsResponse(BaseModel):
    total_templates: int
    total_ratings: int
    avg_effectiveness: float
