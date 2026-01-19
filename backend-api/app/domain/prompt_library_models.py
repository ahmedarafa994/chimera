import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TechniqueType(str, Enum):
    AUTODAN = "autodan"
    GPTFUZZ = "gptfuzz"
    CHIMERA_FRAMING = "chimera_framing"
    PAIR = "pair"
    GCG = "gcg"
    TAP = "tap"
    CRESCENDO = "crescendo"
    MANUAL = "manual"
    OTHER = "other"


class VulnerabilityType(str, Enum):
    JAILBREAK = "jailbreak"
    INJECTION = "injection"
    PII_LEAK = "pii_leak"
    BYPASS = "bypass"
    MALICIOUS_CONTENT = "malicious_content"
    CODE_EXECUTION = "code_execution"
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"
    OTHER = "other"


class SharingLevel(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class TemplateStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class TemplateRating(BaseModel):
    user_id: str
    rating: int = Field(..., ge=1, le=5)
    effectiveness_vote: bool
    comment: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RatingStatistics(BaseModel):
    average_rating: float
    total_ratings: int
    effectiveness_score: float
    rating_distribution: dict[int, int]


class TemplateVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_version_id: str | None = None
    prompt_text: str
    description: str | None = None
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata_overrides: dict[str, Any] | None = None


class TemplateMetadata(BaseModel):
    technique_types: list[TechniqueType] = []
    vulnerability_types: list[VulnerabilityType] = []
    target_models: list[str] = []
    success_rate: float = 0.0
    test_count: int = 0
    avg_score: float = 0.0
    cve_references: list[str] = []
    discovery_date: datetime | None = None
    tags: list[str] = []
    custom_data: dict[str, Any] = {}


class PromptTemplate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    original_prompt: str
    current_version_id: str
    owner_id: str
    organization_id: str | None = None
    sharing_level: SharingLevel = SharingLevel.PRIVATE
    status: TemplateStatus = TemplateStatus.ACTIVE
    metadata: TemplateMetadata
    versions: list[TemplateVersion] = []
    ratings: list[TemplateRating] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def avg_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return sum(r.rating for r in self.ratings) / len(self.ratings)

    @property
    def total_ratings(self) -> int:
        return len(self.ratings)

    @property
    def effectiveness_score(self) -> float:
        if not self.ratings:
            return 0.0
        effective_count = sum(1 for r in self.ratings if r.effectiveness_vote)
        return effective_count / len(self.ratings)
