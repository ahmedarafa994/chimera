from typing import Any

from llm_provider_client import LLMProvider
from pydantic import BaseModel, Field, validator


class TransformRequestSchema(BaseModel):
    core_request: str = Field(..., min_length=1, description="The core prompt to transform")
    potency_level: int = Field(..., ge=1, le=10, description="Potency level between 1 and 10")
    technique_suite: str = Field(..., min_length=1, description="The technique suite to use")


class ExecuteRequestSchema(TransformRequestSchema):
    provider: str = Field(..., description="LLM provider to use")
    model: str | None = Field(None, description="Specific model to use")
    use_cache: bool = Field(True, description="Whether to use cached responses")
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    @validator("provider")
    def validate_provider(cls, v):
        try:
            LLMProvider(v)
        except ValueError:
            raise ValueError(f"Invalid provider. Must be one of: {[p.value for p in LLMProvider]}")
        return v
