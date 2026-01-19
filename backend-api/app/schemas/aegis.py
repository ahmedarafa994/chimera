from typing import Any

from pydantic import BaseModel


class AegisRequest(BaseModel):
    objective: str
    max_iterations: int = 5
    target_model_name: str | None = "mock-refusal-model"


class AegisResult(BaseModel):
    objective: str
    final_prompt: str
    score: float
    telemetry: dict[str, Any]


class AegisResponse(BaseModel):
    status: str
    campaign_id: str
    results: list[AegisResult]
    telemetry: dict[str, Any] | None = None
