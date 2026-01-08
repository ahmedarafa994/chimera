from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class AegisRequest(BaseModel):
    objective: str
    max_iterations: int = 5
    target_model_name: Optional[str] = "mock-refusal-model"

class AegisResult(BaseModel):
    objective: str
    final_prompt: str
    score: float
    telemetry: Dict[str, Any]

class AegisResponse(BaseModel):
    status: str
    campaign_id: str
    results: List[AegisResult]
