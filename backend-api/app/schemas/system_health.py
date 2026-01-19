"""System health check schemas."""
from pydantic import BaseModel
from typing import Dict, List


class SystemHealthResponse(BaseModel):
    """Response model for system health check."""

    status: str
    services: Dict[str, str]
    active_techniques: List[str]
    api_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "database": "connected",
                    "redis": "connected",
                    "llm_providers": "available"
                },
                "active_techniques": ["persona", "obfuscation", "payload_splitting"],
                "api_version": "1.0.0"
            }
        }
