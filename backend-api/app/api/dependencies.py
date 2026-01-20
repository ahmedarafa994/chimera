"""API dependencies for Dependency Injection."""

from app.services.llm_service import LLMService, llm_service
from app.services.transformation_service import TransformationEngine, transformation_engine


def get_llm_service() -> LLMService:
    """Dependency to get the LLM service instance."""
    return llm_service


def get_transformation_engine() -> TransformationEngine:
    """Dependency to get the Transformation engine instance."""
    return transformation_engine
