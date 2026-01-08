from collections.abc import Generator

from app.core.service_registry import service_registry
from app.services.llm_service import LLMService
from app.services.transformation_service import TransformationEngine


def get_service(service_name: str):
    """Generic service dependency factory."""

    def _get_service():
        yield service_registry.get(service_name)

    return _get_service


# Specific service dependencies
get_llm_service = get_service("llm_service")
get_transformation_service = get_service("transformation_engine")
get_metamorph_service = get_service("metamorph_service")


# Legacy support for typing
def get_llm_service_typed() -> Generator[LLMService, None, None]:
    yield service_registry.get("llm_service")


def get_transformation_service_typed() -> Generator[TransformationEngine, None, None]:
    yield service_registry.get("transformation_engine")
