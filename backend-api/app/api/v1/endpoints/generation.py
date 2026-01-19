"""
Generation API Endpoints

UNIFIED-PROVIDER: Updated to use ProviderResolutionService for
    consistent provider/model selection across all requests.
"""

from fastapi import APIRouter, Depends, Request, status

from app.core.dependencies import get_resolved_provider_model
from app.core.logging import logger
from app.domain.models import PromptRequest, PromptResponse
from app.services.llm_service import llm_service

router = APIRouter()


def get_llm_service():
    """Get the LLM service instance."""
    return llm_service


@router.post(
    "/generate",
    response_model=PromptResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_content(
    request: PromptRequest,
    fastapi_request: Request,
    service=Depends(get_llm_service),
):
    """
    Generate text content using the configured LLM provider.

    UNIFIED-PROVIDER: Uses ProviderResolutionService to determine
    the active provider/model based on:
    1. Explicit request parameters (if provided)
    2. Request context from GlobalModelSelectionState
    3. Session selection (if session_id in headers)
    4. Global default configuration
    """
    # UNIFIED-PROVIDER: Resolve provider/model if not explicitly set
    if not request.provider or not request.model:
        try:
            resolved_provider, resolved_model = await get_resolved_provider_model(
                request=fastapi_request,
                session_id=fastapi_request.headers.get("X-Session-ID"),
                explicit_provider=request.provider,
                explicit_model=request.model,
            )

            # Update request with resolved values
            if not request.provider:
                request.provider = resolved_provider
            if not request.model:
                request.model = resolved_model

            logger.debug(
                f"Resolved provider/model for generation: " f"{request.provider}/{request.model}"
            )
        except Exception as e:
            logger.warning(f"Provider resolution failed, using defaults: {e}")

    return await service.generate_text(request)


@router.post(
    "/generate/with-resolution",
    response_model=PromptResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_with_resolution(
    request: PromptRequest,
    fastapi_request: Request,
    provider_model: tuple[str, str] = Depends(get_resolved_provider_model),
    service=Depends(get_llm_service),
):
    """
    Generate text with explicit provider/model resolution.

    This endpoint always uses the ProviderResolutionService to
    determine the provider/model, overriding any explicit values
    in the request unless they are provided.
    """
    provider, model = provider_model

    # Override with resolved values
    request.provider = request.provider or provider
    request.model = request.model or model

    logger.info(f"Generating with resolved selection: " f"{request.provider}/{request.model}")

    return await service.generate_text(request)


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(service=Depends(get_llm_service)):
    """
    Check the health of the LLM provider.
    """
    providers = service.get_available_providers()
    if providers:
        return {"status": "healthy", "providers": providers}
    return {"status": "degraded", "providers": []}


@router.get("/current-selection", status_code=status.HTTP_200_OK)
async def get_current_selection(
    fastapi_request: Request,
):
    """
    Get the currently resolved provider/model selection.

    UNIFIED-PROVIDER: Returns the provider/model that would be used
    for generation based on the current request context.
    """
    try:
        provider, model = await get_resolved_provider_model(
            request=fastapi_request,
            session_id=fastapi_request.headers.get("X-Session-ID"),
        )

        return {
            "provider": provider,
            "model": model,
            "source": "resolved",
            "session_id": fastapi_request.headers.get("X-Session-ID"),
        }
    except Exception as e:
        logger.error(f"Failed to get current selection: {e}")
        return {
            "provider": None,
            "model": None,
            "source": "error",
            "error": str(e),
        }
