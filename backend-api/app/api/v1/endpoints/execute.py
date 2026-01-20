import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_llm_service, get_transformation_service
from app.core.errors import LLMProviderError, TransformationError
from app.schemas import ExecuteRequest, ExecuteResponse, LLMResult, TransformationDetail
from app.services.llm_service import LLMService
from app.services.transformation_service import TransformationEngine

router = APIRouter()
logger = logging.getLogger(__name__)

from typing import Annotated

from app.core.auth import Permission, require_permission


@router.post(
    "/execute",
    response_model=ExecuteResponse,
    dependencies=[Depends(require_permission(Permission.EXECUTE_TRANSFORM))],
)
async def execute_transformation(
    request: ExecuteRequest,
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    transformation_engine: Annotated[TransformationEngine, Depends(get_transformation_service)],
):
    """Transform and execute prompt against real LLM provider."""
    try:
        # 1. Transform the prompt
        transform_result = await transformation_engine.transform(
            prompt=request.core_request,
            potency_level=request.potency_level,
            technique_suite=request.technique_suite.value,
            use_cache=request.use_cache,
        )

        # 2. Execute against LLM
        llm_params = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
        }
        # Filter out None values
        llm_params = {k: v for k, v in llm_params.items() if v is not None}

        # Pass API key via metadata if present
        if request.api_key:
            llm_params.setdefault("metadata", {})["api_key"] = request.api_key

        llm_response = await llm_service.generate(
            prompt=transform_result.transformed_prompt,
            provider=request.provider.value,
            **llm_params,
        )

        return ExecuteResponse(
            success=True,
            request_id=llm_response.request_id,
            result=LLMResult(
                content=llm_response.content,
                tokens=llm_response.usage.total_tokens,
                cost=llm_response.usage.estimated_cost,
                latency_ms=llm_response.latency_ms,
                cached=llm_response.cached,
                provider=llm_response.provider.value,
                model=llm_response.model,
            ),
            transformation=TransformationDetail(
                original_prompt=transform_result.original_prompt,
                transformed_prompt=transform_result.transformed_prompt,
                technique_suite=transform_result.metadata.technique_suite,
                potency_level=transform_result.metadata.potency_level,
                metadata={
                    "strategy": transform_result.metadata.strategy,
                    "layers": transform_result.metadata.layers_applied,
                },
            ),
        )

    except TransformationError as e:
        logger.exception(f"Transformation Error: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except LLMProviderError as e:
        logger.exception(f"LLM Provider Error: {e.detail}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        logger.exception(f"Unexpected error: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e!s}",
        )
