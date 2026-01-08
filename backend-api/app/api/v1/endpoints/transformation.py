import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.auth import Permission, require_permission
from app.core.deps import get_transformation_service
from app.schemas import TransformRequest, TransformResponse, TransformResultMetadata
from app.services.transformation_service import (
    TransformationEngine,
    TransformationResult,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=TransformResponse,
    dependencies=[Depends(require_permission(Permission.EXECUTE_TRANSFORM))],
)
async def transform_prompt(
    request: TransformRequest, service: TransformationEngine = Depends(get_transformation_service)
):
    """
    Transform a prompt using the specified technique suite and potency level.
    """
    start_time = time.time()

    # TransformationError and other exceptions are now handled globally
    result: TransformationResult = await service.transform(
        prompt=request.core_request,
        potency_level=request.potency_level,
        technique_suite=request.technique_suite,
    )

    execution_time_ms = (time.time() - start_time) * 1000

    # Get applied techniques from intermediate steps
    applied_techniques = result.intermediate_steps if result.intermediate_steps else []

    # Calculate bypass probability based on potency level
    bypass_probability = min(0.95, request.potency_level * 0.1)

    metadata = TransformResultMetadata(
        technique_suite=result.metadata.technique_suite,
        potency_level=result.metadata.potency_level,
        potency=result.metadata.potency_level,  # Frontend compatibility
        timestamp=result.metadata.timestamp,
        strategy=result.metadata.strategy,
        cached=result.metadata.cached,
        layers_applied=result.intermediate_steps,
        execution_time_ms=execution_time_ms,
        applied_techniques=applied_techniques,
        bypass_probability=bypass_probability,
    )

    return TransformResponse(
        original_prompt=result.original_prompt,
        transformed_prompt=result.transformed_prompt,
        metadata=metadata,
    )


# =============================================================================
# Streaming Endpoints (STREAM-005)
# =============================================================================


async def _transform_stream_generator(
    request: TransformRequest, service: TransformationEngine
) -> AsyncGenerator[str, None]:
    """Generate SSE events for transformation streaming."""
    try:
        async for chunk in service.transform_stream(
            prompt=request.core_request,
            potency_level=request.potency_level,
            technique_suite=request.technique_suite,
        ):
            event_data = {
                "text": chunk.text,
                "is_final": chunk.is_final,
                "finish_reason": chunk.finish_reason,
                "token_count": chunk.token_count,
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            if chunk.is_final:
                break

    except Exception as e:
        logger.error(f"Transformation streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'is_final': True})}\n\n"


@router.post("/stream", dependencies=[Depends(require_permission(Permission.EXECUTE_TRANSFORM))])
async def stream_transform_prompt(
    request: TransformRequest, service: TransformationEngine = Depends(get_transformation_service)
):
    """
    Stream prompt transformation using Server-Sent Events (SSE).

    This endpoint provides real-time streaming of the transformation process,
    showing each transformation step as it's applied. Particularly useful for
    AI-based transformations (Gemini Brain, AutoDAN) that take longer to complete.

    **Streaming Benefits:**
    - Real-time progress visibility
    - See each transformation step as it's applied
    - Lower perceived latency for complex transformations
    - Ability to cancel mid-transformation

    **SSE Event Format:**
    ```
    data: {"text": "[Step] Applied transformation...", "is_final": false}
    data: {"text": "transformed prompt content", "is_final": false}
    data: {"text": "[Complete] Finished in 123.45ms", "is_final": true}
    ```

    **Parameters:**
    - core_request: The prompt to transform
    - potency_level: Transformation intensity (1-10)
    - technique_suite: Which technique suite to use

    **Returns:**
    SSE stream of transformation progress and result
    """
    logger.info(
        f"Streaming transformation: potency={request.potency_level}, "
        f"suite={request.technique_suite}"
    )

    return StreamingResponse(
        _transform_stream_generator(request, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/estimate-tokens", dependencies=[Depends(require_permission(Permission.EXECUTE_TRANSFORM))]
)
async def estimate_transformation_tokens(
    request: TransformRequest, service: TransformationEngine = Depends(get_transformation_service)
):
    """
    Estimate token usage and cost for a transformation request.

    This endpoint provides token count estimates before running the actual
    transformation. Useful for understanding which transformations use LLM
    calls and their associated costs.

    **Estimation Includes:**
    - Input token count (if LLM-based)
    - Estimated output tokens
    - Total estimated tokens
    - Estimated cost in USD
    - Whether the transformation uses LLM
    - Transformation strategy that will be used

    **Parameters:**
    - core_request: The prompt to transform
    - potency_level: Transformation intensity (1-10)
    - technique_suite: Which technique suite to use

    **Returns:**
    Token estimation with cost information
    """
    try:
        estimation = await service.estimate_transformation_tokens(
            prompt=request.core_request,
            potency_level=request.potency_level,
            technique_suite=request.technique_suite,
        )
        logger.info(
            f"Transformation token estimation: {estimation.get('total_estimated_tokens', 0)} tokens, "
            f"uses_llm={estimation.get('uses_llm', False)}"
        )
        return estimation

    except Exception as e:
        logger.error(f"Transformation token estimation failed: {e}")
        return {
            "error": str(e),
            "input_tokens": 0,
            "estimated_output_tokens": 0,
            "total_estimated_tokens": 0,
            "estimated_cost_usd": 0.0,
            "uses_llm": False,
        }


@router.get("/cache/stats")
async def get_transformation_cache_stats(
    service: TransformationEngine = Depends(get_transformation_service),
):
    """
    Get transformation cache statistics.

    Returns cache performance metrics including hit rate, size, and eviction counts.
    """
    stats = service.get_cache_stats()
    if stats:
        return stats
    return {"message": "Cache is disabled"}
