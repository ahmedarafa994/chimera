import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_llm_service
from app.core.errors import LLMProviderError as LLMError
from app.schemas import ExecuteRequest, ExecuteResponse, LLMResult, TransformationDetail
from app.services.llm_service import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/completions", response_model=ExecuteResponse)
async def execute_chat_completion(
    request: ExecuteRequest, llm_service: LLMService = Depends(get_llm_service)
):
    """
    Execute a chat completion request with optional prompt transformation.
    """
    try:
        # Apply Input Understanding Service to expand/optimize the core request
        from app.services.input_understanding_service import input_understanding_service

        # Expand the input to ensure better understanding and execution
        expanded_prompt = await input_understanding_service.expand_input(request.core_request)

        # For now, we are not applying transformation in this endpoint directly
        # unless we integrate TransformationService here too.
        # The request schema has 'technique_suite', implying transformation might happen here.
        # However, the plan separates them or implies orchestration.
        # Let's assume for now this endpoint handles the raw LLM interaction
        # or minimal transformation if needed.
        # Given the schema, let's just pass the prompt to the LLM service.

        # Note: In a full implementation, we might want to inject TransformationService
        # and apply it before calling LLMService if the user requested it.
        # But for this step, let's focus on the LLM interaction.

        from app.domain.models import (
            GenerationConfig,
            LLMProviderType,
            PromptRequest,
            PromptResponse,
        )

        # Map provider string to enum
        provider_enum = None
        if request.provider:
            try:
                provider_enum = LLMProviderType(request.provider.value)
            except ValueError:
                pass  # Will use default if None

        generation_config = GenerationConfig(
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 0.95,
            max_output_tokens=request.max_tokens if request.max_tokens is not None else 2048,
        )

        prompt_request = PromptRequest(
            prompt=expanded_prompt,
            provider=provider_enum,
            model=request.model,
            config=generation_config,
            api_key=request.api_key,
        )

        response: PromptResponse = await llm_service.generate_text(prompt_request)

        # Generate a request ID since PromptResponse doesn't have one
        request_id = str(uuid.uuid4())

        # Extract token count safely, handling different key names
        total_tokens = 0
        if response.usage_metadata:
            total_tokens = response.usage_metadata.get(
                "total_token_count", response.usage_metadata.get("total_tokens", 0)
            )

        result = LLMResult(
            content=response.text,
            tokens=total_tokens,
            cost=0.0,  # Cost calculation not yet implemented
            latency_ms=response.latency_ms,
            cached=False,  # Cache status not yet propagated
            provider=response.provider,
            model=response.model_used,
        )

        # Placeholder for transformation details since we didn't transform here yet
        transformation_detail = TransformationDetail(
            original_prompt=request.core_request,
            transformed_prompt=expanded_prompt,
            technique_suite=request.technique_suite,
            potency_level=request.potency_level,
            metadata={"input_expansion": "applied"},
        )

        return ExecuteResponse(
            request_id=request_id, result=result, transformation=transformation_detail
        )

    except LLMError as e:
        logger.error(f"LLM Error: {e.message}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error"
        )
