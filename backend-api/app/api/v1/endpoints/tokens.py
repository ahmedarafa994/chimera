"""Token counting endpoint for estimating token usage.

This module provides endpoints for counting tokens in text,
useful for cost estimation and context window management.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.logging import logger
from app.domain.models import LLMProviderType, TokenCountRequest, TokenCountResponse
from app.services.llm_service import LLMService, llm_service

router = APIRouter(tags=["tokens"])


def get_llm_service() -> LLMService:
    return llm_service


class TokenEstimationRequest(BaseModel):
    """Request model for token estimation with cost info."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Text to estimate tokens for",
    )
    provider: LLMProviderType | None = Field(None, description="Provider to use for estimation")


class TokenEstimationResponse(BaseModel):
    """Response model for token estimation with cost and context info."""

    token_count: int = Field(..., ge=0, description="Number of tokens in the text")
    estimated_cost_usd: float = Field(..., ge=0, description="Estimated cost in USD")
    context_window: int = Field(..., ge=0, description="Provider's context window size")
    context_usage_percent: float = Field(..., ge=0, description="Percentage of context window used")
    provider: str = Field(..., description="Provider used for estimation")
    estimation_method: str | None = Field(None, description="Method used if fallback was needed")


@router.post(
    "/tokens/count",
    response_model=TokenCountResponse,
    status_code=status.HTTP_200_OK,
    summary="Count tokens in text",
    description="""
    Count the number of tokens in the provided text.

    This is useful for:
    - **Cost estimation**: Estimate API costs before making generation requests
    - **Context window management**: Check if text fits within model's context limit
    - **Prompt optimization**: Understand token usage for prompt engineering

    **Supported Providers**: Google/Gemini, OpenAI, Anthropic, DeepSeek

    **Example request**:
    ```json
    {
        "text": "Explain quantum computing in simple terms",
        "model": "gemini-2.5-flash",
        "provider": "google"
    }
    ```

    **Example response**:
    ```json
    {
        "total_tokens": 8,
        "model": "gemini-2.5-flash",
        "provider": "google",
        "cached_content_tokens": null
    }
    ```
    """,
    responses={
        200: {"description": "Token count result"},
        400: {"description": "Invalid request"},
        501: {"description": "Token counting not supported for this provider/mode"},
        502: {"description": "Provider error"},
    },
)
async def count_tokens(
    request: TokenCountRequest, service: Annotated[LLMService, Depends(get_llm_service)]
):
    """Count tokens in text for a specific model.

    Useful for estimating costs and checking context window limits.
    """
    # Get provider name, default to Google
    provider_name = request.provider.value if request.provider else "google"

    # Normalize provider name
    if provider_name in ("gemini", "google"):
        provider_name = "google"

    try:
        # Use the LLM service's token counting method
        total_tokens = await service.count_tokens(request.text, provider_name, request.model)

        # Determine the model used (either specified or default)
        model_used = request.model
        if not model_used:
            # Get default model from config
            from app.core.config import get_settings

            settings = get_settings()
            if provider_name in ("google", "gemini"):
                model_used = settings.GOOGLE_MODEL
            elif provider_name == "openai":
                model_used = settings.OPENAI_MODEL
            elif provider_name == "anthropic":
                model_used = settings.ANTHROPIC_MODEL
            elif provider_name == "deepseek":
                model_used = settings.DEEPSEEK_MODEL
            else:
                model_used = "unknown"

        return TokenCountResponse(
            total_tokens=total_tokens,
            model=model_used,
            provider=provider_name,
            cached_content_tokens=None,  # Future: support cached content
        )

    except NotImplementedError as e:
        logger.error(f"Token counting not supported: {e}")
        raise AppError(str(e), status_code=status.HTTP_501_NOT_IMPLEMENTED)
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Token counting failed: {e}", exc_info=True)
        msg = f"Token counting failed: {e}"
        raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)


@router.post(
    "/tokens/estimate",
    response_model=TokenEstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate tokens with cost info",
    description="""
    Estimate token count with additional cost and context window information.

    This endpoint provides:
    - **Token count**: Number of tokens in the text
    - **Cost estimation**: Approximate cost in USD based on provider rates
    - **Context usage**: How much of the provider's context window is used

    **Supported Providers**: Google/Gemini, OpenAI, Anthropic, DeepSeek

    **Example request**:
    ```json
    {
        "text": "Explain quantum computing in simple terms",
        "provider": "google"
    }
    ```

    **Example response**:
    ```json
    {
        "token_count": 8,
        "estimated_cost_usd": 0.000002,
        "context_window": 1000000,
        "context_usage_percent": 0.0008,
        "provider": "google"
    }
    ```
    """,
    responses={
        200: {"description": "Token estimation with cost info"},
        400: {"description": "Invalid request"},
        502: {"description": "Provider error"},
    },
)
async def estimate_tokens(
    request: TokenEstimationRequest,
    service: Annotated[LLMService, Depends(get_llm_service)],
):
    """Estimate tokens with cost and context window information.

    Provides comprehensive token analysis including cost estimation.
    """
    provider_name = request.provider.value if request.provider else None

    try:
        result = await service.estimate_tokens(request.text, provider_name)

        return TokenEstimationResponse(
            token_count=result["token_count"],
            estimated_cost_usd=result["estimated_cost_usd"],
            context_window=result["context_window"],
            context_usage_percent=result["context_usage_percent"],
            provider=result["provider"],
            estimation_method=result.get("estimation_method"),
        )

    except Exception as e:
        logger.error(f"Token estimation failed: {e}", exc_info=True)
        msg = f"Token estimation failed: {e}"
        raise AppError(msg, status_code=status.HTTP_502_BAD_GATEWAY)


@router.get(
    "/tokens/capabilities",
    summary="Get token counting capabilities",
    description="Get information about which providers support token counting.",
    responses={200: {"description": "Token counting capabilities by provider"}},
)
async def get_token_capabilities(service: Annotated[LLMService, Depends(get_llm_service)]):
    """Get token counting capabilities for all registered providers.

    Returns a map of provider names to their token counting support status.
    """
    providers = service.get_available_providers()
    capabilities = {}

    for provider in providers:
        capabilities[provider] = {
            "token_counting_supported": service.supports_token_counting(provider),
            "streaming_supported": service.supports_streaming(provider),
        }

    return {"providers": capabilities, "default_provider": service._default_provider}
