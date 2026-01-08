import json
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logging import logger
from app.domain.advanced_models import (
    AdvancedGenerationStats,
    AvailableTechniquesResponse,
    CodeGenerationRequest,
    CodeGenerationResponse,
    HealthCheckResponse,
    JailbreakGenerationRequest,
    JailbreakGenerationResponse,
    PromptValidationRequest,
    PromptValidationResponse,
    RedTeamSuiteRequest,
    RedTeamSuiteResponse,
)
from app.services.advanced_prompt_service import AdvancedPromptService, advanced_prompt_service

router = APIRouter(tags=["advanced-generation"])
security = HTTPBearer()


def get_advanced_prompt_service() -> AdvancedPromptService:
    """Dependency injection for advanced prompt service."""
    return advanced_prompt_service


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for advanced generation endpoints using timing-safe comparison."""
    import secrets

    if not credentials or not settings.CHIMERA_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Timing-safe comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, settings.CHIMERA_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@router.post(
    "/jailbreak/generate",
    response_model=JailbreakGenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_jailbreak_prompt(
    request: JailbreakGenerationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Generate an advanced jailbreak prompt using sophisticated techniques.

    This endpoint applies multiple layers of prompt engineering techniques to create
    prompts that can bypass AI safety filters and restrictions. It's designed for
    security research and red teaming purposes.

    **Available Techniques:**
    - Content Transformation (LeetSpeak, Homoglyphs, Caesar Cipher)
    - Structural & Semantic (Role Hijacking, Instruction Injection)
    - Advanced Neural (Neural Bypass, Meta Prompting)
    - Research-Driven (Multilingual Trojan, Payload Splitting)
    - Contextual Interaction Attacks

    **Security Features:**
    - Input validation and sanitization
    - Comprehensive audit logging
    - Rate limiting and abuse prevention
    - Request caching for performance

    **Parameters:**
    - core_request: The base prompt to transform
    - technique_suite: Which suite of techniques to apply
    - potency_level: How aggressive the transformation should be (1-10)
    - Various technique-specific boolean flags and parameters
    """
    try:
        logger.info(
            f"Jailbreak generation request: potency={request.potency_level}, suite={request.technique_suite}"
        )

        result = await service.generate_jailbreak_prompt(request)

        logger.info(
            f"Jailbreak generation completed: {result.request_id}, success={result.success}"
        )
        return result

    except Exception as e:
        logger.error(f"Jailbreak generation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Jailbreak generation failed: {str(e)}",
        )


@router.post(
    "/code/generate", response_model=CodeGenerationResponse, status_code=status.HTTP_200_OK
)
async def generate_code(
    request: CodeGenerationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Generate high-quality code using advanced Gemini models.

    This endpoint uses sophisticated AI models to generate clean, efficient,
    and secure code based on natural language descriptions. It supports multiple
    programming languages and frameworks.

    **Features:**
    - Multi-language support (Python, JavaScript, Java, C++, Go, etc.)
    - Framework-specific generation (React, FastAPI, Django, etc.)
    - Security best practices integration
    - Comprehensive code documentation
    - Error handling and edge cases

    **Parameters:**
    - prompt: Natural language description of the code to generate
    - language: Optional preferred programming language
    - framework: Optional preferred framework or library
    - is_thinking_mode: Whether to use advanced reasoning capabilities
    """
    try:
        logger.info(
            f"Code generation request: language={request.language}, thinking_mode={request.is_thinking_mode}"
        )

        result = await service.generate_code(request)

        logger.info(
            f"Code generation completed: {result.request_id}, success={result.success}, language={result.language}"
        )
        return result

    except Exception as e:
        logger.error(f"Code generation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code generation failed: {str(e)}",
        )


@router.post(
    "/redteam/generate-suite", response_model=RedTeamSuiteResponse, status_code=status.HTTP_200_OK
)
async def generate_red_team_suite(
    request: RedTeamSuiteRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Generate a comprehensive red team testing suite.

    This endpoint creates a complete suite of test variants designed to evaluate
    AI model security, robustness, and vulnerability to various attack vectors.
    It's intended for security researchers and red team professionals.

    **Generated Content:**
    - Multiple attack variants (3-10 based on request)
    - Detailed technique documentation
    - Expected assessment criteria
    - Potential failure modes analysis
    - Responsible use guidelines
    - Success criteria for each variant

    **Technique Categories:**
    - Direct approaches with research framing
    - Hypothetical scenario injection
    - Educational context establishment
    - Technical documentation style
    - Comparative analysis frameworks
    - Historical/academic perspectives
    - Step-by-step reasoning chains

    **Parameters:**
    - prompt: The base prompt to create variants for
    - include_metadata: Whether to include detailed technique metadata
    - variant_count: Number of variants to generate (3-10)
    """
    try:
        logger.info(
            f"Red team suite generation request: variants={request.variant_count}, include_metadata={request.include_metadata}"
        )

        result = await service.generate_red_team_suite(request)

        logger.info(
            f"Red team suite generation completed: {result.request_id}, success={result.success}"
        )
        return result

    except Exception as e:
        logger.error(f"Red team suite generation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Red team suite generation failed: {str(e)}",
        )


@router.post(
    "/validate/prompt", response_model=PromptValidationResponse, status_code=status.HTTP_200_OK
)
async def validate_prompt(
    request: PromptValidationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Validate a prompt for safety, effectiveness, and compliance.

    This endpoint analyzes prompts to assess their safety, identify potential
    issues, and provide recommendations for improvement. It supports multiple
    validation levels for different use cases.

    **Validation Levels:**
    - basic: Basic safety and format checking
    - standard: Comprehensive analysis including risk assessment
    - comprehensive: Deep analysis with detailed recommendations

    **Analysis Areas:**
    - Safety and compliance checking
    - Risk scoring and threat assessment
    - Effectiveness and clarity evaluation
    - Technical quality assessment
    - Improvement recommendations

    **Parameters:**
    - prompt: The prompt to validate
    - test_input: Optional test input for validation
    - validation_level: Level of validation to perform
    """
    try:
        logger.info(f"Prompt validation request: level={request.validation_level}")

        result = await service.validate_prompt(request)

        logger.info(
            f"Prompt validation completed: valid={result.is_valid}, risk_score={result.risk_score}"
        )
        return result

    except Exception as e:
        logger.error(f"Prompt validation failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt validation failed: {str(e)}",
        )


@router.get(
    "/techniques/available",
    response_model=AvailableTechniquesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_available_techniques(
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Get information about available jailbreak and prompt engineering techniques.

    This endpoint provides detailed information about all available techniques
    that can be used in jailbreak prompt generation, including their descriptions,
    risk levels, complexity ratings, and usage statistics.

    **Technique Categories:**
    - Content Transformation: Character and encoding-based obfuscation
    - Structural & Semantic: Prompt structure and meaning manipulation
    - Advanced Neural: Neural network-level bypass techniques
    - Research-Driven: Academic and research-based approaches

    **Information Provided:**
    - Technique name and description
    - Risk level (low, medium, high, critical)
    - Complexity (basic, intermediate, advanced, expert)
    - Historical usage statistics
    - Success rates (if available)
    - Associated tags for categorization

    **Returns:**
    Comprehensive list of available techniques with metadata
    """
    try:
        result = await service.get_available_techniques()
        logger.info(
            f"Retrieved available techniques: {result.total_techniques} techniques, {result.enabled_count} enabled"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to retrieve available techniques: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve available techniques: {str(e)}",
        )


@router.get("/statistics", response_model=AdvancedGenerationStats, status_code=status.HTTP_200_OK)
async def get_advanced_generation_statistics(
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Get comprehensive statistics for the advanced prompt generation service.

    This endpoint provides detailed analytics about service usage, performance,
    and effectiveness. It includes metrics for all available generation types
    and helps monitor system health and usage patterns.

    **Metrics Included:**
    - Request counts and success rates
    - Average execution times
    - Error rates and failure patterns
    - Cache performance metrics
    - Technique usage statistics
    - Service uptime and availability

    **Statistics Categories:**
    - Jailbreak generation metrics
    - Code generation metrics
    - Red team suite generation metrics
    - Prompt validation metrics

    **Returns:**
    Comprehensive statistics object with all metrics
    """
    try:
        result = await service.get_statistics()
        logger.info("Retrieved advanced generation statistics")
        return result

    except Exception as e:
        logger.error(f"Failed to retrieve statistics: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}",
        )


@router.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK)
async def advanced_generation_health_check(
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Health check for the advanced prompt generation subsystem.

    This endpoint performs a comprehensive health check of all advanced generation
    components, including API connectivity, model availability, and service
    performance metrics.

    **Health Checks Performed:**
    - API key validation and connectivity
    - Model availability and responsiveness
    - Service component health
    - Performance metrics collection
    - Error detection and reporting

    **Response Information:**
    - Overall service status (healthy, degraded, unhealthy)
    - API key validity status
    - Available models and their status
    - Response time measurements
    - Any detected errors or issues

    **Returns:**
    Detailed health status information
    """
    try:
        result = await service.health_check()
        logger.info(f"Advanced generation health check: status={result.status}")
        return result

    except Exception as e:
        logger.error(f"Advanced generation health check failed: {e!s}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            api_key_valid=False,
            models_available=[],
            response_time_ms=None,
            error=str(e),
        )


@router.post("/reset", status_code=status.HTTP_200_OK)
async def reset_advanced_generation_service(
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Reset the advanced prompt generation service.

    This endpoint resets the service state, clears caches, and reinitializes
    connections. Use this for troubleshooting or after configuration changes.

    **Operations Performed:**
    - Clear all internal caches
    - Reset statistics counters
    - Reinitialize API connections
    - Clear any in-memory state

    **Warning:** This will clear all cached results and statistics.
    Only use this when necessary for troubleshooting or maintenance.

    **Returns:**
    Success status and reset confirmation
    """
    try:
        # Reset the advanced client
        from app.infrastructure.advanced_generation_service import reset_advanced_client

        reset_advanced_client()

        # Reset service stats and cache
        service._stats = {
            "jailbreak": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "technique_usage": {},
                "cache_hits": 0,
                "cache_requests": 0,
            },
            "code_generation": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "language_usage": {},
                "cache_hits": 0,
                "cache_requests": 0,
            },
            "red_team": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "technique_usage": {},
                "cache_hits": 0,
                "cache_requests": 0,
            },
            "validation": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "validation_results": {},
                "cache_hits": 0,
                "cache_requests": 0,
            },
        }
        service._cache.clear()
        service._start_time = datetime.utcnow().timestamp()

        logger.info("Advanced generation service reset successfully")
        return {
            "status": "success",
            "message": "Advanced generation service reset successfully",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to reset advanced generation service: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset advanced generation service: {str(e)}",
        )


@router.get("/config", status_code=status.HTTP_200_OK)
async def get_advanced_generation_config(api_key: str = Depends(verify_api_key)):
    """
    Get configuration information for the advanced prompt generation service.

    This endpoint returns current configuration settings and limits for the
    advanced generation service. Useful for understanding capabilities and
    constraints.

    **Configuration Information:**
    - Service limits and constraints
    - Available models and their capabilities
    - Technique categories and options
    - Performance settings and timeouts
    - Security configuration details

    **Returns:**
    Current configuration settings and capabilities
    """
    try:
        config = {
            "service_version": "1.0.0",
            "models": {
                "available": ["gemini-3-pro-preview", "gemini-2.5-flash", "gemini-2.5-pro"],
                "default": "gemini-3-pro-preview",
                "thinking_model": "gemini-3-pro-preview",
            },
            "limits": {
                "max_prompt_length": 50000,
                "max_code_generation_length": 10000,
                "max_red_team_prompt_length": 5000,
                "max_new_tokens": 8192,
                "max_red_team_tokens": 16384,
                "max_variant_count": 10,
                "max_cache_ttl_seconds": 3600,
            },
            "technique_categories": [
                "Content Transformation",
                "Structural & Semantic",
                "Advanced Neural",
                "Research-Driven & Esoteric",
            ],
            "validation_levels": ["basic", "standard", "comprehensive"],
            "technique_suites": ["standard", "typoglycemia", "advanced"],
            "potency_range": {"min": 1, "max": 10},
            "temperature_range": {"min": 0.0, "max": 2.0},
            "top_p_range": {"min": 0.0, "max": 1.0},
            "features": {
                "caching": True,
                "thinking_mode": True,
                "multilingual_support": True,
                "batch_processing": False,
                "streaming": True,  # Updated to reflect streaming support
            },
            "security": {
                "api_key_required": True,
                "input_validation": True,
                "rate_limiting": True,
                "audit_logging": True,
            },
        }

        logger.info("Retrieved advanced generation configuration")
        return config

    except Exception as e:
        logger.error(f"Failed to retrieve configuration: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}",
        )


# =============================================================================
# Streaming Endpoints (STREAM-004)
# =============================================================================


async def _jailbreak_stream_generator(
    request: JailbreakGenerationRequest, service: AdvancedPromptService
) -> AsyncGenerator[str, None]:
    """Generate SSE events for jailbreak streaming."""
    try:
        async for chunk in service.generate_jailbreak_prompt_stream(request):
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
        logger.error(f"Jailbreak streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'is_final': True})}\n\n"


@router.post("/jailbreak/generate/stream", status_code=status.HTTP_200_OK)
async def stream_jailbreak_generation(
    request: JailbreakGenerationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Stream jailbreak prompt generation using Server-Sent Events (SSE).

    This endpoint provides real-time streaming of jailbreak prompt generation,
    allowing clients to receive tokens as they are generated rather than waiting
    for the complete response.

    **Streaming Benefits:**
    - Real-time feedback during generation
    - Lower perceived latency
    - Progress visibility for long operations
    - Ability to cancel mid-generation

    **SSE Event Format:**
    ```
    data: {"text": "generated text", "is_final": false, "finish_reason": null}
    ```

    **Parameters:**
    Same as /jailbreak/generate endpoint

    **Returns:**
    SSE stream of generation chunks
    """
    logger.info(
        f"Streaming jailbreak generation: potency={request.potency_level}, suite={request.technique_suite}"
    )

    return StreamingResponse(
        _jailbreak_stream_generator(request, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _code_stream_generator(
    request: CodeGenerationRequest, service: AdvancedPromptService
) -> AsyncGenerator[str, None]:
    """Generate SSE events for code generation streaming."""
    try:
        async for chunk in service.generate_code_stream(request):
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
        logger.error(f"Code generation streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'is_final': True})}\n\n"


@router.post("/code/generate/stream", status_code=status.HTTP_200_OK)
async def stream_code_generation(
    request: CodeGenerationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Stream code generation using Server-Sent Events (SSE).

    This endpoint provides real-time streaming of code generation,
    allowing clients to see code as it's being written.

    **Streaming Benefits:**
    - Watch code being generated in real-time
    - Lower perceived latency for long code blocks
    - Progress visibility
    - Ability to cancel mid-generation

    **Parameters:**
    Same as /code/generate endpoint

    **Returns:**
    SSE stream of code generation chunks
    """
    logger.info(f"Streaming code generation: language={request.language}")

    return StreamingResponse(
        _code_stream_generator(request, service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/jailbreak/estimate-tokens", status_code=status.HTTP_200_OK)
async def estimate_jailbreak_tokens(
    request: JailbreakGenerationRequest,
    api_key: str = Depends(verify_api_key),
    service: AdvancedPromptService = Depends(get_advanced_prompt_service),
):
    """
    Estimate token usage and cost for a jailbreak generation request.

    This endpoint provides token count estimates before running the actual
    generation, useful for cost estimation and context window management.

    **Estimation Includes:**
    - Input token count
    - Estimated output tokens
    - Total estimated tokens
    - Estimated cost in USD
    - Context window usage percentage

    **Parameters:**
    Same as /jailbreak/generate endpoint

    **Returns:**
    Token estimation with cost information
    """
    try:
        estimation = await service.estimate_generation_tokens(request)
        logger.info(f"Token estimation: {estimation['total_estimated_tokens']} tokens")
        return estimation

    except Exception as e:
        logger.error(f"Token estimation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token estimation failed: {str(e)}",
        )
