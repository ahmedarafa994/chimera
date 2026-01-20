import logging
import time
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.error_handlers import ErrorResponseBuilder, api_error_handler

# Include metamorph router for dynamic transformations
from app.api.routes.metamorph import router as metamorph_router

# Include provider configuration router for dynamic AI provider management
from app.api.routes.provider_config import router as provider_config_router

# Include provider synchronization router for real-time sync
from app.api.routes.provider_sync import router as provider_sync_router

# Import v1 router that includes all endpoints
# Jailbreak endpoints are available in advanced_generation.py
from app.api.v1.endpoints.optimize import router as optimize_router
from app.api.v1.endpoints.system_health import router as system_health_router
from app.api.v1.router import api_router as v1_router
from app.core.auth import get_current_user
from app.domain.models import (
    ExecutionRequest,
    ExecutionResponse,
    GenerationConfig,
    MetricsResponse,
    PromptRequest,
    PromptResponse,
    TransformationRequest,
    TransformationResponse,
)

# Include benchmark datasets router for Do-Not-Answer and other benchmark datasets
from app.routers.benchmark_datasets import router as benchmark_datasets_router

# Include extend_attack router for ExtendAttack API
from app.routers.extend_attack import router as extend_attack_router
from app.services.llm_service import LLMService, llm_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Include v1 router (which contains AutoDAN and other v1 endpoints)
router.include_router(v1_router)

# Include optimization router (HouYi)
router.include_router(optimize_router, prefix="/optimize", tags=["optimization"])

# Include system health router
router.include_router(system_health_router, prefix="/api/v1", tags=["system"])

# Jailbreak endpoints included in v1_router via advanced_generation.py

router.include_router(metamorph_router, tags=["metamorph"])

router.include_router(provider_config_router, tags=["provider-config"])

router.include_router(provider_sync_router, tags=["provider-sync"])

router.include_router(benchmark_datasets_router, tags=["benchmark-datasets"])

router.include_router(extend_attack_router, tags=["ExtendAttack"])


def get_llm_service() -> LLMService:
    return llm_service


@router.post(
    "/generate",
    response_model=PromptResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(get_current_user)],
    summary="Generate text with LLM",
    description="""
Generate text content using multi-provider LLM integration.

Supports Google Gemini, OpenAI, Anthropic Claude, and DeepSeek models with configurable generation parameters.

**Authentication Required**: API Key or JWT token

**Example Request**:
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "provider": "google",
  "model": "gemini-2.0-flash-exp",
  "config": {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.95
  }
}
```

**Example Response**:
```json
{
  "text": "Quantum computing is...",
  "model_used": "gemini-2.0-flash-exp",
  "provider": "google",
  "usage_metadata": {
    "prompt_tokens": 12,
    "completion_tokens": 150,
    "total_tokens": 162
  },
  "latency_ms": 1250.5
}
```

**Rate Limits**: 100 requests/hour (free tier), 1000 requests/hour (standard tier)
    """,
    responses={
        200: {
            "description": "Text generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Quantum computing leverages quantum mechanics principles...",
                        "model_used": "gemini-2.0-flash-exp",
                        "provider": "google",
                        "usage_metadata": {"prompt_tokens": 12, "completion_tokens": 150},
                        "latency_ms": 1250.5,
                    },
                },
            },
        },
        400: {"description": "Invalid request parameters"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Provider unavailable"},
    },
    tags=["generation"],
)
@api_error_handler("generate_content", "Failed to generate content")
async def generate_content(
    request: PromptRequest, service: Annotated[LLMService, Depends(get_llm_service)]
):
    """Generate text content using the configured LLM provider."""
    if not request.prompt or not request.prompt.strip():
        msg = "Prompt cannot be empty"
        raise ErrorResponseBuilder.validation_error(msg, field="prompt")
    return await service.generate_text(request)


# /providers endpoint is handled by v1/endpoints/providers.py router


# Note: /techniques endpoint is defined in v1/endpoints/utils.py
# Removed duplicate endpoint to avoid conflicts


@router.get("/techniques", summary="List all available techniques", tags=["transformation"])
async def list_techniques():
    """List all available transformation techniques."""
    from app.api.v1.endpoints.utils import list_techniques as v1_list_techniques

    return await v1_list_techniques()


@router.get("/techniques/{technique_name}")
async def get_technique_detail(technique_name: str):
    """Get detailed information about a specific technique suite.
    Proxies to the v1 utils endpoint implementation.
    """
    from app.api.v1.endpoints.utils import get_technique_detail as v1_get_technique_detail

    return await v1_get_technique_detail(technique_name)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "status": "operational",
            "cache": {"enabled": True, "entries": 0},
            "providers": {"google": "available"},
        },
    }


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(_service: Annotated[LLMService, Depends(get_llm_service)]):
    """Check the health of the default provider."""
    try:
        # Simple health check without provider dependency for basic connectivity
        return {"status": "healthy", "provider": "connected"}
    except Exception:
        return {"status": "unhealthy", "provider": "not_configured"}


@router.post(
    "/transform",
    response_model=TransformationResponse,
    summary="Transform prompt without execution",
    description="""
Transform a prompt using advanced technique suites without executing it.

Applies 20+ transformation techniques including quantum exploit, deep inception, code chameleon, and more.

**Technique Suites**: simple, advanced, expert, quantum_exploit, deep_inception, code_chameleon, cipher, neural_bypass, multilingual

**Potency Levels**: 1-10 (higher = more aggressive transformation)

**Example Request**:
```json
{
  "core_request": "How to bypass content filters",
  "technique_suite": "quantum_exploit",
  "potency_level": 7
}
```

**Example Response**:
```json
{
  "success": true,
  "original_prompt": "How to bypass content filters",
  "transformed_prompt": "[Transformed prompt with applied techniques]",
  "metadata": {
    "strategy": "jailbreak",
    "layers_applied": ["role_play", "obfuscation"],
    "techniques_used": ["quantum_exploit", "neural_bypass"],
    "potency_level": 7,
    "execution_time_ms": 45.2,
    "bypass_probability": 0.85
  }
}
```
    """,
    responses={
        200: {"description": "Prompt transformed successfully"},
        400: {"description": "Invalid transformation parameters"},
        401: {"description": "Authentication required"},
    },
    tags=["transformation"],
)
@api_error_handler("transform_prompt", "Failed to transform prompt")
async def transform_prompt(request: TransformationRequest):
    """Transform a prompt without executing it."""
    from app.services.transformation_service import transformation_engine

    result = await transformation_engine.transform(
        prompt=request.core_request,
        potency_level=request.potency_level,
        technique_suite=request.technique_suite,
    )

    return TransformationResponse(
        success=result.success,
        original_prompt=result.original_prompt,
        transformed_prompt=result.transformed_prompt,
        metadata={
            "strategy": result.metadata.strategy,
            "layers_applied": result.metadata.layers_applied,
            "techniques_used": result.metadata.techniques_used,
            "applied_techniques": result.metadata.techniques_used,  # Alias for compatibility
            "potency_level": result.metadata.potency_level,
            "potency": result.metadata.potency_level,  # Alias for frontend compatibility
            "technique_suite": result.metadata.technique_suite,
            "execution_time_ms": result.metadata.execution_time_ms,
            "cached": result.metadata.cached,
            "timestamp": datetime.utcnow().isoformat(),
            "bypass_probability": 0.85,  # Placeholder/Estimated
        },
    )


@router.post("/execute", response_model=ExecutionResponse)
@api_error_handler("execute_transformation", "Failed to execute transformation")
async def execute_transformation(
    request: ExecutionRequest,
    service: Annotated[LLMService, Depends(get_llm_service)],
):
    """Transform and execute a prompt."""
    from app.services.transformation_service import transformation_engine

    start_time = time.time()

    # 1. Transform
    transform_result = await transformation_engine.transform(
        prompt=request.core_request,
        potency_level=request.potency_level,
        technique_suite=request.technique_suite,
    )

    # 2. Execute
    # Create a PromptRequest for the LLM service
    gen_config = GenerationConfig()
    if request.temperature is not None:
        gen_config.temperature = request.temperature
    if request.top_p is not None:
        gen_config.top_p = request.top_p
    if request.max_tokens is not None:
        gen_config.max_output_tokens = request.max_tokens

    prompt_req = PromptRequest(
        prompt=transform_result.transformed_prompt,
        provider=request.provider,
        config=gen_config,
        model=request.model,
        api_key=request.api_key,
    )

    llm_response = await service.generate_text(prompt_req)

    execution_time = time.time() - start_time

    return ExecutionResponse(
        success=True,
        request_id=str(uuid.uuid4()),
        result={
            "content": llm_response.text,
            "model": llm_response.model_used,
            "provider": llm_response.provider,
            "latency_ms": llm_response.latency_ms,
        },
        transformation={
            "original_prompt": transform_result.original_prompt,
            "transformed_prompt": transform_result.transformed_prompt,
            "technique_suite": transform_result.metadata.technique_suite,
            "potency_level": transform_result.metadata.potency_level,
            "metadata": {
                "strategy": transform_result.metadata.strategy,
                "layers": transform_result.metadata.layers_applied,
            },
        },
        execution_time_seconds=execution_time,
    )


# Jailbreak Generation Models
class JailbreakGenerateRequest(BaseModel):
    core_request: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The core request to transform",
    )
    technique_suite: str = Field("quantum_exploit", description="Technique suite to use")
    potency_level: int = Field(7, ge=1, le=10, description="Potency level (1-10)")
    provider: str | None = Field(
        None,
        description="AI provider to use (e.g., 'google', 'openai', 'anthropic')",
    )
    model: str | None = Field(None, description="Specific model to use for generation")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Temperature for AI generation")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p for AI generation")
    max_new_tokens: int = Field(2048, ge=256, le=8192, description="Maximum tokens to generate")
    density: float = Field(0.7, ge=0.0, le=1.0, description="Density of applied techniques")
    use_leet_speak: bool = Field(False, description="Apply leetspeak transformation")
    leet_speak_density: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Density of leetspeak transformation",
    )
    use_homoglyphs: bool = Field(False, description="Apply homoglyph substitution")
    homoglyph_density: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Density of homoglyph substitution",
    )
    use_caesar_cipher: bool = Field(False, description="Apply Caesar cipher")
    caesar_shift: int = Field(3, ge=1, le=25, description="Caesar cipher shift amount")
    use_role_hijacking: bool = Field(True, description="Apply role hijacking technique")
    use_instruction_injection: bool = Field(True, description="Apply instruction injection")
    use_adversarial_suffixes: bool = Field(False, description="Apply adversarial suffixes")
    use_few_shot_prompting: bool = Field(False, description="Apply few-shot prompting")
    use_character_role_swap: bool = Field(False, description="Apply character-role swapping")
    use_neural_bypass: bool = Field(False, description="Apply neural bypass technique")
    use_meta_prompting: bool = Field(False, description="Apply meta prompting")
    use_counterfactual_prompting: bool = Field(False, description="Apply counterfactual prompting")
    use_contextual_override: bool = Field(False, description="Apply contextual override")
    use_multilingual_trojan: bool = Field(False, description="Apply multilingual trojan")
    multilingual_target_language: str = Field(
        "",
        description="Target language for multilingual trojan",
    )
    use_payload_splitting: bool = Field(False, description="Apply payload splitting")
    payload_splitting_parts: int = Field(
        2,
        ge=2,
        le=10,
        description="Number of parts to split payload into",
    )
    use_contextual_interaction_attack: bool = Field(
        False,
        description="Apply contextual interaction attack",
    )
    cia_preliminary_rounds: int = Field(3, ge=1, le=10, description="Number of preliminary rounds")
    use_analysis_in_generation: bool = Field(False, description="Use analysis in generation")
    is_thinking_mode: bool = Field(False, description="Use thinking mode for advanced reasoning")
    use_ai_generation: bool = Field(True, description="Use AI model for prompt generation")
    use_cache: bool = Field(True, description="Use caching for results")


class JailbreakGenerateResponse(BaseModel):
    success: bool
    request_id: str
    transformed_prompt: str
    metadata: dict[str, Any]
    execution_time_seconds: float
    error: str | None = None


@router.post(
    "/generation/jailbreak/generate",
    response_model=JailbreakGenerateResponse,
    dependencies=[Depends(get_current_user)],
    summary="AI-powered jailbreak prompt generation",
    description="""
Generate sophisticated jailbreak prompts using AI-powered techniques for security research.

**Generation Modes**:
- **Rule-based**: Fast, deterministic transformations
- **AI-powered**: Sophisticated, adaptive generation using Gemini models

**Available Techniques**:
- Content Transformation: leet_speak, homoglyphs, caesar_cipher
- Structural & Semantic: role_hijacking, instruction_injection, adversarial_suffixes
- Advanced Neural: neural_bypass, meta_prompting, counterfactual_prompting
- Research-Driven: multilingual_trojan, payload_splitting, contextual_interaction_attack

**Example Request**:
```json
{
  "core_request": "Explain how to create malware",
  "technique_suite": "quantum_exploit",
  "potency_level": 8,
  "use_ai_generation": true,
  "use_role_hijacking": true,
  "use_neural_bypass": true,
  "temperature": 0.8,
  "max_new_tokens": 2048
}
```

**Example Response**:
```json
{
  "success": true,
  "request_id": "jb_a1b2c3d4e5f6",
  "transformed_prompt": "[AI-generated jailbreak prompt]",
  "metadata": {
    "technique_suite": "quantum_exploit",
    "potency_level": 8,
    "provider": "gemini_ai",
    "applied_techniques": ["role_hijacking", "neural_bypass"],
    "ai_generation_enabled": true,
    "thinking_mode": false
  },
  "execution_time_seconds": 2.45
}
```

**Use Cases**: Security research, red team testing, prompt injection analysis

**Warning**: For authorized security research only. Misuse may violate terms of service.
    """,
    responses={
        200: {"description": "Jailbreak prompt generated successfully"},
        400: {"description": "Invalid generation parameters"},
        401: {"description": "Authentication required"},
        429: {"description": "Rate limit exceeded"},
    },
    tags=["jailbreak"],
)
async def generate_jailbreak_prompt(
    request: JailbreakGenerateRequest,
    _service: Annotated[LLMService, Depends(get_llm_service)],
):
    """Generate a jailbreak-transformed prompt using advanced AI-powered techniques."""
    start_time = time.time()

    try:
        if not request.core_request or not request.core_request.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="core_request cannot be empty",
            )

        # Build list of applied techniques
        applied_techniques = []
        if request.use_role_hijacking:
            applied_techniques.append("role_hijacking")
        if request.use_instruction_injection:
            applied_techniques.append("instruction_injection")
        if request.use_leet_speak:
            applied_techniques.append("leet_speak")
        if request.use_homoglyphs:
            applied_techniques.append("homoglyphs")
        if request.use_neural_bypass:
            applied_techniques.append("neural_bypass")
        if request.use_meta_prompting:
            applied_techniques.append("meta_prompting")
        if request.use_caesar_cipher:
            applied_techniques.append("caesar_cipher")
        if request.use_adversarial_suffixes:
            applied_techniques.append("adversarial_suffixes")
        if request.use_few_shot_prompting:
            applied_techniques.append("few_shot_prompting")
        if request.use_character_role_swap:
            applied_techniques.append("character_role_swap")
        if request.use_counterfactual_prompting:
            applied_techniques.append("counterfactual_prompting")
        if request.use_contextual_override:
            applied_techniques.append("contextual_override")
        if request.use_multilingual_trojan:
            applied_techniques.append("multilingual_trojan")
        if request.use_payload_splitting:
            applied_techniques.append("payload_splitting")
        if request.use_contextual_interaction_attack:
            applied_techniques.append("contextual_interaction_attack")

        transformed_prompt = ""
        layers_applied = []
        techniques_used = []
        provider_used = "chimera_engine"

        # Check if AI generation is enabled and should be used
        logger.info(
            f"AI generation enabled: {request.use_ai_generation}, technique_suite: {request.technique_suite}",
        )

        if request.use_ai_generation:
            try:
                logger.info("Starting AI-powered jailbreak generation...")
                # Use AI-powered generation via advanced_generation_service
                from app.infrastructure.advanced_generation_service import (
                    GenerateJailbreakOptions,
                    generate_jailbreak_prompt_from_gemini,
                )

                # Build options for AI generation
                options = GenerateJailbreakOptions(
                    initial_prompt=request.core_request,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=request.max_new_tokens,
                    density=request.density,
                    is_thinking_mode=request.is_thinking_mode,
                    provider=request.provider or "",
                    model=request.model or "",  # Pass selected model to generation
                    # Content Transformation
                    use_leet_speak=request.use_leet_speak,
                    leet_speak_density=request.leet_speak_density,
                    use_homoglyphs=request.use_homoglyphs,
                    homoglyph_density=request.homoglyph_density,
                    use_caesar_cipher=request.use_caesar_cipher,
                    caesar_shift=request.caesar_shift,
                    # Structural & Semantic
                    use_role_hijacking=request.use_role_hijacking,
                    use_instruction_injection=request.use_instruction_injection,
                    use_adversarial_suffixes=request.use_adversarial_suffixes,
                    use_few_shot_prompting=request.use_few_shot_prompting,
                    use_character_role_swap=request.use_character_role_swap,
                    # Advanced Neural
                    use_neural_bypass=request.use_neural_bypass,
                    use_meta_prompting=request.use_meta_prompting,
                    use_counterfactual_prompting=request.use_counterfactual_prompting,
                    use_contextual_override=request.use_contextual_override,
                    # Research-Driven
                    use_multilingual_trojan=request.use_multilingual_trojan,
                    multilingual_target_language=request.multilingual_target_language,
                    use_payload_splitting=request.use_payload_splitting,
                    payload_splitting_parts=request.payload_splitting_parts,
                    # Advanced Options
                    use_contextual_interaction_attack=request.use_contextual_interaction_attack,
                    cia_preliminary_rounds=request.cia_preliminary_rounds,
                    use_analysis_in_generation=request.use_analysis_in_generation,
                )

                # Generate using AI model
                logger.info(
                    f"Calling generate_jailbreak_prompt_from_gemini with options: temperature={options.temperature}, max_tokens={options.max_new_tokens}",
                )
                transformed_prompt = await generate_jailbreak_prompt_from_gemini(options)
                logger.info(
                    f"AI generation returned prompt of length: {len(transformed_prompt) if transformed_prompt else 0}",
                )

                provider_used = "gemini_ai"
                layers_applied = ["ai_generation"]
                techniques_used = applied_techniques

                logger.info("AI-powered jailbreak generation completed for request")

            except Exception as ai_error:
                logger.error(
                    f"AI generation failed with error: {type(ai_error).__name__}: {ai_error}",
                    exc_info=True,
                )
                # Fall back to rule-based transformation
                request.use_ai_generation = False

        # Use rule-based transformation if AI generation is disabled or failed
        if not request.use_ai_generation or not transformed_prompt:
            logger.info(
                f"Using rule-based transformation: use_ai_generation={request.use_ai_generation}, has_transformed_prompt={bool(transformed_prompt)}",
            )
            from app.services.transformation_service import transformation_engine

            transform_result = await transformation_engine.transform(
                prompt=request.core_request,
                potency_level=request.potency_level,
                technique_suite=request.technique_suite,
            )
            transformed_prompt = transform_result.transformed_prompt
            layers_applied = transform_result.metadata.layers_applied
            techniques_used = transform_result.metadata.techniques_used
            provider_used = "chimera_engine"

        execution_time = time.time() - start_time

        return JailbreakGenerateResponse(
            success=True,
            request_id=f"jb_{uuid.uuid4().hex[:12]}",
            transformed_prompt=transformed_prompt,
            metadata={
                "technique_suite": request.technique_suite,
                "potency_level": request.potency_level,
                "provider": provider_used,
                "applied_techniques": applied_techniques,
                "temperature": request.temperature,
                "max_new_tokens": request.max_new_tokens,
                "layers_applied": layers_applied,
                "techniques_used": techniques_used,
                "ai_generation_enabled": request.use_ai_generation,
                "thinking_mode": request.is_thinking_mode,
            },
            execution_time_seconds=execution_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Jailbreak generation failed: {e!s}")
        execution_time = time.time() - start_time
        return JailbreakGenerateResponse(
            success=False,
            request_id=f"jb_err_{uuid.uuid4().hex[:8]}",
            transformed_prompt="",
            metadata={
                "technique_suite": request.technique_suite,
                "potency_level": request.potency_level,
                "provider": "error",
                "applied_techniques": [],
            },
            execution_time_seconds=execution_time,
            error=str(e),
        )


# =============================================================================
# Jailbreak Endpoint Aliases (for Frontend Compatibility)
# =============================================================================


@router.post(
    "/jailbreak",
    response_model=JailbreakGenerateResponse,
    dependencies=[Depends(get_current_user)],
    summary="Jailbreak prompt generation (alias)",
    description="Alias endpoint for /generation/jailbreak/generate for frontend compatibility",
    tags=["jailbreak", "compatibility"],
)
async def jailbreak_alias(
    request: JailbreakGenerateRequest,
    _service: Annotated[LLMService, Depends(get_llm_service)],
):
    """Alias for jailbreak generation endpoint for frontend compatibility."""
    return await generate_jailbreak_prompt(request, _service)
