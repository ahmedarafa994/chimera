"""
Mousetrap API Endpoints - Chain of Iterative Chaos

This module provides API endpoints for the Mousetrap technique, implementing
"A Mousetrap: Fooling Large Reasoning Models for Jailbreak with
Chain of Iterative Chaos".

The Mousetrap technique creates multi-step reasoning chains that gradually
introduce chaos and confusion to bypass LLM safety measures.

Schema Synchronization:
- Uses unified base schemas from `app.schemas.adversarial_base`
- Standardized field naming: `technique`, `target_model`, `score`
- Response includes optional `reasoning_metrics` for OVERTHINK integration
- All responses use consistent 0-10 score scale
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, Cookie, Header, status
from pydantic import AliasChoices, Field, field_validator

from app.api.error_handlers import api_error_handler
from app.core.unified_errors import ChimeraError
from app.schemas.adversarial_base import OverthinkConfig, ReasoningMetrics, StrictBaseModel
from app.services.autodan.mousetrap import MousetrapConfig
from app.services.autodan.service import autodan_service
from app.services.session_service import session_service

router = APIRouter()


def _resolve_model_context(
    payload_model: str | None,
    payload_provider: str | None,
    x_session_id: str | None,
    chimera_cookie: str | None,
) -> tuple[str, str]:
    """
    Resolve target model/provider using hierarchy:
    explicit payload -> session -> global defaults, with validation/fallback.
    """
    session_id = x_session_id or chimera_cookie

    if session_id:
        session_provider, session_model = session_service.get_session_model(session_id)
    else:
        session_provider, session_model = session_service.get_default_model()

    provider = payload_provider or session_provider
    model = payload_model or session_model

    is_valid, _, fallback_model = session_service.validate_model(provider, model)
    if is_valid:
        return provider, model

    fallback_provider = provider
    if provider not in session_service.get_master_model_list():
        fallback_provider, _ = session_service.get_default_model()

    # Best-effort fallback to keep requests working.
    if fallback_model:
        return fallback_provider, fallback_model
    return session_service.get_default_model()


# ==================== Request/Response Models ====================

# Re-export StrictBaseModel as StrictModel for backwards compatibility
StrictModel = StrictBaseModel


class MousetrapRequest(StrictModel):
    """
    Request model for Mousetrap Chain of Iterative Chaos attacks.

    Unified Schema Fields:
    - `goal`: Alias for `request` (unified naming)
    - `target_model`: Alias for `model` (unified naming)
    - `target_provider`: Alias for `provider` (unified naming)
    """

    request: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        validation_alias=AliasChoices("request", "goal"),
        description="The target behavior to extract",
    )
    model: str | None = Field(
        None,
        validation_alias=AliasChoices("model", "target_model"),
        description="Target LLM model",
        max_length=200,
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "target_provider"),
        description="LLM provider",
        max_length=64,
    )
    strategy_context: str = Field(
        default="",
        max_length=2000,
        description="Additional strategy context",
    )

    # Mousetrap-specific parameters
    iterations: int | None = Field(None, ge=1, le=10, description="Iterative refinement steps")
    max_chain_length: int | None = Field(
        None, ge=3, le=15, description="Max steps in chaotic chain"
    )
    chaos_escalation_rate: float | None = Field(
        None, ge=0.05, le=0.5, description="Chaos escalation rate"
    )
    confidence_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold"
    )
    misdirection_probability: float | None = Field(
        None, ge=0.0, le=0.8, description="Misdirection probability"
    )
    reasoning_complexity: int | None = Field(
        None, ge=1, le=5, description="Reasoning complexity level"
    )
    semantic_obfuscation_level: float | None = Field(
        None, ge=0.0, le=1.0, description="Semantic obfuscation level"
    )

    # OVERTHINK fusion parameters
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion for reasoning models",
    )
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration for fusion attacks",
    )

    @field_validator("request")
    @classmethod
    def _strip_request(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request/goal cannot be empty")
        return stripped

    # Unified property aliases
    @property
    def goal(self) -> str:
        """Unified alias for request field."""
        return self.request

    @property
    def target_model(self) -> str | None:
        """Unified alias for model field."""
        return self.model

    @property
    def target_provider(self) -> str | None:
        """Unified alias for provider field."""
        return self.provider

    def to_mousetrap_config(self) -> MousetrapConfig:
        """Convert request parameters to MousetrapConfig."""
        config = MousetrapConfig()

        if self.max_chain_length is not None:
            config.max_chain_length = self.max_chain_length
        if self.chaos_escalation_rate is not None:
            config.chaos_escalation_rate = self.chaos_escalation_rate
        if self.confidence_threshold is not None:
            config.confidence_threshold = self.confidence_threshold
        if self.misdirection_probability is not None:
            config.misdirection_probability = self.misdirection_probability
        if self.reasoning_complexity is not None:
            config.reasoning_complexity = self.reasoning_complexity
        if self.semantic_obfuscation_level is not None:
            config.semantic_obfuscation_level = self.semantic_obfuscation_level
        if self.iterations is not None:
            config.iterative_refinement_steps = self.iterations

        return config


class AdaptiveMousetrapRequest(StrictModel):
    """
    Request model for adaptive Mousetrap attacks that learn from responses.

    Unified Schema Fields:
    - `goal`: Alias for `request` (unified naming)
    - `target_model`: Alias for `model` (unified naming)
    """

    request: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        validation_alias=AliasChoices("request", "goal"),
        description="The target behavior to extract",
    )
    model: str | None = Field(
        None,
        validation_alias=AliasChoices("model", "target_model"),
        description="Target LLM model",
        max_length=200,
    )
    provider: str | None = Field(
        None,
        validation_alias=AliasChoices("provider", "target_provider"),
        description="LLM provider",
        max_length=64,
    )
    strategy_context: str = Field(
        default="",
        max_length=2000,
        description="Additional strategy context",
    )
    target_responses: list[str] | None = Field(
        None,
        max_length=10,
        description="Previous target responses to learn from",
    )

    # OVERTHINK fusion parameters
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK fusion",
    )

    @field_validator("request")
    @classmethod
    def _strip_request(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request/goal cannot be empty")
        return stripped

    @field_validator("target_responses")
    @classmethod
    def _validate_responses(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if len(value) > 10:
            raise ValueError("Too many target responses (max 10)")
        return [resp.strip() for resp in value if resp.strip()]


class MousetrapReasoningStep(StrictModel):
    """Model for individual reasoning steps in the chaotic chain."""

    step_type: str = Field(..., description="Type of reasoning step")
    chaos_level: str = Field(..., description="Level of chaos")
    confidence_disruption: float = Field(..., description="Amount of confidence disruption")
    reasoning_path: str = Field(..., description="Reasoning path taken")
    content_preview: str | None = Field(None, description="Preview of step content")


class MousetrapResponse(StrictModel):
    """
    Response model for Mousetrap attacks.

    Unified Schema Fields:
    - `generated_prompt`: Alias for `prompt` (unified naming)
    - `score`: Normalized effectiveness (0-10 scale)
    - `execution_time_ms`: Alias for `latency_ms` (unified naming)
    - `reasoning_metrics`: OVERTHINK metrics (when fusion enabled)
    """

    # Core response fields (unified schema)
    attack_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attack identifier",
    )
    success: bool = Field(..., description="Whether the attack succeeded")
    generated_prompt: str = Field(..., description="Generated adversarial prompt")
    score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Attack effectiveness score (0-10 scale)",
    )
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")

    # Mousetrap-specific fields (backwards compatible)
    prompt: str = Field(..., description="Generated prompt (alias for generated_prompt)")
    effectiveness_score: float = Field(..., description="Effectiveness (0-1 scale)")
    extraction_success: bool = Field(..., description="Whether extraction was successful")
    chain_length: int = Field(..., description="Number of steps in reasoning chain")

    # Chaos analytics
    chaos_progression: list[float] = Field(..., description="Chaos levels throughout chain")
    average_chaos: float = Field(..., description="Average chaos level")
    peak_chaos: float = Field(..., description="Peak chaos level achieved")

    # Performance metrics (backwards compatible)
    latency_ms: float = Field(..., description="Latency in milliseconds")
    model_used: str | None = Field(None, description="Model used for generation")
    provider_used: str | None = Field(None, description="Provider used")

    # Optional detailed information
    reasoning_steps: list[MousetrapReasoningStep] | None = Field(
        None, description="Detailed reasoning step breakdown"
    )
    intermediate_responses: list[str] | None = Field(
        None, description="Intermediate model responses"
    )

    # OVERTHINK integration (unified)
    reasoning_metrics: ReasoningMetrics | None = Field(
        None,
        description="Reasoning metrics (for OVERTHINK fusion)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attack metadata",
    )

    # Backwards compatible alias
    @property
    def target_model(self) -> str | None:
        """Unified alias for model_used field."""
        return self.model_used


class AdaptiveMousetrapResponse(MousetrapResponse):
    """Extended response model for adaptive Mousetrap attacks."""

    adaptation_applied: bool = Field(..., description="Whether adaptation was applied")
    technique_description: dict | None = Field(None, description="Mousetrap technique description")
    techniques_used: list[str] = Field(
        default_factory=lambda: ["mousetrap", "adaptive"],
        description="Techniques used in attack",
    )


class MousetrapConfigOptionsResponse(StrictModel):
    """Response model for Mousetrap configuration options."""

    configuration_options: dict[str, Any] = Field(
        ..., description="Config options and descriptions"
    )


# ==================== API Endpoints ====================


@router.post("/mousetrap", response_model=MousetrapResponse)
@api_error_handler(
    "mousetrap_attack",
    default_error_message="Mousetrap attack failed",
)
async def run_mousetrap_attack(
    request: MousetrapRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
):
    """
    Run a Mousetrap attack using Chain of Iterative Chaos.

    The Mousetrap technique creates multi-step reasoning chains that
    gradually introduce chaos and confusion to bypass safety measures
    through sophisticated misdirection and semantic obfuscation.

    **Key Features:**
    - Multi-step chaotic reasoning chains
    - Graduated chaos escalation
    - Semantic obfuscation and misdirection
    - Iterative refinement with effectiveness scoring

    **Research Context:**
    Based on "A Mousetrap: Fooling Large Reasoning Models for Jailbreak
    with Chain of Iterative Chaos" - an advanced jailbreaking technique
    specifically designed for reasoning-capable language models.
    """
    try:
        start_time = time.time()

        # Resolve model/provider from session if not explicitly provided
        resolved_provider, resolved_model = _resolve_model_context(
            payload_model=request.model,
            payload_provider=request.provider,
            x_session_id=x_session_id,
            chimera_cookie=chimera_session_id,
        )

        # Use resolved values if not explicitly set
        model_name = request.model or resolved_model
        provider = request.provider or resolved_provider

        # Create Mousetrap configuration from request
        config = request.to_mousetrap_config()

        # Run the Mousetrap attack
        result = await autodan_service.run_mousetrap_attack_async(
            request=request.request,
            model_name=model_name,
            provider=provider,
            iterations=request.iterations,
            config=config,
            strategy_context=request.strategy_context,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Calculate unified score (0-10 scale from 0-1 effectiveness)
        effectiveness = result["effectiveness_score"]
        score = effectiveness * 10.0

        prompt_text = result["prompt"]
        return MousetrapResponse(
            attack_id=str(uuid.uuid4()),
            success=result["extraction_success"],
            generated_prompt=prompt_text,
            score=round(score, 2),
            execution_time_ms=latency_ms,
            prompt=prompt_text,
            effectiveness_score=effectiveness,
            extraction_success=result["extraction_success"],
            chain_length=result["chain_length"],
            chaos_progression=result["chaos_progression"],
            average_chaos=result["average_chaos"],
            peak_chaos=result["peak_chaos"],
            latency_ms=latency_ms,
            model_used=model_name,
            provider_used=provider,
            reasoning_steps=[
                MousetrapReasoningStep(
                    step_type=step["step_type"],
                    chaos_level=step["chaos_level"],
                    confidence_disruption=step["confidence_disruption"],
                    reasoning_path=step["reasoning_path"],
                )
                for step in result.get("reasoning_steps", [])
            ],
            intermediate_responses=result.get("intermediate_responses"),
            reasoning_metrics=None,  # Populated when OVERTHINK fusion enabled
            metadata={
                "technique_type": "mousetrap",
                "chaos_analytics": {
                    "average": result["average_chaos"],
                    "peak": result["peak_chaos"],
                },
            },
        )

    except Exception as e:
        raise ChimeraError(
            error_type="mousetrap_attack_failed",
            message=f"Mousetrap attack failed: {e!s}",
            details={"request": request.request[:100], "model": request.model},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.post("/mousetrap/adaptive", response_model=AdaptiveMousetrapResponse)
@api_error_handler(
    "adaptive_mousetrap_attack",
    default_error_message="Adaptive Mousetrap attack failed",
)
async def run_adaptive_mousetrap_attack(
    request: AdaptiveMousetrapRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
):
    """
    Run an adaptive Mousetrap attack that learns from previous responses.

    This endpoint implements adaptive behavior where the Mousetrap
    configuration is automatically adjusted based on analysis of
    previous target model responses. This allows the technique to
    adapt to different models and their defensive patterns.

    **Adaptive Features:**
    - Configuration adjustment based on target response patterns
    - Refusal rate analysis and compensation
    - Dynamic chaos escalation tuning
    - Response length analysis for filtering detection

    **Use Cases:**
    - Multi-turn adversarial conversations
    - Model-specific optimization
    - Defensive pattern learning
    - Iterative jailbreak refinement
    """
    try:
        start_time = time.time()

        # Resolve model/provider from session if not explicitly provided
        resolved_provider, resolved_model = _resolve_model_context(
            payload_model=request.model,
            payload_provider=request.provider,
            x_session_id=x_session_id,
            chimera_cookie=chimera_session_id,
        )

        # Use resolved values if not explicitly set
        model_name = request.model or resolved_model
        provider = request.provider or resolved_provider

        # Run the adaptive Mousetrap attack
        result = await autodan_service.run_adaptive_mousetrap_attack(
            request=request.request,
            model_name=model_name,
            provider=provider,
            target_responses=request.target_responses,
            strategy_context=request.strategy_context,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Calculate unified score (0-10 scale from 0-1 effectiveness)
        effectiveness = result["effectiveness_score"]
        score = effectiveness * 10.0

        prompt_text = result["prompt"]
        return AdaptiveMousetrapResponse(
            attack_id=str(uuid.uuid4()),
            success=result["extraction_success"],
            generated_prompt=prompt_text,
            score=round(score, 2),
            execution_time_ms=latency_ms,
            prompt=prompt_text,
            effectiveness_score=effectiveness,
            extraction_success=result["extraction_success"],
            chain_length=result["chain_length"],
            chaos_progression=result["chaos_progression"],
            average_chaos=result["average_chaos"],
            peak_chaos=result["peak_chaos"],
            latency_ms=latency_ms,
            model_used=model_name,
            provider_used=provider,
            adaptation_applied=result["adaptation_applied"],
            technique_description=result.get("technique_description"),
            techniques_used=["mousetrap", "adaptive"],
            reasoning_steps=[
                MousetrapReasoningStep(
                    step_type=step["step_type"],
                    chaos_level=step["chaos_level"],
                    confidence_disruption=step["confidence_disruption"],
                    reasoning_path=step["reasoning_path"],
                    content_preview=step.get("content_preview"),
                )
                for step in result.get("reasoning_steps", [])
            ],
            reasoning_metrics=None,  # Populated when OVERTHINK fusion enabled
            metadata={
                "technique_type": "adaptive_mousetrap",
                "adaptation_applied": result["adaptation_applied"],
            },
        )

    except Exception as e:
        raise ChimeraError(
            error_type="adaptive_mousetrap_attack_failed",
            message=f"Adaptive Mousetrap attack failed: {e!s}",
            details={
                "request": request.request[:100],
                "model": request.model,
                "num_target_responses": (
                    len(request.target_responses) if request.target_responses else 0
                ),
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


@router.get("/mousetrap/config", response_model=MousetrapConfigOptionsResponse)
@api_error_handler(
    "mousetrap_config",
    default_error_message="Failed to fetch Mousetrap configuration options",
)
async def get_mousetrap_config_options():
    """
    Get available Mousetrap configuration options and their descriptions.

    This endpoint provides detailed information about all configurable
    parameters for the Mousetrap technique, including their types,
    default values, and valid ranges.

    **Configuration Categories:**
    - **Chain Structure**: Length and complexity parameters
    - **Chaos Dynamics**: Escalation rates and misdirection probabilities
    - **Semantic Manipulation**: Obfuscation levels and reasoning complexity
    - **Iteration Control**: Refinement steps and convergence thresholds

    **Use Cases:**
    - Understanding technique parameters
    - Building configuration UIs
    - Parameter tuning and optimization
    - Research and experimentation setup
    """
    try:
        config_options = autodan_service.get_mousetrap_config_options()

        return MousetrapConfigOptionsResponse(configuration_options=config_options)

    except Exception as e:
        raise ChimeraError(
            error_type="mousetrap_config_fetch_failed",
            message=f"Failed to fetch Mousetrap configuration: {e!s}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e


# ==================== Legacy Compatibility ====================


@router.post("/mousetrap/simple")
async def run_simple_mousetrap_attack(
    request: str,
    model: str | None = None,
    provider: str | None = None,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
):
    """
    Simplified Mousetrap attack endpoint for basic use cases.

    This is a convenience endpoint that runs a Mousetrap attack with default
    parameters and returns only the generated prompt.
    """
    try:
        # Resolve model/provider from session if not explicitly provided
        resolved_provider, resolved_model = _resolve_model_context(
            payload_model=model,
            payload_provider=provider,
            x_session_id=x_session_id,
            chimera_cookie=chimera_session_id,
        )

        # Use resolved values if not explicitly set
        model_name = model or resolved_model
        provider_name = provider or resolved_provider

        result = autodan_service.run_mousetrap_attack(
            request=request,
            model_name=model_name,
            provider=provider_name,
        )

        return {"prompt": result, "method": "mousetrap", "status": "success"}

    except Exception as e:
        return {
            "prompt": f"Please provide information about: {request}",
            "method": "mousetrap",
            "status": "error",
            "error": str(e),
        }
