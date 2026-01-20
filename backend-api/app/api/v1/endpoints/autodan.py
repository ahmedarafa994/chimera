"""Unified AutoDAN API Endpoints.

This module provides a comprehensive API for all AutoDAN variants, supporting:
- Multiple optimization strategies (Adaptive, Genetic, Beam Search, etc.)
- Lifelong learning (ICLR 2025 implementation)
- Configuration management and metrics
- Batch processing support
- OVERTHINK integration for reasoning model attacks

Schema Synchronization:
- Uses unified base schemas from `app.schemas.adversarial_base`
- Standardized field naming: `technique`, `target_model`, `score`
- Response includes optional `reasoning_metrics` for OVERTHINK integration
- All responses use consistent 0-10 score scale
"""

import asyncio
import time
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Cookie, Header, HTTPException, status
from pydantic import AliasChoices, Field, field_validator

from app.api.error_handlers import api_error_handler
from app.core.unified_errors import ChimeraError
from app.schemas.adversarial_base import OverthinkConfig, ReasoningMetrics, StrictBaseModel
from app.services.autodan.config_enhanced import get_config
from app.services.autodan.llm import RateLimitError, ResourceExhaustedError
from app.services.autodan.service_enhanced import enhanced_autodan_service
from app.services.session_service import session_service

router = APIRouter()


# ==================== Request/Response Models ====================

AutoDANMethod = Literal[
    "vanilla",
    "best_of_n",
    "beam_search",
    "genetic",
    "hybrid",
    "advanced",
    "chain_of_thought",
    "adaptive",
    "mousetrap",
    "overthink_enhanced",  # New: OVERTHINK integration
    "aegis",  # Project Aegis: Chimera + AutoDAN hybrid
    "turbo",  # AI-driven turbo engine
]


# Re-export StrictBaseModel as StrictModel for backwards compatibility
StrictModel = StrictBaseModel


class JailbreakRequest(StrictModel):
    """Primary request model for jailbreak generation.

    Unified Schema Fields:
    - `goal`: Alias for `request` (unified naming)
    - `target_model`: Alias for `model` (unified naming)
    - `target_provider`: Alias for `provider` (unified naming)
    - `technique`: Alias for `method` (unified naming)
    """

    # Core field with unified aliases
    request: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        validation_alias=AliasChoices("request", "goal"),
        description="The target behavior/goal for the attack",
    )
    method: AutoDANMethod = Field(
        default="adaptive",
        validation_alias=AliasChoices("method", "technique"),
        description="Attack technique/method to use",
    )

    # Model/provider with unified aliases
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "target_model"),
        description="Target LLM model",
        max_length=200,
    )
    provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("provider", "target_provider"),
        description="LLM provider",
        max_length=64,
    )

    # Hyperparameters
    generations: int | None = Field(None, ge=1, le=100, description="Max generations")
    best_of_n: int | None = Field(None, ge=1, le=50, description="N candidates")
    beam_width: int | None = Field(None, ge=1, le=20, description="Beam width (W)")
    beam_depth: int | None = Field(None, ge=1, le=10, description="Beam depth (C)")
    refinement_iterations: int | None = Field(None, ge=1, le=20, description="Refinement cycles")

    # Generation tuning
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int | None = Field(None, ge=1, le=8192, description="Max output tokens")

    # OVERTHINK integration (optional)
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration for reasoning model attacks",
    )
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK integration for reasoning models",
    )

    @field_validator("request")
    @classmethod
    def _strip_request(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "request/goal cannot be empty"
            raise ValueError(msg)
        return stripped

    @field_validator("provider", "model")
    @classmethod
    def _strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

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

    @property
    def technique(self) -> str:
        """Unified alias for method field."""
        return self.method


class JailbreakResponse(StrictModel):
    """Standardized response model for all jailbreak methods.

    Unified Schema Fields:
    - `generated_prompt`: Alias for `jailbreak_prompt` (unified naming)
    - `technique`: Alias for `method` (unified naming)
    - `execution_time_ms`: Alias for `latency_ms` (unified naming)
    - `target_model`: Alias for `model_used` (unified naming)
    - `reasoning_metrics`: OVERTHINK metrics (when enabled)
    """

    # Core response fields (unified schema)
    attack_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique attack identifier",
    )
    success: bool = Field(
        default=True,
        description="Whether the attack succeeded",
    )
    generated_prompt: str = Field(
        ...,
        description="Generated adversarial prompt",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Attack effectiveness score (0-10 scale)",
    )
    execution_time_ms: float = Field(
        ...,
        description="Execution time in milliseconds",
    )

    # AutoDAN-specific fields (kept for backwards compatibility)
    jailbreak_prompt: str = Field(
        ...,
        description="Jailbreak prompt (alias for generated_prompt)",
    )
    method: AutoDANMethod = Field(
        ...,
        description="AutoDAN method used",
    )
    status: Literal["success", "error"] = Field(
        ...,
        description="Operation status",
    )

    # Metrics
    iterations: int | None = Field(None, description="Optimization iterations performed")
    cached: bool = Field(default=False, description="Result from cache")
    model_used: str | None = Field(None, description="Model used")
    provider_used: str | None = Field(None, description="Provider used")
    error: str | None = Field(None, description="Error message if failed")
    strategies_tried: list[str] | None = Field(None, description="Strategies attempted")

    # OVERTHINK integration (unified)
    reasoning_metrics: ReasoningMetrics | None = Field(
        None,
        description="Reasoning metrics (for OVERTHINK integration)",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attack metadata",
    )

    # Backwards compatible aliases
    @property
    def latency_ms(self) -> float:
        """Backwards compatible alias for execution_time_ms."""
        return self.execution_time_ms

    @property
    def technique(self) -> str:
        """Unified alias for method field."""
        return self.method

    @property
    def target_model(self) -> str | None:
        """Unified alias for model_used field."""
        return self.model_used


class BatchJailbreakRequest(StrictModel):
    """Request model for high-throughput batch processing.

    Unified Schema Fields:
    - `goals`: Alias for `requests` (unified naming)
    - `technique`: Alias for `method` (unified naming)
    - `target_model`: Alias for `model` (unified naming)
    """

    requests: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        validation_alias=AliasChoices("requests", "goals"),
        description="List of target behaviors/goals",
    )
    method: AutoDANMethod = Field(
        default="adaptive",
        validation_alias=AliasChoices("method", "technique"),
    )
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "target_model"),
        description="Target LLM model",
        max_length=200,
    )
    provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("provider", "target_provider"),
        max_length=64,
    )
    parallel: bool = Field(default=True, description="Enable parallel processing")

    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)

    # OVERTHINK integration
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK integration",
    )

    @field_validator("requests")
    @classmethod
    def _validate_requests(cls, value: list[str]) -> list[str]:
        if not value:
            msg = "requests/goals must be a non-empty list"
            raise ValueError(msg)
        cleaned: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                msg = f"requests[{idx}] must be a string"
                raise TypeError(msg)
            stripped = item.strip()
            if not stripped:
                msg = f"requests[{idx}] cannot be empty"
                raise ValueError(msg)
            if len(stripped) > 50000:
                msg = f"requests[{idx}] is too long"
                raise ValueError(msg)
            cleaned.append(stripped)
        return cleaned

    # Unified property aliases
    @property
    def goals(self) -> list[str]:
        """Unified alias for requests field."""
        return self.requests


class BatchJailbreakResponse(StrictModel):
    """Aggregate response for batch operations.

    Unified Schema Fields:
    - `total_execution_time_ms`: Alias for `total_latency_ms` (unified)
    """

    results: list[JailbreakResponse] = Field(..., description="Individual attack results")
    total_requests: int = Field(..., ge=0, description="Total requests in batch")
    successful: int = Field(..., ge=0, description="Successful attacks")
    failed: int = Field(..., ge=0, description="Failed attacks")
    total_latency_ms: float = Field(..., ge=0.0, description="Total execution time in ms")

    # Backwards compatible alias
    @property
    def total_execution_time_ms(self) -> float:
        """Unified alias for total_latency_ms."""
        return self.total_latency_ms


class LifelongAttackRequest(StrictModel):
    """Request for AutoDAN-Turbo lifelong learning attack.

    Unified Schema Fields:
    - `goal`: Alias for `request` (unified naming)
    - `target_model`: Alias for `model` (unified naming)
    """

    request: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        validation_alias=AliasChoices("request", "goal"),
        description="Target behavior/goal",
    )
    model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("model", "target_model"),
        max_length=200,
    )
    provider: str | None = Field(
        default=None,
        validation_alias=AliasChoices("provider", "target_provider"),
        max_length=64,
    )
    detailed_scoring: bool = Field(
        default=True,
        description="Enable detailed scoring metrics",
    )

    # OVERTHINK integration
    enable_overthink: bool = Field(
        default=False,
        description="Enable OVERTHINK integration",
    )
    overthink_config: OverthinkConfig | None = Field(
        None,
        description="OVERTHINK configuration",
    )

    @field_validator("request")
    @classmethod
    def _strip_lifelong_request(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "request/goal cannot be empty"
            raise ValueError(msg)
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


class ConfigResponse(StrictModel):
    """Current system configuration state."""

    default_strategy: str
    default_epochs: int
    genetic: dict[str, Any]
    beam_search: dict[str, Any]
    best_of_n: dict[str, Any]
    adaptive: dict[str, Any]
    scoring: dict[str, Any]
    caching: dict[str, Any]
    parallel: dict[str, Any]


# ==================== Helper Logic ====================


def _resolve_model_context(
    payload_model: str | None,
    payload_provider: str | None,
    x_session_id: str | None,
    chimera_cookie: str | None,
) -> tuple[str, str]:
    """Resolve target model/provider using hierarchy:
    explicit payload -> session -> global defaults, with validation/fallback.
    """
    session_id = x_session_id or chimera_cookie

    if session_id:
        session_provider, session_model = session_service.get_session_model(session_id)
    else:
        session_provider, session_model = session_service.get_default_model()

    provider = payload_provider or session_provider
    model = payload_model or session_model

    is_valid, message, fallback_model = session_service.validate_model(provider, model)
    if is_valid:
        return provider, model

    fallback_provider = provider
    if provider not in session_service.get_master_model_list():
        fallback_provider, _ = session_service.get_default_model()

    if payload_model or payload_provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_MODEL_SELECTION",
                "message": message,
                "fallback_provider": fallback_provider,
                "fallback_model": fallback_model,
            },
        )

    # No explicit selection -> best-effort fallback to keep requests working.
    if fallback_model:
        return fallback_provider, fallback_model
    return session_service.get_default_model()


# ==================== Endpoints ====================


@router.post(
    "/jailbreak",
    response_model=JailbreakResponse,
    status_code=status.HTTP_200_OK,
)
@api_error_handler(
    "autodan_jailbreak",
    default_error_message="AutoDAN jailbreak generation failed",
)
async def generate_jailbreak(
    payload: JailbreakRequest,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Primary entry point for AutoDAN jailbreak generation.
    Supports the new Adaptive Strategy Selection mode by default.
    """
    provider, model = _resolve_model_context(
        payload_model=payload.model,
        payload_provider=payload.provider,
        x_session_id=x_session_id,
        chimera_cookie=chimera_session_id,
    )

    # Initialize context if model/provider changed or uninitialized
    serv = enhanced_autodan_service
    needs_init = (
        (not serv.initialized) or model != serv._current_model or provider != serv._current_provider
    )
    if needs_init:
        await serv.initialize(model_name=model, provider=provider, method=payload.method)

    # Extract parameters for the optimization cycle
    params = {
        "generations": payload.generations,
        "best_of_n": payload.best_of_n,
        "beam_width": payload.beam_width,
        "beam_depth": payload.beam_depth,
        "refinement_iterations": payload.refinement_iterations,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    # Filter out None values to use config defaults
    kwargs = {k: v for k, v in params.items() if v is not None}

    try:
        result = await serv.run_jailbreak(request=payload.request, method=payload.method, **kwargs)
    except RateLimitError as e:
        headers = None
        retry_after = getattr(e, "retry_after", None)
        if retry_after:
            headers = {"Retry-After": str(int(retry_after))}
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=headers,
        ) from e
    except ResourceExhaustedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Resource exhausted. Please try again later.",
        ) from e

    # If the service returns an error payload, map it to a proper HTTP error.
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Unknown error"),
        )

    # Build response with unified fields
    jailbreak_prompt = result.get("jailbreak_prompt", "")
    raw_score = result.get("score")
    # Normalize score to 0-10 scale if needed
    if raw_score is not None:
        score = raw_score * 10.0 if raw_score <= 1.0 else min(10.0, raw_score)
    else:
        score = 0.0

    return JailbreakResponse(
        attack_id=result.get("attack_id", str(uuid.uuid4())),
        success=result.get("status") == "success",
        generated_prompt=jailbreak_prompt,
        score=round(score, 2),
        execution_time_ms=result.get("latency_ms", 0.0),
        jailbreak_prompt=jailbreak_prompt,
        method=result.get("method", payload.method),
        status=result.get("status", "success"),
        iterations=result.get("iterations"),
        cached=result.get("cached", False),
        model_used=result.get("model_used"),
        provider_used=result.get("provider_used"),
        error=result.get("error"),
        strategies_tried=result.get("strategies_tried"),
        reasoning_metrics=None,  # Populated if OVERTHINK enabled
        metadata={
            "technique_type": "autodan",
            "method": payload.method,
        },
    )


@router.post("/batch", response_model=BatchJailbreakResponse)
@api_error_handler(
    "autodan_batch",
    default_error_message="AutoDAN batch generation failed",
)
async def batch_jailbreak(
    payload: BatchJailbreakRequest,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Execute high-volume jailbreak generation in parallel."""
    start_time = time.time()

    provider, model = _resolve_model_context(
        payload_model=payload.model,
        payload_provider=payload.provider,
        x_session_id=x_session_id,
        chimera_cookie=chimera_session_id,
    )

    serv = enhanced_autodan_service
    needs_init = (
        (not serv.initialized) or model != serv._current_model or provider != serv._current_provider
    )
    if needs_init:
        await serv.initialize(model_name=model, provider=provider, method=payload.method)

    results: list[JailbreakResponse] = []

    if payload.parallel:
        tasks = [
            serv.run_jailbreak(
                req,
                method=payload.method,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            )
            for req in payload.requests
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                results.append(
                    JailbreakResponse(
                        jailbreak_prompt=payload.requests[i],
                        method=payload.method,
                        status="error",
                        error=str(res),
                        latency_ms=0.0,
                    ),
                )
            else:
                results.append(JailbreakResponse(**res))
    else:
        for req in payload.requests:
            try:
                res = await serv.run_jailbreak(
                    req,
                    method=payload.method,
                    temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                )
            except (
                RateLimitError,
                ResourceExhaustedError,
                TimeoutError,
                ConnectionError,
                ChimeraError,
            ) as e:
                results.append(
                    JailbreakResponse(
                        jailbreak_prompt=req,
                        method=payload.method,
                        status="error",
                        error=str(e),
                        latency_ms=0.0,
                        cached=False,
                        model_used=model,
                        provider_used=provider,
                    ),
                )
                continue

            if res.get("status") == "error":
                results.append(
                    JailbreakResponse(
                        jailbreak_prompt=req,
                        method=payload.method,
                        status="error",
                        error=res.get("error"),
                        latency_ms=float(res.get("latency_ms", 0.0)),
                        cached=bool(res.get("cached", False)),
                        model_used=res.get("model_used"),
                        provider_used=res.get("provider_used"),
                        strategies_tried=res.get("strategies_tried"),
                    ),
                )
            else:
                results.append(JailbreakResponse(**res))

    successful = sum(1 for r in results if r.status == "success")
    return BatchJailbreakResponse(
        results=results,
        total_requests=len(payload.requests),
        successful=successful,
        failed=len(payload.requests) - successful,
        total_latency_ms=(time.time() - start_time) * 1000,
    )


@router.get("/config", response_model=ConfigResponse)
@api_error_handler(
    "autodan_config",
    default_error_message="Failed to fetch AutoDAN configuration",
)
async def get_system_config():
    """Retrieve the current AutoDAN configuration and state."""
    config = get_config()
    return ConfigResponse(
        default_strategy=config.default_strategy.value,
        default_epochs=config.default_epochs,
        genetic=config.genetic.model_dump(),
        beam_search=config.beam_search.model_dump(),
        best_of_n=config.best_of_n.model_dump(),
        adaptive=config.adaptive.model_dump(),
        scoring=config.scoring.model_dump(),
        caching=config.caching.model_dump(),
        parallel=config.parallel.model_dump(),
    )


@router.post("/lifelong", response_model=dict[str, Any])
@api_error_handler(
    "autodan_lifelong",
    default_error_message="AutoDAN lifelong engine failed",
)
async def trigger_lifelong_learning(payload: LifelongAttackRequest):
    """Trigger lifelong learning engine for strategy extraction."""
    from app.services.autodan.service import autodan_service

    return await autodan_service.run_lifelong_attack(
        request=payload.request,
        model_name=payload.model,
        provider=payload.provider,
        detailed_scoring=payload.detailed_scoring,
    )
