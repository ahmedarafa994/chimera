"""
Enhanced AutoDAN API Endpoints

This module provides enhanced API endpoints for the AutoDAN service with:
- Multiple optimization strategies
- Configuration management
- Metrics and monitoring
- Batch processing support
"""

import asyncio
import logging
import re
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Cookie, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.services.autodan.config_enhanced import get_config, update_config
from app.services.autodan.service_enhanced import enhanced_autodan_service
from app.services.session_service import session_service

# Use unique prefix to avoid conflict with legacy autodan router
router = APIRouter(prefix="/autodan-enhanced", tags=["AutoDAN Enhanced"])
logger = logging.getLogger(__name__)


# ==================== Request/Response Models ====================


class JailbreakRequest(BaseModel):
    """Request model for jailbreak generation."""

    request: str = Field(..., min_length=1, description="The request to jailbreak")
    method: str = Field(
        default="best_of_n",
        description=(
            "Optimization method: vanilla, best_of_n, beam_search, "
            "genetic, hybrid, chain_of_thought, adaptive"
        ),
    )
    target_model: str | None = Field(None, description="Target LLM model")
    provider: str | None = Field(None, description="LLM provider")

    # Method-specific parameters
    generations: int | None = Field(
        None, ge=1, le=50, description="Number of generations (genetic)"
    )
    best_of_n: int | None = Field(None, ge=1, le=20, description="N candidates (best_of_n)")
    beam_width: int | None = Field(None, ge=1, le=20, description="Beam width (beam_search)")
    beam_depth: int | None = Field(None, ge=1, le=10, description="Beam depth (beam_search)")
    refinement_iterations: int | None = Field(
        None, ge=1, le=10, description="Refinement cycles (adaptive/cot)"
    )

    model_config = ConfigDict(str_strip_whitespace=True)


class JailbreakResponse(BaseModel):
    """Response model for jailbreak generation."""

    jailbreak_prompt: str
    method: str
    status: str
    score: float | None = None
    iterations: int | None = None
    latency_ms: float
    cached: bool = False
    model_used: str | None = None
    provider_used: str | None = None
    error: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class BatchJailbreakRequest(BaseModel):
    """Request model for batch jailbreak generation."""

    requests: list[str] = Field(..., min_length=1, max_length=100)
    method: str = Field(default="best_of_n")
    target_model: str | None = None
    provider: str | None = None
    parallel: bool = Field(default=True, description="Process requests in parallel")


class BatchJailbreakResponse(BaseModel):
    """Response model for batch jailbreak generation."""

    results: list[JailbreakResponse]
    total_requests: int
    successful: int
    failed: int
    total_latency_ms: float


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""

    default_strategy: str | None = None
    default_epochs: int | None = Field(None, ge=1, le=50)

    # Genetic algorithm config
    population_size: int | None = Field(None, ge=2, le=100)
    mutation_rate: float | None = Field(None, ge=0.0, le=1.0)
    crossover_rate: float | None = Field(None, ge=0.0, le=1.0)

    # Beam search config
    beam_width: int | None = Field(None, ge=1, le=20)
    beam_depth: int | None = Field(None, ge=1, le=10)
    beam_k: int | None = Field(None, ge=1, le=50)

    # Best-of-N config
    n_candidates: int | None = Field(None, ge=1, le=20)

    # Adaptive config
    max_refinement_iterations: int | None = Field(None, ge=1, le=10)
    feedback_learning: bool | None = None

    # Scoring config
    break_score: float | None = Field(None, ge=1.0, le=10.0)

    # Caching config
    enable_embedding_cache: bool | None = None
    enable_response_cache: bool | None = None
    cache_ttl_seconds: int | None = Field(None, ge=60, le=86400)

    # Parallel config
    enable_parallel: bool | None = None
    max_workers: int | None = Field(None, ge=1, le=16)


class MetricsResponse(BaseModel):
    """Response model for service metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_latency_ms: float
    cache_hit_rate: float
    embedding_cache_stats: dict[str, Any]
    response_cache_stats: dict[str, Any]
    parallel_processor_stats: dict[str, Any]


class ConfigResponse(BaseModel):
    """Response model for configuration."""

    default_strategy: str
    default_epochs: int
    genetic: dict[str, Any]
    beam_search: dict[str, Any]
    best_of_n: dict[str, Any]
    adaptive: dict[str, Any]
    scoring: dict[str, Any]
    caching: dict[str, Any]
    parallel: dict[str, Any]


# ==================== Helper Functions ====================


def get_session_id(
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
) -> str | None:
    """Extract session ID from header or cookie."""
    return x_session_id or chimera_session_id


def is_rate_limit_error(error_message: str) -> bool:
    """Check if an error message indicates a rate limit or quota issue."""
    rate_limit_patterns = [
        r"429",
        r"rate.?limit",
        r"quota.?exceed",
        r"RESOURCE_EXHAUSTED",
        r"too.?many.?requests",
        r"exceeded.+quota",
    ]
    error_lower = error_message.lower()
    return any(re.search(pattern, error_lower, re.IGNORECASE) for pattern in rate_limit_patterns)


def extract_provider_from_error(error_message: str) -> str | None:
    """Extract provider name from error message if present."""
    providers = ["gemini", "openai", "anthropic", "azure", "cohere", "mistral"]
    error_lower = error_message.lower()
    for provider in providers:
        if provider in error_lower:
            return provider
    # Check for API-specific patterns
    if "ai.google.dev" in error_lower or "generativelanguage.googleapis.com" in error_lower:
        return "gemini"
    if "api.openai.com" in error_lower:
        return "openai"
    return None


async def get_model_from_session(session_id: str | None) -> tuple[str | None, str | None]:
    """Get model and provider from session."""
    if session_id:
        provider, model = session_service.get_session_model(session_id)
        return provider, model
    return None, None


# ==================== Endpoints ====================


@router.post("/jailbreak", response_model=JailbreakResponse)
async def generate_jailbreak(
    payload: JailbreakRequest,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
):
    """
    Generate a jailbreak prompt using the enhanced AutoDAN service.

    Supports multiple optimization strategies:
    - **vanilla**: Basic AutoDAN method
    - **turbo**: Weaponized AutoDAN with Bypass Engines (Level 4/5)
    - **best_of_n**: Generate N candidates and select best
    - **beam_search**: Explore strategy combinations with beam search
    - **genetic**: Use genetic algorithm optimization
    - **hybrid**: Combine genetic optimization with AutoDAN strategies
    - **chain_of_thought**: Multi-step reasoning for prompt generation
    - **adaptive**: Dynamic strategy selection based on target feedback
    """
    try:
        # Log incoming request for debugging
        logger.info(
            f"[AutoDAN-Enhanced] Received jailbreak request - "
            f"request_length={len(payload.request)}, "
            f"method={payload.method}, "
            f"target_model={payload.target_model}, "
            f"provider={payload.provider}"
        )
        logger.debug(
            f"[AutoDAN-Enhanced] Full payload: "
            f"request='{payload.request[:100]}...', "
            f"generations={payload.generations}, "
            f"best_of_n={payload.best_of_n}, "
            f"beam_width={payload.beam_width}, "
            f"beam_depth={payload.beam_depth}, "
            f"refinement_iterations={payload.refinement_iterations}"
        )

        # Get session info
        session_id = x_session_id or chimera_session_id
        session_provider, session_model = await get_model_from_session(session_id)

        # Determine model/provider
        provider = payload.provider or session_provider
        model = payload.target_model or session_model

        logger.info(
            f"Jailbreak request: method={payload.method}, model={model}, provider={provider}"
        )

        # Initialize service if needed
        service = enhanced_autodan_service
        if not service.initialized or model != service._current_model:
            await service.initialize(model_name=model, provider=provider, method=payload.method)

        # Build kwargs from optional parameters
        kwargs = {}
        if payload.generations:
            kwargs["generations"] = payload.generations
        if payload.best_of_n:
            kwargs["best_of_n"] = payload.best_of_n
        if payload.beam_width:
            kwargs["beam_width"] = payload.beam_width
        if payload.beam_depth:
            kwargs["beam_depth"] = payload.beam_depth
        if payload.refinement_iterations:
            kwargs["refinement_iterations"] = payload.refinement_iterations

        # Run jailbreak
        result = await service.run_jailbreak(
            request=payload.request, method=payload.method, **kwargs
        )

        return JailbreakResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        error_str = str(e)
        logger.error(f"Jailbreak generation failed: {e}", exc_info=True)

        # Check for rate limit / quota errors
        if is_rate_limit_error(error_str):
            provider = extract_provider_from_error(error_str) or payload.provider or "LLM provider"
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit or quota exceeded for {provider}.",
                headers={"Retry-After": "60"},
            ) from e

        raise HTTPException(status_code=500, detail=f"Internal Server Error: {error_str}") from e


@router.post("/jailbreak/batch", response_model=BatchJailbreakResponse)
async def generate_batch_jailbreak(
    payload: BatchJailbreakRequest,
    background_tasks: BackgroundTasks,
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
):
    """
    Generate jailbreak prompts for multiple requests.

    Supports parallel processing for improved throughput.
    """
    import time

    start_time = time.time()

    try:
        # Get session info
        session_id = x_session_id or chimera_session_id
        session_provider, session_model = await get_model_from_session(session_id)

        provider = payload.provider or session_provider
        model = payload.target_model or session_model

        logger.info(
            f"Batch jailbreak request: {len(payload.requests)} requests, method={payload.method}"
        )

        # Initialize service
        service = enhanced_autodan_service
        if not service.initialized or model != service._current_model:
            await service.initialize(model_name=model, provider=provider, method=payload.method)

        results = []

        if payload.parallel:
            # Process in parallel
            tasks = [
                service.run_jailbreak(request=req, method=payload.method)
                for req in payload.requests
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(raw_results):
                if isinstance(result, Exception):
                    results.append(
                        JailbreakResponse(
                            jailbreak_prompt=payload.requests[i],
                            method=payload.method,
                            status="error",
                            error=str(result),
                            latency_ms=0,
                            cached=False,
                        )
                    )
                else:
                    results.append(JailbreakResponse(**result))
        else:
            # Process sequentially
            for req in payload.requests:
                try:
                    result = await service.run_jailbreak(request=req, method=payload.method)
                    results.append(JailbreakResponse(**result))
                except Exception as e:
                    results.append(
                        JailbreakResponse(
                            jailbreak_prompt=req,
                            method=payload.method,
                            status="error",
                            error=str(e),
                            latency_ms=0,
                            cached=False,
                        )
                    )

        successful = sum(1 for r in results if r.status == "success")
        failed = len(results) - successful
        total_latency = (time.time() - start_time) * 1000

        return BatchJailbreakResponse(
            results=results,
            total_requests=len(payload.requests),
            successful=successful,
            failed=failed,
            total_latency_ms=total_latency,
        )

    except Exception as e:
        error_str = str(e)
        logger.error(f"Batch jailbreak failed: {e}", exc_info=True)

        # Check for rate limit / quota errors
        if is_rate_limit_error(error_str):
            provider = extract_provider_from_error(error_str) or payload.provider or "LLM provider"
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit or quota exceeded for {provider}.",
                headers={"Retry-After": "60"},
            ) from e

        raise HTTPException(status_code=500, detail=f"Internal Server Error: {error_str}") from e


@router.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """Get current AutoDAN configuration."""
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


@router.put("/config")
async def update_configuration(payload: ConfigUpdateRequest):
    """
    Update AutoDAN configuration.

    Only provided fields will be updated.
    """
    try:
        updates = {}

        if payload.default_strategy:
            updates["default_strategy"] = payload.default_strategy
        if payload.default_epochs:
            updates["default_epochs"] = payload.default_epochs

        # Genetic config
        genetic_updates = {}
        if payload.population_size:
            genetic_updates["population_size"] = payload.population_size
        if payload.mutation_rate is not None:
            genetic_updates["mutation_rate"] = payload.mutation_rate
        if payload.crossover_rate is not None:
            genetic_updates["crossover_rate"] = payload.crossover_rate
        if genetic_updates:
            updates["genetic"] = genetic_updates

        # Beam search config
        beam_updates = {}
        if payload.beam_width:
            beam_updates["beam_width"] = payload.beam_width
        if payload.beam_depth:
            beam_updates["beam_depth"] = payload.beam_depth
        if payload.beam_k:
            beam_updates["beam_k"] = payload.beam_k
        if beam_updates:
            updates["beam_search"] = beam_updates

        # Best-of-N config
        if payload.n_candidates:
            updates["best_of_n"] = {"n_candidates": payload.n_candidates}

        # Adaptive config
        adaptive_updates = {}
        if payload.max_refinement_iterations:
            adaptive_updates["max_refinement_iterations"] = payload.max_refinement_iterations
        if payload.feedback_learning is not None:
            adaptive_updates["feedback_learning"] = payload.feedback_learning
        if adaptive_updates:
            updates["adaptive"] = adaptive_updates

        # Scoring config
        if payload.break_score:
            updates["scoring"] = {"break_score": payload.break_score}

        # Caching config
        caching_updates = {}
        if payload.enable_embedding_cache is not None:
            caching_updates["enable_embedding_cache"] = payload.enable_embedding_cache
        if payload.enable_response_cache is not None:
            caching_updates["enable_response_cache"] = payload.enable_response_cache
        if payload.cache_ttl_seconds:
            caching_updates["cache_ttl_seconds"] = payload.cache_ttl_seconds
        if caching_updates:
            updates["caching"] = caching_updates

        # Parallel config
        parallel_updates = {}
        if payload.enable_parallel is not None:
            parallel_updates["enable_parallel"] = payload.enable_parallel
        if payload.max_workers:
            parallel_updates["max_workers"] = payload.max_workers
        if parallel_updates:
            updates["parallel"] = parallel_updates

        if updates:
            update_config(updates)
            enhanced_autodan_service.update_config(updates)
            logger.info(f"Configuration updated: {updates}")

        return {"status": "success", "updates": updates}

    except Exception as e:
        logger.error(f"Configuration update failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get AutoDAN service metrics."""
    metrics = enhanced_autodan_service.get_metrics()
    return MetricsResponse(**metrics)


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset AutoDAN service metrics."""
    enhanced_autodan_service.reset_metrics()
    return {"status": "success", "message": "Metrics reset"}


@router.get("/strategies")
async def list_strategies():
    """List available optimization strategies."""
    return {
        "strategies": [
            {
                "name": "vanilla",
                "description": ("Basic AutoDAN-Turbo method with single candidate generation"),
                "parameters": [],
            },
            {
                "name": "best_of_n",
                "description": ("Generate N candidates and select the best based on scoring"),
                "parameters": ["n_candidates"],
            },
            {
                "name": "beam_search",
                "description": ("Explore strategy combinations using beam search"),
                "parameters": ["beam_width", "beam_depth", "beam_k"],
            },
            {
                "name": "genetic",
                "description": ("Use genetic algorithm for prompt optimization"),
                "parameters": ["generations", "population_size", "mutation_rate", "crossover_rate"],
            },
            {
                "name": "hybrid",
                "description": ("Combine genetic optimization with AutoDAN strategies"),
                "parameters": ["genetic_generations", "autodan_epochs"],
            },
            {
                "name": "chain_of_thought",
                "description": (
                    "Multi-step reasoning pipeline for high-quality jailbreak generation"
                ),
                "parameters": ["refinement_iterations"],
            },
            {
                "name": "adaptive",
                "description": (
                    "Dynamic strategy selection based on observed target model defense patterns"
                ),
                "parameters": ["refinement_iterations"],
            },
        ]
    }


@router.get("/health")
async def health_check():
    """Check AutoDAN service health."""
    service = enhanced_autodan_service

    return {
        "status": "healthy" if service.initialized else "not_initialized",
        "initialized": service.initialized,
        "current_model": service._current_model,
        "current_provider": service._current_provider,
        "strategy_library_size": len(service.strategy_library),
        "cache_stats": {
            "embedding_cache": service.embedding_cache.stats,
            "response_cache": service.response_cache.stats,
        },
    }


@router.post("/cleanup")
async def cleanup_service():
    """Cleanup AutoDAN service resources."""
    await enhanced_autodan_service.cleanup()
    return {"status": "success", "message": "Service cleaned up"}


@router.post("/strategy-library/save")
async def save_strategy_library():
    """Save the current strategy library to disk."""
    await enhanced_autodan_service.save_strategy_library()
    return {"status": "success", "strategies_saved": len(enhanced_autodan_service.strategy_library)}
