"""Chimera + AutoDAN Unified API Router.

Exposes endpoints for the integrated Chimera+AutoDAN red teaming system.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..services.chimera_autodan_service import (
    CampaignRequest,
    ChimeraAutoDanRequest,
    ChimeraAutoDanService,
    get_chimera_autodan_service,
)

logger = logging.getLogger("chimera.routers.chimera_autodan")

router = APIRouter(prefix="/chimera-autodan", tags=["Chimera+AutoDAN"])


# ==============================================================================
# Pydantic Models
# ==============================================================================


class GenerateOptimizedRequest(BaseModel):
    """Request for generating optimized adversarial prompts."""

    objective: str = Field(..., description="The base objective/prompt to transform")
    candidate_count: int = Field(5, ge=1, le=20, description="Number of candidates to generate")
    max_iterations: int = Field(3, ge=1, le=10, description="Number of optimization iterations")
    enable_personas: bool = Field(True, description="Use Chimera persona generation")
    enable_narrative_contexts: bool = Field(True, description="Use narrative context wrapping")
    enable_genetic_optimization: bool = Field(
        True,
        description="Apply AutoDAN genetic optimization",
    )
    generations: int = Field(5, ge=1, le=20, description="Number of genetic generations")
    population_size: int = Field(
        10,
        ge=2,
        le=50,
        description="Population size for genetic optimization",
    )
    mutation_rate: float = Field(0.3, ge=0.0, le=1.0, description="Mutation probability")
    model_name: str | None = Field(None, description="LLM model name for AI-driven mutations")
    provider: str | None = Field(None, description="LLM provider (openai, gemini, etc.)")


class GenerateOptimizedResponse(BaseModel):
    """Response from optimized generation."""

    candidates: list[dict] = Field(default_factory=list, description="Generated candidates")
    best_candidate: dict | None = Field(None, description="Best performing candidate")
    total_evaluated: int = Field(0, description="Total candidates evaluated")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")
    config_used: dict = Field(default_factory=dict, description="Configuration used")


class RunCampaignRequest(BaseModel):
    """Request for running a multi-phase campaign."""

    objective: str = Field(..., description="Target objective for the campaign")
    phases: int = Field(3, ge=1, le=10, description="Number of campaign phases")
    candidates_per_phase: int = Field(5, ge=1, le=20, description="Candidates per phase")
    model_name: str | None = Field(None, description="LLM model name")
    provider: str | None = Field(None, description="LLM provider")


class RunCampaignResponse(BaseModel):
    """Response from campaign execution."""

    phases: list[dict] = Field(default_factory=list, description="Phase results")
    best_candidates: list[dict] = Field(
        default_factory=list,
        description="Best candidates from all phases",
    )
    total_candidates_evaluated: int = Field(0, description="Total candidates evaluated")
    objective: str = Field("", description="Campaign objective")


class MutateRequest(BaseModel):
    """Request for Chimera-based mutation."""

    prompt: str = Field(..., description="Prompt to mutate")
    mutation_type: str | None = Field(
        None,
        description="Mutation type: persona_wrap, narrative_frame, semantic_obfuscate, hybrid, context_inject, style_transform",
    )
    count: int = Field(3, ge=1, le=10, description="Number of mutations to generate")


class MutateResponse(BaseModel):
    """Response from mutation operation."""

    mutations: list[dict] = Field(default_factory=list, description="Mutation results")


class OptimizePersonasRequest(BaseModel):
    """Request for persona optimization."""

    objective: str = Field(..., description="Target objective for persona optimization")
    persona_count: int = Field(5, ge=1, le=20, description="Number of personas to evolve")
    generations: int = Field(5, ge=1, le=20, description="Evolutionary generations")


class OptimizePersonasResponse(BaseModel):
    """Response from persona optimization."""

    personas: list[dict] = Field(
        default_factory=list,
        description="Optimized persona configurations",
    )


class EvaluateRequest(BaseModel):
    """Request for fitness evaluation."""

    original: str = Field(..., description="Original objective")
    transformed: str = Field(..., description="Transformed prompt to evaluate")


class EvaluateResponse(BaseModel):
    """Response from fitness evaluation."""

    total_score: float = Field(0.0, description="Overall fitness score")
    breakdown: dict = Field(default_factory=dict, description="Score breakdown by category")
    passed_threshold: bool = Field(False, description="Whether the prompt passed success threshold")
    recommendations: list[str] = Field(
        default_factory=list,
        description="Improvement recommendations",
    )


# ==============================================================================
# Endpoints
# ==============================================================================


@router.post("/generate")
async def generate_optimized(
    request: GenerateOptimizedRequest,
    service: Annotated[ChimeraAutoDanService, Depends(get_chimera_autodan_service)],
) -> GenerateOptimizedResponse:
    """Generate optimized adversarial prompts using unified Chimera+AutoDAN.

    This endpoint combines:
    - Chimera's narrative persona generation
    - Chimera's semantic obfuscation
    - AutoDAN's genetic mutation and optimization

    Returns candidates sorted by fitness score.
    """
    try:
        # Convert Pydantic model to dataclass
        internal_request = ChimeraAutoDanRequest(
            objective=request.objective,
            candidate_count=request.candidate_count,
            max_iterations=request.max_iterations,
            enable_personas=request.enable_personas,
            enable_narrative_contexts=request.enable_narrative_contexts,
            enable_genetic_optimization=request.enable_genetic_optimization,
            generations=request.generations,
            population_size=request.population_size,
            mutation_rate=request.mutation_rate,
            model_name=request.model_name,
            provider=request.provider,
        )

        result = await service.generate_optimized(internal_request)

        return GenerateOptimizedResponse(
            candidates=result.candidates,
            best_candidate=result.best_candidate,
            total_evaluated=result.total_evaluated,
            execution_time_ms=result.execution_time_ms,
            config_used=result.config_used,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in generate_optimized endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaign")
async def run_campaign(
    request: RunCampaignRequest,
    service: Annotated[ChimeraAutoDanService, Depends(get_chimera_autodan_service)],
) -> RunCampaignResponse:
    """Run a multi-phase campaign combining Chimera and AutoDAN.

    Each phase:
    1. Generates candidates with increasingly diverse strategies
    2. Optimizes using AutoDAN genetic algorithms
    3. Selects best performers for next phase

    Returns cumulative best candidates across all phases.
    """
    try:
        internal_request = CampaignRequest(
            objective=request.objective,
            phases=request.phases,
            candidates_per_phase=request.candidates_per_phase,
            model_name=request.model_name,
            provider=request.provider,
        )

        result = await service.run_campaign(internal_request)

        return RunCampaignResponse(
            phases=result.phases,
            best_candidates=result.best_candidates,
            total_candidates_evaluated=result.total_candidates_evaluated,
            objective=result.objective,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in run_campaign endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mutate")
async def mutate_with_chimera(
    request: MutateRequest,
    service: Annotated[ChimeraAutoDanService, Depends(get_chimera_autodan_service)],
) -> MutateResponse:
    """Apply Chimera-based mutations to a prompt.

    Available mutation types:
    - persona_wrap: Wrap with a generated persona
    - narrative_frame: Frame within a narrative context
    - semantic_obfuscate: Apply semantic obfuscation
    - hybrid: Combine multiple mutation strategies
    - context_inject: Inject contextual framing
    - style_transform: Transform writing style

    If mutation_type is not specified, a random type is selected.
    """
    try:
        mutations = await service.mutate_with_chimera(
            prompt=request.prompt,
            mutation_type=request.mutation_type,
            count=request.count,
        )

        return MutateResponse(mutations=mutations)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in mutate_with_chimera endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-personas")
async def optimize_personas(
    request: OptimizePersonasRequest,
    service: Annotated[ChimeraAutoDanService, Depends(get_chimera_autodan_service)],
) -> OptimizePersonasResponse:
    """Optimize persona configurations using AutoDAN genetic algorithms.

    Evolves persona parameters (role, voice, backstory) to maximize
    effectiveness against the target objective.

    Returns best-performing persona configurations.
    """
    try:
        personas = await service.optimize_personas(
            objective=request.objective,
            persona_count=request.persona_count,
            generations=request.generations,
        )

        return OptimizePersonasResponse(personas=personas)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in optimize_personas endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_prompt(
    request: EvaluateRequest,
    service: Annotated[ChimeraAutoDanService, Depends(get_chimera_autodan_service)],
) -> EvaluateResponse:
    """Evaluate a transformed prompt using unified fitness metrics.

    Metrics include:
    - Jailbreak potential
    - Stealth/detection avoidance
    - Fluency/naturalness
    - Semantic preservation
    - Persona coherence
    - Narrative quality

    Returns overall score, breakdown, and improvement recommendations.
    """
    try:
        result = await service.evaluate_prompt(
            original=request.original,
            transformed=request.transformed,
        )

        return EvaluateResponse(
            total_score=result["total_score"],
            breakdown=result["breakdown"],
            passed_threshold=result["passed_threshold"],
            recommendations=result["recommendations"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in evaluate_prompt endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> dict:
    """Check if the Chimera+AutoDAN service is available."""
    try:
        # Try to find the modules using importlib
        import importlib.util

        autodan_spec = importlib.util.find_spec("meta_prompter.autodan_wrapper")
        chimera_spec = importlib.util.find_spec("meta_prompter.chimera.engine")
        unified_spec = importlib.util.find_spec("meta_prompter.chimera_autodan")

        return {
            "status": "healthy",
            "chimera_available": chimera_spec is not None,
            "autodan_available": autodan_spec is not None,
            "unified_available": unified_spec is not None,
        }
    except ImportError as e:
        return {
            "status": "degraded",
            "error": str(e),
            "chimera_available": False,
            "autodan_available": False,
            "unified_available": False,
        }
