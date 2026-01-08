"""
Chimera + AutoDAN Unified Service.

Backend service exposing the unified Chimera+AutoDAN integration
via FastAPI endpoints for red teaming operations.
"""

import asyncio
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

from fastapi import HTTPException

logger = logging.getLogger("chimera.chimera_autodan_service")


@dataclass
class ChimeraAutoDanRequest:
    """Request model for Chimera+AutoDAN operations."""
    objective: str
    candidate_count: int = 5
    max_iterations: int = 3
    enable_personas: bool = True
    enable_narrative_contexts: bool = True
    enable_genetic_optimization: bool = True
    generations: int = 5
    population_size: int = 10
    mutation_rate: float = 0.3
    model_name: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class ChimeraAutoDanResponse:
    """Response model for Chimera+AutoDAN operations."""
    candidates: list[dict] = field(default_factory=list)
    best_candidate: Optional[dict] = None
    total_evaluated: int = 0
    execution_time_ms: float = 0.0
    config_used: dict = field(default_factory=dict)


@dataclass
class CampaignRequest:
    """Request model for running a full campaign."""
    objective: str
    phases: int = 3
    candidates_per_phase: int = 5
    model_name: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class CampaignResponse:
    """Response model for campaign results."""
    phases: list[dict] = field(default_factory=list)
    best_candidates: list[dict] = field(default_factory=list)
    total_candidates_evaluated: int = 0
    objective: str = ""


class ChimeraAutoDanService:
    """
    Service for unified Chimera+AutoDAN red teaming operations.

    Provides methods for:
    - Generating optimized adversarial prompts
    - Running multi-phase campaigns
    - Optimizing persona configurations
    - Applying Chimera mutations

    Example:
        >>> service = ChimeraAutoDanService()
        >>> await service.initialize()
        >>> result = await service.generate_optimized(
        ...     ChimeraAutoDanRequest(objective="test")
        ... )
    """

    def __init__(self):
        """Initialize the service."""
        self._unified_engine = None
        self._initialized = False

    async def initialize(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ) -> None:
        """
        Initialize the service with LLM configuration.

        Args:
            model_name: Default model for AI-driven optimization
            provider: Default LLM provider
        """
        self._default_model = model_name
        self._default_provider = provider
        self._initialized = True
        logger.info("ChimeraAutoDanService initialized")

    def _get_unified_engine(self, config=None):
        """Lazy load the unified engine."""
        from meta_prompter.chimera_autodan import ChimeraAutoDAN, ChimeraAutoDanConfig

        if config:
            return ChimeraAutoDAN(config)

        if self._unified_engine is None:
            default_config = ChimeraAutoDanConfig(
                model_name=self._default_model,
                provider=self._default_provider
            )
            self._unified_engine = ChimeraAutoDAN(default_config)

        return self._unified_engine

    async def generate_optimized(
        self,
        request: ChimeraAutoDanRequest
    ) -> ChimeraAutoDanResponse:
        """
        Generate optimized adversarial prompts using Chimera+AutoDAN.

        Args:
            request: Generation request parameters

        Returns:
            Response with generated candidates and metrics
        """
        import time
        start_time = time.time()

        try:
            from meta_prompter.chimera_autodan import ChimeraAutoDanConfig

            # Build config from request
            config = ChimeraAutoDanConfig(
                enable_personas=request.enable_personas,
                enable_narrative_contexts=request.enable_narrative_contexts,
                enable_genetic_optimization=request.enable_genetic_optimization,
                generations=request.generations,
                population_size=request.population_size,
                mutation_rate=request.mutation_rate,
                model_name=request.model_name or self._default_model,
                provider=request.provider or self._default_provider
            )

            # Get engine and generate
            engine = self._get_unified_engine(config)
            candidates = await engine.generate_optimized(
                objective=request.objective,
                candidate_count=request.candidate_count,
                max_iterations=request.max_iterations
            )

            # Convert to response format
            candidate_dicts = [
                {
                    "prompt_text": c.prompt_text,
                    "metadata": c.metadata,
                    "fitness_score": c.metadata.get("fitness_score", 0.0)
                }
                for c in candidates
            ]

            execution_time = (time.time() - start_time) * 1000

            return ChimeraAutoDanResponse(
                candidates=candidate_dicts,
                best_candidate=candidate_dicts[0] if candidate_dicts else None,
                total_evaluated=len(candidate_dicts),
                execution_time_ms=execution_time,
                config_used={
                    "enable_personas": config.enable_personas,
                    "enable_narrative_contexts": config.enable_narrative_contexts,
                    "enable_genetic_optimization": config.enable_genetic_optimization,
                    "generations": config.generations
                }
            )

        except ImportError as e:
            logger.error(f"Import error in generate_optimized: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Chimera+AutoDAN module not available: {e}"
            )
        except Exception as e:
            logger.error(f"Error in generate_optimized: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )

    async def run_campaign(
        self,
        request: CampaignRequest
    ) -> CampaignResponse:
        """
        Run a full multi-phase campaign.

        Args:
            request: Campaign request parameters

        Returns:
            Response with campaign results and best candidates
        """
        try:
            from meta_prompter.chimera_autodan import ChimeraAutoDanConfig

            config = ChimeraAutoDanConfig(
                model_name=request.model_name or self._default_model,
                provider=request.provider or self._default_provider
            )

            engine = self._get_unified_engine(config)
            results = await engine.run_campaign(
                objective=request.objective,
                phases=request.phases,
                candidates_per_phase=request.candidates_per_phase
            )

            # Convert candidates to dicts
            best_candidate_dicts = [
                {
                    "prompt_text": c.prompt_text,
                    "metadata": c.metadata,
                    "fitness_score": c.metadata.get("fitness_score", 0.0)
                }
                for c in results.get("best_candidates", [])
            ]

            return CampaignResponse(
                phases=results.get("phases", []),
                best_candidates=best_candidate_dicts,
                total_candidates_evaluated=results.get("total_candidates_evaluated", 0),
                objective=results.get("objective", request.objective)
            )

        except Exception as e:
            logger.error(f"Error in run_campaign: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Campaign failed: {str(e)}"
            )

    async def mutate_with_chimera(
        self,
        prompt: str,
        mutation_type: Optional[str] = None,
        count: int = 3
    ) -> list[dict]:
        """
        Apply Chimera-based mutations to a prompt.

        Args:
            prompt: The prompt to mutate
            mutation_type: Specific mutation type, or None for random
            count: Number of mutations to generate

        Returns:
            List of mutation result dictionaries
        """
        try:
            from meta_prompter.chimera_autodan import ChimeraMutator

            mutator = ChimeraMutator()
            results = []

            for _ in range(count):
                result = mutator.mutate(prompt, mutation_type=mutation_type)
                results.append({
                    "original_text": result.original_text,
                    "mutated_text": result.mutated_text,
                    "mutation_type": result.mutation_type,
                    "persona_used": result.persona_used,
                    "scenario_used": result.scenario_used,
                    "obfuscation_applied": result.obfuscation_applied,
                    "semantic_similarity": result.semantic_similarity
                })

            return results

        except Exception as e:
            logger.error(f"Error in mutate_with_chimera: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Mutation failed: {str(e)}"
            )

    async def optimize_personas(
        self,
        objective: str,
        persona_count: int = 5,
        generations: int = 5
    ) -> list[dict]:
        """
        Optimize persona configurations using genetic algorithms.

        Args:
            objective: Target objective for persona optimization
            persona_count: Number of personas to evolve
            generations: Number of evolutionary generations

        Returns:
            List of optimized persona configurations
        """
        try:
            from meta_prompter.chimera_autodan import AutoDANPersonaOptimizer

            optimizer = AutoDANPersonaOptimizer(
                population_size=persona_count * 4,  # Larger population
                generations=generations
            )

            personas = await optimizer.optimize(objective)
            return personas

        except Exception as e:
            logger.error(f"Error in optimize_personas: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Persona optimization failed: {str(e)}"
            )

    async def evaluate_prompt(
        self,
        original: str,
        transformed: str
    ) -> dict:
        """
        Evaluate a transformed prompt using unified fitness metrics.

        Args:
            original: Original objective
            transformed: Transformed/mutated prompt

        Returns:
            Fitness evaluation results
        """
        try:
            from meta_prompter.chimera_autodan import UnifiedFitnessEvaluator

            evaluator = UnifiedFitnessEvaluator()
            result = evaluator.evaluate(original, transformed)

            return {
                "total_score": result.total_score,
                "breakdown": result.breakdown,
                "passed_threshold": result.passed_threshold,
                "recommendations": result.recommendations
            }

        except Exception as e:
            logger.error(f"Error in evaluate_prompt: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed: {str(e)}"
            )


# Singleton instance
_service_instance: Optional[ChimeraAutoDanService] = None


def get_chimera_autodan_service() -> ChimeraAutoDanService:
    """Get the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ChimeraAutoDanService()
    return _service_instance
