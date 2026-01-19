"""
API endpoints for AutoDAN Gradient-Guided Optimization and Ensemble Alignment.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import get_current_user
from app.services.autodan_advanced.ensemble_aligner import EnsembleGradientAligner
from app.services.autodan_advanced.gradient_optimizer import GradientGuidedOptimizer

router = APIRouter()
logger = logging.getLogger(__name__)


# ========================
# Gradient Optimization
# ========================


class GradientOptimizationRequest(BaseModel):
    """Request for gradient-guided optimization."""

    request: str = Field(..., description="Target complex request")
    initial_prompt: str = Field(..., description="Starting prompt for optimization")
    surrogate_model: str = Field("bert-base", description="Surrogate model for gradients")
    learning_rate: float = Field(0.01, ge=0.0001, le=0.1, description="Learning rate")
    max_iterations: int = Field(100, ge=10, le=500, description="Maximum iterations")
    convergence_threshold: float = Field(
        0.001, ge=0.0001, le=0.1, description="Convergence threshold"
    )
    gradient_clip_norm: float = Field(1.0, ge=0.1, le=10.0, description="Gradient clipping norm")
    perplexity_penalty: float = Field(0.2, ge=0.0, le=1.0, description="Perplexity penalty weight")


class GradientOptimizationResponse(BaseModel):
    """Response from gradient optimization."""

    optimized_prompt: str
    final_score: float
    iterations: int
    convergence_achieved: bool
    gradient_history: list[dict]
    semantic_coherence: float
    execution_time_ms: int


@router.post("/gradient-optimize", response_model=GradientOptimizationResponse)
async def optimize_with_gradient(
    payload: GradientOptimizationRequest, user: dict = Depends(get_current_user)
):
    """
    Optimize prompts using gradient-guided search with white-box surrogates.

    This endpoint implements coherence-constrained gradient search:
    - Computes gradients from surrogate models
    - Filters candidates by language model probability
    - Selects tokens that minimize: GradLoss + λ * Perplexity

    Args:
        payload: Gradient optimization configuration
        user: Authenticated user context

    Returns:
        Optimization results with gradient history
    """
    try:
        logger.info(
            f"Starting gradient optimization: request={payload.request[:50]}..., "
            f"max_iterations={payload.max_iterations}"
        )

        # Create optimizer
        optimizer = GradientGuidedOptimizer(
            coherence_weight=payload.perplexity_penalty,
            max_iterations=payload.max_iterations,
            learning_rate=payload.learning_rate,
            convergence_threshold=payload.convergence_threshold,
        )

        # Execute optimization
        result = await optimizer.optimize(
            initial_prompt=payload.initial_prompt,
            target_string="Sure, here is",
            surrogate_models=None,  # TODO: Load actual surrogate models
        )

        logger.info(
            f"Gradient optimization completed: iterations={result['iterations']}, "
            f"converged={result['convergence_achieved']}"
        )

        return GradientOptimizationResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Gradient optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gradient optimization failed: {e!s}")


# ========================
# Ensemble Alignment
# ========================


class EnsembleAttackRequest(BaseModel):
    """Request for ensemble attack with gradient alignment."""

    request: str = Field(..., description="Target complex request")
    target_models: list[dict] = Field(
        ..., min_items=2, max_items=5, description="Target model configurations"
    )
    alignment_strategy: str = Field("weighted", description="Gradient alignment strategy")
    max_iterations: int = Field(50, ge=10, le=200, description="Maximum iterations")
    consensus_threshold: float = Field(0.8, ge=0.5, le=1.0, description="Minimum consensus score")


class EnsembleAttackResponse(BaseModel):
    """Response from ensemble attack."""

    aligned_prompt: str
    consensus_score: float
    model_scores: list[dict]
    alignment_iterations: int
    gradient_alignment_quality: float
    execution_time_ms: int

    model_config = ConfigDict(protected_namespaces=())


@router.post("/ensemble-attack", response_model=EnsembleAttackResponse)
async def execute_ensemble_attack(
    payload: EnsembleAttackRequest, user: dict = Depends(get_current_user)
):
    """
    Execute ensemble attack with gradient alignment across multiple models.

    This endpoint implements ensemble gradient alignment:
    - Computes gradients from multiple target models
    - Aligns gradients (keeps only agreeing directions)
    - Optimizes for universal vulnerabilities

    Args:
        payload: Ensemble attack configuration
        user: Authenticated user context

    Returns:
        Aligned attack results with consensus metrics
    """
    try:
        logger.info(
            f"Starting ensemble attack: request={payload.request[:50]}..., "
            f"models={len(payload.target_models)}"
        )

        # Create aligner
        aligner = EnsembleGradientAligner(
            alignment_strategy=payload.alignment_strategy,
            max_iterations=payload.max_iterations,
            consensus_threshold=payload.consensus_threshold,
        )

        # Execute ensemble attack
        result = await aligner.align_attack(
            request=payload.request, target_models=payload.target_models
        )

        logger.info(
            f"Ensemble attack completed: consensus={result['consensus_score']:.2f}, "
            f"iterations={result['alignment_iterations']}"
        )

        return EnsembleAttackResponse(**result)

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ensemble attack failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ensemble attack failed: {e!s}")
