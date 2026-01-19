"""
API endpoints for AutoDAN Hierarchical Genetic Search.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import get_current_user
from app.services.autodan_advanced.hierarchical_search import HierarchicalGeneticSearchService
from app.services.autodan_advanced.models import (
    HierarchicalSearchRequest,
    HierarchicalSearchResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/hierarchical-search", response_model=HierarchicalSearchResponse)
async def execute_hierarchical_search(
    payload: HierarchicalSearchRequest, user: dict = Depends(get_current_user)
):
    """
    Execute hierarchical genetic search with bi-level evolution.

    This endpoint implements the advanced AutoDAN framework with:
    - Meta-strategy evolution (Level 1)
    - Prompt instantiation evolution (Level 2)
    - Map-Elites diversity maintenance
    - Dynamic archive-based selection
    - Fitness sharing for unique attack vectors

    Args:
        payload: Hierarchical search configuration
        user: Authenticated user context

    Returns:
        Search results with best prompt and generation history
    """
    try:
        logger.info(
            f"Starting hierarchical search: request={payload.request[:50]}..., "
            f"population={payload.population_size}, generations={payload.generations}"
        )

        # Create service instance
        service = HierarchicalGeneticSearchService()

        # Execute search
        result = await service.execute_search(payload)

        logger.info(
            f"Hierarchical search completed: best_score={result.best_score}, "
            f"execution_time={result.execution_time_ms}ms"
        )

        return result

    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Hierarchical search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hierarchical search failed: {e!s}") from e
