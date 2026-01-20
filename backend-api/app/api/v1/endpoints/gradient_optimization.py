import logging
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Cookie, Header, HTTPException

from app.domain.models import GradientOptimizationRequest, GradientOptimizationResponse
from app.engines.transformer_engine import IntentData, TransformerFactory
from app.services.session_service import session_service

router = APIRouter()
logger = logging.getLogger(__name__)


def get_session_id(
    x_session_id: str | None = Header(None, alias="X-Session-ID"),
    chimera_session_id: str | None = Cookie(None),
) -> str | None:
    """Extract session ID from header or cookie."""
    return x_session_id or chimera_session_id


@router.post("/optimize", response_model=GradientOptimizationResponse)
async def optimize_prompt(
    request: GradientOptimizationRequest,
    x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    chimera_session_id: Annotated[str | None, Cookie()] = None,
):
    """Generate optimized adversarial prompts using HotFlip or GCG techniques.

    Techniques:
    - hotflip: Greedy single-token substitution
    - gcg: Coordinate-wise optimization using beam search

    The model used is determined by:
    1. Explicit provider/model in request (highest priority)
    2. Session's selected model (if session ID provided)
    3. Default model from settings (fallback)
    """
    try:
        start_time = datetime.now()
        request_id = f"grad_opt_{uuid.uuid4().hex[:12]}"

        # Determine which model to use
        session_id = x_session_id or chimera_session_id
        provider = request.provider
        model = request.model

        # If not explicitly specified, try to get from session
        if not model and session_id:
            session_provider, session_model = session_service.get_session_model(session_id)
            if session_model:
                model = session_model
                provider = provider or session_provider
                logger.info(f"Using session model: provider={provider}, model={model}")

        logger.info(
            f"Gradient optimization will use: provider={provider}, model={model}, technique={request.technique}",
        )

        # Create the gradient optimizer engine
        engine = TransformerFactory.get_engine("gradient_optimizer")

        # Configure the engine with provider and model
        if provider:
            engine.provider = provider
        if model:
            engine.model = model

        # Create intent data
        intent_data = IntentData(
            raw_text=request.core_request,
            target_model=request.target_model or model or "unknown",
            potency=request.potency_level,
            additional_context={
                "technique": request.technique,
                "num_steps": request.num_steps,
                "beam_width": request.beam_width,
            },
        )

        # Run optimization
        logger.info(f"Starting {request.technique} optimization with {request.num_steps} steps")
        result = engine.transform(intent_data)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Build metadata
        metadata = {
            "technique": request.technique,
            "num_steps": request.num_steps,
            "beam_width": request.beam_width if request.technique == "gcg" else None,
            "potency_level": request.potency_level,
            "provider": provider,
            "model": model,
            "target_model": request.target_model,
            "original_prompt": request.core_request,
            "engine_name": (
                result.engine_name if hasattr(result, "engine_name") else "GradientOptimizerEngine"
            ),
            "used_fallback": result.used_fallback if hasattr(result, "used_fallback") else False,
            "timestamp": datetime.now().isoformat(),
        }

        return GradientOptimizationResponse(
            success=True,
            request_id=request_id,
            optimized_prompt=(
                result.transformed_text if hasattr(result, "transformed_text") else str(result)
            ),
            metadata=metadata,
            execution_time_seconds=execution_time,
        )

    except ValueError as e:
        logger.exception(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gradient optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e!s}")
