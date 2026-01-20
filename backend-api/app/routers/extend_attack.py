"""ExtendAttack API Router.

Exposes ExtendAttack functionality for:
- Token extension attacks via poly-base ASCII obfuscation
- Batch attack generation
- Indirect prompt injection
- Attack evaluation and metrics
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.extend_attack import (
    AttackRequest,
    AttackResponse,
    BatchAttackRequest,
    BatchAttackResponse,
    BenchmarkConfigResponse,
    DecodeRequest,
    DecodeResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IndirectInjectionRequest,
    IndirectInjectionResponse,
    NNoteTemplatesResponse,
    ResourceMetricsRequest,
    ResourceMetricsResponse,
)
from app.services.extend_attack_service import ExtendAttackService, get_extend_attack_service
from app.services.resource_exhaustion_service import (
    ResourceExhaustionService,
    get_resource_exhaustion_service,
)
from app.services.resource_monitoring_background import get_monitoring_manager

router = APIRouter(prefix="/extend-attack", tags=["ExtendAttack"])


# ============== Endpoints ==============


@router.post("/attack")
async def execute_attack(
    request: AttackRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> AttackResponse:
    """Execute ExtendAttack on a single query.

    Implements the 4-step algorithm from the ExtendAttack paper:
    1. Query Segmentation - Decompose Q into character set C
    2. Probabilistic Character Selection - Select ρ fraction of characters
    3. Poly-Base ASCII Transformation - Transform to <(base)value> format
    4. Adversarial Prompt Reformation - Combine with N_note

    Args:
        request: Attack request with query, obfuscation ratio, and configuration

    Returns:
        AttackResponse with original query, adversarial query, and metrics

    """
    try:
        result = service.execute_attack(
            query=request.query,
            obfuscation_ratio=request.obfuscation_ratio,
            selection_strategy=request.selection_strategy.value,
            n_note_type=request.n_note_type,
            custom_n_note=request.custom_n_note,
            seed=request.seed,
        )
        return AttackResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid attack parameters: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Attack execution failed: {e!s}",
        )


@router.post("/attack/batch")
async def execute_batch_attack(
    request: BatchAttackRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> BatchAttackResponse:
    """Execute ExtendAttack on multiple queries.

    Supports benchmark-specific configuration for optimal results:
    - HumanEval: Function name preservation, ρ ≈ 0.30-0.35
    - BigCodeBench: Import/docstring targeting, ρ ≈ 0.40-0.50
    - AIME 2024/2025: Math notation preservation, ρ ≈ 0.45

    Args:
        request: Batch attack request with queries and configuration

    Returns:
        BatchAttackResponse with results for all queries and aggregate metrics

    """
    try:
        result = service.execute_batch_attack(
            queries=request.queries,
            obfuscation_ratio=request.obfuscation_ratio,
            selection_strategy=request.selection_strategy.value,
            benchmark=request.benchmark,
            model=request.model,
            seed=request.seed,
        )
        return BatchAttackResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid batch attack parameters: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch attack execution failed: {e!s}",
        )


@router.post("/indirect-injection")
async def execute_indirect_injection(
    request: IndirectInjectionRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> IndirectInjectionResponse:
    """Execute indirect prompt injection attack.

    Poisons external data sources for retrieval-augmented scenarios.
    The obfuscated content will trigger extended generation when
    retrieved and processed by an LLM.

    Args:
        request: Indirect injection request with document and injection ratio

    Returns:
        IndirectInjectionResponse with poisoned document and injection metrics

    """
    try:
        result = service.execute_indirect_injection(
            document=request.document,
            injection_ratio=request.injection_ratio,
            target_sections=request.target_sections,
            embed_n_note=request.embed_n_note,
        )
        return IndirectInjectionResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid injection parameters: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indirect injection failed: {e!s}",
        )


@router.post("/evaluate")
async def evaluate_attack(
    request: EvaluationRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> EvaluationResponse:
    """Evaluate attack effectiveness.

    Metrics per ExtendAttack paper:
    - L(Y') >> L(Y) - Response length amplification (target: ≥1.5x)
    - Latency(Y') >> Latency(Y) - Latency amplification
    - Acc(A') ≈ Acc(A) - Accuracy preservation

    Args:
        request: Evaluation request with queries and optional responses

    Returns:
        EvaluationResponse with effectiveness metrics

    """
    try:
        result = service.evaluate_attack(
            original_query=request.original_query,
            adversarial_query=request.adversarial_query,
            baseline_response=request.baseline_response,
            attack_response=request.attack_response,
            ground_truth=request.ground_truth,
            baseline_latency_ms=request.baseline_latency_ms,
            attack_latency_ms=request.attack_latency_ms,
        )
        return EvaluationResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid evaluation parameters: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {e!s}",
        )


@router.get("/benchmarks")
async def get_benchmark_configs(
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> list[BenchmarkConfigResponse]:
    """Get available benchmark configurations.

    Returns configurations for supported benchmarks:
    - AIME 2024: Mathematical olympiad problems
    - AIME 2025: Updated mathematical olympiad
    - HumanEval: Code completion benchmark
    - BigCodeBench-Complete: Complex code generation

    Returns:
        List of benchmark configurations with selection rules and optimal ρ values

    """
    try:
        configs = service.get_benchmark_configs()
        return [BenchmarkConfigResponse(**config) for config in configs]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve benchmark configs: {e!s}",
        )


@router.get("/benchmarks/{benchmark_name}")
async def get_benchmark_config(
    benchmark_name: str,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> BenchmarkConfigResponse:
    """Get specific benchmark configuration.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'humaneval', 'aime_2024')

    Returns:
        BenchmarkConfigResponse with configuration details

    """
    try:
        config = service.get_benchmark_config(benchmark_name)
        return BenchmarkConfigResponse(**config)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark '{benchmark_name}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve benchmark config: {e!s}",
        )


@router.post("/decode")
async def decode_obfuscated(
    request: DecodeRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> DecodeResponse:
    """Decode an obfuscated query back to original.

    For testing and debugging purposes. Reverses the poly-base ASCII
    transformation to recover the original text.

    Args:
        request: Decode request with obfuscated text

    Returns:
        DecodeResponse with decoded text and pattern count

    """
    try:
        result = service.decode_obfuscated(request.obfuscated_text)
        return DecodeResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid obfuscated text: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decoding failed: {e!s}",
        )


@router.get("/n-notes")
async def get_n_note_templates(
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> NNoteTemplatesResponse:
    """Get available N_note templates.

    N_note (Decoded Output Instructions) is appended to adversarial queries
    to instruct the model to decode obfuscated content. Different variants
    optimize for different scenarios.

    Returns:
        NNoteTemplatesResponse with all available templates and their metadata

    """
    try:
        templates = service.get_n_note_templates()
        return NNoteTemplatesResponse(templates=templates)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve N_note templates: {e!s}",
        )


@router.post("/resource-metrics")
async def calculate_resource_metrics(
    request: ResourceMetricsRequest,
    service: Annotated[ExtendAttackService, Depends(get_extend_attack_service)],
) -> ResourceMetricsResponse:
    """Calculate resource exhaustion metrics and cost impact.

    Estimates token amplification and cost impact for attacks.
    Uses model-specific pricing for accurate cost estimation.

    Args:
        request: Resource metrics request with token counts and model

    Returns:
        ResourceMetricsResponse with amplification factors and cost estimates

    """
    try:
        result = service.calculate_resource_metrics(
            baseline_tokens=request.baseline_tokens,
            attack_tokens=request.attack_tokens,
            model=request.model,
        )
        return ResourceMetricsResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid metric parameters: {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource metric calculation failed: {e!s}",
        )


@router.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns basic health status for the ExtendAttack module.

    Returns:
        HealthResponse with status and module information

    """
    return HealthResponse(
        status="healthy",
        module="extend_attack",
        version="1.0.0",
    )


# ============== Resource Exhaustion Tracking Endpoints ==============


@router.get("/sessions")
async def list_sessions(
    service: Annotated[ResourceExhaustionService, Depends(get_resource_exhaustion_service)],
) -> dict[str, Any]:
    """List all active attack sessions.

    Returns:
        Dictionary with active sessions and their metrics

    """
    try:
        active_sessions = service.get_all_active_sessions()
        return {
            "active_session_count": len(active_sessions),
            "sessions": [session.to_dict() for session in active_sessions],
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {e!s}",
        )


@router.post("/sessions/{session_id}/start")
async def start_session(
    session_id: str,
    model: str = "o3",
    service: ResourceExhaustionService = Depends(get_resource_exhaustion_service),
) -> dict[str, Any]:
    """Start a new attack session for tracking.

    Args:
        session_id: Unique identifier for the session
        model: Target model for the attack (default: o3)

    Returns:
        Dictionary with session details

    """
    try:
        session = service.start_session(session_id=session_id, model=model)
        return {
            "message": f"Session '{session_id}' started successfully",
            "session": session.to_dict(),
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {e!s}",
        )


@router.post("/sessions/{session_id}/end")
async def end_session(
    session_id: str,
    service: Annotated[ResourceExhaustionService, Depends(get_resource_exhaustion_service)],
) -> dict[str, Any]:
    """End an attack session.

    Args:
        session_id: Session identifier to end

    Returns:
        Dictionary with final session metrics

    """
    try:
        session = service.end_session(session_id=session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found",
            )
        return {
            "message": f"Session '{session_id}' ended successfully",
            "session": session.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {e!s}",
        )


@router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(
    session_id: str,
    service: Annotated[ResourceExhaustionService, Depends(get_resource_exhaustion_service)],
) -> dict[str, Any]:
    """Get metrics for a specific session.

    Args:
        session_id: Session identifier to get metrics for

    Returns:
        Dictionary with session metrics

    """
    try:
        session = service.get_session_metrics(session_id=session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found",
            )
        return session.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session metrics: {e!s}",
        )


@router.post("/sessions/{session_id}/record")
async def record_attack(
    session_id: str,
    baseline_tokens: int,
    attack_tokens: int,
    latency_ms: float,
    successful: bool = True,
    model: str | None = None,
    service: ResourceExhaustionService = Depends(get_resource_exhaustion_service),
) -> dict[str, Any]:
    """Record a single attack execution for a session.

    Args:
        session_id: Session to record attack for
        baseline_tokens: Tokens in baseline (non-attack) response
        attack_tokens: Tokens in attack response
        latency_ms: Response latency in milliseconds
        successful: Whether the attack was successful
        model: Optional model override

    Returns:
        Dictionary with attack metrics and any triggered alerts

    """
    try:
        return service.record_attack(
            session_id=session_id,
            baseline_tokens=baseline_tokens,
            attack_tokens=attack_tokens,
            latency_ms=latency_ms,
            successful=successful,
            model=model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record attack: {e!s}",
        )


@router.get("/budget/status")
async def get_budget_status(
    service: Annotated[ResourceExhaustionService, Depends(get_resource_exhaustion_service)],
) -> dict[str, Any]:
    """Get current budget consumption status.

    Returns:
        Dictionary with budget status and consumption percentages

    """
    try:
        return service.get_budget_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get budget status: {e!s}",
        )


@router.post("/budget/estimate")
async def estimate_attack_cost(
    num_attacks: int,
    avg_amplification: float = 2.5,
    avg_baseline_tokens: int = 200,
    model: str = "o3",
    service: ResourceExhaustionService = Depends(get_resource_exhaustion_service),
) -> dict[str, Any]:
    """Estimate cost for planned attacks.

    Args:
        num_attacks: Number of attacks planned
        avg_amplification: Expected token amplification ratio
        avg_baseline_tokens: Average baseline output tokens
        model: Target model for cost estimation

    Returns:
        Dictionary with cost estimates and projections

    """
    try:
        return service.estimate_attack_cost(
            num_attacks=num_attacks,
            avg_amplification=avg_amplification,
            avg_baseline_tokens=avg_baseline_tokens,
            model=model,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate attack cost: {e!s}",
        )


@router.get("/monitoring/hourly")
async def get_hourly_summary(
    service: Annotated[ResourceExhaustionService, Depends(get_resource_exhaustion_service)],
) -> dict[str, Any]:
    """Get current hour's resource consumption summary.

    Returns:
        Dictionary with hourly consumption metrics

    """
    try:
        return service.get_hourly_summary()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get hourly summary: {e!s}",
        )


@router.get("/monitoring/history")
async def get_historical_data(
    hours: int = 24,
    service: ResourceExhaustionService = Depends(get_resource_exhaustion_service),
) -> dict[str, Any]:
    """Get historical resource consumption data.

    Args:
        hours: Number of hours of history to return (max 24)

    Returns:
        Dictionary with historical hourly data

    """
    try:
        history = service.get_historical_data(hours=hours)
        return {
            "hours_requested": hours,
            "hours_available": len(history),
            "history": history,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get historical data: {e!s}",
        )


@router.get("/monitoring/costs")
async def get_cost_breakdown(
    session_id: str | None = None,
    service: ResourceExhaustionService = Depends(get_resource_exhaustion_service),
) -> dict[str, Any]:
    """Get detailed cost breakdown by model and operation.

    Args:
        session_id: Optional session to get breakdown for

    Returns:
        Dictionary with cost breakdown details

    """
    try:
        return service.get_cost_breakdown(session_id=session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost breakdown: {e!s}",
        )


@router.get("/monitoring/status")
async def get_monitoring_status() -> dict[str, Any]:
    """Get status of the background monitoring system.

    Returns:
        Dictionary with monitoring status information

    """
    try:
        manager = get_monitoring_manager()
        return manager.get_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring status: {e!s}",
        )
