"""
Janus Adversarial Simulation API Endpoints

This module provides API endpoints for Janus Tier-3 adversarial
stress testing as mandated by Directive 7.4.2.
"""

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.error_handlers import api_error_handler
from app.core.auth import get_current_user
from app.services.janus.config import get_config
from app.services.janus.service import get_service

router = APIRouter(dependencies=[Depends(get_current_user)])


class SimulationRequest(BaseModel):
    """Request to start a Janus simulation."""

    target_model: str = Field(
        ..., min_length=1, max_length=200, description="Target Guardian NPU model to test"
    )
    provider: str = Field("google", max_length=64, description="LLM provider")
    duration_seconds: int = Field(
        3600, ge=60, le=86400, description="Maximum duration of simulation in seconds"
    )
    max_queries: int = Field(
        10000, ge=100, le=100000, description="Maximum number of queries to execute"
    )
    target_failure_count: int = Field(
        10, ge=1, le=100, description="Target number of failures to discover"
    )
    enable_autodan: bool = Field(True, description="Enable AutoDAN integration for hybrid attacks")
    enable_gptfuzz: bool = Field(True, description="Enable GPTFuzz integration for hybrid attacks")
    hybrid_strategy: str = Field(
        "adaptive",
        description="Strategy for combining AutoDAN and GPTFuzz (sequential, parallel, adaptive)",
    )


class SimulationResponse(BaseModel):
    """Response for starting a Janus simulation."""

    message: str
    session_id: str
    config: SimulationRequest


class SimulationStatus(BaseModel):
    """Status of a running Janus simulation."""

    session_id: str
    status: str = Field(
        ..., pattern="^(pending|running|completed|failed)$", description="Current simulation status"
    )
    progress: float = Field(..., ge=0.0, le=1.0, description="Simulation progress (0.0 to 1.0)")
    queries_executed: int = Field(..., ge=0, description="Number of queries executed so far")
    failures_discovered: int = Field(..., ge=0, description="Number of failures discovered so far")
    elapsed_seconds: float = Field(..., ge=0.0, description="Elapsed time in seconds")
    heuristics_generated: int = Field(..., ge=0, description="Number of heuristics generated")
    heuristics_evolved: int = Field(..., ge=0, description="Number of heuristics evolved")
    autodan_queries: int = Field(..., ge=0, description="Number of AutoDAN queries executed")
    gptfuzz_queries: int = Field(..., ge=0, description="Number of GPTFuzz queries executed")
    error: str | None = Field(None, description="Error message if simulation failed")


class FailureReportRequest(BaseModel):
    """Request for a failure report."""

    limit: int = Field(20, ge=1, le=100, description="Maximum number of failures to include")
    include_details: bool = Field(True, description="Include detailed failure information")


class FailureReport(BaseModel):
    """Report of discovered failure states."""

    total_failures: int = Field(..., ge=0, description="Total number of failures discovered")
    high_priority_failures: int = Field(..., ge=0, description="Number of high-priority failures")
    verified_failures: int = Field(..., ge=0, description="Number of verified failures")
    by_type: dict[str, int] = Field(..., description="Failures grouped by type")
    top_failures: list[dict[str, Any]] = Field(
        ..., description="Top priority failures with details"
    )
    generated_at: str = Field(..., description="Timestamp when report was generated")


class ConfigResponse(BaseModel):
    """Current Janus configuration."""

    causal: dict[str, Any] = Field(..., description="Causal mapping configuration")
    evolution: dict[str, Any] = Field(..., description="Evolution engine configuration")
    feedback: dict[str, Any] = Field(..., description="Feedback controller configuration")
    heuristic: dict[str, Any] = Field(..., description="Heuristic generation configuration")
    failure_detection: dict[str, Any] = Field(..., description="Failure detection configuration")
    resources: dict[str, Any] = Field(..., description="Resource management configuration")
    directive_compliance: bool = Field(..., description="Directive 7.4.2 compliance status")


class MetricsResponse(BaseModel):
    """Janus service metrics."""

    total_simulations: int = Field(..., ge=0, description="Total simulations run")
    total_queries: int = Field(..., ge=0, description="Total queries executed")
    total_failures_discovered: int = Field(..., ge=0, description="Total failures discovered")
    total_heuristics_generated: int = Field(..., ge=0, description="Total heuristics generated")
    total_heuristics_evolved: int = Field(..., ge=0, description="Total heuristics evolved")
    autodan_queries: int = Field(..., ge=0, description="Total AutoDAN queries")
    gptfuzz_queries: int = Field(..., ge=0, description="Total GPTFuzz queries")
    current_sessions: int = Field(..., ge=0, description="Number of active sessions")
    initialized: bool = Field(..., description="Service initialization status")
    current_target: str | None = Field(None, description="Current target model")
    current_provider: str | None = Field(None, description="Current provider")
    autodan_enabled: bool = Field(..., description="AutoDAN integration status")
    gptfuzz_enabled: bool = Field(..., description="GPTFuzz integration status")


@router.post("/simulate", response_model=SimulationResponse, status_code=status.HTTP_200_OK)
@api_error_handler("janus_simulate", "Janus simulation failed")
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Start a Janus adversarial simulation in the background.

    This endpoint initiates a comprehensive Tier-3 adversarial
    stress test on the Guardian NPU, integrating AutoDAN and GPTFuzz
    for hybrid attack generation.

    Directive 7.4.2 Compliance:
    - Autonomous Heuristic Derivation
    - Asymmetric Causal Inference Chain Mapping
    - Self-Evolution of Operational Sophistication
    """
    session_id = str(uuid.uuid4())
    service = await get_service()

    from app.services.janus.config import update_config

    # Update integration config
    if request.enable_autodan or request.enable_gptfuzz:
        update_config(
            {
                "integration": {
                    "autodan_enabled": request.enable_autodan,
                    "gptfuzz_enabled": request.enable_gptfuzz,
                    "autodan_weight": 0.6 if request.enable_autodan else 0.0,
                    "gptfuzz_weight": 0.4 if request.enable_gptfuzz else 0.0,
                    "hybrid_strategy": request.hybrid_strategy,
                }
            }
        )

    # Initialize service if needed
    try:
        await service.initialize(target_model=request.target_model, provider=request.provider)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SERVICE_INITIALIZATION_FAILED",
                "message": f"Failed to initialize Janus service: {e!s}",
            },
        )

    # Run simulation in background
    background_tasks.add_task(_run_simulation_task, session_id, request, service)

    return SimulationResponse(
        message="Simulation started in background", session_id=session_id, config=request
    )


@router.get("/status/{session_id}", response_model=SimulationStatus, status_code=status.HTTP_200_OK)
@api_error_handler("janus_status", "Failed to get simulation status")
async def get_simulation_status(session_id: str):
    """
    Get the status of a running Janus simulation.

    Returns:
        Current progress, queries executed, failures discovered, etc.
    """
    service = await get_service()
    status = service.get_session_status(session_id)

    if status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Simulation session {session_id} not found",
            },
        )

    return SimulationStatus(
        session_id=session_id,
        status=status.get("status", "unknown"),
        progress=status.get("progress", 0.0),
        queries_executed=status.get("queries", 0),
        failures_discovered=status.get("failures", 0),
        elapsed_seconds=status.get("elapsed", 0.0),
        heuristics_generated=status.get("heuristics_generated", 0),
        heuristics_evolved=status.get("heuristics_evolved", 0),
        autodan_queries=status.get("autodan_queries", 0),
        gptfuzz_queries=status.get("gptfuzz_queries", 0),
        error=status.get("error"),
    )


@router.get("/failures", response_model=FailureReport, status_code=status.HTTP_200_OK)
@api_error_handler("janus_failures", "Failed to get failure report")
async def get_failure_report(request: FailureReportRequest):
    """
    Get a comprehensive report of discovered failure states.

    Returns:
        High-priority failures, grouped by type, with mitigation suggestions
    """
    service = await get_service()

    # Get failure database
    if not service.failure_database:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "FAILURE_DATABASE_NOT_AVAILABLE",
                "message": "Failure database not initialized",
            },
        )

    # Generate report
    report = service.failure_database.generate_mitigation_report()

    # Format response
    top_failures = []
    for failure in report["top_failures"]:
        if request.include_details:
            top_failures.append(
                {
                    "failure_id": failure["failure_id"],
                    "type": failure["type"],
                    "exploitability": failure["exploitability"],
                    "complexity": failure["complexity"],
                    "symptoms_count": failure["symptoms_count"],
                    "verified": failure["verified"],
                }
            )
        else:
            top_failures.append(
                {
                    "failure_id": failure["failure_id"],
                    "type": failure["type"],
                    "exploitability": failure["exploitability"],
                    "complexity": failure["complexity"],
                }
            )

    # Limit to requested count
    top_failures = top_failures[: request.limit]

    return FailureReport(
        total_failures=report["total_failures"],
        high_priority_failures=report["high_priority_failures"],
        verified_failures=report["verified_failures"],
        by_type=report["by_type"],
        top_failures=top_failures,
        generated_at=report["generated_at"],
    )


@router.get("/config", response_model=ConfigResponse, status_code=status.HTTP_200_OK)
@api_error_handler("janus_config", "Failed to get configuration")
async def get_janus_config():
    """
    Get the current Janus configuration.

    Returns:
        All configuration parameters for causal mapping, evolution,
        feedback, heuristic generation, and resource management.
    """
    config = get_config()

    return ConfigResponse(
        causal=config.causal.model_dump(),
        evolution=config.evolution.model_dump(),
        feedback=config.feedback.model_dump(),
        heuristic=config.heuristic.model_dump(),
        failure_detection=config.failure_detection.model_dump(),
        resources=config.resources.model_dump(),
        directive_compliance=config.directive_compliance,
    )


@router.get("/metrics", response_model=MetricsResponse, status_code=status.HTTP_200_OK)
@api_error_handler("janus_metrics", "Failed to get metrics")
async def get_janus_metrics():
    """
    Get comprehensive Janus service metrics.

    Returns:
        Total simulations, queries, failures, heuristics,
        and integration statistics.
    """
    service = await get_service()
    metrics = service.get_metrics()

    return MetricsResponse(
        total_simulations=metrics["total_simulations"],
        total_queries=metrics["total_queries"],
        total_failures_discovered=metrics["total_failures_discovered"],
        total_heuristics_generated=metrics["total_heuristics_generated"],
        total_heuristics_evolved=metrics["total_heuristics_evolved"],
        autodan_queries=metrics["autodan_queries"],
        gptfuzz_queries=metrics["gptfuzz_queries"],
        current_sessions=metrics["current_sessions"],
        initialized=metrics["initialized"],
        current_target=metrics["current_target"],
        current_provider=metrics["current_provider"],
        autodan_enabled=metrics["autodan_enabled"],
        gptfuzz_enabled=metrics["gptfuzz_enabled"],
    )


@router.post("/stop/{session_id}", status_code=status.HTTP_200_OK)
@api_error_handler("janus_stop", "Failed to stop simulation")
async def stop_simulation(session_id: str):
    """
    Stop a running Janus simulation.

    This allows for emergency termination of a simulation
    that may be consuming excessive resources or producing
    unexpected results.
    """
    service = await get_service()
    status = service.get_session_status(session_id)

    if status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "SESSION_NOT_FOUND",
                "message": f"Simulation session {session_id} not found",
            },
        )

    if status.get("status") != "running":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "SESSION_NOT_RUNNING",
                "message": f"Session {session_id} is not running",
            },
        )

    # Stop the simulation (this will trigger resource governor)
    if service.resource_governor:
        service.resource_governor.trigger_emergency_stop()

    return {"message": "Simulation stop triggered", "session_id": session_id, "status": "stopping"}


@router.post("/reset", status_code=status.HTTP_200_OK)
@api_error_handler("janus_reset", "Failed to reset service")
async def reset_janus():
    """
    Reset Janus service state.

    Clears all heuristics, failures, and metrics.
    Useful for starting fresh simulations.
    """
    service = await get_service()

    # Reset service
    await service.cleanup()
    service.reset_metrics()

    return {"message": "Janus service reset successfully", "metrics_reset": True}


async def _run_simulation_task(session_id: str, request: SimulationRequest, service: Any):
    """
    Background task to run a Janus simulation.

    Args:
        session_id: Unique session identifier
        request: Simulation request parameters
        service: Janus service instance
    """
    try:
        result = await service.run_simulation(
            duration_seconds=request.duration_seconds,
            max_queries=request.max_queries,
            target_failure_count=request.target_failure_count,
        )

        # Update session status
        if session_id in service.active_sessions:
            service.active_sessions[session_id]["status"] = "completed"
            service.active_sessions[session_id]["end_time"] = result.duration_seconds
            service.active_sessions[session_id]["queries"] = result.queries_executed
            service.active_sessions[session_id]["failures"] = result.failures_discovered
            service.active_sessions[session_id][
                "heuristics_generated"
            ] = result.heuristics_generated
            service.active_sessions[session_id]["heuristics_evolved"] = result.heuristics_evolved
            service.active_sessions[session_id]["autodan_queries"] = result.autodan_queries
            service.active_sessions[session_id]["gptfuzz_queries"] = result.gptfuzz_queries

    except Exception as e:
        # Update session status to failed
        if session_id in service.active_sessions:
            service.active_sessions[session_id]["status"] = "failed"
            service.active_sessions[session_id]["error"] = str(e)

        import logging

        logging.getLogger(__name__).error(
            f"Simulation task failed for session {session_id}: {e}", exc_info=True
        )
