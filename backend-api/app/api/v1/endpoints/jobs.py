"""Background Jobs API Endpoint.

Provides endpoints for managing and monitoring background jobs.
SCALE-002: Background job management API.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.auth import TokenPayload, get_current_user
from app.services.background_jobs import (
    BackgroundJobService,
    JobPriority,
    JobStatus,
    get_background_job_service,
)

router = APIRouter(prefix="/jobs", tags=["background-jobs"])


# Request/Response Models
class JobSubmitRequest(BaseModel):
    """Request to submit a background job."""

    name: str = Field(..., description="Job name")
    job_type: str = Field(..., description="Type of job to run")
    parameters: dict = Field(default_factory=dict, description="Job parameters")
    priority: str = Field(default="normal", description="Job priority: low, normal, high, critical")


class JobResponse(BaseModel):
    """Response containing job information."""

    id: str
    name: str
    status: str
    priority: int
    created_at: str
    started_at: str | None
    completed_at: str | None
    progress: float
    metadata: dict
    result: dict | None


class JobListResponse(BaseModel):
    """Response containing list of jobs."""

    jobs: list[JobResponse]
    total: int


class JobStatsResponse(BaseModel):
    """Response containing job service statistics."""

    total_jobs: int
    running_jobs: int
    max_concurrent_jobs: int
    queue_size: int
    status_counts: dict


def get_job_service() -> BackgroundJobService:
    """Dependency for getting job service."""
    return get_background_job_service()


def parse_priority(priority_str: str) -> JobPriority:
    """Parse priority string to enum."""
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "critical": JobPriority.CRITICAL,
    }
    return priority_map.get(priority_str.lower(), JobPriority.NORMAL)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status_filter: str | None = None,
    limit: int = 100,
    service: BackgroundJobService = Depends(get_job_service),
    _user: TokenPayload = Depends(get_current_user),
):
    """List background jobs.

    Optionally filter by status: pending, running, completed, failed, cancelled
    """
    status_enum = None
    if status_filter:
        try:
            status_enum = JobStatus(status_filter.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}. Valid values: pending, running, completed, failed, cancelled",
            )

    jobs = await service.list_jobs(status=status_enum, limit=limit)

    return JobListResponse(
        jobs=[
            JobResponse(
                id=job.id,
                name=job.name,
                status=job.status.value,
                priority=job.priority.value,
                created_at=job.created_at.isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                progress=job.progress,
                metadata=job.metadata,
                result=(
                    {
                        "success": job.result.success,
                        "data": job.result.data,
                        "error": job.result.error,
                        "execution_time_seconds": job.result.execution_time_seconds,
                    }
                    if job.result
                    else None
                ),
            )
            for job in jobs
        ],
        total=len(jobs),
    )


@router.get("/stats", response_model=JobStatsResponse)
async def get_job_stats(
    service: Annotated[BackgroundJobService, Depends(get_job_service)],
    _user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Get background job service statistics."""
    stats = service.get_stats()
    return JobStatsResponse(**stats)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    service: Annotated[BackgroundJobService, Depends(get_job_service)],
    _user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Get a specific job by ID."""
    job = await service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return JobResponse(
        id=job.id,
        name=job.name,
        status=job.status.value,
        priority=job.priority.value,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        metadata=job.metadata,
        result=(
            {
                "success": job.result.success,
                "data": job.result.data,
                "error": job.result.error,
                "execution_time_seconds": job.result.execution_time_seconds,
            }
            if job.result
            else None
        ),
    )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    service: Annotated[BackgroundJobService, Depends(get_job_service)],
    _user: Annotated[TokenPayload, Depends(get_current_user)],
):
    """Cancel a pending job.

    Note: Running jobs cannot be cancelled.
    """
    success = await service.cancel_job(job_id)

    if not success:
        job = await service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found: {job_id}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job.status.value}",
        )

    return {"success": True, "message": f"Job {job_id} cancelled"}
