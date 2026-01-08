"""
Background Job Service

Provides infrastructure for long-running background tasks.
Designed for future Celery integration for AutoDAN and other heavy operations.

SCALE-001: Background job system preparation for horizontal scaling.
"""

import asyncio
import contextlib
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.core.logging import logger


class JobStatus(str, Enum):
    """Status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Priority levels for background jobs."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class JobResult:
    """Result of a completed job."""

    success: bool
    data: Any = None
    error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass
class BackgroundJob:
    """Represents a background job."""

    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: JobResult | None = None
    progress: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "metadata": self.metadata,
            "result": {
                "success": self.result.success,
                "data": self.result.data,
                "error": self.result.error,
                "execution_time_seconds": self.result.execution_time_seconds,
            }
            if self.result
            else None,
        }


class BackgroundJobService:
    """
    Service for managing background jobs.

    This is a simple in-memory implementation for development.
    In production, this should be replaced with Celery + Redis.

    Usage:
        # Submit a job
        job_id = await job_service.submit_job(
            name="autodan_optimization",
            func=run_autodan,
            args=(request_data,),
            priority=JobPriority.HIGH
        )

        # Check job status
        job = await job_service.get_job(job_id)

        # Get job result
        result = await job_service.get_result(job_id)
    """

    def __init__(self, max_concurrent_jobs: int = 5):
        self.max_concurrent_jobs = max_concurrent_jobs
        self._jobs: dict[str, BackgroundJob] = {}
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_jobs: set[str] = set()
        self._worker_task: asyncio.Task | None = None
        self._shutdown = False

    async def start(self):
        """Start the background job worker."""
        if self._worker_task is None:
            self._shutdown = False
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Background job service started")

    async def stop(self):
        """Stop the background job worker."""
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        logger.info("Background job service stopped")

    async def submit_job(
        self,
        name: str,
        func: Callable[..., Coroutine],
        args: tuple = (),
        kwargs: dict | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        metadata: dict | None = None,
    ) -> str:
        """
        Submit a new background job.

        Args:
            name: Human-readable job name
            func: Async function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Job priority (higher = more urgent)
            metadata: Additional metadata to store with the job

        Returns:
            Job ID
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = BackgroundJob(id=job_id, name=name, priority=priority, metadata=metadata or {})

        self._jobs[job_id] = job

        # Add to queue with negative priority (lower number = higher priority in PriorityQueue)
        await self._queue.put((-priority.value, job_id, func, args, kwargs or {}))

        logger.info(f"Job submitted: {job_id} ({name}) with priority {priority.name}")
        return job_id

    async def get_job(self, job_id: str) -> BackgroundJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    async def get_result(self, job_id: str) -> JobResult | None:
        """Get job result by ID."""
        job = self._jobs.get(job_id)
        return job.result if job else None

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Note: Running jobs cannot be cancelled in this simple implementation.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            logger.info(f"Job cancelled: {job_id}")
            return True

        return False

    async def list_jobs(
        self, status: JobStatus | None = None, limit: int = 100
    ) -> list[BackgroundJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def update_progress(self, job_id: str, progress: float):
        """Update job progress (0.0 to 1.0)."""
        job = self._jobs.get(job_id)
        if job:
            job.progress = min(1.0, max(0.0, progress))

    async def _worker_loop(self):
        """Main worker loop that processes jobs from the queue."""
        while not self._shutdown:
            try:
                # Check if we can run more jobs
                if len(self._running_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(0.1)
                    continue

                # Get next job from queue (with timeout to allow shutdown)
                try:
                    _priority, job_id, func, args, kwargs = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job or job.status == JobStatus.CANCELLED:
                    continue

                # Start job execution
                self._running_jobs.add(job_id)
                asyncio.create_task(self._execute_job(job, func, args, kwargs))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(1.0)

    async def _execute_job(
        self, job: BackgroundJob, func: Callable[..., Coroutine], args: tuple, kwargs: dict
    ):
        """Execute a single job."""
        import time

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        start_time = time.time()

        logger.info(f"Job started: {job.id} ({job.name})")

        try:
            result_data = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            job.result = JobResult(
                success=True, data=result_data, execution_time_seconds=execution_time
            )
            job.status = JobStatus.COMPLETED
            job.progress = 1.0

            logger.info(f"Job completed: {job.id} ({job.name}) in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time

            job.result = JobResult(
                success=False, error=str(e), execution_time_seconds=execution_time
            )
            job.status = JobStatus.FAILED

            logger.error(f"Job failed: {job.id} ({job.name}): {e}")

        finally:
            job.completed_at = datetime.utcnow()
            self._running_jobs.discard(job.id)

    def get_stats(self) -> dict:
        """Get service statistics."""
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(1 for j in self._jobs.values() if j.status == status)

        return {
            "total_jobs": len(self._jobs),
            "running_jobs": len(self._running_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "queue_size": self._queue.qsize(),
            "status_counts": status_counts,
        }


# Global instance
_background_job_service: BackgroundJobService | None = None


def get_background_job_service() -> BackgroundJobService:
    """Get the background job service instance."""
    global _background_job_service
    if _background_job_service is None:
        _background_job_service = BackgroundJobService()
    return _background_job_service


async def init_background_job_service():
    """Initialize and start the background job service."""
    service = get_background_job_service()
    await service.start()
    return service


async def shutdown_background_job_service():
    """Shutdown the background job service."""
    global _background_job_service
    if _background_job_service:
        await _background_job_service.stop()
        _background_job_service = None
