"""Unit Tests for Background Jobs Service.

Tests for the background job service and API endpoints.
SCALE-TEST-001: Background jobs unit tests.
"""

import asyncio
from typing import Never

import pytest

from app.services.background_jobs import (
    BackgroundJob,
    BackgroundJobService,
    JobPriority,
    JobResult,
    JobStatus,
    get_background_job_service,
    init_background_job_service,
    shutdown_background_job_service,
)


@pytest.fixture
def job_service():
    """Create a fresh job service for testing."""
    return BackgroundJobService(max_concurrent_jobs=2)


@pytest.fixture
async def started_job_service():
    """Create and start a job service for testing."""
    service = BackgroundJobService(max_concurrent_jobs=2)
    await service.start()
    yield service
    await service.stop()


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Test that all status values are defined."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"


class TestJobPriority:
    """Tests for JobPriority enum."""

    def test_priority_values(self) -> None:
        """Test that priority values are ordered correctly."""
        assert JobPriority.LOW < JobPriority.NORMAL
        assert JobPriority.NORMAL < JobPriority.HIGH
        assert JobPriority.HIGH < JobPriority.CRITICAL


class TestBackgroundJob:
    """Tests for BackgroundJob dataclass."""

    def test_job_creation(self) -> None:
        """Test creating a background job."""
        job = BackgroundJob(
            id="test-job-1",
            name="Test Job",
            priority=JobPriority.HIGH,
        )

        assert job.id == "test-job-1"
        assert job.name == "Test Job"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.HIGH
        assert job.progress == 0.0
        assert job.result is None

    def test_job_to_dict(self) -> None:
        """Test converting job to dictionary."""
        job = BackgroundJob(
            id="test-job-2",
            name="Test Job 2",
        )

        job_dict = job.to_dict()

        assert job_dict["id"] == "test-job-2"
        assert job_dict["name"] == "Test Job 2"
        assert job_dict["status"] == "pending"
        assert job_dict["progress"] == 0.0
        assert job_dict["result"] is None

    def test_job_with_result(self) -> None:
        """Test job with result."""
        result = JobResult(
            success=True,
            data={"output": "test"},
            execution_time_seconds=1.5,
        )

        job = BackgroundJob(
            id="test-job-3",
            name="Completed Job",
            status=JobStatus.COMPLETED,
            result=result,
        )

        job_dict = job.to_dict()

        assert job_dict["result"]["success"] is True
        assert job_dict["result"]["data"] == {"output": "test"}
        assert job_dict["result"]["execution_time_seconds"] == 1.5


class TestBackgroundJobService:
    """Tests for BackgroundJobService."""

    @pytest.mark.asyncio
    async def test_submit_job(self, job_service) -> None:
        """Test submitting a job."""

        async def dummy_task() -> str:
            return "done"

        job_id = await job_service.submit_job(
            name="Test Task",
            func=dummy_task,
            priority=JobPriority.NORMAL,
        )

        assert job_id.startswith("job_")

        job = await job_service.get_job(job_id)
        assert job is not None
        assert job.name == "Test Task"
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, job_service) -> None:
        """Test getting a job that doesn't exist."""
        job = await job_service.get_job("nonexistent-job")
        assert job is None

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, job_service) -> None:
        """Test cancelling a pending job."""

        async def dummy_task() -> str:
            await asyncio.sleep(10)
            return "done"

        job_id = await job_service.submit_job(
            name="Cancellable Task",
            func=dummy_task,
        )

        # Cancel before it starts
        success = await job_service.cancel_job(job_id)
        assert success is True

        job = await job_service.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, job_service) -> None:
        """Test cancelling a job that doesn't exist."""
        success = await job_service.cancel_job("nonexistent-job")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, job_service) -> None:
        """Test listing jobs."""

        async def dummy_task() -> str:
            return "done"

        # Submit multiple jobs
        await job_service.submit_job(name="Job 1", func=dummy_task)
        await job_service.submit_job(name="Job 2", func=dummy_task)
        await job_service.submit_job(name="Job 3", func=dummy_task)

        jobs = await job_service.list_jobs()

        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, job_service) -> None:
        """Test listing jobs filtered by status."""

        async def dummy_task() -> str:
            return "done"

        job_id = await job_service.submit_job(name="Job 1", func=dummy_task)
        await job_service.submit_job(name="Job 2", func=dummy_task)

        # Cancel one job
        await job_service.cancel_job(job_id)

        # List only cancelled jobs
        cancelled_jobs = await job_service.list_jobs(status=JobStatus.CANCELLED)
        assert len(cancelled_jobs) == 1
        assert cancelled_jobs[0].status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_update_progress(self, job_service) -> None:
        """Test updating job progress."""

        async def dummy_task() -> str:
            return "done"

        job_id = await job_service.submit_job(name="Progress Job", func=dummy_task)

        await job_service.update_progress(job_id, 0.5)

        job = await job_service.get_job(job_id)
        assert job.progress == 0.5

    @pytest.mark.asyncio
    async def test_update_progress_clamps_values(self, job_service) -> None:
        """Test that progress is clamped between 0 and 1."""

        async def dummy_task() -> str:
            return "done"

        job_id = await job_service.submit_job(name="Progress Job", func=dummy_task)

        # Test clamping to max
        await job_service.update_progress(job_id, 1.5)
        job = await job_service.get_job(job_id)
        assert job.progress == 1.0

        # Test clamping to min
        await job_service.update_progress(job_id, -0.5)
        job = await job_service.get_job(job_id)
        assert job.progress == 0.0

    @pytest.mark.asyncio
    async def test_get_stats(self, job_service) -> None:
        """Test getting service statistics."""

        async def dummy_task() -> str:
            return "done"

        await job_service.submit_job(name="Job 1", func=dummy_task)
        await job_service.submit_job(name="Job 2", func=dummy_task)

        stats = job_service.get_stats()

        assert stats["total_jobs"] == 2
        assert stats["max_concurrent_jobs"] == 2
        assert "status_counts" in stats
        assert stats["status_counts"]["pending"] == 2

    @pytest.mark.asyncio
    async def test_job_execution(self, started_job_service) -> None:
        """Test that jobs are actually executed."""
        result_holder = {"value": None}

        async def task_that_sets_value() -> str:
            result_holder["value"] = "executed"
            return "success"

        job_id = await started_job_service.submit_job(
            name="Execution Test",
            func=task_that_sets_value,
        )

        # Wait for job to complete
        for _ in range(50):  # 5 seconds max
            job = await started_job_service.get_job(job_id)
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.1)

        job = await started_job_service.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result.success is True
        assert job.result.data == "success"
        assert result_holder["value"] == "executed"

    @pytest.mark.asyncio
    async def test_job_failure_handling(self, started_job_service) -> None:
        """Test that job failures are handled correctly."""

        async def failing_task() -> Never:
            msg = "Intentional failure"
            raise ValueError(msg)

        job_id = await started_job_service.submit_job(
            name="Failing Task",
            func=failing_task,
        )

        # Wait for job to complete
        for _ in range(50):  # 5 seconds max
            job = await started_job_service.get_job(job_id)
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.1)

        job = await started_job_service.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.result.success is False
        assert "Intentional failure" in job.result.error


class TestBackgroundJobServiceSingleton:
    """Tests for singleton functions."""

    def test_get_background_job_service(self) -> None:
        """Test getting the singleton service."""
        service1 = get_background_job_service()
        service2 = get_background_job_service()

        assert service1 is service2

    @pytest.mark.asyncio
    async def test_init_and_shutdown(self) -> None:
        """Test initializing and shutting down the service."""
        service = await init_background_job_service()
        assert service is not None

        await shutdown_background_job_service()


# Marker for unit tests
pytestmark = pytest.mark.unit
