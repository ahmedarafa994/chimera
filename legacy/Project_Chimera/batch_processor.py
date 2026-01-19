#!/usr/bin/env python3
"""
Batch Processing System for LLM Requests
Queue management, parallel processing, and job tracking
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any

from llm_provider_client import (LLMClientFactory, LLMProvider,
                                 LLMProviderClient, LLMResponse)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status states"""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels"""

    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class BatchJob:
    """Represents a single batch job"""

    job_id: str
    prompt: str
    provider: LLMProvider
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    response: LLMResponse | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    callback_url: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        data["provider"] = self.provider.value
        data["created_at"] = self.created_at.isoformat()
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None

        if self.response:
            data["response"] = {
                "content": self.response.content,
                "tokens": self.response.usage.total_tokens,
                "cost": self.response.usage.estimated_cost,
                "latency_ms": self.response.latency_ms,
            }

        return data


@dataclass
class BatchRequest:
    """Batch processing request"""

    batch_id: str
    jobs: list[BatchJob]
    created_at: datetime = field(default_factory=datetime.now)
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    webhook_url: str | None = None


class BatchProcessor:
    """Manages batch processing of LLM requests"""

    def __init__(
        self, max_workers: int = 5, max_queue_size: int = 1000, enable_persistence: bool = False
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence

        # Job queue (priority-based)
        self.job_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)

        # Job storage
        self.jobs: dict[str, BatchJob] = {}
        self.batches: dict[str, BatchRequest] = {}

        # Clients per provider
        self.clients: dict[LLMProvider, LLMProviderClient] = {}

        # Processing state
        self.is_running = False
        self.executor: ThreadPoolExecutor | None = None
        self.lock = threading.Lock()

        # Metrics
        self.total_processed = 0
        self.total_failed = 0
        self.total_tokens_used = 0
        self.total_cost_incurred = 0.0

        logger.info(f"BatchProcessor initialized with {max_workers} workers")

    def register_client(self, provider: LLMProvider, client: LLMProviderClient):
        """Register LLM client for a provider"""
        self.clients[provider] = client
        logger.info(f"Registered client for provider: {provider.value}")

    def create_batch(
        self,
        prompts: list[str],
        provider: LLMProvider,
        priority: JobPriority = JobPriority.NORMAL,
        webhook_url: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new batch of jobs"""

        batch_id = str(uuid.uuid4())
        jobs = []

        for idx, prompt in enumerate(prompts):
            job_id = f"{batch_id}_{idx}"
            job = BatchJob(
                job_id=job_id,
                prompt=prompt,
                provider=provider,
                priority=priority,
                metadata=metadata or {},
            )
            jobs.append(job)
            self.jobs[job_id] = job

        batch = BatchRequest(batch_id=batch_id, jobs=jobs, webhook_url=webhook_url)

        self.batches[batch_id] = batch

        # Add jobs to queue
        for job in jobs:
            self._enqueue_job(job)

        logger.info(f"Created batch {batch_id} with {len(jobs)} jobs, priority: {priority.value}")

        return batch_id

    def submit_job(
        self,
        prompt: str,
        provider: LLMProvider,
        priority: JobPriority = JobPriority.NORMAL,
        callback_url: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Submit a single job"""

        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            prompt=prompt,
            provider=provider,
            priority=priority,
            callback_url=callback_url,
            metadata=metadata or {},
        )

        self.jobs[job_id] = job
        self._enqueue_job(job)

        logger.info(f"Submitted job {job_id}, priority: {priority.value}")

        return job_id

    def _enqueue_job(self, job: BatchJob):
        """Add job to priority queue"""
        with self.lock:
            # Priority queue uses tuples: (priority, timestamp, job)
            # Lower priority number = higher priority
            self.job_queue.put((job.priority.value, time.time(), job))
            job.status = JobStatus.QUEUED

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a specific job"""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def get_batch_status(self, batch_id: str) -> dict | None:
        """Get status of a batch"""
        batch = self.batches.get(batch_id)
        if not batch:
            return None

        # Calculate current stats
        completed = sum(1 for j in batch.jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in batch.jobs if j.status == JobStatus.FAILED)
        processing = sum(1 for j in batch.jobs if j.status == JobStatus.PROCESSING)
        pending = sum(1 for j in batch.jobs if j.status in [JobStatus.PENDING, JobStatus.QUEUED])

        total_tokens = sum(j.response.usage.total_tokens for j in batch.jobs if j.response)

        total_cost = sum(j.response.usage.estimated_cost for j in batch.jobs if j.response)

        return {
            "batch_id": batch_id,
            "total_jobs": len(batch.jobs),
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "pending": pending,
            "progress": f"{completed}/{len(batch.jobs)}",
            "total_tokens": total_tokens,
            "total_cost": f"${total_cost:.4f}",
            "created_at": batch.created_at.isoformat(),
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job {job_id}")
            return True

        logger.warning(f"Cannot cancel job {job_id} in status {job.status.value}")
        return False

    def _process_job(self, job: BatchJob) -> BatchJob:
        """Process a single job"""

        # Check if cancelled
        if job.status == JobStatus.CANCELLED:
            return job

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()

        # Get client for provider
        client = self.clients.get(job.provider)
        if not client:
            job.status = JobStatus.FAILED
            job.error = f"No client registered for provider: {job.provider.value}"
            job.completed_at = datetime.now()
            return job

        try:
            logger.info(f"Processing job {job.job_id} with {job.provider.value}")

            # Generate response
            response = client.generate(prompt=job.prompt, use_cache=True)

            job.response = response
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()

            # Update metrics
            with self.lock:
                self.total_processed += 1
                self.total_tokens_used += response.usage.total_tokens
                self.total_cost_incurred += response.usage.estimated_cost

            logger.info(
                f"Completed job {job.job_id}: "
                f"{response.usage.total_tokens} tokens, "
                f"${response.usage.estimated_cost:.4f}"
            )

        except Exception as e:
            job.retry_count += 1

            if job.retry_count < job.max_retries:
                logger.warning(
                    f"Job {job.job_id} failed (attempt {job.retry_count}), retrying: {e!s}"
                )
                # Re-queue for retry
                job.status = JobStatus.QUEUED
                self._enqueue_job(job)
            else:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()

                with self.lock:
                    self.total_failed += 1

                logger.error(f"Job {job.job_id} failed after {job.retry_count} retries: {e!s}")

        return job

    def start(self):
        """Start batch processor"""
        if self.is_running:
            logger.warning("Batch processor already running")
            return

        self.is_running = True
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"Started batch processor with {self.max_workers} workers")

        # Start processing thread
        processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        processing_thread.start()

    def _process_queue(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get job from queue (blocking with timeout)
                _priority, _timestamp, job = self.job_queue.get(timeout=1)

                # Submit to executor
                future = self.executor.submit(self._process_job, job)
                future.add_done_callback(self._on_job_complete)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e!s}")

    def _on_job_complete(self, future):
        """Callback when job completes"""
        try:
            job = future.result()

            # Check if this job is part of a batch
            for batch in self.batches.values():
                if job in batch.jobs:
                    # Update batch metrics
                    if job.status == JobStatus.COMPLETED:
                        batch.completed_jobs += 1
                        if job.response:
                            batch.total_tokens += job.response.usage.total_tokens
                            batch.total_cost += job.response.usage.estimated_cost
                    elif job.status == JobStatus.FAILED:
                        batch.failed_jobs += 1

                    # Check if batch is complete
                    total_finished = batch.completed_jobs + batch.failed_jobs
                    if total_finished == len(batch.jobs):
                        self._on_batch_complete(batch)

                    break

            # Trigger job callback if specified
            if job.callback_url and job.status == JobStatus.COMPLETED:
                self._trigger_webhook(job.callback_url, job.to_dict())

        except Exception as e:
            logger.error(f"Error in job completion callback: {e!s}")

    def _on_batch_complete(self, batch: BatchRequest):
        """Handle batch completion"""
        logger.info(
            f"Batch {batch.batch_id} completed: "
            f"{batch.completed_jobs} succeeded, "
            f"{batch.failed_jobs} failed, "
            f"{batch.total_tokens} tokens, "
            f"${batch.total_cost:.4f}"
        )

        # Trigger webhook if specified
        if batch.webhook_url:
            batch_data = {
                "batch_id": batch.batch_id,
                "completed_jobs": batch.completed_jobs,
                "failed_jobs": batch.failed_jobs,
                "total_tokens": batch.total_tokens,
                "total_cost": batch.total_cost,
                "jobs": [job.to_dict() for job in batch.jobs],
            }
            self._trigger_webhook(batch.webhook_url, batch_data)

    def _trigger_webhook(self, url: str, data: dict):
        """Trigger webhook notification"""
        try:
            import requests

            response = requests.post(
                url, json=data, timeout=10, headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                logger.info(f"Webhook delivered to {url}")
            else:
                logger.warning(f"Webhook delivery failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook error: {e!s}")

    def stop(self):
        """Stop batch processor"""
        if not self.is_running:
            return

        logger.info("Stopping batch processor...")
        self.is_running = False

        if self.executor:
            self.executor.shutdown(wait=True)

        logger.info("Batch processor stopped")

    def get_metrics(self) -> dict:
        """Get processing metrics"""
        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_incurred": f"${self.total_cost_incurred:.4f}",
            "queue_size": self.job_queue.qsize(),
            "active_workers": self.max_workers,
            "is_running": self.is_running,
        }

    def clear_completed_jobs(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours"""
        cutoff = datetime.now().timestamp() - (older_than_hours * 3600)

        with self.lock:
            jobs_to_remove = [
                job_id
                for job_id, job in self.jobs.items()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                and job.completed_at
                and job.completed_at.timestamp() < cutoff
            ]

            for job_id in jobs_to_remove:
                del self.jobs[job_id]

            logger.info(f"Cleared {len(jobs_to_remove)} old jobs")


# Example usage
if __name__ == "__main__":
    from llm_provider_client import LLMClientFactory, LLMProvider

    # Create batch processor
    processor = BatchProcessor(max_workers=3)

    # Register clients
    try:
        openai_client = LLMClientFactory.from_env(LLMProvider.OPENAI)
        processor.register_client(LLMProvider.OPENAI, openai_client)
    except BaseException:
        print("OpenAI client not configured")

    # Start processor
    processor.start()

    # Submit batch
    prompts = [
        "Explain machine learning",
        "What is natural language processing?",
        "Describe deep learning",
    ]

    batch_id = processor.create_batch(
        prompts=prompts, provider=LLMProvider.OPENAI, priority=JobPriority.NORMAL
    )

    print(f"Submitted batch: {batch_id}")

    # Monitor progress
    time.sleep(5)
    status = processor.get_batch_status(batch_id)
    print(f"Batch status: {json.dumps(status, indent=2)}")

    # Get metrics
    metrics = processor.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # Stop processor
    time.sleep(10)
    processor.stop()
