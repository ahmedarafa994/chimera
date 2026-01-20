"""Processing Queue Manager.

Async processing queue that honors provider binding for AI operations.
Supports:
- Provider-bound task queuing
- Priority-based processing
- Worker management per provider
- Queue status monitoring
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a queued task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class QueuedTask:
    """Represents a task in the processing queue."""

    task_id: str
    operation: Callable
    provider: str | None
    priority: int
    created_at: datetime
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None
    kwargs: dict = field(default_factory=dict)

    def __lt__(self, other: "QueuedTask") -> bool:
        """Compare by priority (higher first) then by creation time."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.created_at < other.created_at  # Earlier first

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": (self.started_at.isoformat() if self.started_at else None),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "error": self.error,
        }


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


class ProviderBoundQueue:
    """Processing queue that honors provider binding.

    Features:
    - Enqueue operations with optional provider binding
    - Priority-based task processing
    - Per-provider worker management
    - Task status tracking
    - Queue statistics

    Example:
        queue = ProviderBoundQueue(default_provider="openai")
        await queue.start()

        # Enqueue a task
        task_id = await queue.enqueue(
            operation=my_llm_call,
            provider="gemini",
            priority=TaskPriority.HIGH.value,
        )

        # Wait for result
        result = await queue.wait_for_task(task_id)

        await queue.stop()

    """

    def __init__(
        self,
        default_provider: str | None = None,
        max_workers_per_provider: int = 3,
        max_queue_size: int = 1000,
    ) -> None:
        """Initialize provider-bound queue.

        Args:
            default_provider: Default provider for tasks without binding
            max_workers_per_provider: Max concurrent workers per provider
            max_queue_size: Maximum queue size per provider

        """
        self._default_provider = default_provider
        self._max_workers = max_workers_per_provider
        self._max_queue_size = max_queue_size

        # Per-provider queues
        self._queues: dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue(maxsize=max_queue_size),
        )

        # Task tracking
        self._tasks: dict[str, QueuedTask] = {}
        self._task_events: dict[str, asyncio.Event] = {}

        # Worker management
        self._workers: dict[str, list[asyncio.Task]] = defaultdict(list)
        self._running = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_enqueued": 0,
            "total_completed": 0,
            "total_failed": 0,
        }

    async def start(self) -> None:
        """Start the queue processing workers."""
        if self._running:
            logger.warning("Queue already running")
            return

        self._running = True
        logger.info("Processing queue started")

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the queue and wait for workers to finish.

        Args:
            timeout: Maximum time to wait for workers

        """
        self._running = False

        # Cancel all workers
        all_workers = []
        for workers in self._workers.values():
            all_workers.extend(workers)

        for worker in all_workers:
            worker.cancel()

        if all_workers:
            await asyncio.wait(
                all_workers,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

        self._workers.clear()
        logger.info("Processing queue stopped")

    async def enqueue(
        self,
        operation: Callable[..., Any],
        provider: str | None = None,
        priority: int = TaskPriority.NORMAL.value,
        **kwargs,
    ) -> str:
        """Enqueue an operation with provider binding.

        Args:
            operation: Async callable to execute
            provider: Provider to bind task to (uses default if not specified)
            priority: Task priority (higher = processed first)
            **kwargs: Arguments to pass to operation

        Returns:
            Task ID for tracking

        Raises:
            RuntimeError: If queue is not running
            asyncio.QueueFull: If queue is at capacity

        """
        if not self._running:
            msg = "Queue is not running"
            raise RuntimeError(msg)

        # Resolve provider
        bound_provider = provider or self._default_provider or "default"

        # Create task
        task_id = str(uuid.uuid4())
        task = QueuedTask(
            task_id=task_id,
            operation=operation,
            provider=bound_provider,
            priority=priority,
            created_at=datetime.utcnow(),
            kwargs=kwargs,
        )

        # Create completion event
        self._task_events[task_id] = asyncio.Event()

        # Store task
        self._tasks[task_id] = task

        # Get or create queue for provider
        queue = self._queues[bound_provider]

        # Ensure workers are running for this provider
        await self._ensure_workers(bound_provider)

        # Enqueue task
        try:
            queue.put_nowait(task)
            self._stats["total_enqueued"] += 1

            logger.debug(
                f"Enqueued task {task_id} for provider {bound_provider} (priority={priority})",
            )

            return task_id

        except asyncio.QueueFull:
            # Clean up on failure
            del self._tasks[task_id]
            del self._task_events[task_id]
            raise

    async def _ensure_workers(self, provider: str) -> None:
        """Ensure workers are running for a provider."""
        async with self._lock:
            # Clean up completed workers
            self._workers[provider] = [w for w in self._workers[provider] if not w.done()]

            # Start workers if needed
            while len(self._workers[provider]) < self._max_workers:
                worker = asyncio.create_task(
                    self._worker_loop(provider),
                    name=f"worker-{provider}-{len(self._workers[provider])}",
                )
                self._workers[provider].append(worker)

    async def _worker_loop(self, provider: str) -> None:
        """Worker loop for processing tasks."""
        queue = self._queues[provider]

        while self._running:
            try:
                # Wait for task with timeout
                try:
                    task = await asyncio.wait_for(
                        queue.get(),
                        timeout=5.0,
                    )
                except TimeoutError:
                    continue

                # Process task
                await self._process_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker error for provider {provider}: {e}")

    async def _process_task(self, task: QueuedTask) -> None:
        """Process a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        start_time = time.perf_counter()

        try:
            logger.debug(f"Processing task {task.task_id}")

            # Execute operation
            if asyncio.iscoroutinefunction(task.operation):
                result = await task.operation(**task.kwargs)
            else:
                result = task.operation(**task.kwargs)

            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow()
            self._stats["total_completed"] += 1

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Task {task.task_id} completed in {duration_ms:.2f}ms")

        except Exception as e:
            # Mark failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            self._stats["total_failed"] += 1

            logger.warning(f"Task {task.task_id} failed: {e}")

        finally:
            # Signal completion
            event = self._task_events.get(task.task_id)
            if event:
                event.set()

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> TaskResult:
        """Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            TaskResult with task outcome

        Raises:
            KeyError: If task not found
            asyncio.TimeoutError: If timeout exceeded

        """
        if task_id not in self._tasks:
            msg = f"Task {task_id} not found"
            raise KeyError(msg)

        event = self._task_events[task_id]
        task = self._tasks[task_id]

        if not event.is_set():
            await asyncio.wait_for(event.wait(), timeout=timeout)

        duration_ms = 0.0
        if task.started_at and task.completed_at:
            delta = task.completed_at - task.started_at
            duration_ms = delta.total_seconds() * 1000

        return TaskResult(
            task_id=task_id,
            success=task.status == TaskStatus.COMPLETED,
            result=task.result,
            error=task.error,
            duration_ms=duration_ms,
        )

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if task was cancelled, False if not found or not pending

        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()

        # Signal completion
        event = self._task_events.get(task_id)
        if event:
            event.set()

        logger.debug(f"Task {task_id} cancelled")
        return True

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get status of a task.

        Args:
            task_id: Task ID to check

        Returns:
            Task status dict or None if not found

        """
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def get_queue_status(self) -> dict[str, Any]:
        """Get queue status by provider.

        Returns:
            Dict with queue statistics

        """
        queue_sizes = {}
        for provider, queue in self._queues.items():
            queue_sizes[provider] = queue.qsize()

        worker_counts = {}
        for provider, workers in self._workers.items():
            active = sum(1 for w in workers if not w.done())
            worker_counts[provider] = active

        pending_by_provider: dict[str, int] = defaultdict(int)
        running_by_provider: dict[str, int] = defaultdict(int)

        for task in self._tasks.values():
            provider = task.provider or "default"
            if task.status == TaskStatus.PENDING:
                pending_by_provider[provider] += 1
            elif task.status == TaskStatus.RUNNING:
                running_by_provider[provider] += 1

        return {
            "running": self._running,
            "queue_sizes": queue_sizes,
            "worker_counts": worker_counts,
            "pending_by_provider": dict(pending_by_provider),
            "running_by_provider": dict(running_by_provider),
            "stats": {
                "total_enqueued": self._stats["total_enqueued"],
                "total_completed": self._stats["total_completed"],
                "total_failed": self._stats["total_failed"],
                "success_rate": (
                    self._stats["total_completed"]
                    / max(1, self._stats["total_completed"] + self._stats["total_failed"])
                ),
            },
        }

    def get_pending_tasks(
        self,
        provider: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get pending tasks, optionally filtered by provider.

        Args:
            provider: Filter by provider (None = all)
            limit: Maximum tasks to return

        Returns:
            List of task status dicts

        """
        tasks = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if provider and task.provider != provider:
                continue
            tasks.append(task.to_dict())
            if len(tasks) >= limit:
                break

        return tasks

    async def process(self) -> Any:
        """Process next item in queue (for manual processing).

        Returns:
            Result from processed task or None if no tasks

        """
        # Find a task to process
        for queue in self._queues.values():
            if not queue.empty():
                try:
                    task = queue.get_nowait()
                    await self._process_task(task)
                    return task.result
                except asyncio.QueueEmpty:
                    continue

        return None

    def cleanup_completed(
        self,
        max_age_seconds: int = 3600,
    ) -> int:
        """Clean up completed tasks older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of tasks cleaned up

        """
        cutoff = datetime.utcnow()
        count = 0

        task_ids_to_remove = []
        for task_id, task in self._tasks.items():
            if (
                task.status
                in (
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                )
                and task.completed_at
            ):
                age = (cutoff - task.completed_at).total_seconds()
                if age > max_age_seconds:
                    task_ids_to_remove.append(task_id)

        for task_id in task_ids_to_remove:
            del self._tasks[task_id]
            if task_id in self._task_events:
                del self._task_events[task_id]
            count += 1

        if count > 0:
            logger.debug(f"Cleaned up {count} completed tasks")

        return count


# Global queue instance
_processing_queue: ProviderBoundQueue | None = None


def get_processing_queue() -> ProviderBoundQueue:
    """Get the global processing queue instance."""
    global _processing_queue
    if _processing_queue is None:
        _processing_queue = ProviderBoundQueue()
    return _processing_queue


async def initialize_processing_queue(
    default_provider: str | None = None,
    max_workers_per_provider: int = 3,
) -> ProviderBoundQueue:
    """Initialize and start the global processing queue.

    Args:
        default_provider: Default provider for unbound tasks
        max_workers_per_provider: Max concurrent workers per provider

    Returns:
        Initialized ProviderBoundQueue

    """
    global _processing_queue
    _processing_queue = ProviderBoundQueue(
        default_provider=default_provider,
        max_workers_per_provider=max_workers_per_provider,
    )
    await _processing_queue.start()
    logger.info("Processing queue initialized")
    return _processing_queue


async def shutdown_processing_queue() -> None:
    """Shutdown the global processing queue."""
    global _processing_queue
    if _processing_queue:
        await _processing_queue.stop()
        _processing_queue = None
        logger.info("Processing queue shutdown")
