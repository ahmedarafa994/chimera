"""Background Task Queue - Reliable async task processing.

Provides:
- Task scheduling with priority levels
- Guaranteed delivery with configurable retry
- Dead-letter queue for failed tasks
- Task status monitoring and metrics
- Graceful shutdown handling

Usage:
    from app.core.task_queue import task_queue, Task

    # Register a task handler
    @task_queue.handler("send_email")
    async def send_email_handler(payload: dict):
        await send_email(payload["to"], payload["subject"])
        return {"sent": True}

    # Enqueue a task
    task_id = await task_queue.enqueue(Task(
        name="send_email",
        payload={"to": "user@example.com", "subject": "Hello"}
    ))

    # Check task status
    status = task_queue.get_task_status(task_id)
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class TaskPriority(int, Enum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Background task definition.

    Attributes:
        name: Handler name for this task
        payload: Task payload/arguments
        task_id: Unique task identifier
        status: Current task status
        priority: Task priority level
        created_at: ISO timestamp of creation
        started_at: ISO timestamp when execution started
        completed_at: ISO timestamp when execution finished
        attempts: Number of execution attempts
        max_retries: Maximum retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        result: Task result (if completed)
        error: Error message (if failed)
        correlation_id: ID for distributed tracing

    """

    name: str
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    started_at: str | None = None
    completed_at: str | None = None
    attempts: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay in seconds
    result: Any = None
    error: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "result": self.result,
            "error": self.error,
            "correlation_id": self.correlation_id,
        }


@dataclass
class TaskQueueStats:
    """Task queue statistics."""

    total_enqueued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_retried: int = 0
    total_dead_letter: int = 0
    current_queue_size: int = 0
    current_running: int = 0


class TaskQueue:
    """In-memory task queue with retry logic and dead-letter handling.

    Features:
    - Priority-based processing
    - Exponential backoff retry
    - Dead-letter queue for persistent failures
    - Concurrent worker support
    - Graceful shutdown
    """

    def __init__(self, max_workers: int = 5, dead_letter_size: int = 100) -> None:
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._handlers: dict[str, Callable] = {}
        self._tasks: dict[str, Task] = {}
        self._dead_letter: list[Task] = []
        self._dead_letter_size = dead_letter_size
        self._max_workers = max_workers
        self._running = False
        self._workers: list[asyncio.Task] = []
        self._stats = TaskQueueStats()
        self._lock = asyncio.Lock()

    def handler(self, task_name: str):
        """Decorator for registering task handlers.

        Usage:
            @task_queue.handler("process_data")
            async def process_data(payload: dict):
                return {"processed": True}
        """

        def decorator(func: Callable[[dict], Awaitable[Any] | Any]):
            self.register_handler(task_name, func)
            return func

        return decorator

    def register_handler(
        self,
        task_name: str,
        handler: Callable[[dict], Awaitable[Any] | Any],
    ) -> None:
        """Register a handler for a task type."""
        self._handlers[task_name] = handler
        logger.info(f"Registered task handler: {task_name}")

    async def enqueue(self, task: Task, delay: float = 0) -> str:
        """Add a task to the queue.

        Args:
            task: Task to enqueue
            delay: Optional delay before task becomes available

        Returns:
            Task ID for tracking

        """
        task.status = TaskStatus.QUEUED
        self._tasks[task.task_id] = task

        async with self._lock:
            self._stats.total_enqueued += 1
            self._stats.current_queue_size += 1

        if delay > 0:
            asyncio.create_task(self._delayed_enqueue(task, delay))
        else:
            # Priority queue uses (priority, timestamp, task_id) tuple
            # Lower priority number = higher priority (CRITICAL=3 processed first)
            priority_score = -task.priority.value
            await self._queue.put((priority_score, time.time(), task.task_id))

        logger.debug(f"Enqueued task {task.task_id}: {task.name}")
        return task.task_id

    async def _delayed_enqueue(self, task: Task, delay: float) -> None:
        """Enqueue task after a delay."""
        await asyncio.sleep(delay)
        priority_score = -task.priority.value
        await self._queue.put((priority_score, time.time(), task.task_id))

    async def start_workers(self, num_workers: int | None = None) -> None:
        """Start worker tasks for processing."""
        if self._running:
            logger.warning("Task queue workers already running")
            return

        self._running = True
        workers = num_workers or self._max_workers

        for i in range(workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Started {workers} task queue workers")

    async def stop_workers(self, timeout: float = 30.0) -> None:
        """Stop workers gracefully."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.wait(self._workers, timeout=timeout, return_when=asyncio.ALL_COMPLETED)

        self._workers.clear()
        logger.info("Task queue workers stopped")

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine for processing tasks."""
        logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                # Get task with timeout for graceful shutdown checks
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                _, _, task_id = item
                task = self._tasks.get(task_id)

                if not task:
                    logger.warning(f"Task {task_id} not found")
                    continue

                async with self._lock:
                    self._stats.current_queue_size -= 1
                    self._stats.current_running += 1

                await self._process_task(task)

                async with self._lock:
                    self._stats.current_running -= 1

            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} error: {e}")

    async def _process_task(self, task: Task) -> None:
        """Process a single task with retry logic."""
        handler = self._handlers.get(task.name)

        if not handler:
            task.status = TaskStatus.FAILED
            task.error = f"No handler registered for task: {task.name}"
            task.completed_at = datetime.utcnow().isoformat() + "Z"
            await self._send_to_dead_letter(task)
            return

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow().isoformat() + "Z"
        task.attempts += 1

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload)
            else:
                result = handler(task.payload)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.utcnow().isoformat() + "Z"

            async with self._lock:
                self._stats.total_completed += 1

            logger.debug(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.warning(f"Task {task.task_id} failed: {e}")
            task.error = str(e)

            if task.attempts < task.max_retries:
                # Retry with exponential backoff
                task.status = TaskStatus.RETRYING
                delay = task.retry_delay * (2 ** (task.attempts - 1))

                async with self._lock:
                    self._stats.total_retried += 1

                logger.info(
                    f"Retrying task {task.task_id} in {delay:.1f}s "
                    f"(attempt {task.attempts}/{task.max_retries})",
                )
                await self.enqueue(task, delay=delay)
            else:
                # Move to dead-letter queue
                task.status = TaskStatus.DEAD_LETTER
                task.completed_at = datetime.utcnow().isoformat() + "Z"
                await self._send_to_dead_letter(task)

    async def _send_to_dead_letter(self, task: Task) -> None:
        """Send failed task to dead-letter queue."""
        async with self._lock:
            self._stats.total_failed += 1
            self._stats.total_dead_letter += 1

        self._dead_letter.append(task)

        # Enforce max size
        if len(self._dead_letter) > self._dead_letter_size:
            self._dead_letter = self._dead_letter[-self._dead_letter_size :]

        logger.warning(f"Task {task.task_id} sent to dead-letter queue")

    def get_task_status(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_dead_letter_tasks(self, limit: int = 50) -> list[Task]:
        """Get tasks from dead-letter queue."""
        return self._dead_letter[-limit:]

    async def retry_dead_letter(self, task_id: str) -> bool:
        """Retry a task from dead-letter queue."""
        for i, task in enumerate(self._dead_letter):
            if task.task_id == task_id:
                task.attempts = 0
                task.error = None
                task.status = TaskStatus.PENDING
                self._dead_letter.pop(i)
                await self.enqueue(task)
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "running": self._running,
            "workers": len(self._workers),
            "handlers_registered": len(self._handlers),
            "total_enqueued": self._stats.total_enqueued,
            "total_completed": self._stats.total_completed,
            "total_failed": self._stats.total_failed,
            "total_retried": self._stats.total_retried,
            "dead_letter_count": len(self._dead_letter),
            "current_queue_size": self._stats.current_queue_size,
            "current_running": self._stats.current_running,
        }

    def clear(self) -> None:
        """Clear all tasks and queues."""
        self._tasks.clear()
        self._dead_letter.clear()
        self._stats = TaskQueueStats()
        # Clear the priority queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


# Global task queue instance
task_queue = TaskQueue()


def get_task_queue() -> TaskQueue:
    """Get the global task queue instance."""
    return task_queue
