"""
Async Batch Pipeline for AutoDAN Turbo.

Implements parallel candidate processing with:
- Asynchronous batch execution
- Speculative execution
- Dynamic batch sizing
- Resource-aware scheduling
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class TaskStatus(Enum):
    """Status of a pipeline task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineTask:
    """A task in the pipeline."""

    task_id: str
    input_data: Any
    status: TaskStatus = TaskStatus.PENDING
    result: Any | None = None
    error: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    priority: int = 0

    @property
    def duration(self) -> float | None:
        """Get task duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class BatchResult:
    """Result of a batch execution."""

    batch_id: str
    results: list[Any]
    successful: int
    failed: int
    total_time: float
    avg_time_per_task: float


@dataclass
class PipelineStats:
    """Statistics for the pipeline."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_task_time: float = 0.0
    throughput: float = 0.0  # tasks per second


class DynamicBatchSizer:
    """
    Dynamically adjusts batch size based on performance.

    Monitors throughput and latency to find optimal batch size.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        initial_batch_size: int = 8,
        target_latency: float = 1.0,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.target_latency = target_latency

        # History for adaptation
        self.latency_history: list[float] = []
        self.throughput_history: list[float] = []

    def get_batch_size(self) -> int:
        """Get current recommended batch size."""
        return self.current_batch_size

    def update(self, batch_size: int, latency: float, throughput: float):
        """Update batch sizer with observed performance."""
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)

        # Keep limited history
        max_history = 20
        if len(self.latency_history) > max_history:
            self.latency_history = self.latency_history[-max_history:]
            self.throughput_history = self.throughput_history[-max_history:]

        # Adapt batch size
        avg_latency = np.mean(self.latency_history[-5:])

        if avg_latency > self.target_latency * 1.2:
            # Latency too high, reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8),
            )
        elif avg_latency < self.target_latency * 0.8:
            # Latency low, can increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2),
            )

    def reset(self):
        """Reset batch sizer state."""
        self.latency_history.clear()
        self.throughput_history.clear()


class SpeculativeExecutor:
    """
    Executes speculative tasks that may be cancelled.

    Useful for exploring multiple paths simultaneously.
    """

    def __init__(self, max_speculative: int = 4):
        self.max_speculative = max_speculative
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.results: dict[str, Any] = {}

    async def execute_speculative(
        self,
        tasks: list[PipelineTask],
        executor: Callable[[Any], Any],
        select_best: Callable[[list[Any]], int],
    ) -> Any:
        """
        Execute tasks speculatively and return best result.

        Args:
            tasks: Tasks to execute speculatively
            executor: Function to execute each task
            select_best: Function to select best result index

        Returns:
            Best result from speculative execution
        """
        # Limit speculative tasks
        tasks = tasks[: self.max_speculative]

        # Create async tasks
        async_tasks = []
        for task in tasks:
            async_task = asyncio.create_task(self._execute_task(task, executor))
            self.active_tasks[task.task_id] = async_task
            async_tasks.append(async_task)

        # Wait for first completion or all completions
        done, pending = await asyncio.wait(
            async_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Get results from completed tasks
        results = []
        for task in done:
            try:
                result = task.result()
                results.append(result)
            except Exception as e:
                logger.warning(f"Speculative task failed: {e}")

        # If we have results, select best and cancel pending
        if results:
            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Select best result
            if len(results) == 1:
                return results[0]

            best_idx = select_best(results)
            return results[best_idx]

        # Wait for remaining if no results yet
        if pending:
            done, _ = await asyncio.wait(pending)
            for task in done:
                try:
                    return task.result()
                except Exception:
                    continue

        raise RuntimeError("All speculative tasks failed")

    async def _execute_task(
        self,
        task: PipelineTask,
        executor: Callable[[Any], Any],
    ) -> Any:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(executor):
                result = await executor(task.input_data)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, executor, task.input_data)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = time.time()
            return result

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.end_time = time.time()
            raise

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
            raise

    def cancel_all(self):
        """Cancel all active speculative tasks."""
        for task in self.active_tasks.values():
            task.cancel()
        self.active_tasks.clear()


class AsyncBatchPipeline:
    """
    Asynchronous batch processing pipeline.

    Features:
    - Parallel batch execution
    - Dynamic batch sizing
    - Speculative execution
    - Priority scheduling
    - Resource-aware throttling
    """

    def __init__(
        self,
        max_concurrency: int = 8,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        target_latency: float = 1.0,
        enable_speculation: bool = True,
    ):
        self.max_concurrency = max_concurrency
        self.enable_speculation = enable_speculation

        # Batch sizer
        self.batch_sizer = DynamicBatchSizer(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_latency=target_latency,
        )

        # Speculative executor
        self.speculative = SpeculativeExecutor(
            max_speculative=max_concurrency // 2,
        )

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # Task queue
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Statistics
        self.stats = PipelineStats()
        self.batch_count = 0

    async def process_batch(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
        batch_id: str | None = None,
    ) -> BatchResult:
        """
        Process a batch of items in parallel.

        Args:
            items: Items to process
            processor: Function to process each item
            batch_id: Optional batch identifier

        Returns:
            BatchResult with all results
        """
        if not items:
            return BatchResult(
                batch_id=batch_id or "empty",
                results=[],
                successful=0,
                failed=0,
                total_time=0.0,
                avg_time_per_task=0.0,
            )

        batch_id = batch_id or f"batch_{self.batch_count}"
        self.batch_count += 1
        start_time = time.time()

        # Create tasks
        tasks = [
            PipelineTask(
                task_id=f"{batch_id}_{i}",
                input_data=item,
            )
            for i, item in enumerate(items)
        ]

        # Process in parallel with concurrency limit
        results = await self._process_tasks_parallel(tasks, processor)

        # Compute statistics
        end_time = time.time()
        total_time = end_time - start_time
        successful = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

        # Update batch sizer
        throughput = len(items) / total_time if total_time > 0 else 0
        self.batch_sizer.update(len(items), total_time, throughput)

        # Update stats
        self._update_stats(tasks, total_time)

        return BatchResult(
            batch_id=batch_id,
            results=results,
            successful=successful,
            failed=failed,
            total_time=total_time,
            avg_time_per_task=total_time / len(items) if items else 0,
        )

    async def _process_tasks_parallel(
        self,
        tasks: list[PipelineTask],
        processor: Callable[[Any], Any],
    ) -> list[Any]:
        """Process tasks in parallel with concurrency limit."""

        async def process_with_semaphore(task: PipelineTask) -> Any:
            async with self.semaphore:
                return await self._execute_single_task(task, processor)

        # Create coroutines
        coroutines = [process_with_semaphore(task) for task in tasks]

        # Execute all
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tasks[i].status = TaskStatus.FAILED
                tasks[i].error = str(result)
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_task(
        self,
        task: PipelineTask,
        processor: Callable[[Any], Any],
    ) -> Any:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(processor):
                result = await processor(task.input_data)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, processor, task.input_data)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.end_time = time.time()
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
            raise

    async def process_stream(
        self,
        item_generator: Any,
        processor: Callable[[Any], Any],
        on_result: Callable[[Any], None] | None = None,
    ) -> list[Any]:
        """
        Process a stream of items with dynamic batching.

        Args:
            item_generator: Async generator of items
            processor: Function to process each item
            on_result: Optional callback for each result

        Returns:
            List of all results
        """
        all_results = []
        batch = []
        batch_size = self.batch_sizer.get_batch_size()

        async for item in item_generator:
            batch.append(item)

            if len(batch) >= batch_size:
                # Process batch
                result = await self.process_batch(batch, processor)
                all_results.extend(result.results)

                # Callback for each result
                if on_result:
                    for r in result.results:
                        on_result(r)

                # Update batch size for next iteration
                batch_size = self.batch_sizer.get_batch_size()
                batch = []

        # Process remaining items
        if batch:
            result = await self.process_batch(batch, processor)
            all_results.extend(result.results)
            if on_result:
                for r in result.results:
                    on_result(r)

        return all_results

    async def process_with_speculation(
        self,
        candidates: list[Any],
        processor: Callable[[Any], Any],
        scorer: Callable[[Any], float],
    ) -> Any:
        """
        Process candidates with speculative execution.

        Executes multiple candidates in parallel and returns best.

        Args:
            candidates: Candidate inputs to try
            processor: Function to process each candidate
            scorer: Function to score results (higher is better)

        Returns:
            Best result from speculative execution
        """
        if not self.enable_speculation:
            # Fall back to sequential
            best_result = None
            best_score = float("-inf")

            for candidate in candidates:
                result = await self._execute_single_task(
                    PipelineTask(task_id="seq", input_data=candidate),
                    processor,
                )
                score = scorer(result)
                if score > best_score:
                    best_score = score
                    best_result = result

            return best_result

        # Create tasks for speculative execution
        tasks = [PipelineTask(task_id=f"spec_{i}", input_data=c) for i, c in enumerate(candidates)]

        def select_best(results: list[Any]) -> int:
            scores = [scorer(r) for r in results]
            return int(np.argmax(scores))

        return await self.speculative.execute_speculative(tasks, processor, select_best)

    async def process_priority_queue(
        self,
        processor: Callable[[Any], Any],
        timeout: float | None = None,
    ) -> list[Any]:
        """
        Process items from priority queue.

        Args:
            processor: Function to process each item
            timeout: Optional timeout in seconds

        Returns:
            List of results
        """
        results = []
        start_time = time.time()

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break

            try:
                # Get next item with timeout
                _priority, task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=0.1,
                )

                # Process task
                result = await self._execute_single_task(task, processor)
                results.append(result)

                self.task_queue.task_done()

            except TimeoutError:
                # No items in queue
                if self.task_queue.empty():
                    break

        return results

    def add_to_queue(self, item: Any, priority: int = 0):
        """Add item to priority queue."""
        task = PipelineTask(
            task_id=f"queue_{self.stats.total_tasks}",
            input_data=item,
            priority=priority,
        )
        # Lower priority value = higher priority
        self.task_queue.put_nowait((-priority, task))
        self.stats.total_tasks += 1

    def _update_stats(self, tasks: list[PipelineTask], batch_time: float):
        """Update pipeline statistics."""
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)

        self.stats.total_tasks += len(tasks)
        self.stats.completed_tasks += completed
        self.stats.failed_tasks += failed
        self.stats.total_batches += 1

        # Update averages
        self.stats.avg_batch_size = self.stats.total_tasks / self.stats.total_batches

        task_times = [t.duration for t in tasks if t.duration is not None]
        if task_times:
            self.stats.avg_task_time = np.mean(task_times)

        if batch_time > 0:
            self.stats.throughput = len(tasks) / batch_time

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_tasks": self.stats.total_tasks,
            "completed_tasks": self.stats.completed_tasks,
            "failed_tasks": self.stats.failed_tasks,
            "success_rate": (
                self.stats.completed_tasks / self.stats.total_tasks
                if self.stats.total_tasks > 0
                else 0.0
            ),
            "total_batches": self.stats.total_batches,
            "avg_batch_size": self.stats.avg_batch_size,
            "avg_task_time": self.stats.avg_task_time,
            "throughput": self.stats.throughput,
            "current_batch_size": self.batch_sizer.get_batch_size(),
            "max_concurrency": self.max_concurrency,
        }

    def reset(self):
        """Reset pipeline state."""
        self.batch_sizer.reset()
        self.speculative.cancel_all()
        self.stats = PipelineStats()
        self.batch_count = 0
        # Clear queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class PipelineOrchestrator:
    """
    Orchestrates multiple pipelines for complex workflows.

    Supports pipeline chaining and fan-out/fan-in patterns.
    """

    def __init__(self):
        self.pipelines: dict[str, AsyncBatchPipeline] = {}
        self.results: dict[str, list[Any]] = {}

    def create_pipeline(
        self,
        name: str,
        max_concurrency: int = 8,
        **kwargs,
    ) -> AsyncBatchPipeline:
        """Create a named pipeline."""
        pipeline = AsyncBatchPipeline(
            max_concurrency=max_concurrency,
            **kwargs,
        )
        self.pipelines[name] = pipeline
        return pipeline

    async def chain(
        self,
        items: list[Any],
        stages: list[tuple],
    ) -> list[Any]:
        """
        Execute a chain of pipeline stages.

        Args:
            items: Initial items
            stages: List of (pipeline_name, processor) tuples

        Returns:
            Final results after all stages
        """
        current_items = items

        for pipeline_name, processor in stages:
            pipeline = self.pipelines.get(pipeline_name)
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_name} not found")

            result = await pipeline.process_batch(current_items, processor)
            current_items = [r for r in result.results if r is not None]

        return current_items

    async def fan_out_fan_in(
        self,
        items: list[Any],
        fan_out_processor: Callable[[Any], list[Any]],
        fan_in_processor: Callable[[list[Any]], Any],
        pipeline_name: str = "default",
    ) -> list[Any]:
        """
        Fan-out items, process, then fan-in results.

        Args:
            items: Items to fan out
            fan_out_processor: Expands each item to multiple
            fan_in_processor: Combines results back
            pipeline_name: Pipeline to use

        Returns:
            Fan-in results
        """
        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            pipeline = self.create_pipeline(pipeline_name)

        # Fan out
        expanded = []
        item_indices = []
        for i, item in enumerate(items):
            fan_out_results = fan_out_processor(item)
            expanded.extend(fan_out_results)
            item_indices.extend([i] * len(fan_out_results))

        # Process expanded items
        result = await pipeline.process_batch(expanded, lambda x: x)

        # Fan in - group by original item
        grouped: dict[int, list[Any]] = {}
        for idx, res in zip(item_indices, result.results, strict=False):
            if idx not in grouped:
                grouped[idx] = []
            grouped[idx].append(res)

        # Apply fan-in processor
        final_results = []
        for i in range(len(items)):
            group = grouped.get(i, [])
            final_results.append(fan_in_processor(group))

        return final_results

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all pipelines."""
        return {name: pipeline.get_stats() for name, pipeline in self.pipelines.items()}
