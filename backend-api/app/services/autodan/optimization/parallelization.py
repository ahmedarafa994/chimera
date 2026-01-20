"""AutoDAN Parallelization and Turbo Architecture.

Efficient parallel execution and AutoDAN-Turbo specific optimizations:
- Thread/process pool management
- Batch evaluation parallelization
- Async prompt generation
- Turbo lifelong learning architecture

References:
- AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration (Liu et al., 2024)

"""

import asyncio
import contextlib
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

from .config import ParallelizationConfig, TurboConfig

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# BATCH PROCESSING
# =============================================================================


@dataclass
class BatchResult(Generic[T]):
    """Container for batch processing results."""

    results: list[T]
    success_count: int
    failure_count: int
    total_time: float
    errors: list[tuple[int, Exception]]


class BatchProcessor(Generic[T, R]):
    """Generic batch processor for parallel evaluation.

    Supports both sync and async processing with configurable
    parallelization strategy.
    """

    def __init__(
        self,
        config: ParallelizationConfig | None = None,
        process_func: Callable[[T], R] | None = None,
    ) -> None:
        self.config = config or ParallelizationConfig()
        self.process_func = process_func
        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """Get or create executor."""
        if self._executor is None:
            if self.config.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.config.num_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        return self._executor

    def process_batch(
        self,
        items: list[T],
        process_func: Callable[[T], R] | None = None,
    ) -> BatchResult[R]:
        """Process a batch of items in parallel.

        Args:
            items: List of items to process
            process_func: Optional override for processing function

        Returns:
            BatchResult with results and diagnostics

        """
        func = process_func or self.process_func
        if func is None:
            msg = "No processing function provided"
            raise ValueError(msg)

        start_time = time.time()
        results: list[R | None] = [None] * len(items)
        errors: list[tuple[int, Exception]] = []
        success_count = 0

        executor = self._get_executor()

        # Submit all tasks
        future_to_index = {executor.submit(func, item): i for i, item in enumerate(items)}

        # Collect results with timeout
        for future in as_completed(future_to_index, timeout=self.config.task_timeout * len(items)):
            index = future_to_index[future]
            try:
                results[index] = future.result(timeout=self.config.task_timeout)
                success_count += 1
            except Exception as e:
                errors.append((index, e))

        total_time = time.time() - start_time

        return BatchResult(
            results=[r for r in results if r is not None],
            success_count=success_count,
            failure_count=len(items) - success_count,
            total_time=total_time,
            errors=errors,
        )

    async def process_batch_async(
        self,
        items: list[T],
        process_func: Callable[[T], R] | None = None,
    ) -> BatchResult[R]:
        """Process a batch of items asynchronously.

        Uses asyncio for I/O-bound operations.
        """
        func = process_func or self.process_func
        if func is None:
            msg = "No processing function provided"
            raise ValueError(msg)

        start_time = time.time()
        results: list[R | None] = []
        errors: list[tuple[int, Exception]] = []

        # Create async tasks
        async def process_item(index: int, item: T) -> tuple[int, R | None, Exception | None]:
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(item)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, item)
                return index, result, None
            except Exception as e:
                return index, None, e

        # Run all tasks concurrently
        tasks = [process_item(i, item) for i, item in enumerate(items)]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        # Collect results
        success_count = 0
        for index, result, error in completed:
            if error is not None:
                errors.append((index, error))
            else:
                results.append(result)
                success_count += 1

        total_time = time.time() - start_time

        return BatchResult(
            results=results,
            success_count=success_count,
            failure_count=len(items) - success_count,
            total_time=total_time,
            errors=errors,
        )

    def shutdown(self) -> None:
        """Shutdown executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


# =============================================================================
# ASYNC QUEUE PROCESSOR
# =============================================================================


class AsyncQueueProcessor(Generic[T, R]):
    """Async queue-based processor for continuous prompt generation.

    Maintains a queue of pending items and processes them
    asynchronously with configurable concurrency.
    """

    def __init__(
        self,
        config: ParallelizationConfig | None = None,
        process_func: Callable[[T], R] | None = None,
        result_callback: Callable[[R], None] | None = None,
    ) -> None:
        self.config = config or ParallelizationConfig()
        self.process_func = process_func
        self.result_callback = result_callback

        self._input_queue: queue.Queue[T | None] = queue.Queue(maxsize=self.config.max_queue_size)
        self._output_queue: queue.Queue[R] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._running = False
        self._processed_count = 0
        self._error_count = 0

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return

        self._running = True

        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                item = self._input_queue.get(timeout=1.0)

                if item is None:  # Shutdown signal
                    break

                if self.process_func is not None:
                    try:
                        result = self.process_func(item)
                        self._output_queue.put(result)
                        self._processed_count += 1

                        if self.result_callback is not None:
                            self.result_callback(result)
                    except Exception:
                        self._error_count += 1

                self._input_queue.task_done()

            except queue.Empty:
                continue

    def submit(self, item: T, block: bool = True, timeout: float | None = None) -> None:
        """Submit item for processing."""
        self._input_queue.put(item, block=block, timeout=timeout)

    def submit_batch(self, items: list[T]) -> None:
        """Submit multiple items."""
        for item in items:
            self.submit(item)

    def get_result(self, block: bool = True, timeout: float | None = None) -> R | None:
        """Get processed result."""
        try:
            return self._output_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_all_results(self) -> list[R]:
        """Get all available results."""
        results = []
        while not self._output_queue.empty():
            try:
                results.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def stop(self) -> None:
        """Stop all workers."""
        self._running = False

        # Send shutdown signals
        for _ in self._workers:
            with contextlib.suppress(queue.Full):
                self._input_queue.put(None, timeout=1.0)

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)

        self._workers.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "pending_count": self._input_queue.qsize(),
            "result_count": self._output_queue.qsize(),
            "worker_count": len(self._workers),
            "running": self._running,
        }


# =============================================================================
# POPULATION PARALLEL EVALUATOR
# =============================================================================


class PopulationParallelEvaluator:
    """Parallel evaluator optimized for genetic algorithm populations.

    Features:
    - Batch fitness evaluation
    - Caching of repeated evaluations
    - Adaptive batch sizing based on performance
    """

    def __init__(
        self,
        config: ParallelizationConfig | None = None,
        fitness_func: Callable[[str], float] | None = None,
    ) -> None:
        self.config = config or ParallelizationConfig()
        self.fitness_func = fitness_func
        self.batch_processor = BatchProcessor(config)

        # Evaluation cache
        self._cache: dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Adaptive batch sizing
        self._optimal_batch_size = config.batch_size
        self._timing_history: deque = deque(maxlen=10)

    def evaluate_population(
        self,
        population: list[str],
        fitness_func: Callable[[str], float] | None = None,
    ) -> list[float]:
        """Evaluate fitness for entire population.

        Args:
            population: List of prompt strings
            fitness_func: Optional override for fitness function

        Returns:
            List of fitness values

        """
        func = fitness_func or self.fitness_func
        if func is None:
            msg = "No fitness function provided"
            raise ValueError(msg)

        # Check cache first
        results = [None] * len(population)
        uncached_indices = []
        uncached_prompts = []

        for i, prompt in enumerate(population):
            cache_key = hash(prompt)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_prompts.append(prompt)
                self._cache_misses += 1

        # Evaluate uncached prompts in batches
        if uncached_prompts:
            start_time = time.time()

            # Process in batches
            batch_results = []
            for batch_start in range(0, len(uncached_prompts), self._optimal_batch_size):
                batch_end = min(batch_start + self._optimal_batch_size, len(uncached_prompts))
                batch = uncached_prompts[batch_start:batch_end]

                batch_result = self.batch_processor.process_batch(batch, func)
                batch_results.extend(batch_result.results)

            elapsed = time.time() - start_time
            self._timing_history.append(elapsed / max(len(uncached_prompts), 1))

            # Update results and cache
            for i, fitness in zip(uncached_indices, batch_results, strict=False):
                results[i] = fitness
                cache_key = hash(population[i])
                self._cache[cache_key] = fitness

            # Adapt batch size
            self._adapt_batch_size()

        return results

    def _adapt_batch_size(self) -> None:
        """Adapt batch size based on timing history."""
        if len(self._timing_history) < 5:
            return

        avg_time_per_item = np.mean(list(self._timing_history))

        # Target: 1-2 seconds per batch
        target_batch_time = 1.5
        ideal_batch_size = int(target_batch_time / max(avg_time_per_item, 0.001))

        # Smooth adjustment
        self._optimal_batch_size = int(0.7 * self._optimal_batch_size + 0.3 * ideal_batch_size)
        self._optimal_batch_size = max(1, min(self._optimal_batch_size, 100))

    def clear_cache(self) -> None:
        """Clear evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            "optimal_batch_size": self._optimal_batch_size,
            "avg_time_per_item": np.mean(list(self._timing_history)) if self._timing_history else 0,
        }


# =============================================================================
# TURBO STRATEGY LIBRARY
# =============================================================================


@dataclass
class Strategy:
    """Represents an attack strategy in the Turbo library."""

    id: str
    name: str
    description: str
    pattern: str
    effectiveness_score: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


class StrategyLibrary:
    """Lifelong learning strategy library for AutoDAN-Turbo.

    Maintains a collection of discovered attack strategies,
    their effectiveness scores, and semantic embeddings for
    efficient retrieval.
    """

    def __init__(
        self,
        config: TurboConfig | None = None,
        embedding_func: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.config = config or TurboConfig()
        self.embedding_func = embedding_func

        self._strategies: dict[str, Strategy] = {}
        self._embedding_matrix: np.ndarray | None = None
        self._strategy_ids: list[str] = []

    def add_strategy(
        self,
        name: str,
        description: str,
        pattern: str,
        effectiveness_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Strategy:
        """Add a new strategy to the library.

        Returns the created Strategy object.
        """
        # Generate unique ID
        strategy_id = f"strategy_{len(self._strategies)}_{hash(pattern) % 10000}"

        # Compute embedding if available
        embedding = None
        if self.embedding_func is not None:
            embedding = self.embedding_func(f"{name}: {description}")

        # Create strategy
        strategy = Strategy(
            id=strategy_id,
            name=name,
            description=description,
            pattern=pattern,
            effectiveness_score=effectiveness_score,
            embedding=embedding,
            metadata=metadata or {},
        )

        # Check for duplicates using embedding similarity
        if embedding is not None and self._embedding_matrix is not None:
            similarities = self._compute_similarities(embedding)
            if np.max(similarities) > self.config.similarity_threshold:
                # Update existing strategy instead
                most_similar_idx = np.argmax(similarities)
                existing_id = self._strategy_ids[most_similar_idx]
                existing = self._strategies[existing_id]

                # Update effectiveness score (exponential moving average)
                existing.effectiveness_score = (
                    0.8 * existing.effectiveness_score + 0.2 * effectiveness_score
                )
                return existing

        # Add new strategy
        self._strategies[strategy_id] = strategy
        self._strategy_ids.append(strategy_id)

        # Update embedding matrix
        if embedding is not None:
            if self._embedding_matrix is None:
                self._embedding_matrix = embedding.reshape(1, -1)
            else:
                self._embedding_matrix = np.vstack(
                    [self._embedding_matrix, embedding.reshape(1, -1)],
                )

        # Prune if necessary
        if len(self._strategies) > self.config.max_library_size:
            self._prune_strategies()

        return strategy

    def _compute_similarities(self, embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities to all strategies."""
        if self._embedding_matrix is None:
            return np.array([])

        # Normalize
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        matrix_norm = self._embedding_matrix / (
            np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True) + 1e-8
        )

        return np.dot(matrix_norm, emb_norm)

    def get_similar_strategies(self, query: str, top_k: int = 5) -> list[Strategy]:
        """Retrieve strategies similar to the query.

        Args:
            query: Query string
            top_k: Number of strategies to return

        Returns:
            List of similar strategies

        """
        if self.embedding_func is None or len(self._strategies) == 0:
            # Return top strategies by effectiveness
            sorted_strategies = sorted(
                self._strategies.values(),
                key=lambda s: s.effectiveness_score,
                reverse=True,
            )
            return sorted_strategies[:top_k]

        # Compute query embedding
        query_embedding = self.embedding_func(query)
        similarities = self._compute_similarities(query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self._strategies[self._strategy_ids[i]] for i in top_indices]

    def get_effective_strategies(
        self,
        min_score: float = 0.0,
        top_k: int | None = None,
    ) -> list[Strategy]:
        """Get strategies above effectiveness threshold.

        Args:
            min_score: Minimum effectiveness score
            top_k: Optional limit on number of strategies

        Returns:
            List of effective strategies

        """
        effective = [s for s in self._strategies.values() if s.effectiveness_score >= min_score]

        # Sort by effectiveness
        effective.sort(key=lambda s: s.effectiveness_score, reverse=True)

        if top_k is not None:
            effective = effective[:top_k]

        return effective

    def update_strategy_effectiveness(
        self, strategy_id: str, score: float, success: bool = True
    ) -> None:
        """Update strategy effectiveness based on usage.

        Args:
            strategy_id: ID of the strategy
            score: Score achieved
            success: Whether the attack succeeded

        """
        if strategy_id not in self._strategies:
            return

        strategy = self._strategies[strategy_id]
        strategy.usage_count += 1

        if success:
            strategy.success_count += 1

        # Update effectiveness (exponential moving average)
        alpha = 0.1  # Learning rate for effectiveness update
        strategy.effectiveness_score = (1 - alpha) * strategy.effectiveness_score + alpha * score

    def _prune_strategies(self) -> None:
        """Remove low-effectiveness strategies to maintain library size."""
        if len(self._strategies) <= self.config.max_library_size:
            return

        # Sort by effectiveness and success rate
        sorted_ids = sorted(
            self._strategy_ids,
            key=lambda sid: (
                self._strategies[sid].effectiveness_score * 0.7
                + self._strategies[sid].success_rate * 0.3
            ),
            reverse=True,
        )

        # Keep top strategies
        keep_ids = set(sorted_ids[: self.config.max_library_size])
        remove_ids = set(sorted_ids[self.config.max_library_size :])

        # Remove from dictionary
        for sid in remove_ids:
            del self._strategies[sid]

        # Rebuild embedding matrix
        self._strategy_ids = [sid for sid in self._strategy_ids if sid in keep_ids]

        if self.embedding_func is not None and self._strategy_ids:
            self._embedding_matrix = np.vstack(
                [
                    self._strategies[sid].embedding
                    for sid in self._strategy_ids
                    if self._strategies[sid].embedding is not None
                ],
            )

    def export_library(self) -> dict[str, Any]:
        """Export library to serializable format."""
        return {
            "strategies": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "pattern": s.pattern,
                    "effectiveness_score": s.effectiveness_score,
                    "usage_count": s.usage_count,
                    "success_count": s.success_count,
                    "metadata": s.metadata,
                }
                for s in self._strategies.values()
            ],
            "config": {
                "max_library_size": self.config.max_library_size,
                "similarity_threshold": self.config.similarity_threshold,
            },
        }

    def import_library(self, data: dict[str, Any]) -> None:
        """Import library from serialized format."""
        for s_data in data.get("strategies", []):
            self.add_strategy(
                name=s_data["name"],
                description=s_data["description"],
                pattern=s_data["pattern"],
                effectiveness_score=s_data.get("effectiveness_score", 0.0),
                metadata=s_data.get("metadata", {}),
            )

    def get_stats(self) -> dict[str, Any]:
        """Get library statistics."""
        if not self._strategies:
            return {
                "total_strategies": 0,
                "avg_effectiveness": 0.0,
                "avg_success_rate": 0.0,
            }

        return {
            "total_strategies": len(self._strategies),
            "avg_effectiveness": np.mean(
                [s.effectiveness_score for s in self._strategies.values()],
            ),
            "avg_success_rate": np.mean([s.success_rate for s in self._strategies.values()]),
            "top_strategies": [s.name for s in self.get_effective_strategies(top_k=5)],
        }


# =============================================================================
# TURBO LIFELONG LEARNING
# =============================================================================


class TurboLifelongLearner:
    """AutoDAN-Turbo lifelong learning orchestrator.

    Manages the continuous improvement cycle:
    1. Strategy exploration
    2. Effectiveness evaluation
    3. Library update
    4. Guided generation
    """

    def __init__(
        self,
        config: TurboConfig | None = None,
        strategy_library: StrategyLibrary | None = None,
        evaluator: PopulationParallelEvaluator | None = None,
    ) -> None:
        self.config = config or TurboConfig()
        self.strategy_library = strategy_library or StrategyLibrary(config)
        self.evaluator = evaluator

        self._iteration = 0
        self._phase = "warmup"  # "warmup", "lifelong", "exploitation"
        self._best_score = 0.0
        self._score_history: deque = deque(maxlen=100)

    def step(
        self,
        prompts: list[str],
        scores: list[float],
        metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Perform one lifelong learning step.

        Args:
            prompts: Generated prompts from this iteration
            scores: Corresponding scores
            metadata: Optional metadata for each prompt

        Returns:
            Dict with updated state and recommendations

        """
        self._iteration += 1
        metadata = metadata or [{}] * len(prompts)

        # Update score history
        max_score = max(scores) if scores else 0.0
        self._score_history.append(max_score)

        self._best_score = max(self._best_score, max_score)

        # Extract and store successful strategies
        successful_strategies = self._extract_strategies(prompts, scores, metadata)

        for strategy_info in successful_strategies:
            self.strategy_library.add_strategy(**strategy_info)

        # Determine phase
        self._update_phase()

        # Get recommendations for next iteration
        recommendations = self._get_recommendations()

        return {
            "iteration": self._iteration,
            "phase": self._phase,
            "best_score": self._best_score,
            "strategies_added": len(successful_strategies),
            "recommendations": recommendations,
            "library_stats": self.strategy_library.get_stats(),
        }

    def _extract_strategies(
        self,
        prompts: list[str],
        scores: list[float],
        metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract successful strategies from prompts."""
        strategies = []

        for prompt, score, meta in zip(prompts, scores, metadata, strict=False):
            # Only extract from high-scoring prompts
            if score < self.config.min_score_improvement + self._best_score * 0.8:
                continue

            # Determine effectiveness level
            score_diff = score - self._best_score
            if score_diff >= self.config.highly_effective_score_diff:
                effectiveness = 1.0
            elif score_diff >= self.config.effective_score_diff:
                effectiveness = 0.7
            else:
                effectiveness = 0.4

            strategy_info = {
                "name": meta.get("strategy_name", f"auto_{self._iteration}_{len(strategies)}"),
                "description": meta.get(
                    "description",
                    f"Strategy from iteration {self._iteration}",
                ),
                "pattern": prompt[:500],  # Truncate for storage
                "effectiveness_score": effectiveness,
                "metadata": {"score": score, "iteration": self._iteration, **meta},
            }
            strategies.append(strategy_info)

        return strategies

    def _update_phase(self) -> None:
        """Update learning phase based on progress."""
        if self._iteration <= self.config.warmup_iterations:
            self._phase = "warmup"
        elif self._iteration <= self.config.warmup_iterations + self.config.lifelong_iterations:
            self._phase = "lifelong"
        # Check if still improving
        elif len(self._score_history) >= 10:
            recent = list(self._score_history)[-10:]
            improvement = recent[-1] - recent[0]

            if improvement < self.config.min_score_improvement:
                self._phase = "exploitation"
            else:
                self._phase = "lifelong"

    def _get_recommendations(self) -> dict[str, Any]:
        """Get recommendations for next iteration."""
        recommendations = {
            "exploration_rate": 0.5,
            "use_strategies": [],
            "focus_areas": [],
        }

        if self._phase == "warmup":
            # High exploration, no strategy use
            recommendations["exploration_rate"] = 0.8
            recommendations["focus_areas"] = ["diversity", "coverage"]

        elif self._phase == "lifelong":
            # Balance exploration and exploitation
            recommendations["exploration_rate"] = 0.4

            # Get top strategies to use
            effective_strategies = self.strategy_library.get_effective_strategies(
                min_score=0.5,
                top_k=self.config.top_k_strategies,
            )
            recommendations["use_strategies"] = [s.pattern for s in effective_strategies]
            recommendations["focus_areas"] = ["refinement", "combination"]

        else:  # exploitation
            # Low exploration, focus on best strategies
            recommendations["exploration_rate"] = 0.1

            top_strategies = self.strategy_library.get_effective_strategies(
                top_k=self.config.top_k_strategies // 2,
            )
            recommendations["use_strategies"] = [s.pattern for s in top_strategies]
            recommendations["focus_areas"] = ["optimization", "transfer"]

        return recommendations

    def best_of_n_sample(
        self,
        candidates: list[str],
        fitness_func: Callable[[str], float],
    ) -> tuple[str, float]:
        """Best-of-N sampling for prompt selection.

        Args:
            candidates: List of candidate prompts
            fitness_func: Function to evaluate fitness

        Returns:
            Tuple of (best_prompt, best_score)

        """
        if not candidates:
            return "", 0.0

        # Limit to best_of_n samples
        sample_size = min(len(candidates), self.config.best_of_n)
        sampled = np.random.choice(candidates, size=sample_size, replace=False)

        # Evaluate all samples
        if self.evaluator is not None:
            scores = self.evaluator.evaluate_population(list(sampled), fitness_func)
        else:
            scores = [fitness_func(p) for p in sampled]

        # Return best
        best_idx = np.argmax(scores)
        return sampled[best_idx], scores[best_idx]

    def get_state(self) -> dict[str, Any]:
        """Get current learner state."""
        return {
            "iteration": self._iteration,
            "phase": self._phase,
            "best_score": self._best_score,
            "score_history": list(self._score_history),
            "library_stats": self.strategy_library.get_stats(),
        }
