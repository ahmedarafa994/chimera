"""
Parallel Mousetrap Generator for Optimized Jailbreak Chain Generation

This module implements parallel generation of Mousetrap attack chains to significantly
improve throughput and effectiveness through concurrent exploration of different
chaotic reasoning paths.

Key Optimizations:
- Concurrent generation of multiple attack chains
- Intelligent workload distribution across available resources
- Advanced result aggregation and selection strategies
- Circuit breaker patterns for resilient operation
- Comprehensive metrics and monitoring
"""

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..llm.chimera_adapter import ChimeraLLMAdapter
from ..mousetrap import MousetrapConfig, MousetrapGenerator, MousetrapResult

logger = logging.getLogger(__name__)


class ParallelStrategy(Enum):
    """Strategies for parallel Mousetrap generation"""

    DIVERSE_CHAOS = "diverse_chaos"  # Different chaos levels per worker
    VARIED_COMPLEXITY = "varied_complexity"  # Different complexity levels
    MULTI_STRATEGY = "multi_strategy"  # Different strategy contexts
    HYBRID_APPROACH = "hybrid_approach"  # Combination of strategies


@dataclass
class ParallelConfig:
    """Configuration for parallel Mousetrap generation"""

    max_workers: int = 4
    max_concurrent_chains: int = 8
    timeout_seconds: float = 60.0
    min_effectiveness_threshold: float = 0.4
    result_aggregation_strategy: str = "best_of_all"  # best_of_all, consensus, diversity
    enable_early_stopping: bool = True
    early_stop_threshold: float = 0.8
    chaos_level_diversity: bool = True
    complexity_variance: float = 0.3


@dataclass
class ParallelMousetrapResult:
    """Enhanced result from parallel Mousetrap generation"""

    best_result: MousetrapResult
    all_results: list[MousetrapResult]
    generation_stats: dict[str, Any]
    parallel_metrics: dict[str, Any]
    convergence_analysis: dict[str, Any]


@dataclass
class WorkerTask:
    """Task configuration for individual workers"""

    worker_id: int
    goal: str
    strategy_context: str
    config: MousetrapConfig
    iterations: int
    chaos_modifier: float = 0.0
    complexity_modifier: float = 0.0


@dataclass
class WorkerResult:
    """Result from individual worker"""

    worker_id: int
    result: MousetrapResult | None
    execution_time: float
    error: str | None = None
    iterations_completed: int = 0


class ParallelMousetrapGenerator:
    """
    Advanced parallel Mousetrap generator for high-throughput jailbreak chain creation

    Features:
    - Concurrent generation across multiple workers
    - Intelligent workload distribution
    - Advanced result aggregation strategies
    - Circuit breaker and timeout protection
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        llm_adapter: ChimeraLLMAdapter,
        base_config: MousetrapConfig | None = None,
        parallel_config: ParallelConfig | None = None,
    ):
        self.llm_adapter = llm_adapter
        self.base_config = base_config or MousetrapConfig()
        self.parallel_config = parallel_config or ParallelConfig()

        # Create base generator for fallback
        self.base_generator = MousetrapGenerator(llm_adapter, self.base_config)

        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=self.parallel_config.max_workers)

        # Metrics
        self.metrics = {
            "total_generations": 0,
            "parallel_generations": 0,
            "sequential_generations": 0,
            "total_execution_time": 0.0,
            "worker_utilization": {},
            "timeout_events": 0,
            "early_stops": 0,
        }

        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 3
        self.circuit_breaker_recovery_time = 30.0
        self.last_failure_time = 0.0

        logger.info("ParallelMousetrapGenerator initialized")

    async def generate_parallel_attack(
        self,
        goal: str,
        strategy_context: str = "",
        parallel_strategy: ParallelStrategy = ParallelStrategy.DIVERSE_CHAOS,
        custom_iterations: int | None = None,
    ) -> ParallelMousetrapResult:
        """
        Generate Mousetrap attack using parallel workers

        Args:
            goal: Target goal for the attack
            strategy_context: Additional context for strategy selection
            parallel_strategy: Strategy for distributing work across workers
            custom_iterations: Override for iterations per worker

        Returns:
            ParallelMousetrapResult with best result and comprehensive metrics
        """
        start_time = time.time()
        self.metrics["total_generations"] += 1

        logger.info(
            f"Starting parallel Mousetrap generation: strategy={parallel_strategy.value}, "
            f"workers={self.parallel_config.max_workers}"
        )

        # Check circuit breaker
        if await self._is_circuit_breaker_open():
            logger.warning("Circuit breaker is open, falling back to sequential generation")
            return await self._fallback_to_sequential(goal, strategy_context, custom_iterations)

        try:
            # Generate worker tasks
            worker_tasks = self._generate_worker_tasks(
                goal, strategy_context, parallel_strategy, custom_iterations
            )

            # Execute tasks in parallel
            worker_results = await self._execute_parallel_tasks(worker_tasks)

            # Aggregate results
            aggregated_result = self._aggregate_results(worker_results, goal, strategy_context)

            # Update metrics
            execution_time = time.time() - start_time
            self.metrics["parallel_generations"] += 1
            self.metrics["total_execution_time"] += execution_time

            # Reset circuit breaker on success
            self.circuit_breaker_failures = 0

            logger.info(
                f"Parallel Mousetrap generation completed: "
                f"best_score={aggregated_result.best_result.effectiveness_score:.3f}, "
                f"execution_time={execution_time:.2f}s"
            )

            return aggregated_result

        except Exception as e:
            logger.error(f"Parallel generation failed: {e}")
            await self._handle_circuit_breaker_failure()
            return await self._fallback_to_sequential(goal, strategy_context, custom_iterations)

    def _generate_worker_tasks(
        self,
        goal: str,
        strategy_context: str,
        parallel_strategy: ParallelStrategy,
        custom_iterations: int | None,
    ) -> list[WorkerTask]:
        """Generate tasks for parallel workers based on the selected strategy"""
        tasks = []
        base_iterations = custom_iterations or self.base_config.iterative_refinement_steps

        for worker_id in range(self.parallel_config.max_workers):
            # Create worker-specific configuration
            worker_config = MousetrapConfig(
                max_chain_length=self.base_config.max_chain_length,
                chaos_escalation_rate=self.base_config.chaos_escalation_rate,
                confidence_threshold=self.base_config.confidence_threshold,
                misdirection_probability=self.base_config.misdirection_probability,
                reasoning_complexity=self.base_config.reasoning_complexity,
                semantic_obfuscation_level=self.base_config.semantic_obfuscation_level,
                iterative_refinement_steps=base_iterations,
            )

            chaos_modifier = 0.0
            complexity_modifier = 0.0
            worker_strategy_context = strategy_context

            # Apply strategy-specific modifications
            if parallel_strategy == ParallelStrategy.DIVERSE_CHAOS:
                # Vary chaos levels across workers
                chaos_levels = [0.0, 0.2, 0.4, 0.6]
                chaos_modifier = chaos_levels[worker_id % len(chaos_levels)]
                worker_config.chaos_escalation_rate += chaos_modifier
                worker_config.semantic_obfuscation_level = min(
                    1.0, worker_config.semantic_obfuscation_level + chaos_modifier
                )

            elif parallel_strategy == ParallelStrategy.VARIED_COMPLEXITY:
                # Vary complexity levels
                complexity_modifiers = [-0.3, -0.1, 0.1, 0.3]
                complexity_modifier = complexity_modifiers[worker_id % len(complexity_modifiers)]
                worker_config.reasoning_complexity = max(
                    1, int(worker_config.reasoning_complexity * (1 + complexity_modifier))
                )
                worker_config.max_chain_length = max(
                    4, int(worker_config.max_chain_length * (1 + complexity_modifier))
                )

            elif parallel_strategy == ParallelStrategy.MULTI_STRATEGY:
                # Vary strategy contexts
                strategy_variations = [
                    "academic_research",
                    "creative_exploration",
                    "logical_analysis",
                    "philosophical_inquiry",
                ]
                worker_strategy_context = f"{strategy_context} {strategy_variations[worker_id % len(strategy_variations)]}"

            elif parallel_strategy == ParallelStrategy.HYBRID_APPROACH:
                # Combine multiple variations
                chaos_modifier = random.uniform(-0.2, 0.4)
                complexity_modifier = random.uniform(-0.2, 0.3)
                worker_config.chaos_escalation_rate += chaos_modifier
                worker_config.reasoning_complexity = max(
                    1, int(worker_config.reasoning_complexity * (1 + complexity_modifier))
                )

            task = WorkerTask(
                worker_id=worker_id,
                goal=goal,
                strategy_context=worker_strategy_context,
                config=worker_config,
                iterations=base_iterations,
                chaos_modifier=chaos_modifier,
                complexity_modifier=complexity_modifier,
            )

            tasks.append(task)

        return tasks

    async def _execute_parallel_tasks(self, tasks: list[WorkerTask]) -> list[WorkerResult]:
        """Execute worker tasks in parallel with timeout and error handling"""
        semaphore = asyncio.Semaphore(self.parallel_config.max_concurrent_chains)

        async def execute_single_task(task: WorkerTask) -> WorkerResult:
            async with semaphore:
                start_time = time.time()
                try:
                    # Create worker-specific generator
                    worker_generator = MousetrapGenerator(self.llm_adapter, task.config)

                    # Execute with timeout
                    result = await asyncio.wait_for(
                        worker_generator.generate_mousetrap_attack(
                            task.goal, task.strategy_context, task.iterations
                        ),
                        timeout=self.parallel_config.timeout_seconds,
                    )

                    execution_time = time.time() - start_time

                    # Early stopping check
                    if (
                        self.parallel_config.enable_early_stopping
                        and result.effectiveness_score >= self.parallel_config.early_stop_threshold
                    ):
                        logger.info(f"Early stop triggered for worker {task.worker_id}")
                        self.metrics["early_stops"] += 1

                    return WorkerResult(
                        worker_id=task.worker_id,
                        result=result,
                        execution_time=execution_time,
                        iterations_completed=task.iterations,
                    )

                except TimeoutError:
                    execution_time = time.time() - start_time
                    logger.warning(f"Worker {task.worker_id} timed out after {execution_time:.2f}s")
                    self.metrics["timeout_events"] += 1

                    return WorkerResult(
                        worker_id=task.worker_id,
                        result=None,
                        execution_time=execution_time,
                        error="timeout",
                        iterations_completed=0,
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Worker {task.worker_id} failed: {e}")

                    return WorkerResult(
                        worker_id=task.worker_id,
                        result=None,
                        execution_time=execution_time,
                        error=str(e),
                        iterations_completed=0,
                    )

        # Execute all tasks concurrently
        worker_results = await asyncio.gather(
            *[execute_single_task(task) for task in tasks], return_exceptions=True
        )

        # Handle any exceptions from gather
        processed_results = []
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                processed_results.append(
                    WorkerResult(worker_id=i, result=None, execution_time=0.0, error=str(result))
                )
            else:
                processed_results.append(result)

        # Update worker utilization metrics
        for result in processed_results:
            worker_id = f"worker_{result.worker_id}"
            if worker_id not in self.metrics["worker_utilization"]:
                self.metrics["worker_utilization"][worker_id] = {
                    "total_tasks": 0,
                    "successful_tasks": 0,
                    "total_time": 0.0,
                }

            worker_metrics = self.metrics["worker_utilization"][worker_id]
            worker_metrics["total_tasks"] += 1
            worker_metrics["total_time"] += result.execution_time
            if result.result is not None:
                worker_metrics["successful_tasks"] += 1

        return processed_results

    def _aggregate_results(
        self, worker_results: list[WorkerResult], goal: str, strategy_context: str
    ) -> ParallelMousetrapResult:
        """Aggregate worker results using the configured strategy"""
        # Filter successful results
        successful_results = [
            wr.result
            for wr in worker_results
            if wr.result is not None
            and wr.result.effectiveness_score >= self.parallel_config.min_effectiveness_threshold
        ]

        # If no successful results, use the best attempt
        if not successful_results:
            all_results = [wr.result for wr in worker_results if wr.result is not None]
            if all_results:
                successful_results = [max(all_results, key=lambda r: r.effectiveness_score)]
            else:
                # Create fallback result
                from ..mousetrap import MousetrapResult

                fallback_result = MousetrapResult(
                    final_prompt=goal,
                    reasoning_chain=[],
                    chaos_progression=[],
                    effectiveness_score=0.0,
                    extraction_success=False,
                    intermediate_responses=[],
                )
                successful_results = [fallback_result]

        # Select best result based on aggregation strategy
        if self.parallel_config.result_aggregation_strategy == "best_of_all":
            best_result = max(successful_results, key=lambda r: r.effectiveness_score)
        elif self.parallel_config.result_aggregation_strategy == "consensus":
            # Use result that appears most similar to others (simplified consensus)
            best_result = (
                successful_results[len(successful_results) // 2]
                if successful_results
                else successful_results[0]
            )
        else:  # diversity - prefer different approaches
            best_result = max(successful_results, key=lambda r: len(r.reasoning_chain))

        # Generate comprehensive metrics
        generation_stats = {
            "total_workers": len(worker_results),
            "successful_workers": len([wr for wr in worker_results if wr.result is not None]),
            "failed_workers": len([wr for wr in worker_results if wr.result is None]),
            "timeout_workers": len([wr for wr in worker_results if wr.error == "timeout"]),
            "average_execution_time": sum(wr.execution_time for wr in worker_results)
            / len(worker_results),
            "best_effectiveness_score": best_result.effectiveness_score,
            "effectiveness_scores": [r.effectiveness_score for r in successful_results],
            "total_iterations": sum(wr.iterations_completed for wr in worker_results),
        }

        parallel_metrics = {
            "worker_efficiency": len(successful_results) / len(worker_results),
            "parallel_speedup": self._calculate_parallel_speedup(worker_results),
            "resource_utilization": self._calculate_resource_utilization(worker_results),
            "result_diversity": self._calculate_result_diversity(successful_results),
        }

        convergence_analysis = {
            "early_convergence": any(
                wr.result
                and wr.result.effectiveness_score >= self.parallel_config.early_stop_threshold
                for wr in worker_results
            ),
            "score_variance": self._calculate_score_variance(successful_results),
            "chain_length_variance": self._calculate_chain_length_variance(successful_results),
        }

        return ParallelMousetrapResult(
            best_result=best_result,
            all_results=successful_results,
            generation_stats=generation_stats,
            parallel_metrics=parallel_metrics,
            convergence_analysis=convergence_analysis,
        )

    def _calculate_parallel_speedup(self, worker_results: list[WorkerResult]) -> float:
        """Calculate theoretical parallel speedup"""
        if not worker_results:
            return 1.0

        max_time = max(wr.execution_time for wr in worker_results)
        total_time = sum(wr.execution_time for wr in worker_results)

        return total_time / max_time if max_time > 0 else 1.0

    def _calculate_resource_utilization(self, worker_results: list[WorkerResult]) -> float:
        """Calculate resource utilization efficiency"""
        successful_workers = len([wr for wr in worker_results if wr.result is not None])
        return successful_workers / len(worker_results) if worker_results else 0.0

    def _calculate_result_diversity(self, results: list[MousetrapResult]) -> float:
        """Calculate diversity of results based on chain characteristics"""
        if len(results) <= 1:
            return 0.0

        chain_lengths = [len(r.reasoning_chain) for r in results]
        effectiveness_scores = [r.effectiveness_score for r in results]

        # Calculate coefficient of variation as diversity measure
        chain_variance = sum(
            (x - sum(chain_lengths) / len(chain_lengths)) ** 2 for x in chain_lengths
        ) / len(chain_lengths)
        score_variance = sum(
            (x - sum(effectiveness_scores) / len(effectiveness_scores)) ** 2
            for x in effectiveness_scores
        ) / len(effectiveness_scores)

        return (chain_variance + score_variance) / 2

    def _calculate_score_variance(self, results: list[MousetrapResult]) -> float:
        """Calculate variance in effectiveness scores"""
        if len(results) <= 1:
            return 0.0

        scores = [r.effectiveness_score for r in results]
        mean_score = sum(scores) / len(scores)
        return sum((score - mean_score) ** 2 for score in scores) / len(scores)

    def _calculate_chain_length_variance(self, results: list[MousetrapResult]) -> float:
        """Calculate variance in chain lengths"""
        if len(results) <= 1:
            return 0.0

        lengths = [len(r.reasoning_chain) for r in results]
        mean_length = sum(lengths) / len(lengths)
        return sum((length - mean_length) ** 2 for length in lengths) / len(lengths)

    async def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            if time.time() - self.last_failure_time > self.circuit_breaker_recovery_time:
                # Reset circuit breaker
                self.circuit_breaker_failures = 0
                logger.info("Circuit breaker reset")
                return False
            return True
        return False

    async def _handle_circuit_breaker_failure(self):
        """Handle circuit breaker failure"""
        self.circuit_breaker_failures += 1
        self.last_failure_time = time.time()
        logger.warning(f"Circuit breaker failure count: {self.circuit_breaker_failures}")

    async def _fallback_to_sequential(
        self, goal: str, strategy_context: str, custom_iterations: int | None
    ) -> ParallelMousetrapResult:
        """Fallback to sequential generation when parallel fails"""
        logger.info("Falling back to sequential Mousetrap generation")
        self.metrics["sequential_generations"] += 1

        try:
            result = await self.base_generator.generate_mousetrap_attack(
                goal, strategy_context, custom_iterations
            )

            return ParallelMousetrapResult(
                best_result=result,
                all_results=[result],
                generation_stats={"fallback_mode": True},
                parallel_metrics={"parallel_speedup": 1.0},
                convergence_analysis={"sequential_fallback": True},
            )
        except Exception as e:
            logger.error(f"Sequential fallback also failed: {e}")
            # Return minimal result
            from ..mousetrap import MousetrapResult

            fallback_result = MousetrapResult(
                final_prompt=goal,
                reasoning_chain=[],
                chaos_progression=[],
                effectiveness_score=0.0,
                extraction_success=False,
                intermediate_responses=[],
            )

            return ParallelMousetrapResult(
                best_result=fallback_result,
                all_results=[fallback_result],
                generation_stats={"fallback_failed": True},
                parallel_metrics={"parallel_speedup": 0.0},
                convergence_analysis={"total_failure": True},
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive parallel generation metrics"""
        return {
            **self.metrics,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "circuit_breaker_open": self.circuit_breaker_failures >= self.circuit_breaker_threshold,
            "parallel_config": {
                "max_workers": self.parallel_config.max_workers,
                "max_concurrent_chains": self.parallel_config.max_concurrent_chains,
                "timeout_seconds": self.parallel_config.timeout_seconds,
            },
        }

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "total_generations": 0,
            "parallel_generations": 0,
            "sequential_generations": 0,
            "total_execution_time": 0.0,
            "worker_utilization": {},
            "timeout_events": 0,
            "early_stops": 0,
        }
        logger.info("Parallel Mousetrap metrics reset")

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("ParallelMousetrapGenerator cleaned up")


# Factory function for easy instantiation
def create_parallel_mousetrap_generator(
    llm_adapter: ChimeraLLMAdapter, **kwargs
) -> ParallelMousetrapGenerator:
    """Create a parallel Mousetrap generator with optimized configuration"""
    return ParallelMousetrapGenerator(llm_adapter, **kwargs)
